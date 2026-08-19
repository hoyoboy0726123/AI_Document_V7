import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Button, Spin, Typography } from "antd";
import { BarChartOutlined } from "@ant-design/icons";

const { Text } = Typography;

/**
 * 答案富內容三件套：Mermaid 圖、表格轉圖表、KG 關係小圖。
 *
 * 共同原則：
 * - 重依賴（mermaid/echarts/force-graph）一律動態載入 —— 沒用到的對話不付大小
 * - LLM 產物必須有失敗回退：mermaid 語法炸掉就顯示原始碼，不能讓整則答案白屏
 * - 串流中不渲染：半截的 mermaid/表格每個 token 都會觸發 parse 失敗又閃爍，
 *   用 debounce「內容穩定 500ms 才畫」一招同時解掉串流與抖動
 */

// ── Mermaid ────────────────────────────────────────────────────────────────

let _mermaidP = null;
const loadMermaid = () => {
  if (!_mermaidP) {
    _mermaidP = import("mermaid").then((m) => {
      const mermaid = m.default;
      mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: "neutral" });
      return mermaid;
    });
  }
  return _mermaidP;
};

let _mermaidSeq = 0;

// LLM 生成的 mermaid 最常見的死法：節點標籤裡帶 ASCII 括號/百分比/度數
// （「100 mm (4 in.)」「±2.5 %」），未加引號的標籤遇到 () 就是語法錯誤。
// 確定性硬化：把 [标签]、{标签}、|邊標籤| 一律補上雙引號 —— 引號內幾乎
// 什麼字元都合法。已含引號的標籤跳過，不會二次包裝。
const hardenMermaid = (code) =>
  code
    .replace(/\[([^[\]"\n]+)\]/g, (m, inner) => `["${inner.replace(/"/g, "'")}"]`)
    .replace(/\{([^{}"\n]+)\}/g, (m, inner) => `{"${inner.replace(/"/g, "'")}"}`)
    .replace(/\|([^|"\n]+)\|/g, (m, inner) => `|"${inner.replace(/"/g, "'")}"|`);

export const MermaidBlock = ({ code }) => {
  const [svg, setSvg] = useState("");
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let dead = false;
    setFailed(false);
    // 內容穩定 500ms 才 parse：串流中每個 token 都是半截語法
    const t = setTimeout(async () => {
      try {
        const mermaid = await loadMermaid();
        let src = hardenMermaid(code);
        try {
          await mermaid.parse(src);           // 先語法預檢，失敗不進 render
        } catch {
          src = code;                          // 硬化反而弄壞冷門語法時退回原文再試
          await mermaid.parse(src);
        }
        const { svg: out } = await mermaid.render(`mmd-${++_mermaidSeq}`, src);
        if (!dead) setSvg(out);
      } catch {
        if (!dead) { setFailed(true); setSvg(""); }
      }
    }, 500);
    return () => { dead = true; clearTimeout(t); };
  }, [code]);

  if (failed) {
    return (
      <pre style={{ background: "#fafafa", padding: 8, borderRadius: 6, overflowX: "auto" }}>
        <code>{code}</code>
      </pre>
    );
  }
  if (!svg) return <Spin size="small" style={{ margin: 8 }} />;
  // mermaid securityLevel=strict 已消毒；svg 直接內嵌
  return (
    <div
      style={{ overflowX: "auto", margin: "8px 0" }}
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
};

// ── 表格轉圖表（opt-in 按鈕，確定性解析，零 LLM） ──────────────────────────

// 整格必須「就是一個數值」（可帶±、範圍前綴與短單位字尾）才算數據。
// 實測教訓：說明欄散文裡的【來源6】、±2.5% 會被寬鬆抽取撈成數據，
// 畫出 6,6,6,2.5 的長條圖 —— 純垃圾。散文嵌數字一律不算。
const _CELL_NUM_RE = /^[-+±≦≧<>約~\s]*(-?\d[\d,]*(?:\.\d+)?)\s*(?:[a-zA-Z°%µ/·.-]{0,10}|小時|分鐘|秒|次|天|個|項)?\s*$/;
const _UNIT_HINT_RE = /°C|°F|℃|℉|%|m\/s|km\/h|Hz|kHz|dB|kPa|MPa|psi|\bg\b|kg|mg|mm|cm|\bm\b|in\b|ft\b|min|hr|hours?|sec|ms\b|[VAW]\b|小時|分鐘|秒|溫度|濕度|速度|壓力|高度|重量|時間|頻率|加速度/i;

const parseTableEl = (tableEl) => {
  if (!tableEl) return null;
  const headers = [...tableEl.querySelectorAll("thead th")].map((th) => th.textContent.trim());
  const rows = [...tableEl.querySelectorAll("tbody tr")].map((tr) =>
    [...tr.querySelectorAll("td")].map((td) => td.textContent.trim()));
  if (headers.length < 2 || rows.length < 3) return null;
  const numOf = (s) => {
    const cell = String(s || "").trim();
    if (!cell || cell.length > 20) return null;          // 長格＝散文，不是數值
    const m = cell.match(_CELL_NUM_RE);
    return m ? parseFloat(m[1].replace(/,/g, "")) : null;
  };
  const numericCols = [];
  for (let c = 1; c < headers.length; c += 1) {
    const cells = rows.map((r) => r[c] ?? "");
    const nums = cells.map(numOf);
    const valid = nums.filter((v) => v !== null);
    if (valid.length < Math.max(3, Math.ceil(rows.length * 2 / 3))) continue;
    // 代號/流水號排除：全整數且近乎等差（01,02,03… / 頁碼）不是量值
    const ints = valid.every((v) => Number.isInteger(v));
    if (ints) {
      const sorted = [...valid].sort((a, b) => a - b);
      const seqLike = sorted.every((v, i) => i === 0 || v - sorted[i - 1] <= 2);
      const zeroPadded = cells.some((s) => /^0\d/.test(s.trim()));
      if (seqLike || zeroPadded) continue;
    }
    // 量值證據：表頭或儲存格帶單位；沒有單位就要求有小數/負數（避免任意整數欄）
    const hasUnit = _UNIT_HINT_RE.test(headers[c] || "") || cells.some((s) => _UNIT_HINT_RE.test(s));
    const hasQuantityShape = valid.some((v) => !Number.isInteger(v) || v < 0);
    if (!hasUnit && !hasQuantityShape) continue;
    numericCols.push({ index: c, values: nums });
  }
  if (!numericCols.length) return null;
  return {
    labels: rows.map((r) => r[0] || ""),
    series: numericCols.map((c) => ({ name: headers[c.index] || `欄${c.index + 1}`, data: c.values })),
  };
};

const EChart = ({ option, height = 300 }) => {
  const ref = useRef(null);
  useEffect(() => {
    let chart = null;
    let dead = false;
    (async () => {
      // 按需註冊而非 full bundle（gzip 差 200KB+）
      const [{ init, use }, charts, components, renderers] = await Promise.all([
        import("echarts/core"), import("echarts/charts"),
        import("echarts/components"), import("echarts/renderers"),
      ]);
      use([charts.BarChart, charts.LineChart, components.GridComponent,
           components.TooltipComponent, components.LegendComponent, renderers.CanvasRenderer]);
      if (dead || !ref.current) return;
      chart = init(ref.current);
      chart.setOption(option);
    })();
    const onResize = () => chart && chart.resize();
    window.addEventListener("resize", onResize);
    return () => { dead = true; window.removeEventListener("resize", onResize); chart && chart.dispose(); };
  }, [option]);
  return <div ref={ref} style={{ width: "100%", height }} />;
};

export const TableWithChartButton = ({ children, tableStyle }) => {
  const wrapRef = useRef(null);
  const [chartData, setChartData] = useState(null);
  const [canChart, setCanChart] = useState(false);

  useEffect(() => {
    // 渲染完成後從 DOM 讀表格判斷可否轉圖 —— 確定性、不需要 LLM 參與
    const t = setTimeout(() => {
      setCanChart(Boolean(parseTableEl(wrapRef.current?.querySelector("table"))));
    }, 500);
    return () => clearTimeout(t);
  }, [children]);

  const option = useMemo(() => {
    if (!chartData) return null;
    return {
      tooltip: { trigger: "axis" },
      legend: chartData.series.length > 1 ? { top: 0 } : undefined,
      grid: { left: 48, right: 16, top: chartData.series.length > 1 ? 32 : 16, bottom: 56 },
      xAxis: { type: "category", data: chartData.labels,
               axisLabel: { rotate: chartData.labels.some((l) => l.length > 6) ? 30 : 0 } },
      yAxis: { type: "value" },
      series: chartData.series.map((s) => ({ name: s.name, type: "bar", data: s.data })),
    };
  }, [chartData]);

  return (
    <div ref={wrapRef}>
      <div style={{ overflowX: "auto" }}>
        <table style={tableStyle}>{children}</table>
      </div>
      {canChart && !chartData && (
        <Button size="small" icon={<BarChartOutlined />} style={{ marginTop: 4 }}
                onClick={() => setChartData(parseTableEl(wrapRef.current?.querySelector("table")))}>
          轉圖表
        </Button>
      )}
      {chartData && option && (
        <div style={{ marginTop: 8 }}>
          <EChart option={option} />
          <Button size="small" type="text" onClick={() => setChartData(null)}>收起圖表</Button>
        </div>
      )}
    </div>
  );
};

// ── KG 關係小圖（Agent 圖譜查詢結果視覺化） ────────────────────────────────

const ForceGraph2D = lazy(() => import("react-force-graph-2d"));

const REL_COLORS = {
  references: "#1677ff", supersedes: "#fa541c", defines: "#52c41a",
  requires: "#722ed1", derives_from: "#13c2c2", contains: "#8c8c8c", part_of: "#8c8c8c",
};

export const MiniKgGraph = ({ data }) => {
  const wrapRef = useRef(null);
  const fgRef = useRef(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return undefined;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    setW(el.clientWidth);
    return () => ro.disconnect();
  }, []);
  return (
    <div ref={wrapRef} style={{ width: "100%", height: 300, border: "1px solid #f0f0f0",
                                borderRadius: 6, overflow: "hidden", background: "#fff" }}>
      {w > 0 && (
        <Suspense fallback={<Spin style={{ margin: 24 }} />}>
          <ForceGraph2D
            ref={fgRef}
            graphData={data}
            width={w}
            height={298}
            nodeLabel="id"
            nodeRelSize={4}
            // 標籤直接畫在 canvas 上 —— 只靠 hover 提示的話，整張圖就是
            // 一團無資訊量的藍點（實測使用者直接回報「不正常」）
            nodeCanvasObject={(node, ctx, globalScale) => {
              const r = node.main ? 6 : 4;
              ctx.beginPath();
              ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
              ctx.fillStyle = node.main ? "#fa8c16" : "#1677ff";
              ctx.fill();
              const label = String(node.id).length > 22
                ? `${String(node.id).slice(0, 21)}…` : String(node.id);
              const fs = Math.max(10 / globalScale, 2.6);
              ctx.font = `${node.main ? "bold " : ""}${fs}px sans-serif`;
              ctx.textAlign = "center";
              ctx.textBaseline = "top";
              ctx.fillStyle = "#333";
              ctx.fillText(label, node.x, node.y + r + 1.5);
            }}
            nodePointerAreaPaint={(node, color, ctx) => {
              ctx.beginPath();
              ctx.arc(node.x, node.y, 8, 0, 2 * Math.PI);
              ctx.fillStyle = color;
              ctx.fill();
            }}
            linkColor={(l) => REL_COLORS[l.rel] || "#bfbfbf"}
            linkDirectionalArrowLength={3.5}
            linkLabel={(l) => l.rel || ""}
            cooldownTicks={90}
            // 佈局收斂後自動縮放到全圖可見 —— 沒有這個，版本鏈長尾直接跑出視窗
            onEngineStop={() => fgRef.current?.zoomToFit(300, 28)}
          />
        </Suspense>
      )}
      <Text type="secondary" style={{ fontSize: 11, position: "relative", top: -22, left: 8 }}>
        知識圖譜關聯（可拖曳／滾輪縮放）
      </Text>
    </div>
  );
};
