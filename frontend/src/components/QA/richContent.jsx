import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Button, Spin, Typography } from "antd";
import { graphFromObservation } from "./kgGraphData";
export { graphFromObservation };
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
        await mermaid.parse(code);            // 先語法預檢，失敗不進 render
        const { svg: out } = await mermaid.render(`mmd-${++_mermaidSeq}`, code);
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

const parseTableEl = (tableEl) => {
  if (!tableEl) return null;
  const headers = [...tableEl.querySelectorAll("thead th")].map((th) => th.textContent.trim());
  const rows = [...tableEl.querySelectorAll("tbody tr")].map((tr) =>
    [...tr.querySelectorAll("td")].map((td) => td.textContent.trim()));
  if (headers.length < 2 || rows.length < 2) return null;
  // 找數值欄：該欄至少 2/3 的儲存格能抽出數字（容忍單位字尾，如 "15 m/s"、"-51°C"）
  const numOf = (s) => {
    const m = String(s).replace(/,/g, "").match(/-?\d+(?:\.\d+)?/);
    return m ? parseFloat(m[0]) : null;
  };
  const numericCols = [];
  for (let c = 1; c < headers.length; c += 1) {
    const nums = rows.map((r) => numOf(r[c]));
    if (nums.filter((v) => v !== null).length >= Math.ceil(rows.length * 2 / 3)) {
      numericCols.push({ index: c, values: nums });
    }
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
    <div ref={wrapRef} style={{ width: "100%", height: 260, border: "1px solid #f0f0f0",
                                borderRadius: 6, overflow: "hidden", background: "#fff" }}>
      {w > 0 && (
        <Suspense fallback={<Spin style={{ margin: 24 }} />}>
          <ForceGraph2D
            graphData={data}
            width={w}
            height={258}
            nodeLabel="id"
            nodeColor={(n) => (n.main ? "#fa8c16" : "#1677ff")}
            nodeRelSize={5}
            linkColor={(l) => REL_COLORS[l.rel] || "#bfbfbf"}
            linkDirectionalArrowLength={4}
            linkLabel={(l) => l.rel || ""}
            cooldownTicks={90}
          />
        </Suspense>
      )}
      <Text type="secondary" style={{ fontSize: 11, position: "relative", top: -22, left: 8 }}>
        知識圖譜關聯（可拖曳）
      </Text>
    </div>
  );
};
