import { useEffect, useState } from "react";
import {
  Button, Card, Col, Input, InputNumber, Modal, Row, Select, Slider,
  Space, Spin, Table, Tag, Tooltip, Typography, message,
} from "antd";
import { EditOutlined, FileTextOutlined, SearchOutlined, ThunderboltOutlined } from "@ant-design/icons";
import { useNavigate } from "react-router-dom";
import apiClient from "../services/api";
import AppLayout from "../components/Layout/AppLayout";

const { TextArea } = Input;

const scoreColor = (score) => {
  if (score >= 0.8) return "#52c41a";
  if (score >= 0.6) return "#1677ff";
  if (score >= 0.4) return "#faad14";
  return "#ff4d4f";
};

const VectorSearchTestPage = () => {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [minScore, setMinScore] = useState(0.3);
  const [documentId, setDocumentId] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [expandedId, setExpandedId] = useState(null);

  // 編輯向量塊
  const [editingChunk, setEditingChunk] = useState(null);
  const [editText, setEditText] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    apiClient.get("documents/?page=1&page_size=200").then((r) => {
      setDocuments(r.data.items ?? []);
    }).catch(() => {});
  }, []);

  const handleSearch = async () => {
    if (!query.trim()) { message.warning("請輸入查詢關鍵字"); return; }
    try {
      setLoading(true);
      setResults(null);
      const resp = await apiClient.post("vector-search/test", {
        query: query.trim(),
        top_k: topK,
        min_score: minScore,
        document_id: documentId || null,
      });
      setResults(resp.data);
    } catch (error) {
      message.error(error.response?.data?.detail ?? "查詢失敗");
    } finally {
      setLoading(false);
    }
  };

  const handleSaveChunk = async () => {
    if (!editingChunk || !editText.trim()) return;
    try {
      setSaving(true);
      await apiClient.put(
        `documents/${editingChunk.document_id}/chunks/${editingChunk.chunk_id}`,
        { text: editText }
      );
      message.success("向量塊已更新並重新向量化");
      setEditingChunk(null);
      // 重新搜尋以反映最新結果
      handleSearch();
    } catch (error) {
      message.error(error.response?.data?.detail ?? "更新失敗");
    } finally {
      setSaving(false);
    }
  };

  const columns = [
    {
      title: "排名",
      dataIndex: "rank",
      width: 60,
      render: (v) => <Typography.Text strong>#{v}</Typography.Text>,
    },
    {
      title: "相似度",
      dataIndex: "score",
      width: 90,
      render: (v) => (
        <Tag color={scoreColor(v)} style={{ fontWeight: 700, fontSize: 13 }}>
          {(v * 100).toFixed(1)}%
        </Tag>
      ),
    },
    {
      title: "文件",
      dataIndex: "document_title",
      width: 200,
      ellipsis: true,
    },
    {
      title: "頁碼",
      dataIndex: "page",
      width: 70,
      render: (v) => v ?? "-",
    },
    {
      title: "內容",
      dataIndex: "text",
      render: (text, record) => (
        <div>
          <Typography.Text
            style={{ cursor: "pointer", whiteSpace: "pre-wrap", wordBreak: "break-word" }}
            onClick={() => setExpandedId(expandedId === record.chunk_id ? null : record.chunk_id)}
          >
            {expandedId === record.chunk_id
              ? text
              : `${text.slice(0, 200)}${text.length > 200 ? "…" : ""}`}
          </Typography.Text>
          {text.length > 200 && (
            <Typography.Link
              style={{ display: "block", marginTop: 4, fontSize: 12 }}
              onClick={() => setExpandedId(expandedId === record.chunk_id ? null : record.chunk_id)}
            >
              {expandedId === record.chunk_id ? "收起" : "展開全文"}
            </Typography.Link>
          )}
        </div>
      ),
    },
    {
      title: "操作",
      width: 100,
      render: (_, record) => (
        <Space>
          <Tooltip title="前往文件">
            <Button
              size="small"
              icon={<FileTextOutlined />}
              onClick={() => navigate(`/documents/${record.document_id}`)}
            />
          </Tooltip>
          <Tooltip title="編輯此塊">
            <Button
              size="small"
              icon={<EditOutlined />}
              onClick={() => { setEditingChunk(record); setEditText(record.text); }}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  return (
    <AppLayout>
    <div style={{ padding: 24 }}>
      <Typography.Title level={4} style={{ marginBottom: 20 }}>
        <ThunderboltOutlined style={{ marginRight: 8, color: "#1677ff" }} />
        向量查詢測試台
      </Typography.Title>

      <Card style={{ marginBottom: 20 }}>
        <Row gutter={[16, 16]}>
          <Col span={24}>
            <Typography.Text strong>查詢內容：</Typography.Text>
            <TextArea
              rows={3}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="輸入查詢關鍵字或句子，測試向量搜尋的召回結果..."
              onPressEnter={(e) => { if (e.ctrlKey) handleSearch(); }}
              style={{ marginTop: 6 }}
            />
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>Ctrl + Enter 送出</Typography.Text>
          </Col>

          <Col xs={24} sm={8}>
            <Typography.Text strong>篩選文件：</Typography.Text>
            <Select
              allowClear
              placeholder="全部文件"
              value={documentId}
              onChange={setDocumentId}
              style={{ width: "100%", marginTop: 6 }}
              showSearch
              filterOption={(input, option) =>
                (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
              }
              options={documents.map((d) => ({ value: d.id, label: d.title }))}
            />
          </Col>

          <Col xs={12} sm={4}>
            <Typography.Text strong>Top-K：</Typography.Text>
            <InputNumber
              min={1} max={20}
              value={topK}
              onChange={setTopK}
              style={{ width: "100%", marginTop: 6 }}
            />
          </Col>

          <Col xs={12} sm={8}>
            <Typography.Text strong>最低相似度門檻：{(minScore * 100).toFixed(0)}%</Typography.Text>
            <Slider
              min={0} max={1} step={0.05}
              value={minScore}
              onChange={setMinScore}
              marks={{ 0: "0", 0.3: "0.3", 0.6: "0.6", 1: "1" }}
              style={{ marginTop: 10 }}
            />
          </Col>

          <Col xs={24} sm={4} style={{ display: "flex", alignItems: "flex-end" }}>
            <Button
              type="primary"
              icon={<SearchOutlined />}
              onClick={handleSearch}
              loading={loading}
              style={{ width: "100%" }}
            >
              查詢
            </Button>
          </Col>
        </Row>
      </Card>

      {loading && (
        <div style={{ textAlign: "center", padding: 40 }}>
          <Spin size="large" />
          <div style={{ marginTop: 12, color: "#888" }}>正在計算向量並搜尋...</div>
        </div>
      )}

      {results && !loading && (
        <Card
          title={
            <Space>
              <span>查詢結果</span>
              <Tag color="blue">{results.results.length} 筆</Tag>
              <Typography.Text type="secondary" style={{ fontSize: 13 }}>
                耗時 {results.elapsed_ms} ms
              </Typography.Text>
            </Space>
          }
        >
          {results.results.length === 0 ? (
            <Typography.Text type="secondary">
              未找到符合條件的向量塊，請嘗試降低相似度門檻或更換查詢內容。
            </Typography.Text>
          ) : (
            <Table
              dataSource={results.results}
              rowKey="chunk_id"
              columns={columns}
              size="small"
              pagination={false}
            />
          )}
        </Card>
      )}

      {/* 編輯向量塊 Modal */}
      <Modal
        title="編輯向量塊"
        open={!!editingChunk}
        onOk={handleSaveChunk}
        onCancel={() => setEditingChunk(null)}
        confirmLoading={saving}
        okText="儲存並重新向量化"
        cancelText="取消"
        width={800}
      >
        {editingChunk && (
          <>
            <Typography.Text type="secondary" style={{ display: "block", marginBottom: 8 }}>
              文件：{editingChunk.document_title}｜第 {editingChunk.page ?? "?"} 頁｜{editText.length} 字
            </Typography.Text>
            <Input.TextArea
              rows={14}
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              style={{ fontFamily: "monospace", fontSize: 13 }}
            />
          </>
        )}
      </Modal>
    </div>
    </AppLayout>
  );
};

export default VectorSearchTestPage;
