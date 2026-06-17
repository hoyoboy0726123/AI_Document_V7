import { useEffect, useState } from "react";
import {
  Alert, Badge, Button, Card, Col, Row, Spin, Statistic, Table, Tag, Tooltip, Typography, message,
} from "antd";
import { HeartOutlined, ReloadOutlined, WarningOutlined } from "@ant-design/icons";
import { useNavigate } from "react-router-dom";
import apiClient from "../services/api";
import AppLayout from "../components/Layout/AppLayout";

const VectorHealthPage = () => {
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchHealth = async () => {
    try {
      setLoading(true);
      const resp = await apiClient.get("vector-search/health");
      setData(resp.data);
    } catch (error) {
      message.error(error.response?.data?.detail ?? "載入失敗");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchHealth(); }, []);

  const abnormalColumns = [
    {
      title: "原因",
      dataIndex: "reason",
      width: 90,
      render: (v) => (
        <Tag color={v === "too_short" ? "red" : "orange"}>
          {v === "too_short" ? "過短" : "過長"}
        </Tag>
      ),
    },
    {
      title: "字數",
      dataIndex: "char_count",
      width: 70,
    },
    {
      title: "文件",
      dataIndex: "document_title",
      width: 180,
      ellipsis: true,
    },
    {
      title: "頁碼",
      dataIndex: "page",
      width: 70,
      render: (v) => v ?? "-",
    },
    {
      title: "內容預覽",
      dataIndex: "text_preview",
      render: (v) => (
        <Typography.Text style={{ fontFamily: "monospace", fontSize: 12 }}>
          {v}
        </Typography.Text>
      ),
    },
    {
      title: "操作",
      width: 100,
      render: (_, record) => (
        <Tooltip title="前往文件向量塊管理">
          <Button
            size="small"
            onClick={() => navigate(`/documents/${record.document_id}`)}
          >
            前往文件
          </Button>
        </Tooltip>
      ),
    },
  ];

  const docStatsColumns = [
    {
      title: "文件",
      dataIndex: "document_title",
      ellipsis: true,
      render: (v, record) => (
        <Typography.Link onClick={() => navigate(`/documents/${record.document_id}`)}>
          {v}
        </Typography.Link>
      ),
    },
    { title: "塊數", dataIndex: "chunk_count", width: 80 },
    { title: "總字數", dataIndex: "total_chars", width: 90 },
    { title: "平均字數", dataIndex: "avg_chars", width: 90 },
    {
      title: "空向量",
      dataIndex: "empty_embedding_count",
      width: 90,
      render: (v) => (
        v > 0 ? <Tag color="red">{v}</Tag> : <Tag color="green">0</Tag>
      ),
    },
  ];

  return (
    <AppLayout>
      <div style={{ padding: 24 }}>
        <div style={{ display: "flex", alignItems: "center", marginBottom: 20, gap: 12 }}>
          <Typography.Title level={4} style={{ margin: 0 }}>
            <HeartOutlined style={{ marginRight: 8, color: "#ff4d4f" }} />
            向量庫健康儀表板
          </Typography.Title>
          <Button icon={<ReloadOutlined />} onClick={fetchHealth} loading={loading}>
            重新整理
          </Button>
        </div>

        {loading && !data && (
          <div style={{ textAlign: "center", padding: 60 }}>
            <Spin size="large" />
          </div>
        )}

        {data && (
          <>
            {/* 總覽統計 */}
            <Row gutter={16} style={{ marginBottom: 24 }}>
              <Col xs={12} sm={6}><Card><Statistic title="向量塊總數" value={data.total_chunks} /></Card></Col>
              <Col xs={12} sm={6}><Card><Statistic title="文件數量" value={data.total_documents} /></Card></Col>
              <Col xs={12} sm={6}><Card><Statistic title="總字數" value={data.total_chars} /></Card></Col>
              <Col xs={12} sm={6}><Card><Statistic title="平均每塊字數" value={data.avg_chars_per_chunk} /></Card></Col>
            </Row>

            {data.empty_embedding_count > 0 && (
              <Alert
                type="error"
                showIcon
                icon={<WarningOutlined />}
                message={`發現 ${data.empty_embedding_count} 個空向量塊，這些塊無法被搜尋到，請重新向量化相關文件。`}
                style={{ marginBottom: 20 }}
              />
            )}

            {/* 異常向量塊 */}
            <Card
              title={
                <span>
                  異常向量塊
                  {data.abnormal_chunks.length > 0 && (
                    <Badge count={data.abnormal_chunks.length} style={{ marginLeft: 8, backgroundColor: "#ff4d4f" }} />
                  )}
                </span>
              }
              style={{ marginBottom: 20 }}
            >
              {data.abnormal_chunks.length === 0 ? (
                <Typography.Text type="secondary">無異常向量塊</Typography.Text>
              ) : (
                <Table
                  dataSource={data.abnormal_chunks}
                  rowKey="chunk_id"
                  columns={abnormalColumns}
                  size="small"
                  pagination={{ pageSize: 20 }}
                />
              )}
            </Card>

            {/* 各文件統計 */}
            <Card title="各文件向量塊統計">
              <Table
                dataSource={data.document_stats}
                rowKey="document_id"
                columns={docStatsColumns}
                size="small"
                pagination={{ pageSize: 20 }}
              />
            </Card>
          </>
        )}
      </div>
    </AppLayout>
  );
};

export default VectorHealthPage;
