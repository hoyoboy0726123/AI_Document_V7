import { useEffect, useState } from "react";
import { Button, Divider, Input, List, Modal, Select, Space, Tag, Typography } from "antd";
import { EditOutlined } from "@ant-design/icons";

const { Paragraph, Text, Title } = Typography;

const AISuggestionModal = ({
  open,
  suggestion,
  suggestedMetadata,
  segments,
  onApply,
  onClose,
}) => {
  // 可編輯本地狀態
  const [editSummary, setEditSummary] = useState("");
  const [editClassification, setEditClassification] = useState("");
  const [editKeywords, setEditKeywords] = useState([]);

  // 每次 Modal 開啟時重設為最新的建議內容
  useEffect(() => {
    if (open && suggestion) {
      setEditSummary(suggestion.summary ?? "");
      setEditClassification(suggestion.classification ?? "");
      const combined = Array.from(
        new Set([
          ...(suggestion.keywords ?? []),
          ...(suggestedMetadata?.keywords ?? []),
        ])
      );
      setEditKeywords(combined);
    }
  }, [open, suggestion, suggestedMetadata]);

  const hasSuggestion =
    suggestion &&
    (suggestion.summary ||
      suggestion.classification ||
      suggestion.project ||
      (suggestion.keywords && suggestion.keywords.length > 0));

  const handleApply = () => {
    if (!hasSuggestion) { onClose(); return; }
    onApply({
      ...suggestion,
      summary: editSummary,
      classification: editClassification || suggestion.classification,
      keywords: editKeywords,
    });
  };

  return (
    <Modal
      open={open}
      title={
        <Space>
          <EditOutlined style={{ color: "#1677ff" }} />
          AI 智慧建議（可在套用前修改）
        </Space>
      }
      onCancel={onClose}
      width={860}
      footer={[
        <Button key="close" onClick={onClose}>
          關閉
        </Button>,
        <Button key="apply" type="primary" onClick={handleApply} disabled={!hasSuggestion}>
          套用建議
        </Button>,
      ]}
    >
      {!hasSuggestion && (
        <Paragraph type="secondary">
          目前沒有可套用的 AI 建議，您仍可依照上方欄位自行填寫並儲存文件。
        </Paragraph>
      )}

      {hasSuggestion && (
        <>
          {/* 摘要 */}
          <Title level={5} style={{ marginTop: 0 }}>摘要</Title>
          <Input.TextArea
            rows={4}
            value={editSummary}
            onChange={(e) => setEditSummary(e.target.value)}
            placeholder="可在此修改 AI 建議的摘要..."
            style={{ marginBottom: 4 }}
          />
          <Text type="secondary" style={{ fontSize: 12 }}>{editSummary.length} 字</Text>

          <Divider />

          {/* 分類與專案 */}
          <Title level={5} style={{ marginTop: 0 }}>分類與專案</Title>
          <div style={{ marginBottom: 12 }}>
            <Text strong>分類：</Text>
            <Space size="small" style={{ marginLeft: 8 }} align="center">
              <Input
                value={editClassification}
                onChange={(e) => setEditClassification(e.target.value)}
                style={{ width: 240 }}
                placeholder="修改分類名稱..."
              />
              {suggestion.classification_is_new
                ? <Tag color="volcano">新增</Tag>
                : <Tag color="blue">既有</Tag>}
            </Space>
            {suggestion.classification_reason && (
              <Paragraph style={{ marginBottom: 0, marginLeft: 24, marginTop: 4 }} type="secondary">
                {suggestion.classification_reason}
              </Paragraph>
            )}
          </div>
          {suggestion.project && (
            <div style={{ marginBottom: 12 }}>
              <Text strong>專案：</Text>
              <Space size="small" style={{ marginLeft: 8 }}>
                <Text>{suggestion.project}</Text>
                {suggestion.project_is_new
                  ? <Tag color="volcano">新增</Tag>
                  : <Tag color="blue">既有</Tag>}
              </Space>
              {suggestion.project_reason && (
                <Paragraph style={{ marginBottom: 0, marginLeft: 24, marginTop: 4 }} type="secondary">
                  {suggestion.project_reason}
                </Paragraph>
              )}
            </div>
          )}

          <Divider />

          {/* 關鍵字 */}
          <Title level={5}>關鍵字</Title>
          <Select
            mode="tags"
            style={{ width: "100%", marginBottom: 4 }}
            value={editKeywords}
            onChange={setEditKeywords}
            placeholder="可新增或移除關鍵字..."
            tokenSeparators={[","]}
          />
          <Text type="secondary" style={{ fontSize: 12 }}>可直接輸入新關鍵字後按 Enter 新增，點 × 移除</Text>

          <Divider />

          {/* 文字分段預覽（唯讀） */}
          <Title level={5}>文字分段預覽（前 10 筆）</Title>
          {segments && segments.length > 0 ? (
            <List
              size="small"
              dataSource={segments.slice(0, 10)}
              renderItem={(item) => (
                <List.Item>
                  <List.Item.Meta
                    title={
                      <Text strong>第 {item.page} 頁，第 {item.paragraph_index} 段</Text>
                    }
                    description={<Text style={{ fontSize: 12 }}>{item.text}</Text>}
                  />
                </List.Item>
              )}
            />
          ) : (
            <Paragraph type="secondary">目前沒有可顯示的分段內容。</Paragraph>
          )}
          {segments && segments.length > 10 && (
            <Paragraph type="secondary" style={{ marginTop: 8 }}>
              僅顯示前 10 筆，共 {segments.length} 筆分段。
            </Paragraph>
          )}
        </>
      )}
    </Modal>
  );
};

export default AISuggestionModal;
