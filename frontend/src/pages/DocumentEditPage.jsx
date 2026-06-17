import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { message } from "antd";
import AppLayout from "../components/Layout/AppLayout";
import DocumentForm from "../components/Documents/DocumentForm";
import apiClient from "../services/api";

const DocumentEditPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [document, setDocument] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchDocument = async () => {
      try {
        const resp = await apiClient.get(`documents/${id}`);
        setDocument(resp.data);
      } catch (error) {
        message.error(error.response?.data?.detail ?? "載入文件失敗");
        navigate("/documents");
      } finally {
        setLoading(false);
      }
    };

    fetchDocument();
  }, [id, navigate]);

  return (
    <AppLayout>
      <DocumentForm
        document={document}
        loading={loading}
        onSuccess={() => navigate(`/documents/${id}`)}
        onCancel={() => navigate(`/documents/${id}`)}
      />
    </AppLayout>
  );
};

export default DocumentEditPage;

