
import { useNavigate, useParams, useLocation, useSearchParams } from 'react-router-dom';
import AppLayout from '../components/Layout/AppLayout';
import DocumentDetail from '../components/Documents/DocumentDetail';

const DocumentDetailPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const { initialPage: statePage, initialHighlightKeyword } = location.state || {};
  const initialPage = statePage || (searchParams.get("page") ? parseInt(searchParams.get("page")) : undefined);

  return (
    <AppLayout>
      <DocumentDetail
        documentId={id}
        initialPage={initialPage}
        initialHighlightKeyword={initialHighlightKeyword}
        onBack={() => navigate('/documents')}
        onEdit={(doc) => navigate(`/documents/${doc.id}/edit`)}
      />
    </AppLayout>
  );
};

export default DocumentDetailPage;
