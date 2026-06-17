
import { useNavigate } from 'react-router-dom';
import AppLayout from '../components/Layout/AppLayout';
import DocumentList from '../components/Documents/DocumentList';

const DocumentsPage = () => {
  const navigate = useNavigate();

  const handleViewDocument = (doc) => {
    // 如果有搜尋相關參數，通過 state 傳遞
    if (doc._searchPage || doc._searchKeyword) {
      navigate(`/documents/${doc.id}`, {
        state: {
          initialPage: doc._searchPage,
          initialHighlightKeyword: doc._searchKeyword,
        },
      });
    } else {
      navigate(`/documents/${doc.id}`);
    }
  };

  return (
    <AppLayout>
      <DocumentList
        onCreate={() => navigate('/documents/new')}
        onView={handleViewDocument}
      />
    </AppLayout>
  );
};

export default DocumentsPage;
