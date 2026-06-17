
import { useNavigate } from 'react-router-dom';
import AppLayout from '../components/Layout/AppLayout';
import DocumentForm from '../components/Documents/DocumentForm';

const DocumentCreatePage = () => {
  const navigate = useNavigate();

  return (
    <AppLayout>
      <DocumentForm
        onSuccess={() => navigate('/documents')}
        onCancel={() => navigate('/documents')}
      />
    </AppLayout>
  );
};

export default DocumentCreatePage;
