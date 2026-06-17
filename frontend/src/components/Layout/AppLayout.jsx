
import { Layout } from 'antd';
import AppHeader from './AppHeader';
import AppSidebar from './AppSidebar';

const { Content } = Layout;

const AppLayout = ({ children }) => (
  <Layout style={{ minHeight: '100vh' }}>
    <AppSidebar />
    <Layout>
      <AppHeader />
      <Content style={{ padding: 24, background: '#f5f7fa' }}>
        {children}
      </Content>
    </Layout>
  </Layout>
);

export default AppLayout;
