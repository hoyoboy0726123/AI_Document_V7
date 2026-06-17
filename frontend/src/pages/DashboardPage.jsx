import { Layout, Menu } from 'antd';

const { Header, Content, Footer, Sider } = Layout;

const DashboardPage = () => {
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider>
        <div className="logo" />
        <Menu theme="dark" defaultSelectedKeys={['1']} mode="inline">
          <Menu.Item key="1">文件列表</Menu.Item>
          <Menu.Item key="2">管理介面</Menu.Item>
        </Menu>
      </Sider>
      <Layout>
        <Header style={{ background: '#fff', padding: 0 }} />
        <Content style={{ margin: '0 16px' }}>
          <h1>文件儀表板</h1>
          <p>此處將顯示文件列表...</p>
        </Content>
        <Footer style={{ textAlign: 'center' }}>智慧文件管理系統 ©2025</Footer>
      </Layout>
    </Layout>
  );
};

export default DashboardPage;
