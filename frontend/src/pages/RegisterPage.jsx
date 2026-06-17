import { useState } from 'react';
import { Form, Input, Button, Card, message, Typography } from 'antd';
import { Link, useNavigate } from 'react-router-dom';

import apiClient from '../services/api';

const RegisterPage = () => {
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const onFinish = async ({ username, email, password, confirmPassword }) => {
    if (password !== confirmPassword) {
      message.error('兩次輸入的密碼不一致');
      return;
    }

    setLoading(true);
    try {
      await apiClient.post('/auth/register', {
        username,
        email,
        password,
      });
      message.success('註冊成功，請使用帳號密碼登入');
      navigate('/login');
    } catch (error) {
      const errMsg = error.response?.data?.detail ?? '註冊失敗，請稍後再試。';
      message.error(errMsg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
      <Card title="建立新帳號" style={{ width: 380 }}>
        <Form name="register" layout="vertical" onFinish={onFinish}>
          <Form.Item
            label="使用者名稱"
            name="username"
            rules={[{ required: true, message: '請輸入使用者名稱!' }]}
          >
            <Input />
          </Form.Item>

          <Form.Item
            label="電子郵件"
            name="email"
            rules={[
              { required: true, message: '請輸入電子郵件!' },
              { type: 'email', message: '請輸入正確的電子郵件格式!' },
            ]}
          >
            <Input />
          </Form.Item>

          <Form.Item
            label="密碼"
            name="password"
            rules={[{ required: true, message: '請輸入密碼!' }]}
            hasFeedback
          >
            <Input.Password />
          </Form.Item>

          <Form.Item
            label="確認密碼"
            name="confirmPassword"
            dependencies={['password']}
            hasFeedback
            rules={[{ required: true, message: '請再次輸入密碼!' }]}
          >
            <Input.Password />
          </Form.Item>

          <Form.Item>
            <Button type="primary" htmlType="submit" block loading={loading}>
              註冊
            </Button>
          </Form.Item>
        </Form>
        <Typography.Paragraph style={{ marginTop: 16, textAlign: 'center' }}>
          已經有帳號？ <Link to="/login">返回登入</Link>
        </Typography.Paragraph>
      </Card>
    </div>
  );
};

export default RegisterPage;
