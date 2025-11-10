import React, { useState, useEffect } from 'react';
import {
  Layout,
  Menu,
  Avatar,
  Dropdown,
  Badge,
  Button,
  Space,
  Typography,
  Card,
  Row,
  Col,
  Statistic,
  Progress
} from 'antd';
import {
  DashboardOutlined,
  DollarOutlined,
  ProjectOutlined,
  BarChartOutlined,
  UserOutlined,
  BellOutlined,
  SettingOutlined,
  RiseOutlined,
  FallOutlined,
  TeamOutlined,
  FileTextOutlined
} from '@ant-design/icons';

import PriceMonitoringDashboard from './components/PriceMonitoringDashboard';
import CustomerRelationshipManager from './components/CustomerRelationshipManager';
import ProjectKanbanBoard from './components/ProjectKanbanBoard';
import DrawingViewer from './components/DrawingViewer';
import QuotationManager from './components/QuotationManager';
import SupplierAnalytics from './components/SupplierAnalytics';
import ProfitabilityReports from './components/ProfitabilityReports';

const { Header, Sider, Content } = Layout;
const { Title, Text } = Typography;

interface TradingDashboardProps {}

const TradingDashboard: React.FC<TradingDashboardProps> = () => {
  const [collapsed, setCollapsed] = useState(false);
  const [selectedMenu, setSelectedMenu] = useState('dashboard');
  const [notifications, setNotifications] = useState(8);

  // 模拟实时数据更新
  useEffect(() => {
    const interval = setInterval(() => {
      // 模拟新通知
      setNotifications(prev => prev + Math.floor(Math.random() * 3));
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  // 菜单项配置
  const menuItems = [
    {
      key: 'dashboard',
      icon: <DashboardOutlined />,
      label: '总览仪表板',
    },
    {
      key: 'price-monitoring',
      icon: <DollarOutlined />,
      label: '价格监控',
    },
    {
      key: 'customer-management',
      icon: <UserOutlined />,
      label: '客户管理',
    },
    {
      key: 'project-tracking',
      icon: <ProjectOutlined />,
      label: '项目跟踪',
    },
    {
      key: 'drawing-viewer',
      icon: <FileTextOutlined />,
      label: '图纸查看器',
    },
    {
      key: 'quotation',
      icon: <BarChartOutlined />,
      label: '报价管理',
    },
    {
      key: 'supplier-analytics',
      icon: <TeamOutlined />,
      label: '供应商分析',
    },
    {
      key: 'profitability',
      icon: <RiseOutlined />,
      label: '利润分析',
    },
  ];

  // 用户下拉菜单
  const userMenuItems = [
    {
      key: 'profile',
      label: '个人资料',
      icon: <UserOutlined />,
    },
    {
      key: 'settings',
      label: '系统设置',
      icon: <SettingOutlined />,
    },
    {
      type: 'divider' as const,
    },
    {
      key: 'logout',
      label: '退出登录',
      danger: true,
    },
  ];

  // 渲染主要内容区域
  const renderContent = () => {
    switch (selectedMenu) {
      case 'dashboard':
        return <OverviewDashboard />;
      case 'price-monitoring':
        return <PriceMonitoringDashboard />;
      case 'customer-management':
        return <CustomerRelationshipManager />;
      case 'project-tracking':
        return <ProjectKanbanBoard />;
      case 'drawing-viewer':
        return <DrawingViewer />;
      case 'quotation':
        return <QuotationManager />;
      case 'supplier-analytics':
        return <SupplierAnalytics />;
      case 'profitability':
        return <ProfitabilityReports />;
      default:
        return <OverviewDashboard />;
    }
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      {/* 侧边栏 */}
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        style={{
          background: '#001529',
          boxShadow: '2px 0 8px rgba(0,0,0,0.15)',
        }}
      >
        <div style={{
          height: 64,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'rgba(255,255,255,0.1)',
          marginBottom: 16
        }}>
          {!collapsed ? (
            <Title level={4} style={{ color: 'white', margin: 0 }}>
              🏭 贸易公司
            </Title>
          ) : (
            <Title level={4} style={{ color: 'white', margin: 0 }}>🏭</Title>
          )}
        </div>

        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[selectedMenu]}
          items={menuItems}
          onClick={({ key }) => setSelectedMenu(key)}
        />
      </Sider>

      <Layout>
        {/* 顶部导航 */}
        <Header style={{
          padding: '0 24px',
          background: '#fff',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <Button
            type="text"
            icon={collapsed ? <DashboardOutlined /> : <DashboardOutlined />}
            onClick={() => setCollapsed(!collapsed)}
            style={{ fontSize: '16px', width: 64, height: 64 }}
          />

          <Space size="large">
            {/* 实时状态指示器 */}
            <Space>
              <Badge status="processing" text="实时数据" />
              <Badge count={notifications} size="small">
                <Button
                  type="text"
                  icon={<BellOutlined />}
                  style={{ fontSize: '16px' }}
                />
              </Badge>
            </Space>

            {/* 用户信息 */}
            <Dropdown menu={{ items: userMenuItems }} placement="bottomRight">
              <Space style={{ cursor: 'pointer' }}>
                <Avatar size="small" icon={<UserOutlined />} />
                <span>销售经理</span>
              </Space>
            </Dropdown>
          </Space>
        </Header>

        {/* 主要内容区域 */}
        <Content style={{
          margin: '16px',
          padding: 0,
          minHeight: 280,
          background: '#f0f2f5'
        }}>
          {renderContent()}
        </Content>
      </Layout>
    </Layout>
  );
};

// 总览仪表板组件
const OverviewDashboard: React.FC = () => {
  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>🏭 贸易公司智能仪表板</Title>

      {/* 核心指标卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="今日询价"
              value={28}
              precision={0}
              valueStyle={{ color: '#3f8600' }}
              prefix={<RiseOutlined />}
              suffix="个"
            />
            <Progress percent={78} size="small" style={{ marginTop: 8 }} />
          </Card>
        </Col>

        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="待处理报价"
              value={15}
              precision={0}
              valueStyle={{ color: '#cf1322' }}
              prefix={<FileTextOutlined />}
              suffix="份"
            />
            <Text type="secondary">平均处理时间: 2.5小时</Text>
          </Card>
        </Col>

        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="本月利润"
              value={256000}
              precision={0}
              valueStyle={{ color: '#3f8600' }}
              prefix={<DollarOutlined />}
              suffix="元"
            />
            <Text type="success">+12.5% vs 上月</Text>
          </Card>
        </Col>

        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="准时交付率"
              value={94.8}
              precision={1}
              valueStyle={{ color: '#3f8600' }}
              suffix="%"
            />
            <Text type="secondary">目标: 95%</Text>
          </Card>
        </Col>
      </Row>

      {/* 实时市场动态 */}
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <Card title="📈 原材料价格走势" extra={<Button type="link">查看详情</Button>}>
            <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Text type="secondary">价格监控图表组件将在此显示</Text>
            </div>
          </Card>
        </Col>

        <Col xs={24} lg={12}>
          <Card title="💱 汇率变动" extra={<Button type="link">查看详情</Button>}>
            <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Text type="secondary">汇率监控图表组件将在此显示</Text>
            </div>
          </Card>
        </Col>
      </Row>

      {/* 最新活动和提醒 */}
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} lg={8}>
          <Card title="🔥 最新询价" size="small">
            <div style={{ height: 150, overflow: 'auto' }}>
              <Text type="secondary">最新询价列表组件将在此显示</Text>
            </div>
          </Card>
        </Col>

        <Col xs={24} lg={8}>
          <Card title="⚠️ 紧急提醒" size="small">
            <div style={{ height: 150, overflow: 'auto' }}>
              <Text type="secondary">紧急提醒列表组件将在此显示</Text>
            </div>
          </Card>
        </Col>

        <Col xs={24} lg={8}>
          <Card title="🎯 重点客户" size="small">
            <div style={{ height: 150, overflow: 'auto' }}>
              <Text type="secondary">重点客户列表组件将在此显示</Text>
            </div>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default TradingDashboard;