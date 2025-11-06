import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Table,
  Tag,
  Button,
  Modal,
  Descriptions,
  Timeline,
  Progress,
  Alert,
  Space,
  Tooltip,
  Badge,
  message,
  Switch,
  Tabs
} from 'antd';
import {
  BellOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ClockCircleOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  EyeOutlined,
  MailOutlined,
  DashboardOutlined,
  LineChartOutlined
} from '@ant-design/icons';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const { TabPane } = Tabs;

const RemindersDashboard = () => {
  const [dashboardData, setDashboardData] = useState({});
  const [rules, setRules] = useState([]);
  const [records, setRecords] = useState([]);
  const [notifications, setNotifications] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedRule, setSelectedRule] = useState(null);
  const [ruleDetailVisible, setRuleDetailVisible] = useState(false);

  // 颜色配置
  const COLORS = ['#52c41a', '#faad14', '#ff4d4f', '#1890ff', '#722ed1', '#13c2c2'];

  // 优先级标签
  const priorityLabels = {
    1: { text: '高', color: 'red' },
    2: { text: '中', color: 'orange' },
    3: { text: '低', color: 'blue' }
  };

  // 状态标签
  const statusLabels = {
    pending: { text: '待处理', color: 'orange' },
    processing: { text: '处理中', color: 'blue' },
    completed: { text: '已完成', color: 'green' },
    failed: { text: '失败', color: 'red' }
  };

  // 加载仪表板数据
  const loadDashboardData = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/v1/reminders/dashboard');
      const result = await response.json();
      if (result.success) {
        setDashboardData(result.data);
      }
    } catch (error) {
      message.error('加载仪表板数据失败');
    } finally {
      setLoading(false);
    }
  };

  // 加载提醒规则
  const loadRules = async () => {
    try {
      const response = await fetch('/api/v1/reminders/rules?limit=10');
      const result = await response.json();
      if (result.success) {
        setRules(result.data.rules);
      }
    } catch (error) {
      message.error('加载提醒规则失败');
    }
  };

  // 加载提醒记录
  const loadRecords = async () => {
    try {
      const response = await fetch('/api/v1/reminders/records?limit=10');
      const result = await response.json();
      if (result.success) {
        setRecords(result.data.records);
      }
    } catch (error) {
      message.error('加载提醒记录失败');
    }
  };

  // 加载通知历史
  const loadNotifications = async () => {
    try {
      const response = await fetch('/api/v1/reminders/notifications?limit=10');
      const result = await response.json();
      if (result.success) {
        setNotifications(result.data.notifications);
      }
    } catch (error) {
      message.error('加载通知历史失败');
    }
  };

  // 初始化数据
  useEffect(() => {
    loadDashboardData();
    loadRules();
    loadRecords();
    loadNotifications();
  }, []);

  // 手动触发规则
  const triggerRule = async (ruleId) => {
    try {
      const response = await fetch('/api/v1/reminders/trigger', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rule_id: ruleId })
      });
      const result = await response.json();
      if (result.success) {
        message.success('规则触发成功');
        loadRecords();
        loadDashboardData();
      } else {
        message.error(result.error?.message || '触发失败');
      }
    } catch (error) {
      message.error('触发规则失败');
    }
  };

  // 切换规则状态
  const toggleRule = async (ruleId) => {
    try {
      const response = await fetch(`/api/v1/reminders/rules/${ruleId}/toggle`, {
        method: 'POST'
      });
      const result = await response.json();
      if (result.success) {
        message.success(result.data.message);
        loadRules();
        loadDashboardData();
      } else {
        message.error(result.error?.message || '操作失败');
      }
    } catch (error) {
      message.error('操作失败');
    }
  };

  // 查看规则详情
  const viewRuleDetail = async (ruleId) => {
    try {
      const response = await fetch(`/api/v1/reminders/rules/${ruleId}`);
      const result = await response.json();
      if (result.success) {
        setSelectedRule(result.data);
        setRuleDetailVisible(true);
      } else {
        message.error('加载规则详情失败');
      }
    } catch (error) {
      message.error('加载规则详情失败');
    }
  };

  // 规则表格列
  const ruleColumns = [
    {
      title: '规则ID',
      dataIndex: 'rule_id',
      key: 'rule_id',
      width: 100,
    },
    {
      title: '规则名称',
      dataIndex: 'name',
      key: 'name',
      ellipsis: true,
    },
    {
      title: '优先级',
      dataIndex: 'priority',
      key: 'priority',
      width: 80,
      render: (priority) => {
        const label = priorityLabels[priority];
        return <Tag color={label.color}>{label.text}</Tag>;
      }
    },
    {
      title: '状态',
      dataIndex: 'is_active',
      key: 'is_active',
      width: 80,
      render: (isActive) => (
        <Switch
          size="small"
          checked={isActive}
          onChange={() => toggleRule(selectedRule?.rule_id)}
        />
      )
    },
    {
      title: '触发次数',
      dataIndex: 'trigger_count',
      key: 'trigger_count',
      width: 100,
    },
    {
      title: '最后触发',
      dataIndex: 'last_triggered',
      key: 'last_triggered',
      width: 150,
      render: (time) => time ? new Date(time).toLocaleString() : '-'
    },
    {
      title: '操作',
      key: 'actions',
      width: 150,
      render: (_, record) => (
        <Space size="small">
          <Tooltip title="查看详情">
            <Button
              type="link"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => viewRuleDetail(record.rule_id)}
            />
          </Tooltip>
          <Tooltip title="手动触发">
            <Button
              type="link"
              size="small"
              icon={<PlayCircleOutlined />}
              onClick={() => triggerRule(record.rule_id)}
            />
          </Tooltip>
        </Space>
      )
    }
  ];

  // 记录表格列
  const recordColumns = [
    {
      title: '执行ID',
      dataIndex: 'execution_id',
      key: 'execution_id',
      width: 180,
      ellipsis: true,
    },
    {
      title: '规则名称',
      dataIndex: 'rule_name',
      key: 'rule_name',
      ellipsis: true,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status) => {
        const label = statusLabels[status];
        return <Tag color={label.color}>{label.text}</Tag>;
      }
    },
    {
      title: '触发时间',
      dataIndex: 'triggered_at',
      key: 'triggered_at',
      width: 150,
      render: (time) => new Date(time).toLocaleString()
    },
    {
      title: '通知数量',
      dataIndex: 'notification_count',
      key: 'notification_count',
      width: 100,
    },
    {
      title: '执行耗时',
      dataIndex: 'execution_time_ms',
      key: 'execution_time_ms',
      width: 100,
      render: (time) => `${time}ms`
    }
  ];

  // 通知表格列
  const notificationColumns = [
    {
      title: '类型',
      dataIndex: 'notification_type',
      key: 'notification_type',
      width: 120,
      render: (type) => {
        const icons = {
          email: <MailOutlined />,
          dashboard: <DashboardOutlined />,
          system_message: <BellOutlined />
        };
        return (
          <Space>
            {icons[type]}
            <span>{type}</span>
          </Space>
        );
      }
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status) => (
        <Tag color={status === 'sent' ? 'green' : 'red'}>
          {status === 'sent' ? '已发送' : '失败'}
        </Tag>
      )
    },
    {
      title: '规则名称',
      dataIndex: 'rule_name',
      key: 'rule_name',
      ellipsis: true,
    },
    {
      title: '发送时间',
      dataIndex: 'sent_at',
      key: 'sent_at',
      width: 150,
      render: (time) => time ? new Date(time).toLocaleString() : '-'
    },
    {
      title: '重试次数',
      dataIndex: 'retry_count',
      key: 'retry_count',
      width: 100,
    }
  ];

  return (
    <div className="reminders-dashboard">
      {/* 系统健康状态 */}
      {dashboardData.system_health && (
        <Alert
          message={`系统状态: ${dashboardData.system_health.status === 'healthy' ? '正常' : dashboardData.system_health.status === 'warning' ? '警告' : '严重'}`}
          description={`失败率: ${dashboardData.system_health.failure_rate}% | 运行时间: ${dashboardData.system_health.uptime}`}
          type={dashboardData.system_health.status === 'healthy' ? 'success' : dashboardData.system_health.status === 'warning' ? 'warning' : 'error'}
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

      {/* 统计卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="总规则数"
              value={dashboardData.rules?.total_rules || 0}
              prefix={<SettingOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="活跃规则"
              value={dashboardData.rules?.active_rules || 0}
              prefix={<PlayCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="今日提醒"
              value={dashboardData.today?.total_reminders || 0}
              prefix={<BellOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="成功率"
              value={dashboardData.today?.total_reminders ?
                Math.round((dashboardData.today.completed_reminders / dashboardData.today.total_reminders) * 100) : 0}
              suffix="%"
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
      </Row>

      <Tabs defaultActiveKey="rules">
        {/* 提醒规则 */}
        <TabPane tab="提醒规则" key="rules">
          <Card title="提醒规则列表" extra={
            <Button type="primary" onClick={loadRules}>
              刷新
            </Button>
          }>
            <Table
              columns={ruleColumns}
              dataSource={rules}
              rowKey="id"
              pagination={false}
              size="small"
            />
          </Card>
        </TabPane>

        {/* 提醒记录 */}
        <TabPane tab="提醒记录" key="records">
          <Card title="提醒记录列表" extra={
            <Button type="primary" onClick={loadRecords}>
              刷新
            </Button>
          }>
            <Table
              columns={recordColumns}
              dataSource={records}
              rowKey="id"
              pagination={false}
              size="small"
            />
          </Card>
        </TabPane>

        {/* 通知历史 */}
        <TabPane tab="通知历史" key="notifications">
          <Card title="通知历史列表" extra={
            <Button type="primary" onClick={loadNotifications}>
              刷新
            </Button>
          }>
            <Table
              columns={notificationColumns}
              dataSource={notifications}
              rowKey="id"
              pagination={false}
              size="small"
            />
          </Card>
        </TabPane>

        {/* 数据分析 */}
        <TabPane tab="数据分析" key="analytics">
          <Row gutter={[16, 16]}>
            {/* 趋势图 */}
            <Col xs={24} lg={12}>
              <Card title="本周提醒趋势">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={dashboardData.weekly_trend || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <RechartsTooltip />
                    <Line type="monotone" dataKey="count" stroke="#1890ff" name="总提醒数" />
                    <Line type="monotone" dataKey="success_count" stroke="#52c41a" name="成功数" />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </Col>

            {/* 通知类型分布 */}
            <Col xs={24} lg={12}>
              <Card title="通知类型分布">
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={dashboardData.notifications || []}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="total_count"
                    >
                      {(dashboardData.notifications || []).map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <RechartsTooltip />
                  </PieChart>
                </ResponsiveContainer>
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      {/* 规则详情弹窗 */}
      <Modal
        title="规则详情"
        visible={ruleDetailVisible}
        onCancel={() => setRuleDetailVisible(false)}
        footer={null}
        width={800}
      >
        {selectedRule && (
          <div>
            <Descriptions bordered column={2} style={{ marginBottom: 16 }}>
              <Descriptions.Item label="规则ID">{selectedRule.rule_id}</Descriptions.Item>
              <Descriptions.Item label="规则名称">{selectedRule.name}</Descriptions.Item>
              <Descriptions.Item label="优先级">
                <Tag color={priorityLabels[selectedRule.priority]?.color}>
                  {priorityLabels[selectedRule.priority]?.text}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="状态">
                <Tag color={selectedRule.is_active ? 'green' : 'red'}>
                  {selectedRule.is_active ? '启用' : '禁用'}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="描述" span={2}>{selectedRule.description}</Descriptions.Item>
              <Descriptions.Item label="触发次数">{selectedRule.trigger_count}</Descriptions.Item>
              <Descriptions.Item label="最后触发">
                {selectedRule.last_triggered ? new Date(selectedRule.last_triggered).toLocaleString() : '-'}
              </Descriptions.Item>
            </Descriptions>

            {/* 执行历史 */}
            {selectedRule.recent_executions && selectedRule.recent_executions.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <h4>最近执行记录</h4>
                <Timeline>
                  {selectedRule.recent_executions.map((execution) => (
                    <Timeline.Item
                      key={execution.id}
                      color={statusLabels[execution.status]?.color}
                      dot={statusLabels[execution.status]?.text === '已完成' ? <CheckCircleOutlined /> : <ExclamationCircleOutlined />}
                    >
                      <div>
                        <strong>{execution.execution_id}</strong>
                        <br />
                        状态: <Tag color={statusLabels[execution.status]?.color}>
                          {statusLabels[execution.status]?.text}
                        </Tag>
                        <br />
                        时间: {new Date(execution.triggered_at).toLocaleString()}
                        <br />
                        通知数量: {execution.notification_count || 0}
                        {execution.execution_time_ms && (
                          <> | 耗时: {execution.execution_time_ms}ms</>
                        )}
                      </div>
                    </Timeline.Item>
                  ))}
                </Timeline>
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
};

export default RemindersDashboard;