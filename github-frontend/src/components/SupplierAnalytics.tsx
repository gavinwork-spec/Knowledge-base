import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Table,
  Tag,
  Button,
  Space,
  Select,
  DatePicker,
  Progress,
  Rate,
  Tabs,
  Badge,
  Tooltip,
  Alert,
  Modal,
  Form,
  Input,
  InputNumber,
  message,
  Typography
} from 'antd';
import {
  TeamOutlined,
  StarOutlined,
  TrophyOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ClockCircleOutlined,
  DollarOutlined,
  TruckOutlined,
  FileTextOutlined,
  TrendingUpOutlined,
  TrendingDownOutlined,
  BarChartOutlined,
  SettingOutlined,
  EyeOutlined
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import dayjs from 'dayjs';

const { Option } = Select;
const { TabPane } = Tabs;
const { Text } = Typography;

interface Supplier {
  id: string;
  name: string;
  category: string;
  country: string;
  tier: 'A+' | 'A' | 'B' | 'C' | 'D';
  status: 'active' | 'inactive' | 'suspended';
  totalOrders: number;
  totalValue: number;
  onTimeDeliveryRate: number;
  qualityScore: number;
  priceCompetitiveness: number;
  responseTime: number;
  lastOrderDate: string;
  contactPerson: string;
  phone: string;
  email: string;
  specialties: string[];
  certifications: string[];
}

interface SupplierPerformance {
  supplierId: string;
  supplierName: string;
  period: string;
  ordersCompleted: number;
  onTimeDeliveries: number;
  qualityIssues: number;
  averageResponseTime: number;
  totalPrice: number;
  performanceScore: number;
  trend: 'up' | 'down' | 'stable';
}

interface SupplierRisk {
  supplierId: string;
  supplierName: string;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  riskFactors: string[];
  lastIncident: string;
  impact: 'low' | 'medium' | 'high';
}

const SupplierAnalytics: React.FC = () => {
  const [suppliers, setSuppliers] = useState<Supplier[]>([]);
  const [performance, setPerformance] = useState<SupplierPerformance[]>([]);
  const [risks, setRisks] = useState<SupplierRisk[]>([]);
  const [selectedSupplier, setSelectedSupplier] = useState<Supplier | null>(null);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  const [filterTier, setFilterTier] = useState<string>('all');
  const [filterCategory, setFilterCategory] = useState<string>('all');

  // 模拟数据
  useEffect(() => {
    const mockSuppliers: Supplier[] = [
      {
        id: '1',
        name: '东方金属制品厂',
        category: '标准紧固件',
        country: '中国',
        tier: 'A+',
        status: 'active',
        totalOrders: 156,
        totalValue: 12500000,
        onTimeDeliveryRate: 96.5,
        qualityScore: 4.7,
        priceCompetitiveness: 8.5,
        responseTime: 2.5,
        lastOrderDate: '2024-01-15',
        contactPerson: '张经理',
        phone: '+86 21 1234 5678',
        email: 'zhang@dongfang-metal.com',
        specialties: ['高强度螺栓', '不锈钢紧固件', '特殊定制'],
        certifications: ['ISO 9001', 'ISO 14001', 'IATF 16949']
      },
      {
        id: '2',
        name: '精密不锈钢公司',
        category: '不锈钢制品',
        country: '中国',
        tier: 'A',
        status: 'active',
        totalOrders: 89,
        totalValue: 8900000,
        onTimeDeliveryRate: 92.3,
        qualityScore: 4.5,
        priceCompetitiveness: 7.8,
        responseTime: 3.2,
        lastOrderDate: '2024-01-14',
        contactPerson: '李总监',
        phone: '+86 755 8765 4321',
        email: 'li@precision-ss.com',
        specialties: ['不锈钢螺栓', '耐腐蚀紧固件', '精密零件'],
        certifications: ['ISO 9001', 'ASTM', 'DIN']
      },
      {
        id: '3',
        name: '德国FastTech GmbH',
        category: '高端紧固件',
        country: '德国',
        tier: 'A',
        status: 'active',
        totalOrders: 45,
        totalValue: 15600000,
        onTimeDeliveryRate: 98.2,
        qualityScore: 4.9,
        priceCompetitiveness: 6.5,
        responseTime: 4.8,
        lastOrderDate: '2024-01-12',
        contactPerson: 'Herr Schmidt',
        phone: '+49 30 9876 5432',
        email: 'schmidt@fasttech.de',
        specialties: ['汽车级紧固件', '航空航天零件', '高性能材料'],
        certifications: ['ISO 9001', 'VDA 6.1', 'AS9100']
      }
    ];

    const mockPerformance: SupplierPerformance[] = [
      {
        supplierId: '1',
        supplierName: '东方金属制品厂',
        period: '2024-Q1',
        ordersCompleted: 42,
        onTimeDeliveries: 40,
        qualityIssues: 1,
        averageResponseTime: 2.3,
        totalPrice: 3200000,
        performanceScore: 94.5,
        trend: 'up'
      },
      {
        supplierId: '2',
        supplierName: '精密不锈钢公司',
        period: '2024-Q1',
        ordersCompleted: 28,
        onTimeDeliveries: 26,
        qualityIssues: 2,
        averageResponseTime: 3.1,
        totalPrice: 2100000,
        performanceScore: 88.7,
        trend: 'stable'
      }
    ];

    const mockRisks: SupplierRisk[] = [
      {
        supplierId: '2',
        supplierName: '精密不锈钢公司',
        riskLevel: 'medium',
        riskFactors: ['交付时间波动', '原材料价格上涨'],
        lastIncident: '2024-01-08',
        impact: 'medium'
      }
    ];

    setSuppliers(mockSuppliers);
    setPerformance(mockPerformance);
    setRisks(mockRisks);
  }, []);

  // 供应商表格列定义
  const supplierColumns: ColumnsType<Supplier> = [
    {
      title: '供应商名称',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <div>
          <div style={{ fontWeight: 600 }}>{text}</div>
          <div style={{ fontSize: 12, color: '#8c8c8c' }}>{record.category}</div>
        </div>
      ),
    },
    {
      title: '等级',
      dataIndex: 'tier',
      key: 'tier',
      render: (tier) => {
        const tierConfig = {
          'A+': { color: 'gold', text: 'A+ 顶级' },
          'A': { color: 'blue', text: 'A 优秀' },
          'B': { color: 'green', text: 'B 良好' },
          'C': { color: 'orange', text: 'C 一般' },
          'D': { color: 'red', text: 'D 待改进' }
        };
        const config = tierConfig[tier];
        return <Tag color={config.color}>{config.text}</Tag>;
      },
    },
    {
      title: '准交率',
      dataIndex: 'onTimeDeliveryRate',
      key: 'onTimeDeliveryRate',
      render: (rate) => (
        <div>
          <Progress percent={rate} size="small" />
          <span style={{ fontSize: 12, color: '#8c8c8c' }}>{rate}%</span>
        </div>
      ),
    },
    {
      title: '质量评分',
      dataIndex: 'qualityScore',
      key: 'qualityScore',
      render: (score) => <Rate disabled defaultValue={score} style={{ fontSize: 14 }} />,
    },
    {
      title: '价格竞争力',
      dataIndex: 'priceCompetitiveness',
      key: 'priceCompetitiveness',
      render: (score) => (
        <div>
          <Progress percent={score * 10} size="small" strokeColor={score > 8 ? '#52c41a' : score > 6 ? '#fa8c16' : '#ff4d4f'} />
          <span style={{ fontSize: 12, color: '#8c8c8c' }}>{score}/10</span>
        </div>
      ),
    },
    {
      title: '订单数/金额',
      key: 'orders',
      render: (_, record) => (
        <div>
          <div>{record.totalOrders} 订单</div>
          <div>¥{(record.totalValue / 10000).toFixed(0)}万</div>
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => {
        const statusConfig = {
          active: { color: 'success', text: '活跃' },
          inactive: { color: 'default', text: '非活跃' },
          suspended: { color: 'error', text: '暂停' }
        };
        const config = statusConfig[status];
        return <Badge status={status as any} text={config.text} />;
      },
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Tooltip title="查看详情">
            <Button
              type="text"
              icon={<EyeOutlined />}
              onClick={() => {
                setSelectedSupplier(record);
                setDetailModalVisible(true);
              }}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  // 风险预警表格列定义
  const riskColumns: ColumnsType<SupplierRisk> = [
    {
      title: '供应商',
      dataIndex: 'supplierName',
      key: 'supplierName',
    },
    {
      title: '风险等级',
      dataIndex: 'riskLevel',
      key: 'riskLevel',
      render: (level) => {
        const levelConfig = {
          low: { color: 'green', text: '低风险' },
          medium: { color: 'orange', text: '中风险' },
          high: { color: 'red', text: '高风险' },
          critical: { color: 'purple', text: '严重' }
        };
        const config = levelConfig[level];
        return <Tag color={config.color}>{config.text}</Tag>;
      },
    },
    {
      title: '风险因素',
      dataIndex: 'riskFactors',
      key: 'riskFactors',
      render: (factors) => (
        <Space direction="vertical" size="small">
          {factors.map((factor, index) => (
            <Tag key={index} size="small">{factor}</Tag>
          ))}
        </Space>
      ),
    },
    {
      title: '最后事件',
      dataIndex: 'lastIncident',
      key: 'lastIncident',
    },
    {
      title: '影响程度',
      dataIndex: 'impact',
      key: 'impact',
      render: (impact) => {
        const impactConfig = {
          low: { color: 'green', text: '低' },
          medium: { color: 'orange', text: '中' },
          high: { color: 'red', text: '高' }
        };
        const config = impactConfig[impact];
        return <Tag color={config.color}>{config.text}</Tag>;
      },
    },
  ];

  // 过滤供应商
  const filteredSuppliers = suppliers.filter(supplier => {
    const matchesTier = filterTier === 'all' || supplier.tier === filterTier;
    const matchesCategory = filterCategory === 'all' || supplier.category === filterCategory;
    return matchesTier && matchesCategory;
  });

  // 计算统计数据
  const statistics = {
    totalSuppliers: suppliers.length,
    activeSuppliers: suppliers.filter(s => s.status === 'active').length,
    topTierSuppliers: suppliers.filter(s => s.tier === 'A+' || s.tier === 'A').length,
    averageOnTimeDelivery: suppliers.length > 0
      ? suppliers.reduce((sum, s) => sum + s.onTimeDeliveryRate, 0) / suppliers.length
      : 0,
    totalOrders: suppliers.reduce((sum, s) => sum + s.totalOrders, 0),
    totalValue: suppliers.reduce((sum, s) => sum + s.totalValue, 0),
    riskCount: risks.length
  };

  return (
    <div style={{ padding: '24px' }}>
      {/* 头部 */}
      <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <h1 style={{ fontSize: 24, fontWeight: 600, margin: 0 }}>
            🏢 供应商分析仪表板
          </h1>
          <p style={{ color: '#8c8c8c', margin: 0 }}>
            供应商绩效评估与风险管理 - 管理{statistics.totalSuppliers}家供应商
          </p>
        </Col>
        <Col>
          <Space>
            <Button icon={<SettingOutlined />}>
              分析设置
            </Button>
            <Button icon={<FileTextOutlined />}>
              导出报告
            </Button>
            <Button type="primary" icon={<TeamOutlined />}>
              添加供应商
            </Button>
          </Space>
        </Col>
      </Row>

      {/* 统计卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="供应商总数"
              value={statistics.totalSuppliers}
              prefix={<TeamOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
            <Text type="secondary">活跃: {statistics.activeSuppliers}</Text>
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="顶级供应商"
              value={statistics.topTierSuppliers}
              prefix={<TrophyOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
            <Text type="secondary">A+和A等级</Text>
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="平均准交率"
              value={statistics.averageOnTimeDelivery}
              precision={1}
              suffix="%"
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
            <Progress percent={statistics.averageOnTimeDelivery} size="small" style={{ marginTop: 8 }} />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="风险预警"
              value={statistics.riskCount}
              prefix={<WarningOutlined />}
              valueStyle={{ color: '#ff4d4f' }}
            />
            <Text type="warning">需要关注</Text>
          </Card>
        </Col>
      </Row>

      {/* 风险预警 */}
      {statistics.riskCount > 0 && (
        <Alert
          message="供应商风险提醒"
          description={`发现 ${statistics.riskCount} 个供应商存在风险，请及时关注并采取相应措施。`}
          type="warning"
          showIcon
          closable
          style={{ marginBottom: 24 }}
        />
      )}

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="供应商列表" key="suppliers">
          {/* 过滤器 */}
          <Card style={{ marginBottom: 24 }}>
            <Row gutter={[16, 16]}>
              <Col>
                <Select
                  placeholder="供应商等级"
                  style={{ width: 120 }}
                  value={filterTier}
                  onChange={setFilterTier}
                >
                  <Option value="all">全部</Option>
                  <Option value="A+">A+</Option>
                  <Option value="A">A</Option>
                  <Option value="B">B</Option>
                  <Option value="C">C</Option>
                  <Option value="D">D</Option>
                </Select>
              </Col>
              <Col>
                <Select
                  placeholder="产品类别"
                  style={{ width: 150 }}
                  value={filterCategory}
                  onChange={setFilterCategory}
                >
                  <Option value="all">全部</Option>
                  <Option value="标准紧固件">标准紧固件</Option>
                  <Option value="不锈钢制品">不锈钢制品</Option>
                  <Option value="高端紧固件">高端紧固件</Option>
                </Select>
              </Col>
            </Row>
          </Card>

          <Card>
            <Table
              columns={supplierColumns}
              dataSource={filteredSuppliers}
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total) => `共 ${total} 个供应商`,
              }}
            />
          </Card>
        </TabPane>

        <TabPane tab="绩效分析" key="performance">
          <Row gutter={[16, 16]}>
            <Col span={24}>
              <Card
                title="📈 供应商绩效趋势"
                extra={<DatePicker.RangePicker />}
                style={{ marginBottom: 24 }}
              >
                <div style={{
                  height: 400,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: '#fafafa',
                  borderRadius: 6
                }}>
                  <Text type="secondary">
                    供应商绩效趋势图表
                    <br />
                    (集成 Recharts 或 ECharts)
                  </Text>
                </div>
              </Card>
            </Col>
          </Row>

          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="📊 准交率排行">
                <div style={{ height: 300 }}>
                  <Text type="secondary">准交率排行榜图表</Text>
                </div>
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title="💰 价格竞争力对比">
                <div style={{ height: 300 }}>
                  <Text type="secondary">价格竞争力对比图表</Text>
                </div>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="风险管理" key="risks">
          <Card title="⚠️ 供应商风险监控" style={{ marginBottom: 24 }}>
            <Table
              columns={riskColumns}
              dataSource={risks}
              pagination={false}
              size="small"
            />
          </Card>

          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="🎯 风险分布">
                <div style={{ height: 300 }}>
                  <Text type="secondary">风险分布图表</Text>
                </div>
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title="📈 风险趋势">
                <div style={{ height: 300 }}>
                  <Text type="secondary">风险趋势图表</Text>
                </div>
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      {/* 供应商详情模态框 */}
      <Modal
        title={selectedSupplier ? `供应商详情 - ${selectedSupplier.name}` : '供应商详情'}
        visible={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        width={1000}
        footer={[
          <Button key="contact" icon={<MessageOutlined />}>
            联系供应商
          </Button>,
          <Button key="edit" icon={<EditOutlined />}>
            编辑信息
          </Button>,
          <Button key="export" icon={<FileTextOutlined />}>
            导出报告
          </Button>,
        ]}
      >
        {selectedSupplier && (
          <Tabs defaultActiveKey="basic">
            <TabPane tab="基本信息" key="basic">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>公司名称:</Text> {selectedSupplier.name}
                  </div>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>产品类别:</Text> {selectedSupplier.category}
                  </div>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>国家/地区:</Text> {selectedSupplier.country}
                  </div>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>供应商等级:</Text>
                    <Tag color={selectedSupplier.tier === 'A+' ? 'gold' : 'blue'}>
                      {selectedSupplier.tier}
                    </Tag>
                  </div>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>联系人:</Text> {selectedSupplier.contactPerson}
                  </div>
                </Col>
                <Col span={12}>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>电话:</Text> {selectedSupplier.phone}
                  </div>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>邮箱:</Text> {selectedSupplier.email}
                  </div>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>总订单数:</Text> {selectedSupplier.totalOrders}
                  </div>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>总金额:</Text> ¥{(selectedSupplier.totalValue / 10000).toFixed(0)}万
                  </div>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>最后订单:</Text> {selectedSupplier.lastOrderDate}
                  </div>
                </Col>
              </Row>

              <div style={{ marginBottom: 16 }}>
                <Text strong>专业领域:</Text>
                <div style={{ marginTop: 8 }}>
                  {selectedSupplier.specialties.map((specialty, index) => (
                    <Tag key={index} color="blue">{specialty}</Tag>
                  ))}
                </div>
              </div>

              <div style={{ marginBottom: 16 }}>
                <Text strong>认证资质:</Text>
                <div style={{ marginTop: 8 }}>
                  {selectedSupplier.certifications.map((cert, index) => (
                    <Tag key={index} color="green">{cert}</Tag>
                  ))}
                </div>
              </div>
            </TabPane>

            <TabPane tab="绩效指标" key="performance">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Card size="small" title="交付绩效">
                    <div style={{ marginBottom: 16 }}>
                      <Text strong>准交率:</Text>
                      <Progress percent={selectedSupplier.onTimeDeliveryRate} />
                    </div>
                    <div style={{ marginBottom: 16 }}>
                      <Text strong>平均响应时间:</Text> {selectedSupplier.responseTime}小时
                    </div>
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small" title="质量绩效">
                    <div style={{ marginBottom: 16 }}>
                      <Text strong>质量评分:</Text>
                      <Rate disabled defaultValue={selectedSupplier.qualityScore} />
                    </div>
                    <div style={{ marginBottom: 16 }}>
                      <Text strong>价格竞争力:</Text>
                      <Progress percent={selectedSupplier.priceCompetitiveness * 10} />
                    </div>
                  </Card>
                </Col>
              </Row>
            </TabPane>

            <TabPane tab="历史记录" key="history">
              <Timeline>
                <Timeline.Item color="green">
                  <Text strong>最后订单完成</Text>
                  <br />
                  <Text type="secondary">{selectedSupplier.lastOrderDate} - 准时交付</Text>
                </Timeline.Item>
                <Timeline.Item color="blue">
                  <Text strong>质量审核通过</Text>
                  <br />
                  <Text type="secondary">2024-01-10 - 年度审核</Text>
                </Timeline.Item>
                <Timeline.Item>
                  <Text strong>合作开始</Text>
                  <br />
                  <Text type="secondary">2023-06-15 - 签订合作协议</Text>
                </Timeline.Item>
              </Timeline>
            </TabPane>
          </Tabs>
        )}
      </Modal>
    </div>
  );
};

export default SupplierAnalytics;