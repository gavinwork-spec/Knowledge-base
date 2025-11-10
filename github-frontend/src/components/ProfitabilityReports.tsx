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
  Tabs,
  Progress,
  Tooltip,
  Alert,
  Modal,
  Form,
  Input,
  message,
  Typography
} from 'antd';
import {
  DollarOutlined,
  RiseOutlined,
  FallOutlined,
  BarChartOutlined,
  PieChartOutlined,
  LineChartOutlined,
  FileTextOutlined,
  DownloadOutlined,
  SettingOutlined,
  WarningOutlined,
  InfoCircleOutlined,
  RocketOutlined,
  EyeOutlined
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import dayjs from 'dayjs';

const { Option } = Select;
const { TabPane } = Tabs;
const { RangePicker } = DatePicker;
const { Text } = Typography;

interface ProfitabilityData {
  period: string;
  revenue: number;
  cost: number;
  grossProfit: number;
  grossMargin: number;
  netProfit: number;
  netMargin: number;
  operatingExpense: number;
  ordersCount: number;
  averageOrderValue: number;
}

interface ProductProfitability {
  productCategory: string;
  revenue: number;
  cost: number;
  profit: number;
  margin: number;
  volume: number;
  growthRate: number;
  trend: 'up' | 'down' | 'stable';
}

interface CustomerProfitability {
  customerName: string;
  revenue: number;
  cost: number;
  profit: number;
  margin: number;
  orders: number;
  avgOrderValue: number;
  profitability: 'high' | 'medium' | 'low';
  risk: 'low' | 'medium' | 'high';
}

interface CostBreakdown {
  category: string;
  amount: number;
  percentage: number;
  trend: 'up' | 'down' | 'stable';
  description: string;
}

const ProfitabilityReports: React.FC = () => {
  const [profitData, setProfitData] = useState<ProfitabilityData[]>([]);
  const [productProfitability, setProductProfitability] = useState<ProductProfitability[]>([]);
  const [customerProfitability, setCustomerProfitability] = useState<CustomerProfitability[]>([]);
  const [costBreakdown, setCostBreakdown] = useState<CostBreakdown[]>([]);
  const [selectedPeriod, setSelectedPeriod] = useState('monthly');
  const [loading, setLoading] = useState(false);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');

  // 模拟数据加载
  useEffect(() => {
    const mockProfitData: ProfitabilityData[] = [
      {
        period: '2024-01',
        revenue: 2850000,
        cost: 2137500,
        grossProfit: 712500,
        grossMargin: 25.0,
        netProfit: 342000,
        netMargin: 12.0,
        operatingExpense: 370500,
        ordersCount: 156,
        averageOrderValue: 18269
      },
      {
        period: '2023-12',
        revenue: 2680000,
        cost: 2080400,
        grossProfit: 599600,
        grossMargin: 22.4,
        netProfit: 285600,
        netMargin: 10.7,
        operatingExpense: 314000,
        ordersCount: 142,
        averageOrderValue: 18873
      },
      {
        period: '2023-11',
        revenue: 2520000,
        cost: 1957800,
        grossProfit: 562200,
        grossMargin: 22.3,
        netProfit: 274800,
        netMargin: 10.9,
        operatingExpense: 287400,
        ordersCount: 138,
        averageOrderValue: 18261
      }
    ];

    const mockProductProfitability: ProductProfitability[] = [
      {
        productCategory: '高强度螺栓',
        revenue: 1250000,
        cost: 937500,
        profit: 312500,
        margin: 25.0,
        volume: 450000,
        growthRate: 15.2,
        trend: 'up'
      },
      {
        productCategory: '不锈钢紧固件',
        revenue: 890000,
        cost: 667500,
        profit: 222500,
        margin: 25.0,
        volume: 125000,
        growthRate: 8.7,
        trend: 'up'
      },
      {
        productCategory: '标准紧固件',
        revenue: 710000,
        cost: 532500,
        profit: 177500,
        margin: 25.0,
        volume: 890000,
        growthRate: -3.2,
        trend: 'down'
      }
    ];

    const mockCustomerProfitability: CustomerProfitability[] = [
      {
        customerName: '上海汽车制造有限公司',
        revenue: 1250000,
        cost: 937500,
        profit: 312500,
        margin: 25.0,
        orders: 45,
        avgOrderValue: 27778,
        profitability: 'high',
        risk: 'low'
      },
      {
        customerName: '德国AutoParts GmbH',
        revenue: 890000,
        cost: 667500,
        profit: 222500,
        margin: 25.0,
        orders: 28,
        avgOrderValue: 31786,
        profitability: 'high',
        risk: 'medium'
      },
      {
        customerName: '深圳精密仪器公司',
        revenue: 456000,
        cost: 380000,
        profit: 76000,
        margin: 16.7,
        orders: 32,
        avgOrderValue: 14250,
        profitability: 'medium',
        risk: 'low'
      }
    ];

    const mockCostBreakdown: CostBreakdown[] = [
      {
        category: '原材料成本',
        amount: 1282500,
        percentage: 60.0,
        trend: 'up',
        description: '钢材、不锈钢等主要原材料'
      },
      {
        category: '人工成本',
        amount: 342000,
        percentage: 16.0,
        trend: 'stable',
        description: '生产人员工资及相关费用'
      },
      {
        category: '运营费用',
        amount: 370500,
        percentage: 17.3,
        trend: 'up',
        description: '销售、管理及行政费用'
      },
      {
        category: '物流运输',
        amount: 142500,
        percentage: 6.7,
        trend: 'down',
        description: '运输、仓储及报关费用'
      }
    ];

    setProfitData(mockProfitData);
    setProductProfitability(mockProductProfitability);
    setCustomerProfitability(mockCustomerProfitability);
    setCostBreakdown(mockCostBreakdown);
  }, []);

  // 客户利润率表格列定义
  const customerProfitColumns: ColumnsType<CustomerProfitability> = [
    {
      title: '客户名称',
      dataIndex: 'customerName',
      key: 'customerName',
      render: (text) => <strong>{text}</strong>,
    },
    {
      title: '营收',
      dataIndex: 'revenue',
      key: 'revenue',
      render: (value) => `¥${(value / 10000).toFixed(1)}万`,
    },
    {
      title: '成本',
      dataIndex: 'cost',
      key: 'cost',
      render: (value) => `¥${(value / 10000).toFixed(1)}万`,
    },
    {
      title: '利润',
      dataIndex: 'profit',
      key: 'profit',
      render: (value) => (
        <span style={{ color: value > 0 ? '#52c41a' : '#ff4d4f' }}>
          ¥{(value / 10000).toFixed(1)}万
        </span>
      ),
    },
    {
      title: '利润率',
      dataIndex: 'margin',
      key: 'margin',
      render: (margin) => (
        <Tag color={margin > 20 ? 'green' : margin > 10 ? 'orange' : 'red'}>
          {margin.toFixed(1)}%
        </Tag>
      ),
    },
    {
      title: '订单数',
      dataIndex: 'orders',
      key: 'orders',
    },
    {
      title: '平均订单价值',
      dataIndex: 'avgOrderValue',
      key: 'avgOrderValue',
      render: (value) => `¥${value.toLocaleString()}`,
    },
    {
      title: '风险等级',
      dataIndex: 'risk',
      key: 'risk',
      render: (risk) => {
        const riskConfig = {
          low: { color: 'green', text: '低风险' },
          medium: { color: 'orange', text: '中风险' },
          high: { color: 'red', text: '高风险' }
        };
        const config = riskConfig[risk];
        return <Tag color={config.color}>{config.text}</Tag>;
      },
    },
  ];

  // 产品利润率表格列定义
  const productProfitColumns: ColumnsType<ProductProfitability> = [
    {
      title: '产品类别',
      dataIndex: 'productCategory',
      key: 'productCategory',
    },
    {
      title: '营收',
      dataIndex: 'revenue',
      key: 'revenue',
      render: (value) => `¥${(value / 10000).toFixed(1)}万`,
    },
    {
      title: '销量',
      dataIndex: 'volume',
      key: 'volume',
      render: (value) => `${(value / 1000).toFixed(0)}K件`,
    },
    {
      title: '利润',
      dataIndex: 'profit',
      key: 'profit',
      render: (value) => (
        <span style={{ color: '#52c41a' }}>
          ¥{(value / 10000).toFixed(1)}万
        </span>
      ),
    },
    {
      title: '利润率',
      dataIndex: 'margin',
      key: 'margin',
      render: (margin) => `${margin.toFixed(1)}%`,
    },
    {
      title: '增长率',
      dataIndex: 'growthRate',
      key: 'growthRate',
      render: (rate) => (
        <span style={{ color: rate > 0 ? '#52c41a' : '#ff4d4f' }}>
          {rate > 0 ? '+' : ''}{rate.toFixed(1)}%
        </span>
      ),
    },
    {
      title: '趋势',
      dataIndex: 'trend',
      key: 'trend',
      render: (trend) => (
        <span>
          {trend === 'up' && <RiseOutlined style={{ color: '#52c41a' }} />}
          {trend === 'down' && <FallOutlined style={{ color: '#ff4d4f' }} />}
          {trend === 'stable' && <span style={{ color: '#8c8c8c' }}>—</span>}
        </span>
      ),
    },
  ];

  // 计算当前月度关键指标
  const currentMonthData = profitData[0] || {
    revenue: 0,
    cost: 0,
    grossProfit: 0,
    grossMargin: 0,
    netProfit: 0,
    netMargin: 0,
    ordersCount: 0,
    averageOrderValue: 0
  };

  // 计算同比增长
  const calculateGrowth = (current: number, previous: number) => {
    if (previous === 0) return 0;
    return ((current - previous) / previous) * 100;
  };

  const revenueGrowth = profitData.length > 1
    ? calculateGrowth(currentMonthData.revenue, profitData[1].revenue)
    : 0;

  const profitGrowth = profitData.length > 1
    ? calculateGrowth(currentMonthData.netProfit, profitData[1].netProfit)
    : 0;

  // 导出报告
  const exportReport = (type: string) => {
    message.success(`正在导出${type}报告...`);
  };

  return (
    <div style={{ padding: '24px' }}>
      {/* 头部 */}
      <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <h1 style={{ fontSize: 24, fontWeight: 600, margin: 0 }}>
            💹 利润分析报告
          </h1>
          <p style={{ color: '#8c8c8c', margin: 0 }}>
            综合利润分析与成本管理 - 智能商业洞察
          </p>
        </Col>
        <Col>
          <Space>
            <Select
              defaultValue="monthly"
              style={{ width: 120 }}
              onChange={setSelectedPeriod}
            >
              <Option value="daily">日报</Option>
              <Option value="weekly">周报</Option>
              <Option value="monthly">月报</Option>
              <Option value="quarterly">季报</Option>
              <Option value="yearly">年报</Option>
            </Select>
            <RangePicker />
            <Button icon={<SettingOutlined />}>
              报告设置
            </Button>
            <Button icon={<DownloadOutlined />} onClick={() => exportReport('利润分析')}>
              导出报告
            </Button>
          </Space>
        </Col>
      </Row>

      {/* 关键指标卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="本月营收"
              value={currentMonthData.revenue / 10000}
              precision={1}
              suffix="万"
              prefix={<DollarOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
            <div style={{ marginTop: 8 }}>
              <span style={{ color: revenueGrowth > 0 ? '#52c41a' : '#ff4d4f' }}>
                {revenueGrowth > 0 ? <RiseOutlined /> : <FallOutlined />}
                {' '}
                {revenueGrowth > 0 ? '+' : ''}{revenueGrowth.toFixed(1)}% 同比
              </span>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="毛利润"
              value={currentMonthData.grossProfit / 10000}
              precision={1}
              suffix="万"
              prefix={<DollarOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
            <div style={{ marginTop: 8 }}>
              <Progress percent={currentMonthData.grossMargin} size="small" />
              <span style={{ fontSize: 12, color: '#8c8c8c' }}>
                毛利率: {currentMonthData.grossMargin.toFixed(1)}%
              </span>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="净利润"
              value={currentMonthData.netProfit / 10000}
              precision={1}
              suffix="万"
              prefix={<RocketOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
            <div style={{ marginTop: 8 }}>
              <span style={{ color: profitGrowth > 0 ? '#52c41a' : '#ff4d4f' }}>
                {profitGrowth > 0 ? <RiseOutlined /> : <FallOutlined />}
                {' '}
                {profitGrowth > 0 ? '+' : ''}{profitGrowth.toFixed(1)}% 同比
              </span>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="净利率"
              value={currentMonthData.netMargin}
              precision={1}
              suffix="%"
              prefix={<BarChartOutlined />}
              valueStyle={{ color: '#13c2c2' }}
            />
            <div style={{ marginTop: 8 }}>
              <Text type="secondary">
                目标: 15.0% | 差距: {(15 - currentMonthData.netMargin).toFixed(1)}%
              </Text>
            </div>
          </Card>
        </Col>
      </Row>

      {/* 利润洞察提醒 */}
      {currentMonthData.netMargin < 15 && (
        <Alert
          message="利润率提醒"
          description={
            <Space>
              <span>
                当前净利率{currentMonthData.netMargin.toFixed(1)}%低于目标值15%，
                建议优化成本结构或提升高利润产品销售比例。
              </span>
              <Button type="link" size="small">查看优化建议</Button>
            </Space>
          }
          type="warning"
          showIcon
          closable
          style={{ marginBottom: 24 }}
        />
      )}

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="利润概览" key="overview">
          {/* 利润趋势图 */}
          <Card
            title="📈 利润趋势分析"
            extra={<Button type="link" icon={<EyeOutlined />}>全屏查看</Button>}
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
              <div style={{ textAlign: 'center' }}>
                <BarChartOutlined style={{ fontSize: 48, color: '#8c8c8c', marginBottom: 16 }} />
                <div style={{ color: '#8c8c8c', fontSize: 16 }}>
                  利润趋势图表
                  <br />
                  (集成 Recharts 或 ECharts)
                  <br />
                  <br />
                  显示营收、成本、毛利润、净利润趋势
                </div>
              </div>
            </div>
          </Card>

          {/* 成本结构分析 */}
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="💰 成本结构分析">
                <div style={{ height: 300 }}>
                  <div style={{ textAlign: 'center', padding: 40 }}>
                    <PieChartOutlined style={{ fontSize: 36, color: '#8c8c8c', marginBottom: 16 }} />
                    <div style={{ color: '#8c8c8c' }}>
                      成本结构饼图
                      <br />
                      显示原材料、人工、运营、运输等成本占比
                    </div>
                  </div>
                </div>
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title="📊 成本构成明细">
                <div style={{ height: 300, overflow: 'auto' }}>
                  {costBreakdown.map((item, index) => (
                    <div key={index} style={{ marginBottom: 16 }}>
                      <Row justify="space-between" align="middle">
                        <Col flex="auto">
                          <div style={{ fontWeight: 600 }}>{item.category}</div>
                          <div style={{ fontSize: 12, color: '#8c8c8c' }}>
                            {item.description}
                          </div>
                        </Col>
                        <Col>
                          <div style={{ textAlign: 'right' }}>
                            <div style={{ fontWeight: 600 }}>
                              ¥{(item.amount / 10000).toFixed(1)}万
                            </div>
                            <Tag color="blue" size="small">
                              {item.percentage.toFixed(1)}%
                            </Tag>
                          </div>
                        </Col>
                      </Row>
                      <Progress
                        percent={item.percentage}
                        size="small"
                        strokeColor={
                          item.trend === 'up' ? '#ff4d4f' :
                          item.trend === 'down' ? '#52c41a' : '#8c8c8c'
                        }
                        style={{ marginTop: 4 }}
                      />
                    </div>
                  ))}
                </div>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="产品利润分析" key="products">
          <Card
            title="🏭 产品类别利润分析"
            extra={
              <Space>
                <Button size="small">导出数据</Button>
                <Button size="small" type="primary">产品优化建议</Button>
              </Space>
            }
          >
            <Table
              columns={productProfitColumns}
              dataSource={productProfitability}
              pagination={false}
              size="middle"
            />
          </Card>

          <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
            <Col xs={24} lg={12}>
              <Card title="📈 产品销量趋势">
                <div style={{ height: 250 }}>
                  <div style={{ textAlign: 'center', padding: 40 }}>
                    <LineChartOutlined style={{ fontSize: 32, color: '#8c8c8c', marginBottom: 16 }} />
                    <div style={{ color: '#8c8c8c' }}>产品销量趋势图</div>
                  </div>
                </div>
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title="💎 高利润产品推荐">
                <div style={{ height: 250, padding: 20 }}>
                  <Alert
                    message="产品优化建议"
                    description={
                      <div>
                        <div style={{ marginBottom: 8 }}>
                          <strong>高强度螺栓</strong> - 利润率25%，增长强劲
                        </div>
                        <div style={{ marginBottom: 8 }}>
                          <strong>不锈钢紧固件</strong> - 利润率25%，稳定增长
                        </div>
                        <div>
                          <strong>建议:</strong> 加大高利润产品推广，优化标准件产品定价策略
                        </div>
                      </div>
                    }
                    type="info"
                    showIcon
                  />
                </div>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="客户利润分析" key="customers">
          <Card
            title="👥 客户利润贡献分析"
            extra={
              <Space>
                <Select defaultValue="all" style={{ width: 120 }} size="small">
                  <Option value="all">全部客户</Option>
                  <Option value="vip">VIP客户</Option>
                  <Option value="potential">潜力客户</Option>
                </Select>
                <Button size="small">客户分级管理</Button>
              </Space>
            }
          >
            <Table
              columns={customerProfitColumns}
              dataSource={customerProfitability}
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showTotal: (total) => `共 ${total} 个客户`,
              }}
              size="middle"
            />
          </Card>
        </TabPane>

        <TabPane tab="财务报告" key="reports">
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={8}>
              <Card
                title="📄 月度利润报告"
                extra={
                  <Button
                    type="link"
                    icon={<DownloadOutlined />}
                    onClick={() => exportReport('月度利润')}
                  >
                    下载
                  </Button>
                }
              >
                <div style={{ textAlign: 'center', padding: 20 }}>
                  <FileTextOutlined style={{ fontSize: 36, color: '#1890ff', marginBottom: 16 }} />
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ fontSize: 16, fontWeight: 600 }}>2024年1月</div>
                    <div style={{ color: '#8c8c8c' }}>月度利润报告</div>
                  </div>
                  <Button type="primary" block>生成报告</Button>
                </div>
              </Card>
            </Col>
            <Col xs={24} lg={8}>
              <Card
                title="📊 季度分析报告"
                extra={
                  <Button
                    type="link"
                    icon={<DownloadOutlined />}
                    onClick={() => exportReport('季度分析')}
                  >
                    下载
                  </Button>
                }
              >
                <div style={{ textAlign: 'center', padding: 20 }}>
                  <BarChartOutlined style={{ fontSize: 36, color: '#52c41a', marginBottom: 16 }} />
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ fontSize: 16, fontWeight: 600 }}>Q1 2024</div>
                    <div style={{ color: '#8c8c8c' }}>季度综合分析</div>
                  </div>
                  <Button type="primary" block>生成报告</Button>
                </div>
              </Card>
            </Col>
            <Col xs={24} lg={8}>
              <Card
                title="🎯 年度预测报告"
                extra={
                  <Button
                    type="link"
                    icon={<DownloadOutlined />}
                    onClick={() => exportReport('年度预测')}
                  >
                    下载
                  </Button>
                }
              >
                <div style={{ textAlign: 'center', padding: 20 }}>
                  <RocketOutlined style={{ fontSize: 36, color: '#722ed1', marginBottom: 16 }} />
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ fontSize: 16, fontWeight: 600 }}>2024年度</div>
                    <div style={{ color: '#8c8c8c' }}>业绩预测与目标</div>
                  </div>
                  <Button type="primary" block>生成报告</Button>
                </div>
              </Card>
            </Col>
          </Row>

          <Card title="📋 自定义报告" style={{ marginTop: 24 }}>
            <Form layout="inline">
              <Form.Item label="报告类型">
                <Select style={{ width: 150 }} defaultValue="profit">
                  <Option value="profit">利润分析</Option>
                  <Option value="cost">成本分析</Option>
                  <Option value="customer">客户分析</Option>
                  <Option value="product">产品分析</Option>
                </Select>
              </Form.Item>
              <Form.Item label="时间范围">
                <RangePicker />
              </Form.Item>
              <Form.Item label="输出格式">
                <Select style={{ width: 100 }} defaultValue="pdf">
                  <Option value="pdf">PDF</Option>
                  <Option value="excel">Excel</Option>
                  <Option value="word">Word</Option>
                </Select>
              </Form.Item>
              <Form.Item>
                <Button type="primary">生成自定义报告</Button>
              </Form.Item>
            </Form>
          </Card>
        </TabPane>
      </Tabs>
    </div>
  );
};

export default ProfitabilityReports;