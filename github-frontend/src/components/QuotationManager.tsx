import React, { useState, useEffect } from 'react';
import {
  Card,
  Table,
  Button,
  Space,
  Tag,
  Modal,
  Form,
  Input,
  InputNumber,
  Select,
  Row,
  Col,
  Statistic,
  Timeline,
  Badge,
  Tooltip,
  Tabs,
  message,
  Progress,
  Divider,
  Popconfirm,
  DatePicker,
  Typography
} from 'antd';
import {
  DollarOutlined,
  FileTextOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  EyeOutlined,
  CopyOutlined,
  SendOutlined,
  CalculatorOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  BarChartOutlined
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import dayjs from 'dayjs';

const { Option } = Select;
const { TextArea } = Input;
const { TabPane } = Tabs;
const { Text } = Typography;

interface QuotationItem {
  id: string;
  customerId: string;
  customerName: string;
  inquiryId: string;
  quotationNumber: string;
  items: QuotationLineItem[];
  totalAmount: number;
  currency: string;
  margin: number;
  status: 'draft' | 'sent' | 'accepted' | 'rejected' | 'expired';
  priority: 'high' | 'medium' | 'low';
  validUntil: string;
  createdAt: string;
  lastModified: string;
  assignedTo: string;
  notes: string;
  attachments: string[];
}

interface QuotationLineItem {
  id: string;
  productCode: string;
  description: string;
  specification: string;
  quantity: number;
  unitPrice: number;
  totalPrice: number;
  supplier: string;
  leadTime: number;
  material: string;
  standard: string;
}

interface QuotationTemplate {
  id: string;
  name: string;
  category: string;
  items: Omit<QuotationLineItem, 'totalPrice'>[];
  marginRate: number;
  description: string;
}

const QuotationManager: React.FC = () => {
  const [quotations, setQuotations] = useState<QuotationItem[]>([]);
  const [templates, setTemplates] = useState<QuotationTemplate[]>([]);
  const [selectedQuotation, setSelectedQuotation] = useState<QuotationItem | null>(null);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [templateModalVisible, setTemplateModalVisible] = useState(false);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('list');

  // 模拟数据
  useEffect(() => {
    const mockQuotations: QuotationItem[] = [
      {
        id: '1',
        customerId: '1',
        customerName: '上海汽车制造有限公司',
        inquiryId: 'INQ-2024-001',
        quotationNumber: 'QT-2024-001',
        items: [
          {
            id: '1',
            productCode: 'FB-001',
            description: '高强度螺栓 M16x80',
            specification: 'M16x80, 8.8级，镀锌',
            quantity: 10000,
            unitPrice: 2.85,
            totalPrice: 28500,
            supplier: '东方金属制品厂',
            leadTime: 15,
            material: '碳钢 Q235',
            standard: 'ISO 4014'
          },
          {
            id: '2',
            productCode: 'FB-002',
            description: '螺母 M16',
            specification: 'M16, 8级，镀锌',
            quantity: 10000,
            unitPrice: 0.85,
            totalPrice: 8500,
            supplier: '东方金属制品厂',
            leadTime: 15,
            material: '碳钢 Q235',
            standard: 'ISO 4032'
          }
        ],
        totalAmount: 37000,
        currency: 'CNY',
        margin: 25.5,
        status: 'sent',
        priority: 'high',
        validUntil: '2024-02-15',
        createdAt: '2024-01-15',
        lastModified: '2024-01-15',
        assignedTo: '李销售',
        notes: '新能源汽车项目，紧急询价',
        attachments: ['技术规格书.pdf', '图纸.dwg']
      },
      {
        id: '2',
        customerId: '2',
        customerName: '德国AutoParts GmbH',
        inquiryId: 'INQ-2024-002',
        quotationNumber: 'QT-2024-002',
        items: [
          {
            id: '3',
            productCode: 'SS-001',
            description: '不锈钢螺栓 A2-70 M12x50',
            specification: 'M12x50, A2-70',
            quantity: 5000,
            unitPrice: 4.25,
            totalPrice: 21250,
            supplier: '精密不锈钢公司',
            leadTime: 20,
            material: '不锈钢 304',
            standard: 'DIN 933'
          }
        ],
        totalAmount: 21250,
        currency: 'EUR',
        margin: 32.8,
        status: 'accepted',
        priority: 'medium',
        validUntil: '2024-02-20',
        createdAt: '2024-01-18',
        lastModified: '2024-01-19',
        assignedTo: '王经理',
        notes: '出口订单，需要特殊包装',
        attachments: ['质量证书.pdf']
      }
    ];

    const mockTemplates: QuotationTemplate[] = [
      {
        id: '1',
        name: '高强度螺栓标准模板',
        category: '标准件',
        items: [
          {
            id: '1',
            productCode: 'FB-STD-001',
            description: '高强度螺栓',
            specification: '标准规格',
            quantity: 1000,
            unitPrice: 2.50,
            supplier: '标准供应商',
            leadTime: 14,
            material: '碳钢',
            standard: 'ISO'
          }
        ],
        marginRate: 25.0,
        description: '适用于标准高强度螺栓报价'
      }
    ];

    setQuotations(mockQuotations);
    setTemplates(mockTemplates);
  }, []);

  // 报价单表格列定义
  const quotationColumns: ColumnsType<QuotationItem> = [
    {
      title: '报价单号',
      dataIndex: 'quotationNumber',
      key: 'quotationNumber',
      render: (text) => <strong>{text}</strong>,
    },
    {
      title: '客户',
      dataIndex: 'customerName',
      key: 'customerName',
    },
    {
      title: '询价单号',
      dataIndex: 'inquiryId',
      key: 'inquiryId',
    },
    {
      title: '总金额',
      key: 'amount',
      render: (_, record) => (
        <span>
          {record.currency === 'CNY' ? '¥' : '€'}
          {record.totalAmount.toLocaleString()}
        </span>
      ),
    },
    {
      title: '毛利率',
      dataIndex: 'margin',
      key: 'margin',
      render: (margin) => (
        <Tag color={margin > 30 ? 'green' : margin > 20 ? 'orange' : 'red'}>
          {margin.toFixed(1)}%
        </Tag>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => {
        const statusConfig = {
          draft: { color: 'default', text: '草稿' },
          sent: { color: 'processing', text: '已发送' },
          accepted: { color: 'success', text: '已接受' },
          rejected: { color: 'error', text: '已拒绝' },
          expired: { color: 'warning', text: '已过期' }
        };
        const config = statusConfig[status];
        return <Badge status={status as any} text={config.text} />;
      },
    },
    {
      title: '有效期至',
      dataIndex: 'validUntil',
      key: 'validUntil',
      render: (date) => (
        <span style={{
          color: dayjs(date).isBefore(dayjs()) ? '#ff4d4f' : 'inherit'
        }}>
          {date}
        </span>
      ),
    },
    {
      title: '负责销售',
      dataIndex: 'assignedTo',
      key: 'assignedTo',
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
                setSelectedQuotation(record);
                setDetailModalVisible(true);
              }}
            />
          </Tooltip>
          <Tooltip title="编辑">
            <Button
              type="text"
              icon={<EditOutlined />}
              onClick={() => message.info('编辑功能开发中')}
            />
          </Tooltip>
          <Tooltip title="复制模板">
            <Button
              type="text"
              icon={<CopyOutlined />}
              onClick={() => message.info('复制模板功能开发中')}
            />
          </Tooltip>
          <Tooltip title="发送">
            <Button
              type="text"
              icon={<SendOutlined />}
              onClick={() => message.success('报价单已发送')}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  // 计算统计数据
  const statistics = {
    totalQuotations: quotations.length,
    pendingQuotations: quotations.filter(q => q.status === 'sent').length,
    acceptedQuotations: quotations.filter(q => q.status === 'accepted').length,
    totalValue: quotations.reduce((sum, q) => sum + q.totalAmount, 0),
    averageMargin: quotations.length > 0
      ? quotations.reduce((sum, q) => sum + q.margin, 0) / quotations.length
      : 0
  };

  // 计算器功能
  const openCalculator = () => {
    message.info('价格计算器功能开发中');
  };

  return (
    <div style={{ padding: '24px' }}>
      {/* 头部 */}
      <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <h1 style={{ fontSize: 24, fontWeight: 600, margin: 0 }}>
            💼 报价管理系统
          </h1>
          <p style={{ color: '#8c8c8c', margin: 0 }}>
            智能报价生成与模板管理 - {statistics.totalQuotations}个报价单
          </p>
        </Col>
        <Col>
          <Space>
            <Button icon={<CalculatorOutlined />} onClick={openCalculator}>
              价格计算器
            </Button>
            <Button icon={<BarChartOutlined />}>
              成本分析
            </Button>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => setCreateModalVisible(true)}
            >
              新建报价
            </Button>
          </Space>
        </Col>
      </Row>

      {/* 统计卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="总报价单"
              value={statistics.totalQuotations}
              prefix={<FileTextOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
            <Text type="secondary">待处理: {statistics.pendingQuotations}</Text>
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="已接受"
              value={statistics.acceptedQuotations}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
            <Text type="secondary">成功率: {statistics.totalQuotations > 0 ? ((statistics.acceptedQuotations / statistics.totalQuotations) * 100).toFixed(0) : 0}%</Text>
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="总金额"
              value={statistics.totalValue / 10000}
              precision={1}
              suffix="万"
              prefix={<DollarOutlined />}
              valueStyle={{ color: '#13c2c2' }}
            />
            <Text type="secondary">累计报价</Text>
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="平均毛利率"
              value={statistics.averageMargin}
              precision={1}
              suffix="%"
              prefix={<BarChartOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
            <Text type="secondary">利润分析</Text>
          </Card>
        </Col>
      </Row>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="报价单列表" key="list">
          <Card>
            <Table
              columns={quotationColumns}
              dataSource={quotations}
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total) => `共 ${total} 个报价单`,
              }}
              loading={loading}
            />
          </Card>
        </TabPane>

        <TabPane tab="模板库" key="templates">
          <Card
            title="报价模板库"
            extra={
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={() => setTemplateModalVisible(true)}
              >
                新建模板
              </Button>
            }
          >
            <Row gutter={[16, 16]}>
              {templates.map(template => (
                <Col xs={24} sm={12} lg={8} key={template.id}>
                  <Card
                    size="small"
                    title={template.name}
                    extra={<Tag color="blue">{template.category}</Tag>}
                    actions={[
                      <EditOutlined key="edit" onClick={() => message.info('编辑模板')} />,
                      <CopyOutlined key="copy" onClick={() => message.info('使用模板')} />,
                      <DeleteOutlined key="delete" onClick={() => message.info('删除模板')} />
                    ]}
                  >
                    <p><strong>描述:</strong> {template.description}</p>
                    <p><strong>毛利率:</strong> {template.marginRate}%</p>
                    <p><strong>项目数:</strong> {template.items.length}</p>
                  </Card>
                </Col>
              ))}
            </Row>
          </Card>
        </TabPane>
      </Tabs>

      {/* 报价详情模态框 */}
      <Modal
        title={selectedQuotation ? `报价详情 - ${selectedQuotation.quotationNumber}` : '报价详情'}
        visible={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        width={1200}
        footer={[
          <Button key="edit" icon={<EditOutlined />}>
            编辑
          </Button>,
          <Button key="export" icon={<FileTextOutlined />}>
            导出PDF
          </Button>,
          <Button key="send" type="primary" icon={<SendOutlined />}>
            发送客户
          </Button>,
        ]}
      >
        {selectedQuotation && (
          <Tabs defaultActiveKey="overview">
            <TabPane tab="基本信息" key="overview">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>报价单号:</Text> {selectedQuotation.quotationNumber}
                  </div>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>询价单号:</Text> {selectedQuotation.inquiryId}
                  </div>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>客户:</Text> {selectedQuotation.customerName}
                  </div>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>负责销售:</Text> {selectedQuotation.assignedTo}
                  </div>
                </Col>
                <Col span={12}>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>总金额:</Text> {selectedQuotation.currency === 'CNY' ? '¥' : '€'}{selectedQuotation.totalAmount.toLocaleString()}
                  </div>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>毛利率:</Text> {selectedQuotation.margin}%
                  </div>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>有效期至:</Text> {selectedQuotation.validUntil}
                  </div>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>创建时间:</Text> {selectedQuotation.createdAt}
                  </div>
                </Col>
              </Row>

              <div style={{ marginBottom: 16 }}>
                <Text strong>备注:</Text>
                <p style={{ marginTop: 8 }}>{selectedQuotation.notes}</p>
              </div>

              <div>
                <Text strong>状态:</Text>
                <Badge
                  status={selectedQuotation.status as any}
                  text={selectedQuotation.status}
                  style={{ marginLeft: 8 }}
                />
              </div>
            </TabPane>

            <TabPane tab="报价项目" key="items">
              <Table
                dataSource={selectedQuotation.items}
                columns={[
                  { title: '产品编码', dataIndex: 'productCode' },
                  { title: '描述', dataIndex: 'description' },
                  { title: '规格', dataIndex: 'specification' },
                  {
                    title: '数量',
                    dataIndex: 'quantity',
                    render: (text) => text.toLocaleString()
                  },
                  {
                    title: '单价',
                    dataIndex: 'unitPrice',
                    render: (text) => `${selectedQuotation.currency === 'CNY' ? '¥' : '€'}${text}`
                  },
                  {
                    title: '总价',
                    dataIndex: 'totalPrice',
                    render: (text) => `${selectedQuotation.currency === 'CNY' ? '¥' : '€'}${text.toLocaleString()}`
                  },
                  { title: '供应商', dataIndex: 'supplier' },
                  { title: '交期(天)', dataIndex: 'leadTime' }
                ]}
                pagination={false}
                size="small"
              />
            </TabPane>

            <TabPane tab="历史记录" key="history">
              <Timeline>
                <Timeline.Item color="blue">
                  <Text strong>创建报价单</Text>
                  <br />
                  <Text type="secondary">{selectedQuotation.createdAt} - {selectedQuotation.assignedTo}</Text>
                </Timeline.Item>
                <Timeline.Item color="green">
                  <Text strong>发送给客户</Text>
                  <br />
                  <Text type="secondary">2024-01-16 - 系统自动发送</Text>
                </Timeline.Item>
                <Timeline.Item color="orange">
                  <Text strong>等待客户反馈</Text>
                  <br />
                  <Text type="secondary">预计3个工作日内得到回复</Text>
                </Timeline.Item>
              </Timeline>
            </TabPane>
          </Tabs>
        )}
      </Modal>

      {/* 新建报价单模态框 */}
      <Modal
        title="新建报价单"
        visible={createModalVisible}
        onCancel={() => setCreateModalVisible(false)}
        width={800}
        footer={[
          <Button key="cancel" onClick={() => setCreateModalVisible(false)}>
            取消
          </Button>,
          <Button key="draft" type="default">
            保存草稿
          </Button>,
          <Button key="create" type="primary">
            创建报价单
          </Button>,
        ]}
      >
        <Form layout="vertical">
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="选择客户" required>
                <Select placeholder="请选择客户">
                  <Option value="1">上海汽车制造有限公司</Option>
                  <Option value="2">德国AutoParts GmbH</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="询价单号">
                <Input placeholder="系统自动生成" />
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="货币">
                <Select defaultValue="CNY">
                  <Option value="CNY">人民币 (CNY)</Option>
                  <Option value="USD">美元 (USD)</Option>
                  <Option value="EUR">欧元 (EUR)</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="有效期">
                <DatePicker style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item label="使用模板">
            <Select placeholder="选择报价模板（可选）">
              <Option value="1">高强度螺栓标准模板</Option>
            </Select>
          </Form.Item>
          <Form.Item label="备注">
            <TextArea rows={3} placeholder="报价备注信息" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default QuotationManager;