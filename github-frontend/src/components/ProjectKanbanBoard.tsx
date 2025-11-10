import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Button,
  Space,
  Tag,
  Progress,
  Timeline,
  Modal,
  Form,
  Input,
  DatePicker,
  Select,
  InputNumber,
  Avatar,
  Tooltip,
  Badge,
  Tabs,
  List,
  Alert,
  Typography
} from 'antd';
import {
  ProjectOutlined,
  CalendarOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  WarningOutlined,
  PlusOutlined,
  EditOutlined,
  FilterOutlined,
  UserOutlined,
  DollarOutlined,
  FileTextOutlined,
  TeamOutlined,
  TruckOutlined,
  StarOutlined
} from '@ant-design/icons';
import type { DragEndEvent, DragStartEvent } from '@dnd-kit/core';
import dayjs from 'dayjs';

const { TabPane } = Tabs;
const { Option } = Select;
const { TextArea } = Input;
const { Text } = Typography;

interface Project {
  id: string;
  title: string;
  customer: string;
  stage: 'inquiry' | 'quotation' | 'order_confirmed' | 'production' | 'quality_inspection' | 'shipment' | 'delivered';
  priority: 'high' | 'medium' | 'low';
  status: 'on_track' | 'at_risk' | 'delayed';
  value: number;
  progress: number;
  startDate: string;
  expectedDelivery: string;
  assignedTo: string;
  description: string;
  tags: string[];
  milestones: Milestone[];
  documents: Document[];
}

interface Milestone {
  id: string;
  title: string;
  completed: boolean;
  dueDate: string;
  completedDate?: string;
}

interface Document {
  id: string;
  name: string;
  type: 'contract' | 'drawing' | 'quality_report' | 'invoice';
  uploadDate: string;
  url: string;
}

const ProjectKanbanBoard: React.FC = () => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [activeTab, setActiveTab] = useState('board');

  // 项目阶段配置
  const stages = [
    { key: 'inquiry', title: '询价阶段', color: '#6366f1', icon: <FileTextOutlined /> },
    { key: 'quotation', title: '报价阶段', color: '#8b5cf6', icon: <DollarOutlined /> },
    { key: 'order_confirmed', title: '订单确认', color: '#ec4899', icon: <CheckCircleOutlined /> },
    { key: 'production', title: '生产制造', color: '#84cc16', icon: <TeamOutlined /> },
    { key: 'quality_inspection', title: '质量检验', color: '#10b981', icon: <StarOutlined /> },
    { key: 'shipment', title: '包装发货', color: '#06b6d4', icon: <TruckOutlined /> },
    { key: 'delivered', title: '交付完成', color: '#22c55e', icon: <CheckCircleOutlined /> },
  ];

  // 模拟数据加载
  useEffect(() => {
    const mockProjects: Project[] = [
      {
        id: '1',
        title: '汽车紧固件项目 - Q1订单',
        customer: '上海汽车制造有限公司',
        stage: 'production',
        priority: 'high',
        status: 'on_track',
        value: 1250000,
        progress: 65,
        startDate: '2024-01-10',
        expectedDelivery: '2024-02-15',
        assignedTo: '张项目经理',
        description: '新能源汽车专用高强度螺栓项目，包含8种规格，总数量50万件',
        tags: ['新能源汽车', '高强度螺栓', '大批量'],
        milestones: [
          { id: '1', title: '技术确认完成', completed: true, dueDate: '2024-01-15', completedDate: '2024-01-14' },
          { id: '2', title: '报价完成', completed: true, dueDate: '2024-01-18', completedDate: '2024-01-17' },
          { id: '3', title: '订单确认', completed: true, dueDate: '2024-01-20', completedDate: '2024-01-19' },
          { id: '4', title: '生产启动', completed: true, dueDate: '2024-01-25', completedDate: '2024-01-24' },
          { id: '5', title: '生产完成50%', completed: true, dueDate: '2024-02-05', completedDate: '2024-02-03' },
          { id: '6', title: '生产完成100%', completed: false, dueDate: '2024-02-10' },
        ],
        documents: [
          { id: '1', name: '技术规格书.pdf', type: 'drawing', uploadDate: '2024-01-12', url: '#' },
          { id: '2', name: '销售合同.docx', type: 'contract', uploadDate: '2024-01-19', url: '#' },
        ]
      },
      {
        id: '2',
        title: '机械设备出口项目',
        customer: '德国AutoParts GmbH',
        stage: 'quality_inspection',
        priority: 'medium',
        status: 'at_risk',
        value: 850000,
        progress: 85,
        startDate: '2024-01-05',
        expectedDelivery: '2024-02-01',
        assignedTo: '李项目经理',
        description: '工业机械设备专用不锈钢紧固件，出口德国，需要特殊表面处理',
        tags: ['出口', '不锈钢', '特殊处理'],
        milestones: [
          { id: '1', title: '技术确认完成', completed: true, dueDate: '2024-01-08', completedDate: '2024-01-07' },
          { id: '2', title: '生产完成', completed: true, dueDate: '2024-01-20', completedDate: '2024-01-18' },
          { id: '3', title: '质量检验', completed: false, dueDate: '2024-01-25' },
          { id: '4', title: '包装发货', completed: false, dueDate: '2024-01-28' },
        ],
        documents: [
          { id: '1', name: '质量检验报告.pdf', type: 'quality_report', uploadDate: '2024-01-18', url: '#' },
          { id: '2', name: '出口文件.zip', type: 'contract', uploadDate: '2024-01-15', url: '#' },
        ]
      },
      {
        id: '3',
        title: '建筑工程项目',
        customer: '深圳建筑集团',
        stage: 'quotation',
        priority: 'low',
        status: 'on_track',
        value: 560000,
        progress: 25,
        startDate: '2024-01-18',
        expectedDelivery: '2024-03-15',
        assignedTo: '王项目经理',
        description: '大型建筑工程项目用高强度螺栓和锚固件，包含多种规格定制产品',
        tags: ['建筑工程', '定制产品', '大批量'],
        milestones: [
          { id: '1', title: '技术确认完成', completed: true, dueDate: '2024-01-22', completedDate: '2024-01-21' },
          { id: '2', title: '报价完成', completed: false, dueDate: '2024-01-25' },
        ],
        documents: [
          { id: '1', name: '技术图纸.dwg', type: 'drawing', uploadDate: '2024-01-19', url: '#' },
        ]
      }
    ];

    setProjects(mockProjects);
  }, []);

  // 获取阶段项目
  const getProjectsByStage = (stage: string) => {
    return projects.filter(project => project.stage === stage);
  };

  // 获取优先级颜色
  const getPriorityColor = (priority: string) => {
    const colors = {
      high: 'red',
      medium: 'orange',
      low: 'blue'
    };
    return colors[priority] || 'default';
  };

  // 获取状态图标
  const getStatusIcon = (status: string) => {
    const icons = {
      on_track: <CheckCircleOutlined style={{ color: '#52c41a' }} />,
      at_risk: <ExclamationCircleOutlined style={{ color: '#fa8c16' }} />,
      delayed: <WarningOutlined style={{ color: '#ff4d4f' }} />
    };
    return icons[status] || <ClockCircleOutlined />;
  };

  // 统计数据
  const statistics = {
    totalProjects: projects.length,
    onTrackProjects: projects.filter(p => p.status === 'on_track').length,
    atRiskProjects: projects.filter(p => p.status === 'at_risk').length,
    delayedProjects: projects.filter(p => p.status === 'delayed').length,
    totalValue: projects.reduce((sum, p) => sum + p.value, 0),
    averageProgress: projects.length > 0
      ? Math.round(projects.reduce((sum, p) => sum + p.progress, 0) / projects.length)
      : 0
  };

  // 渲染项目卡片
  const renderProjectCard = (project: Project) => (
    <Card
      key={project.id}
      size="small"
      style={{
        marginBottom: 16,
        backgroundColor: '#fff',
        cursor: 'pointer',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
      }}
      bodyStyle={{ padding: 12 }}
      onClick={() => {
        setSelectedProject(project);
        setDetailModalVisible(true);
      }}
    >
      <div style={{ marginBottom: 8 }}>
        <Space direction="vertical" size="small" style={{ width: '100%' }}>
          <div style={{ fontWeight: 600, fontSize: 14 }}>{project.title}</div>
          <div style={{ fontSize: 12, color: '#666' }}>{project.customer}</div>
        </Space>
      </div>

      <div style={{ marginBottom: 8 }}>
        <Space size="small">
          <Tag color={getPriorityColor(project.priority)} size="small">
            {project.priority === 'high' ? '高' : project.priority === 'medium' ? '中' : '低'}
          </Tag>
          {getStatusIcon(project.status)}
          <span style={{ fontSize: 12, color: '#666' }}>
            ¥{(project.value / 10000).toFixed(0)}万
          </span>
        </Space>
      </div>

      <div style={{ marginBottom: 8 }}>
        <Progress percent={project.progress} size="small" showInfo={false} />
        <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
          {project.progress}% 完成
        </div>
      </div>

      <div style={{ marginBottom: 8 }}>
        <div style={{ fontSize: 12, color: '#666' }}>
          <UserOutlined style={{ marginRight: 4 }} />
          {project.assignedTo}
        </div>
        <div style={{ fontSize: 12, color: '#666' }}>
          <CalendarOutlined style={{ marginRight: 4 }} />
          {project.expectedDelivery}
        </div>
      </div>

      {project.tags.length > 0 && (
        <div>
          {project.tags.slice(0, 2).map(tag => (
            <Tag key={tag} size="small" style={{ fontSize: 10, marginBottom: 4 }}>
              {tag}
            </Tag>
          ))}
          {project.tags.length > 2 && (
            <Tag size="small" style={{ fontSize: 10 }}>+{project.tags.length - 2}</Tag>
          )}
        </div>
      )}
    </Card>
  );

  return (
    <div style={{ padding: '24px' }}>
      {/* 头部 */}
      <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <h1 style={{ fontSize: 24, fontWeight: 600, margin: 0 }}>
            📋 项目跟踪看板
          </h1>
          <p style={{ color: '#8c8c8c', margin: 0 }}>
            拖拽式项目管理 - 实时跟踪{statistics.totalProjects}个项目
          </p>
        </Col>
        <Col>
          <Space>
            <Button icon={<PlusOutlined />} type="primary">
              新建项目
            </Button>
            <Button icon={<FilterOutlined />}>
              筛选
            </Button>
            <Button icon={<CalendarOutlined />}>
              甘特图
            </Button>
          </Space>
        </Col>
      </Row>

      {/* 统计卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="总项目数"
              value={statistics.totalProjects}
              prefix={<ProjectOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
            <Text type="secondary">进行中: {projects.filter(p => p.stage !== 'delivered').length}</Text>
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="正常进度"
              value={statistics.onTrackProjects}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
            <Text type="secondary">占总数 {((statistics.onTrackProjects / statistics.totalProjects) * 100).toFixed(0)}%</Text>
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="风险项目"
              value={statistics.atRiskProjects + statistics.delayedProjects}
              prefix={<WarningOutlined />}
              valueStyle={{ color: '#fa8c16' }}
            />
            <Text type="secondary">需要关注</Text>
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="平均进度"
              value={statistics.averageProgress}
              suffix="%"
              prefix={<FileTextOutlined />}
              valueStyle={{ color: '#13c2c2' }}
            />
            <Text type="secondary">整体完成度</Text>
          </Card>
        </Col>
      </Row>

      {/* 风险提醒 */}
      {(statistics.atRiskProjects > 0 || statistics.delayedProjects > 0) && (
        <Alert
          message="项目风险提醒"
          description={`发现 ${statistics.atRiskProjects} 个项目存在风险，${statistics.delayedProjects} 个项目已延期。请及时关注并采取相应措施。`}
          type="warning"
          showIcon
          closable
          style={{ marginBottom: 24 }}
        />
      )}

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="看板视图" key="board">
          {/* 看板布局 */}
          <Row gutter={[16, 16]}>
            {stages.map((stage, index) => (
              <Col xs={24} sm={12} lg={8} xl={6} key={stage.key}>
                <Card
                  title={
                    <Space>
                      {stage.icon}
                      <span>{stage.title}</span>
                      <Badge count={getProjectsByStage(stage.key).length} size="small" />
                    </Space>
                  }
                  style={{
                    backgroundColor: `${stage.color}08`,
                    borderLeft: `4px solid ${stage.color}`,
                    minHeight: 500
                  }}
                  bodyStyle={{ padding: 16 }}
                >
                  {getProjectsByStage(stage.key).map(project => renderProjectCard(project))}
                </Card>
              </Col>
            ))}
          </Row>
        </TabPane>

        <TabPane tab="列表视图" key="list">
          <Card>
            {/* 项目列表表格将在此显示 */}
            <div style={{
              height: 400,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: '#fafafa',
              borderRadius: 6
            }}>
              <Text type="secondary">项目列表表格组件将在此显示</Text>
            </div>
          </Card>
        </TabPane>

        <TabPane tab="甘特图" key="gantt">
          <Card>
            {/* 甘特图组件将在此显示 */}
            <div style={{
              height: 600,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: '#fafafa',
              borderRadius: 6
            }}>
              <Text type="secondary">甘特图组件将在此显示 (集成 dhtmlx-gantt 或 react-gantt-timeline)</Text>
            </div>
          </Card>
        </TabPane>
      </Tabs>

      {/* 项目详情模态框 */}
      <Modal
        title={selectedProject ? `项目详情 - ${selectedProject.title}` : '项目详情'}
        visible={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        width={1000}
        footer={[
          <Button key="edit" type="primary" icon={<EditOutlined />}>
            编辑项目
          </Button>,
          <Button key="export" icon={<FileTextOutlined />}>
            导出报告
          </Button>
        ]}
      >
        {selectedProject && (
          <Tabs defaultActiveKey="overview">
            <TabPane tab="概览" key="overview">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>客户:</Text> {selectedProject.customer}
                  </div>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>项目经理:</Text> {selectedProject.assignedTo}
                  </div>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>项目价值:</Text> ¥{(selectedProject.value / 10000).toFixed(0)}万
                  </div>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>进度:</Text> {selectedProject.progress}%
                  </div>
                </Col>
                <Col span={12}>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>开始日期:</Text> {selectedProject.startDate}
                  </div>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>预计交付:</Text> {selectedProject.expectedDelivery}
                  </div>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>优先级:</Text>
                    <Tag color={getPriorityColor(selectedProject.priority)}>
                      {selectedProject.priority === 'high' ? '高' : selectedProject.priority === 'medium' ? '中' : '低'}
                    </Tag>
                  </div>
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>状态:</Text> {getStatusIcon(selectedProject.status)}
                  </div>
                </Col>
              </Row>

              <div style={{ marginBottom: 16 }}>
                <Text strong>项目描述:</Text>
                <p style={{ marginTop: 8 }}>{selectedProject.description}</p>
              </div>

              <div style={{ marginBottom: 16 }}>
                <Text strong>标签:</Text>
                <div style={{ marginTop: 8 }}>
                  {selectedProject.tags.map(tag => (
                    <Tag key={tag} color="blue">{tag}</Tag>
                  ))}
                </div>
              </div>

              <div>
                <Text strong>整体进度:</Text>
                <Progress
                  percent={selectedProject.progress}
                  status={selectedProject.status === 'delayed' ? 'exception' : 'active'}
                  style={{ marginTop: 8 }}
                />
              </div>
            </TabPane>

            <TabPane tab="里程碑" key="milestones">
              <Timeline>
                {selectedProject.milestones.map(milestone => (
                  <Timeline.Item
                    key={milestone.id}
                    color={milestone.completed ? 'green' : 'blue'}
                    dot={milestone.completed ? <CheckCircleOutlined /> : <ClockCircleOutlined />}
                  >
                    <div style={{ fontWeight: milestone.completed ? 600 : 400 }}>
                      {milestone.title}
                    </div>
                    <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
                      截止日期: {milestone.dueDate}
                    </div>
                    {milestone.completed && milestone.completedDate && (
                      <div style={{ fontSize: 12, color: '#52c41a', marginTop: 4 }}>
                        完成日期: {milestone.completedDate}
                      </div>
                    )}
                  </Timeline.Item>
                ))}
              </Timeline>
            </TabPane>

            <TabPane tab="文档" key="documents">
              <List
                dataSource={selectedProject.documents}
                renderItem={doc => (
                  <List.Item key={doc.id}>
                    <List.Item.Meta
                      avatar={<Avatar icon={<FileTextOutlined />} />}
                      title={doc.name}
                      description={`${doc.type} - 上传于 ${doc.uploadDate}`}
                    />
                    <Button type="link" size="small">下载</Button>
                  </List.Item>
                )}
              />
            </TabPane>
          </Tabs>
        )}
      </Modal>
    </div>
  );
};

export default ProjectKanbanBoard;