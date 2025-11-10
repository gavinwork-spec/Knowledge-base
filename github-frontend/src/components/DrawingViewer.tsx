import React, { useState } from 'react';
import {
  Card,
  Row,
  Col,
  Button,
  Upload,
  Space,
  Tag,
  Tabs,
  Table,
  Form,
  Input,
  Select,
  InputNumber,
  Alert,
  message
} from 'antd';
import {
  UploadOutlined,
  FileImageOutlined,
  ZoomInOutlined,
  ZoomOutOutlined,
  RotateLeftOutlined,
  RotateRightOutlined,
  SaveOutlined,
  DownloadOutlined,
  EyeOutlined,
  ToolOutlined,
  EditOutlined
} from '@ant-design/icons';

const { TabPane } = Tabs;
const { Dragger } = Upload;

interface DrawingSpecification {
  id: string;
  category: string;
  parameter: string;
  value: string;
  tolerance: string;
  standard: string;
}

interface Measurement {
  id: string;
  x: number;
  y: number;
  type: 'dimension' | 'angle' | 'radius';
  value: number;
  unit: string;
  description: string;
}

const DrawingViewer: React.FC = () => {
  const [drawingUrl, setDrawingUrl] = useState<string>('');
  const [uploading, setUploading] = useState(false);
  const [zoom, setZoom] = useState(100);
  const [rotation, setRotation] = useState(0);
  const [activeTab, setActiveTab] = useState('viewer');
  const [specifications, setSpecifications] = useState<DrawingSpecification[]>([]);
  const [measurements, setMeasurements] = useState<Measurement[]>([]);
  const [editingMode, setEditingMode] = useState(false);

  // 模拟规格数据
  const mockSpecifications: DrawingSpecification[] = [
    {
      id: '1',
      category: '螺栓类型',
      parameter: '螺栓直径',
      value: 'M16',
      tolerance: 'h6',
      standard: 'ISO 4014'
    },
    {
      id: '2',
      category: '螺栓类型',
      parameter: '螺栓长度',
      value: '80',
      tolerance: '±0.2',
      standard: 'ISO 4014'
    },
    {
      id: '3',
      category: '材料规格',
      parameter: '材料等级',
      value: '8.8',
      tolerance: 'N/A',
      standard: 'ISO 898'
    },
    {
      id: '4',
      category: '表面处理',
      parameter: '镀锌层',
      value: 'Zn8-C',
      tolerance: '≥8μm',
      standard: 'ISO 4042'
    }
  ];

  // 模拟测量数据
  const mockMeasurements: Measurement[] = [
    {
      id: '1',
      x: 150,
      y: 100,
      type: 'dimension',
      value: 16,
      unit: 'mm',
      description: '螺栓直径'
    },
    {
      id: '2',
      x: 300,
      y: 200,
      type: 'dimension',
      value: 80,
      unit: 'mm',
      description: '螺栓长度'
    }
  ];

  // 上传配置
  const uploadProps = {
    name: 'file',
    multiple: false,
    accept: '.dwg,.dxf,.pdf,.jpg,.jpeg,.png,.step,.stp',
    beforeUpload: (file) => {
      setUploading(true);
      // 模拟上传过程
      setTimeout(() => {
        const fileUrl = URL.createObjectURL(file);
        setDrawingUrl(fileUrl);
        setSpecifications(mockSpecifications);
        setMeasurements(mockMeasurements);
        setUploading(false);
        message.success('图纸上传成功！');
      }, 2000);
      return false; // 阻止默认上传行为
    },
  };

  // 工具栏操作
  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev + 20, 200));
  };

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev - 20, 50));
  };

  const handleRotateLeft = () => {
    setRotation(prev => prev - 90);
  };

  const handleRotateRight = () => {
    setRotation(prev => prev + 90);
  };

  const handleReset = () => {
    setZoom(100);
    setRotation(0);
  };

  const handleSave = () => {
    message.success('图纸和测量数据已保存');
  };

  const handleDownload = () => {
    message.info('下载功能开发中');
  };

  // 添加测量
  const handleAddMeasurement = () => {
    message.info('点击图纸添加测量点功能开发中');
  };

  // OCR识别
  const handleOCR = () => {
    message.loading('正在识别图纸内容...');
    setTimeout(() => {
      message.success('OCR识别完成！已提取规格参数');
      // 模拟OCR结果更新规格
    }, 3000);
  };

  // 规格表格列定义
  const specificationColumns = [
    {
      title: '类别',
      dataIndex: 'category',
      key: 'category',
    },
    {
      title: '参数',
      dataIndex: 'parameter',
      key: 'parameter',
    },
    {
      title: '值',
      dataIndex: 'value',
      key: 'value',
    },
    {
      title: '公差',
      dataIndex: 'tolerance',
      key: 'tolerance',
    },
    {
      title: '标准',
      dataIndex: 'standard',
      key: 'standard',
      render: (text) => <Tag color="blue">{text}</Tag>,
    },
  ];

  // 测量表格列定义
  const measurementColumns = [
    {
      title: 'X坐标',
      dataIndex: 'x',
      key: 'x',
    },
    {
      title: 'Y坐标',
      dataIndex: 'y',
      key: 'y',
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type) => {
        const typeMap = {
          dimension: '尺寸',
          angle: '角度',
          radius: '半径'
        };
        return typeMap[type] || type;
      },
    },
    {
      title: '值',
      dataIndex: 'value',
      key: 'value',
    },
    {
      title: '单位',
      dataIndex: 'unit',
      key: 'unit',
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      {/* 头部 */}
      <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <h1 style={{ fontSize: 24, fontWeight: 600, margin: 0 }}>
            📐 图纸查看器
          </h1>
          <p style={{ color: '#8c8c8c', margin: 0 }}>
            支持多种格式图纸查看与测量工具 - 集成OCR识别
          </p>
        </Col>
        <Col>
          <Space>
            <Button icon={<ToolOutlined />} onClick={handleOCR}>
              OCR识别
            </Button>
            <Button icon={<EditOutlined />} onClick={() => setEditingMode(!editingMode)}>
              {editingMode ? '退出编辑' : '编辑模式'}
            </Button>
            <Button icon={<SaveOutlined />} onClick={handleSave}>
              保存
            </Button>
            <Button icon={<DownloadOutlined />} onClick={handleDownload}>
              下载
            </Button>
          </Space>
        </Col>
      </Row>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="图纸查看" key="viewer">
          <Row gutter={[16, 16]}>
            <Col span={18}>
              {/* 图纸查看区域 */}
              <Card style={{ height: 600, position: 'relative' }}>
                {/* 工具栏 */}
                <div style={{
                  position: 'absolute',
                  top: 16,
                  right: 16,
                  zIndex: 10,
                  background: 'rgba(255,255,255,0.9)',
                  padding: '8px',
                  borderRadius: 6,
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                }}>
                  <Space>
                    <Button
                      size="small"
                      icon={<ZoomInOutlined />}
                      onClick={handleZoomIn}
                      disabled={zoom >= 200}
                    />
                    <Button
                      size="small"
                      icon={<ZoomOutOutlined />}
                      onClick={handleZoomOut}
                      disabled={zoom <= 50}
                    />
                    <Button
                      size="small"
                      icon={<RotateLeftOutlined />}
                      onClick={handleRotateLeft}
                    />
                    <Button
                      size="small"
                      icon={<RotateRightOutlined />}
                      onClick={handleRotateRight}
                    />
                    <Button
                      size="small"
                      onClick={handleReset}
                    >
                      重置
                    </Button>
                    {editingMode && (
                      <Button
                        size="small"
                        icon={<ToolOutlined />}
                        onClick={handleAddMeasurement}
                        type="primary"
                      >
                        添加测量
                      </Button>
                    )}
                  </Space>
                </div>

                {/* 图纸显示区域 */}
                <div
                  style={{
                    height: 600,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    backgroundColor: '#f5f5f5',
                    position: 'relative'
                  }}
                >
                  {drawingUrl ? (
                    <img
                      src={drawingUrl}
                      alt="图纸"
                      style={{
                        maxWidth: '100%',
                        maxHeight: '100%',
                        objectFit: 'contain',
                        transform: `scale(${zoom / 100}) rotate(${rotation}deg)`,
                        transition: 'transform 0.3s ease'
                      }}
                    />
                  ) : (
                    <Dragger {...uploadProps} style={{ width: '100%', height: 300 }}>
                      <p className="ant-upload-drag-icon">
                        <UploadOutlined style={{ fontSize: 48, color: '#1890ff' }} />
                      </p>
                      <p className="ant-upload-text">
                        点击或拖拽文件到此区域上传
                      </p>
                      <p className="ant-upload-hint">
                        支持 DWG, DXF, PDF, JPG, PNG, STEP 文件
                      </p>
                    </Dragger>
                  )}

                  {/* 缩放信息 */}
                  {drawingUrl && (
                    <div
                      style={{
                        position: 'absolute',
                        bottom: 16,
                        left: 16,
                        background: 'rgba(0,0,0,0.7)',
                        color: 'white',
                        padding: '4px 8px',
                        borderRadius: 4,
                        fontSize: 12
                      }}
                    >
                      缩放: {zoom}% | 旋转: {rotation}°
                    </div>
                  )}
                </div>
              </Card>
            </Col>

            <Col span={6}>
              {/* 快速操作面板 */}
              <Card title="快速操作" style={{ marginBottom: 16 }}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Button block icon={<EyeOutlined />}>
                    全屏查看
                  </Button>
                  <Button block icon={<FileImageOutlined />}>
                    截图工具
                  </Button>
                  <Button block icon={<ToolOutlined />}>
                    测量工具
                  </Button>
                  <Button block icon={<SaveOutlined />}>
                    导出PDF
                  </Button>
                </Space>
              </Card>

              {/* 图纸信息 */}
              <Card title="图纸信息" style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 14 }}>
                  <p><strong>文件名:</strong> {drawingUrl ? 'drawing_file' : '未上传'}</p>
                  <p><strong>文件类型:</strong> {drawingUrl ? 'Image' : '-'}</p>
                  <p><strong>文件大小:</strong> {drawingUrl ? '~2.5MB' : '-'}</p>
                  <p><strong>上传时间:</strong> {drawingUrl ? new Date().toLocaleString() : '-'}</p>
                </div>
              </Card>

              {/* 测量信息 */}
              <Card title="测量工具" style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 14 }}>
                  <p><strong>总测量点:</strong> {measurements.length}</p>
                  <p><strong>测量模式:</strong> {editingMode ? '开启' : '关闭'}</p>
                  <p><strong>精度设置:</strong> 0.1mm</p>
                </div>
              </Card>

              {/* 快捷键说明 */}
              <Card title="快捷键">
                <div style={{ fontSize: 12 }}>
                  <p><strong>鼠标滚轮:</strong> 缩放</p>
                  <p><strong>Shift + 拖拽:</strong> 移动</p>
                  <p><strong>双击:</strong> 全屏</p>
                  <p><strong>R:</strong> 重置视图</p>
                  <p><strong>M:</strong> 测量模式</p>
                </div>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="规格参数" key="specifications">
          <Card title="提取的规格参数" extra={
            <Button icon={<ToolOutlined />} onClick={handleOCR}>
              重新识别
            </Button>
          }>
            <Table
              columns={specificationColumns}
              dataSource={specifications}
              pagination={false}
              size="small"
            />
          </Card>
        </TabPane>

        <TabPane tab="测量数据" key="measurements">
          <Card title="测量记录">
            <Table
              columns={measurementColumns}
              dataSource={measurements}
              pagination={false}
              size="small"
            />
          </Card>
        </TabPane>
      </Tabs>

      {/* 提示信息 */}
      <Alert
        message="图纸查看器功能说明"
        description={
          <Space direction="vertical">
            <span>• 支持拖拽上传图纸文件</span>
            <span>• 集成OCR文字识别，自动提取规格参数</span>
            <span>• 提供测量工具，支持尺寸、角度、半径等测量</span>
            <span>• 支持缩放、旋转等视图操作</span>
            <span>• 可导出标注和测量结果</span>
          </Space>
        }
        type="info"
        showIcon
        style={{ marginTop: 16 }}
      />
    </div>
  );
};

export default DrawingViewer;