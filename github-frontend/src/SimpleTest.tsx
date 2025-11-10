import React from 'react';

const SimpleTest: React.FC = () => {
  return (
    <div style={{
      padding: '40px',
      fontFamily: 'Arial, sans-serif',
      backgroundColor: '#f5f5f5',
      minHeight: '100vh'
    }}>
      <h1 style={{ color: '#1890ff', textAlign: 'center' }}>
        🏭 贸易公司管理系统
      </h1>
      <div style={{
        maxWidth: '800px',
        margin: '20px auto',
        padding: '20px',
        backgroundColor: 'white',
        borderRadius: '8px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
      }}>
        <h2>系统状态：✅ 正常运行</h2>
        <p>如果您能看到这个页面，说明React应用已经成功启动！</p>
        <div style={{ marginTop: '20px' }}>
          <h3>功能模块：</h3>
          <ul>
            <li>📊 实时价格监控</li>
            <li>👥 客户关系管理</li>
            <li>📋 项目跟踪看板</li>
            <li>📐 图纸查看器</li>
            <li>💰 报价管理系统</li>
            <li>🏭 供应商分析</li>
            <li>📈 利润分析报告</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default SimpleTest;