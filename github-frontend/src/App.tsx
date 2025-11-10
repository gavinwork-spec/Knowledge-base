import React from 'react';
import TradingDashboard from './TradingDashboard';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import 'antd/dist/reset.css';

function App() {
  return (
    <ConfigProvider locale={zhCN}>
      <div className="App">
        <TradingDashboard />
      </div>
    </ConfigProvider>
  );
}

export default App;