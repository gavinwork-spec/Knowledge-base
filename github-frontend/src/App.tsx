import React from 'react';
import TradingDashboardSimple from './TradingDashboardSimple';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import 'antd/dist/reset.css';

function App() {
  return (
    <ConfigProvider locale={zhCN}>
      <div className="App">
        <TradingDashboardSimple />
      </div>
    </ConfigProvider>
  );
}

export default App;