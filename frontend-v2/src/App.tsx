import React from 'react';

function App() {
  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-600 to-blue-800 text-white shadow-lg">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">XAgent 制造业智能系统</h1>
              <p className="text-blue-100">Manufacturing Intelligence Platform</p>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-sm bg-white/20 px-3 py-1 rounded-full">
                系统状态: 正常运行
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {/* Hero Section */}
        <section className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            欢迎使用 XAgent 制造业智能系统
          </h2>
          <p className="text-xl text-gray-600 mb-8">
            先进的多智能体编排平台，专为制造业优化
          </p>
        </section>

        {/* Status Cards */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          {/* Frontend Status */}
          <div className="bg-white p-6 rounded-lg shadow-lg border-l-4 border-green-500">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">前端应用</h3>
            <p className="text-green-600 mb-4">✅ 正常运行</p>
            <p className="text-sm text-gray-600">http://localhost:3000</p>
            <button
              onClick={() => window.open('http://localhost:3000', '_blank')}
              className="mt-2 w-full bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-colors"
            >
              打开前端
            </button>
          </div>

          {/* Knowledge API Status */}
          <div className="bg-white p-6 rounded-lg shadow-lg border-l-4 border-blue-500">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">知识管理 API</h3>
            <p className="text-blue-600 mb-4">✅ 正常运行</p>
            <p className="text-sm text-gray-600">http://localhost:8001</p>
            <a
              href="http://localhost:8001/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="mt-2 inline-block w-full bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-colors text-center"
            >
              API 文档
            </a>
          </div>

          {/* Chat API Status */}
          <div className="bg-white p-6 rounded-lg shadow-lg border-l-4 border-purple-500">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">聊天界面</h3>
            <p className="text-purple-600 mb-4">✅ 正在运行</p>
            <p className="text-sm text-gray-600">http://localhost:8002</p>
            <button
              onClick={() => window.open('http://localhost:8002', '_blank')}
              className="mt-2 w-full bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700 transition-colors"
            >
              打开聊天
            </button>
          </div>

          {/* XAgent API Status */}
          <div className="bg-white p-6 rounded-lg shadow-lg border-l-4 border-orange-500">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">XAgent API</h3>
            <p className="text-orange-600 mb-4">✅ 可访问</p>
            <p className="text-sm text-gray-600">http://localhost:8003</p>
            <button
              onClick={() => window.open('http://localhost:8003/api/health', '_blank')}
              className="mt-2 w-full bg-orange-600 text-white px-4 py-2 rounded hover:bg-orange-700 transition-colors"
            >
              检查状态
            </button>
          </div>
        </section>

        {/* Features */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">系统功能</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">

            <div className="bg-white p-6 rounded-lg shadow-md">
              <div className="text-blue-600 text-2xl mb-3">🏭</div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">制造业专用智能体</h3>
              <ul className="text-gray-600 space-y-1">
                <li>• 安全检查员 - Safety Inspector</li>
                <li>• 质量控制器 - Quality Controller</li>
                <li>• 维护技术员 - Maintenance Technician</li>
                <li>• 生产经理 - Production Manager</li>
              </ul>
            </div>

            <div className="bg-white p-6 rounded-lg shadow-md">
              <div className="text-green-600 text-2xl mb-3">🔄</div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">智能任务编排</h3>
              <ul className="text-gray-600 space-y-1">
                <li>• 自动任务分配</li>
                <li>• 智能优先级管理</li>
                <li>• 工作流程自动化</li>
                <li>• 实时协作协调</li>
              </ul>
            </div>

            <div className="bg-white p-6 rounded-lg shadow-md">
              <div className="text-purple-600 text-2xl mb-3">📊</div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">实时监控分析</h3>
              <ul className="text-gray-600 space-y-1">
                <li>• 性能指标监控</li>
                <li>• 健康状态检查</li>
                <li>• 智能警报系统</li>
                <li>• 数据分析报告</li>
              </ul>
            </div>

            <div className="bg-white p-6 rounded-lg shadow-md">
              <div className="text-orange-600 text-2xl mb-3">🔒</div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">安全通信协议</h3>
              <ul className="text-gray-600 space-y-1">
                <li>• 加密消息传递</li>
                <li>• 优先级路由</li>
                <li>• 可靠交付确认</li>
                <li>• 实时协作</li>
              </ul>
            </div>

            <div className="bg-white p-6 rounded-lg shadow-md">
              <div className="text-red-600 text-2xl mb-3">🚨</div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">安全合规管理</h3>
              <ul className="text-gray-600 space-y-1">
                <li>• OSHA 标准检查</li>
                <li>• ISO 质量认证</li>
                <li>• 实时安全监控</li>
                <li>• 风险评估报告</li>
              </ul>
            </div>

            <div className="bg-white p-6 rounded-lg shadow-md">
              <div className="text-cyan-600 text-2xl mb-3">🔧</div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">配置管理</h3>
              <ul className="text-gray-600 space-y-1">
                <li>• YAML 配置支持</li>
                <li>• 自动迁移工具</li>
                <li>• 热加载功能</li>
                <li>• 配置验证</li>
              </ul>
            </div>

          </div>
        </section>

        {/* Technology Stack */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">技术架构</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">

            <div className="text-center p-4">
              <div className="text-blue-600 text-3xl mb-2">⚛️</div>
              <h4 className="font-semibold text-gray-900">前端技术</h4>
              <p className="text-sm text-gray-600">React + TypeScript + Vite</p>
            </div>

            <div className="text-center p-4">
              <div className="text-green-600 text-3xl mb-2">🔧</div>
              <h4 className="font-semibold text-gray-900">后端服务</h4>
              <p className="text-sm text-gray-600">Python + Flask + FastAPI</p>
            </div>

            <div className="text-center p-4">
              <div className="text-purple-600 text-3xl mb-2">🤖</div>
              <h4 className="font-semibold text-gray-900">智能体系统</h4>
              <p className="text-sm text-gray-600">XAgent 多智能体编排</p>
            </div>

            <div className="text-center p-4">
              <div className="text-orange-600 text-3xl mb-2">📡</div>
              <h4 className="font-semibold text-gray-900">集成架构</h4>
              <p className="text-sm text-gray-600">LangChain + LobeChat</p>
            </div>

          </div>
        </section>

        {/* Quick Start */}
        <section className="text-center">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">快速开始</h2>
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-8">
            <p className="text-gray-700 mb-6">
              所有服务已启动并运行正常！您可以点击上方按钮访问各个服务。
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button
                onClick={() => window.open('http://localhost:3000', '_blank')}
                className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors font-medium"
              >
                🎨 打开前端应用
              </button>
              <button
                onClick={() => window.open('http://localhost:8001/docs', '_blank')}
                className="bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 transition-colors font-medium"
              >
                📚 API 文档
              </button>
              <button
                onClick={() => window.open('http://localhost:8002', '_blank')}
                className="bg-purple-600 text-white px-6 py-3 rounded-lg hover:bg-purple-700 transition-colors font-medium"
              >
                💬 聊天界面
              </button>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-8 mt-12">
        <div className="container mx-auto px-4 text-center">
          <p className="text-gray-400">
            © 2024 XAgent Manufacturing Intelligence System. All rights reserved.
          </p>
          <p className="text-gray-500 text-sm mt-2">
            Powered by React, Python, and Advanced AI Technologies
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;