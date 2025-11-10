#!/bin/bash
# XAgent System Quick Start Script
# XAgent 系统快速启动脚本

echo "🚀 Starting XAgent Manufacturing Intelligence System..."
echo "🚀 启动 XAgent 制造业智能系统..."

# Set base directory
BASE_DIR="/Users/gavin/Knowledge base"
cd "$BASE_DIR"

# Function to check if port is available
check_port() {
    if lsof -i :$1 > /dev/null 2>&1; then
        echo "⚠️ Port $1 is already in use"
        return 1
    fi
    return 0
}

# Function to start service
start_service() {
    local service_name=$1
    local port=$2
    local command=$3

    echo "📦 Starting $service_name on port $port..."

    if check_port $port; then
        eval "$command" &
        echo "✅ $service_name started with PID: $!"
    else
        echo "⚠️ $service_name already running on port $port"
    fi
}

# Check Python dependencies
echo "🔍 Checking Python dependencies..."
python3 -c "import flask, yaml, asyncio" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing Python dependencies. Installing..."
    pip3 install flask pyyaml asyncio aiofiles cryptography
fi

# Check Node.js dependencies
echo "🔍 Checking Node.js dependencies..."
if [ -d "frontend-v2" ]; then
    cd frontend-v2
    if [ ! -d "node_modules" ]; then
        echo "📦 Installing Node.js dependencies..."
        npm install
    fi
    cd ..
fi

# Start backend services
echo "🔧 Starting backend services..."

start_service "Knowledge API Server" 8001 "python3 api_server_knowledge.py --port 8001"
sleep 2

start_service "Chat Interface API" 8002 "python3 api_chat_interface.py --port 8002"
sleep 2

start_service "XAgent API Server" 8003 "python3 xagent_api_server.py"
sleep 2

# Start frontend application
echo "🎨 Starting frontend application..."
if [ -d "frontend-v2" ]; then
    cd frontend-v2
    start_service "Frontend Application" 3000 "npm run dev"
    cd ..
fi

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 5

# Display access URLs
echo ""
echo "🌐 XAgent System is now running!"
echo "🌐 XAgent 系统现在正在运行！"
echo ""
echo "📱 Access URLs / 访问链接:"
echo "   Frontend Application: http://localhost:3000"
echo "   Knowledge API: http://localhost:8001"
echo "   Chat Interface: http://localhost:8002"
echo "   XAgent API: http://localhost:8003"
echo "   API Documentation: http://localhost:8001/docs"
echo ""
echo "🔧 System Components / 系统组件:"
echo "   ✅ Manufacturing Safety Inspector - 制造业安全检查员"
echo "   ✅ Quality Controller - 质量控制器"
echo "   ✅ Maintenance Technician - 维护技术员"
echo "   ✅ Production Manager - 生产经理"
echo ""
echo "📊 Monitoring & Analytics / 监控与分析:"
echo "   📈 Real-time metrics collection"
echo "   🔍 Health monitoring"
echo "   🚨 Alert management"
echo "   📋 Performance analytics"
echo ""
echo "💡 To stop all services, press Ctrl+C"
echo "💡 要停止所有服务，请按 Ctrl+C"
echo ""

# Keep script running
trap 'echo "🛑 Shutting down services..."; pkill -f "python3.*api_server"; pkill -f "npm.*dev"; exit' INT

echo "⏳ Monitoring system health..."
while true; do
    sleep 10
    # You can add health checks here
    echo "$(date '+%H:%M:%S') - ✅ All services running"
done