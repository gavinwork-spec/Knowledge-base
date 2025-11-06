/**
 * Cloudflare Workers API代理
 * 解决CORS问题，代理API请求到本地服务器
 */

// 配置本地API服务器地址
const LOCAL_API_URL = 'http://YOUR_LOCAL_IP:8001';

// 允许的域名列表（用于安全控制）
const ALLOWED_ORIGINS = [
    'https://yourusername.github.io',
    'https://your-custom-domain.com',
    'http://localhost:3000',
    'http://127.0.0.1:3000'
];

// CORS设置
const CORS_HEADERS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
    'Access-Control-Max-Age': '86400'
};

/**
 * 主处理函数
 */
export default {
    async fetch(request, env, ctx) {
        try {
            const url = new URL(request.url);

            // 处理CORS预检请求
            if (request.method === 'OPTIONS') {
                return handleCORS();
            }

            // API请求代理
            if (url.pathname.startsWith('/api/')) {
                return handleApiProxy(request, url);
            }

            // 静态资源服务
            return handleStaticResource(request, env);

        } catch (error) {
            console.error('Worker error:', error);
            return new Response(JSON.stringify({
                error: 'Internal Server Error',
                message: error.message,
                timestamp: new Date().toISOString()
            }), {
                status: 500,
                headers: {
                    'Content-Type': 'application/json',
                    ...CORS_HEADERS
                }
            });
        }
    }
};

/**
 * 处理CORS预检请求
 */
function handleCORS() {
    return new Response(null, {
        status: 200,
        headers: CORS_HEADERS
    });
}

/**
 * 处理API代理请求
 */
async function handleApiProxy(request, url) {
    try {
        // 构建目标URL
        const targetUrl = LOCAL_API_URL + url.pathname + url.search;

        console.log('Proxying request to:', targetUrl);

        // 复制请求头，移除可能导致问题的头
        const headers = new Headers();
        for (const [key, value] of request.headers.entries()) {
            // 跳过一些可能导致问题的头
            if (!['host', 'origin'].includes(key.toLowerCase())) {
                headers.append(key, value);
            }
        }

        // 添加客户端信息
        headers.append('X-Forwarded-For', request.headers.get('CF-Connecting-IP') || '');
        headers.append('X-Forwarded-Proto', url.protocol);
        headers.append('X-Forwarded-Host', url.host);

        // 转发请求
        const response = await fetch(targetUrl, {
            method: request.method,
            headers: headers,
            body: request.body,
            redirect: 'manual'
        });

        // 处理响应
        const responseHeaders = new Headers();
        for (const [key, value] of response.headers.entries()) {
            responseHeaders.append(key, value);
        }

        // 添加CORS头
        for (const [key, value] of Object.entries(CORS_HEADERS)) {
            responseHeaders.set(key, value);
        }

        // 处理响应体
        let responseBody;
        const contentType = response.headers.get('content-type') || '';

        if (contentType.includes('application/json')) {
            responseBody = await response.text();

            // 尝试解析JSON并添加调试信息
            try {
                const jsonData = JSON.parse(responseBody);
                if (url.pathname.includes('/dashboard')) {
                    jsonData._debug = {
                        proxied: true,
                        timestamp: new Date().toISOString(),
                        worker_version: '1.0.0'
                    };
                    responseBody = JSON.stringify(jsonData);
                }
            } catch (e) {
                // JSON解析失败，保持原样
            }
        } else {
            responseBody = await response.text();
        }

        // 返回代理响应
        return new Response(responseBody, {
            status: response.status,
            statusText: response.statusText,
            headers: responseHeaders
        });

    } catch (error) {
        console.error('API proxy error:', error);

        // 返回错误响应
        return new Response(JSON.stringify({
            error: 'Proxy Error',
            message: 'Failed to proxy request to API server',
            details: error.message,
            timestamp: new Date().toISOString(),
            suggestions: [
                '检查本地API服务器是否运行',
                '确认LOCAL_API_URL配置正确',
                '检查网络连接'
            ]
        }), {
            status: 502,
            headers: {
                'Content-Type': 'application/json',
                ...CORS_HEADERS
            }
        });
    }
}

/**
 * 处理静态资源
 */
async function handleStaticResource(request, env) {
    // 这里可以处理一些静态资源或提供默认页面
    const url = new URL(request.url);

    // 根路径返回主页
    if (url.pathname === '/' || url.pathname === '/index.html') {
        return new Response(getHomePage(), {
            headers: {
                'Content-Type': 'text/html; charset=utf-8',
                ...CORS_HEADERS
            }
        });
    }

    // API健康检查端点
    if (url.pathname === '/health') {
        return new Response(JSON.stringify({
            status: 'healthy',
            service: 'API Proxy Worker',
            version: '1.0.0',
            timestamp: new Date().toISOString(),
            local_api: LOCAL_API_URL
        }), {
            headers: {
                'Content-Type': 'application/json',
                ...CORS_HEADERS
            }
        });
    }

    // 404响应
    return new Response(JSON.stringify({
        error: 'Not Found',
        message: 'The requested resource was not found',
        path: url.pathname,
        timestamp: new Date().toISOString()
    }), {
        status: 404,
        headers: {
            'Content-Type': 'application/json',
            ...CORS_HEADERS
        }
    });
}

/**
 * 获取主页HTML
 */
function getHomePage() {
    return `
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API代理服务</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            line-height: 1.6;
        }
        .header {
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #e1e5e9;
        }
        .status {
            background: #e8f5e8;
            color: #2d6a2d;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .warning {
            background: #fff3cd;
            color: #856404;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .api-info {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            font-family: monospace;
            margin: 1rem 0;
        }
        .instructions {
            background: #e3f2fd;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        code {
            background: #f1f3f4;
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔧 知识库提醒系统 API代理</h1>
        <p>Cloudflare Workers API代理服务</p>
    </div>

    <div class="status">
        <h3>✅ 服务状态</h3>
        <p>代理服务正在运行中</p>
    </div>

    <div class="api-info">
        <h3>📡 API配置</h3>
        <p>本地API服务器: <code>${LOCAL_API_URL}</code></p>
        <p>代理端点: <code>/api/v1/*</code></p>
    </div>

    <div class="instructions">
        <h3>📋 使用说明</h3>
        <p>前端应用应该配置API基础URL为:</p>
        <p><code>https://your-proxy.your-domain.com/api/v1</code></p>

        <h4>可用端点:</h4>
        <ul>
            <li><code>GET /api/v1/health</code> - 健康检查</li>
            <li><code>GET /api/v1/reminders/dashboard</code> - 仪表板数据</li>
            <li><code>GET /api/v1/reminders/records</code> - 提醒记录</li>
            <li><code>GET /api/v1/reminders/rules</code> - 规则列表</li>
        </ul>
    </div>

    <div class="warning">
        <h3>⚠️ 注意事项</h3>
        <ul>
            <li>确保本地API服务器正在运行</li>
            <li>检查防火墙设置，允许外部访问</li>
            <li>确保LOCAL_API_URL配置正确</li>
        </ul>
    </div>

    <div style="text-align: center; margin-top: 2rem; color: #666;">
        <p>最后更新: ${new Date().toLocaleString('zh-CN')}</p>
    </div>
</body>
</html>
    `;
}

/**
 * 请求日志中间件
 */
function logRequest(request, response, startTime) {
    const duration = Date.now() - startTime;
    const url = new URL(request.url);

    console.log({
        method: request.method,
        url: url.pathname + url.search,
        status: response.status,
        duration: `${duration}ms`,
        timestamp: new Date().toISOString(),
        userAgent: request.headers.get('user-agent') || 'unknown',
        ip: request.headers.get('cf-connecting-ip') || 'unknown'
    });
}

/**
 * 健康检查端点
 */
async function handleHealthCheck() {
    try {
        // 尝试连接本地API服务器
        const healthResponse = await fetch(`${LOCAL_API_URL}/api/v1/health`, {
            method: 'GET',
            headers: { 'User-Agent': 'Cloudflare-Worker-Health-Check' }
        });

        const localApiStatus = healthResponse.ok ? 'healthy' : 'unhealthy';
        const localApiData = healthResponse.ok ? await healthResponse.json() : null;

        return new Response(JSON.stringify({
            proxy_status: 'healthy',
            local_api_status: localApiStatus,
            local_api_data: localApiData,
            proxy_version: '1.0.0',
            timestamp: new Date().toISOString(),
            uptime: process.uptime ? `${Math.floor(process.uptime() / 1000)}s` : 'unknown'
        }), {
            headers: {
                'Content-Type': 'application/json',
                ...CORS_HEADERS
            }
        });

    } catch (error) {
        return new Response(JSON.stringify({
            proxy_status: 'degraded',
            local_api_status: 'unreachable',
            error: error.message,
            proxy_version: '1.0.0',
            timestamp: new Date().toISOString()
        }), {
            status: 503,
            headers: {
                'Content-Type': 'application/json',
                ...CORS_HEADERS
            }
        });
    }
}