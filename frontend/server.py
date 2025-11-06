#!/usr/bin/env python3
"""
简单的静态文件服务器
用于提供前端页面访问
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

# 设置端口和目录
PORT = 3000
DIRECTORY = Path(__file__).parent / "public"

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # 添加CORS头
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

def start_server():
    """启动静态文件服务器"""
    os.chdir(DIRECTORY)

    with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
        print(f"🚀 前端服务器启动成功!")
        print(f"📡 服务地址: http://localhost:{PORT}")
        print(f"📁 服务目录: {DIRECTORY}")
        print(f"🔗 API地址: http://localhost:8000/api/v1")
        print("按 Ctrl+C 停止服务器")

        # 自动打开浏览器
        try:
            webbrowser.open(f'http://localhost:{PORT}')
            print("✅ 已自动打开浏览器")
        except:
            print("⚠️ 无法自动打开浏览器，请手动访问上述地址")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 服务器已停止")

if __name__ == "__main__":
    start_server()