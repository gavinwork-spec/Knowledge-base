#!/usr/bin/env python3
"""
文件监控系统
监控数据库表的变化，自动触发相应的处理脚本
"""

import sqlite3
import logging
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Callable, Optional
import subprocess
import json

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/file_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('FileMonitor')

class FileChangeMonitor:
    """文件变化监控器"""

    def __init__(self, db_path: str = "./data/db.sqlite"):
        self.db_path = db_path
        self.monitoring = False
        self.check_interval = 30  # 检查间隔（秒）
        self.monitored_tables = {}
        self.callbacks = {}
        self.last_table_states = {}

    def add_table_monitor(self, table_name: str, check_sql: str, callback: Callable):
        """添加表监控"""
        self.monitored_tables[table_name] = {
            'check_sql': check_sql,
            'callback': callback
        }
        logger.info(f"✅ 添加表监控: {table_name}")

    def get_table_state(self, table_name: str) -> Dict:
        """获取表的当前状态"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(self.monitored_tables[table_name]['check_sql'])
            result = cursor.fetchone()
            conn.close()

            if result:
                return dict(result)
            else:
                return {'count': 0}
        except Exception as e:
            logger.error(f"获取表 {table_name} 状态失败: {e}")
            return {'count': 0}

    def check_table_changes(self):
        """检查表变化"""
        for table_name, config in self.monitored_tables.items():
            try:
                current_state = self.get_table_state(table_name)
                last_state = self.last_table_states.get(table_name, {})

                # 检查是否有变化
                if current_state != last_state:
                    logger.info(f"🔍 检测到表 {table_name} 发生变化")

                    # 调用回调函数
                    try:
                        callback(table_name, current_state, last_state)
                    except Exception as e:
                        logger.error(f"执行 {table_name} 回调失败: {e}")

                    # 更新最后状态
                    self.last_table_states[table_name] = current_state

            except Exception as e:
                logger.error(f"检查表 {table_name} 变化时出错: {e}")

    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            logger.warning("监控已经在运行中")
            return

        self.monitoring = True
        logger.info("🚀 启动文件监控...")

        # 初始化所有表的状态
        for table_name in self.monitored_tables:
            self.last_table_states[table_name] = self.get_table_state(table_name)

        # 启动监控线程
        def monitor_loop():
            while self.monitoring:
                try:
                    self.check_table_changes()
                    time.sleep(self.check_interval)
                except Exception as e:
                    logger.error(f"监控循环出错: {e}")
                    time.sleep(self.check_interval)

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info(f"✅ 文件监控已启动，检查间隔: {self.check_interval}秒")

    def stop_monitoring(self):
        """停止监控"""
        if not self.monitoring:
            logger.warning("监控未在运行")
            return

        self.monitoring = False
        logger.info("👋 文件监控已停止")

class AgentManager:
    """Agent管理器"""

    def __init__(self):
        self.scripts = {
            'classify_drawings': {
                'path': './classify_drawings_enhanced.py',
                'description': '图纸分类脚本'
            },
            'analyze_trends': {
                'path': './analyze_factory_quote_trends.py',
                'description': '趋势分析脚本'
            },
            'export_statistics': {
                'path': './export_statistics.py',
                'description': '统计导出脚本'
            },
            'check_reminders': {
                'path': './check_reminders.py',
                'description': '提醒检查脚本'
            }
        }

    def execute_script(self, script_name: str, **kwargs) -> Dict:
        """执行指定脚本"""
        if script_name not in self.scripts:
            raise ValueError(f"未知脚本: {script_name}")

        script_info = self.scripts[script_name]
        script_path = script_info['path']

        try:
            logger.info(f"🚀 执行脚本: {script_info['description']} ({script_path})")

            # 记录开始时间
            start_time = datetime.now()

            # 执行脚本
            result = subprocess.run(
                ['python3', script_path],
                capture_output=True,
                text=True,
                cwd='./'
            )

            # 记录结束时间
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            execution_result = {
                'script_name': script_name,
                'description': script_info['description'],
                'success': result.returncode == 0,
                'execution_time': execution_time,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }

            if result.returncode == 0:
                logger.info(f"✅ 脚本执行成功: {script_info['description']} (耗时: {execution_time:.2f}秒)")
            else:
                logger.error(f"❌ 脚本执行失败: {script_info['description']} (返回码: {result.returncode})")
                logger.error(f"错误输出: {result.stderr}")

            return execution_result

        except Exception as e:
            logger.error(f"执行脚本 {script_name} 时出错: {e}")
            return {
                'script_name': script_name,
                'success': False,
                'error': str(e),
                'execution_time': 0
            }

def setup_drawing_monitoring(monitor: FileChangeMonitor):
    """设置图纸相关监控"""

    # 监控新图纸
    monitor.add_table_monitor(
        'drawings',
        "SELECT COUNT(*) as count, MAX(created_at) as latest_created FROM drawings",
        lambda table, current, last: handle_new_drawings(current, last)
    )

    # 监控工厂报价变化
    monitor.add_table_monitor(
        'factory_quotes',
        "SELECT COUNT(*) as count, MAX(quote_date) as latest_quote FROM factory_quotes",
        lambda table, current, last: handle_new_quotes(current, last)
    )

    # 监控客户变化
    monitor.add_table_monitor(
        'customers',
        "SELECT COUNT(*) as count, MAX(created_at) as latest_customer FROM customers",
        lambda table, current, last: handle_new_customers(current, last)
    )

def handle_new_drawings(current: Dict, last: Dict) -> bool:
    """处理新图纸"""
    current_count = current.get('count', 0)
    last_count = last.get('count', 0)

    # 如果新增了超过5个图纸，触发分类
    if current_count - last_count >= 5:
        logger.info(f"🆕 检测到 {current_count - last_count} 个新图纸，触发自动分类")

        # 执行分类脚本
        manager = AgentManager()
        result = manager.execute_script('classify_drawings')

        if result['success']:
            # 分类成功后执行统计导出
            export_result = manager.execute_script('export_statistics')
            if export_result['success']:
                logger.info("✅ 自动分类和统计导出完成")
            else:
                logger.error("❌ 统计导出失败")
        else:
            logger.error("❌ 自动分类失败")

        return True
    return False

def handle_new_quotes(current: Dict, last: Dict) -> bool:
    """处理新报价"""
    current_count = current.get('count', 0)
    last_count = last.get('count', 0)

    # 如果新增了超过3个报价，触发分析
    if current_count - last_count >= 3:
        logger.info(f"💰 检测到 {current_count - last_count} 个新报价，触发趋势分析")

        manager = AgentManager()
        result = manager.execute_script('analyze_trends')

        if result['success']:
            logger.info("✅ 报价趋势分析完成")
        else:
            logger.error("❌ 报价趋势分析失败")

        return True
    return False

def handle_new_customers(current: Dict, last: Dict) -> bool:
    """处理新客户"""
    current_count = current.get('count', 0)
    last_count = last.get('count', 0)

    # 如果新增了超过2个客户，触发提醒
    if current_count - last_count >= 2:
        logger.info(f"👤 检测到 {current_count - last_count} 个新客户，检查提醒规则")

        manager = AgentManager()
        result = manager.execute_script('check_reminders')

        if result['success']:
            logger.info("✅ 提醒规则检查完成")
        else:
            logger.error("❌ 提醒规则检查失败")

        return True
    return False

def start_file_monitoring_system():
    """启动文件监控系统"""
    logger.info("🔧 启动文件监控系统...")

    # 创建监控器
    monitor = FileChangeMonitor()

    # 设置监控
    setup_drawing_monitoring(monitor)

    # 启动监控
    monitor.start_monitoring()

    return monitor

def main():
    """主函数"""
    try:
        logger.info("🚀 启动Agent文件监控系统...")

        # 启动文件监控
        monitor = start_file_monitoring_system()

        print("🎉 Agent监控系统已启动!")
        print("📊 监控对象: 图纸、报价、客户表")
        print("⚡ 自动触发: 分类、分析、导出、提醒")
        print("📋 检查间隔: 30秒")
        print("按 Ctrl+C 停止监控")

        # 保持运行
        try:
            while True:
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("👋 收到停止信号...")
            monitor.stop_monitoring()
            print("👋 Agent监控系统已停止")

    except Exception as e:
        logger.error(f"启动监控系统失败: {e}")

if __name__ == "__main__":
    main()