#!/usr/bin/env python3
"""
提醒系统验证脚本
验证提醒系统的各个组件是否正常工作
"""

import sqlite3
import logging
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ReminderVerifier')

class ReminderSystemVerifier:
    """提醒系统验证器"""

    def __init__(self, db_path: str = "./data/db.sqlite", api_url: str = "http://localhost:8000"):
        self.db_path = db_path
        self.api_url = api_url
        self.verification_results = []

    def add_result(self, component: str, test_name: str, success: bool, message: str, details: dict = None):
        """添加验证结果"""
        result = {
            'component': component,
            'test_name': test_name,
            'success': success,
            'message': message,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        }
        self.verification_results.append(result)
        
        status = "✅" if success else "❌"
        logger.info(f"{status} {component} - {test_name}: {message}")

    def verify_database_tables(self):
        """验证数据库表结构"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 检查必需的表是否存在
            required_tables = ['reminder_rules', 'reminder_records', 'reminder_settings']
            existing_tables = []

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            for row in cursor.fetchall():
                existing_tables.append(row[0])

            for table in required_tables:
                if table in existing_tables:
                    # 检查表结构
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = cursor.fetchall()
                    self.add_result(
                        'Database', 
                        f'Table {table}', 
                        True, 
                        f'表存在，包含 {len(columns)} 个字段',
                        {'columns': [col[1] for col in columns]}
                    )
                else:
                    self.add_result('Database', f'Table {table}', False, '表不存在')

            conn.close()

        except Exception as e:
            self.add_result('Database', 'Table Structure', False, f'验证失败: {str(e)}')

    def verify_reminder_rules(self):
        """验证提醒规则配置"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 检查规则数量
            cursor.execute("SELECT COUNT(*) FROM reminder_rules WHERE is_active = 1")
            active_rules_count = cursor.fetchone()[0]

            if active_rules_count > 0:
                self.add_result(
                    'Reminder Rules',
                    'Active Rules Count',
                    True,
                    f'发现 {active_rules_count} 个活跃规则'
                )

                # 检查规则配置完整性
                cursor.execute("""
                    SELECT rule_id, name, priority, trigger_config, schedule_config 
                    FROM reminder_rules 
                    WHERE is_active = 1
                """)
                rules = cursor.fetchall()

                for rule in rules:
                    rule_id, name, priority, trigger_config, schedule_config = rule
                    
                    # 验证触发配置
                    try:
                        trigger_data = json.loads(trigger_config) if trigger_config else {}
                        if 'type' in trigger_data and 'condition' in trigger_data:
                            self.add_result(
                                'Reminder Rules',
                                f'Rule {rule_id} Trigger Config',
                                True,
                                '触发配置完整'
                            )
                        else:
                            self.add_result(
                                'Reminder Rules',
                                f'Rule {rule_id} Trigger Config',
                                False,
                                '触发配置不完整'
                            )
                    except json.JSONDecodeError:
                        self.add_result(
                            'Reminder Rules',
                            f'Rule {rule_id} Trigger Config',
                            False,
                            '触发配置JSON格式错误'
                        )

            else:
                self.add_result('Reminder Rules', 'Active Rules Count', False, '没有活跃的提醒规则')

            conn.close()

        except Exception as e:
            self.add_result('Reminder Rules', 'Configuration', False, f'验证失败: {str(e)}')

    def verify_api_endpoints(self):
        """验证API端点"""
        try:
            # 测试健康检查
            try:
                response = requests.get(f"{self.api_url}/api/v1/health", timeout=5)
                if response.status_code == 200:
                    self.add_result(
                        'API Endpoints',
                        'Health Check',
                        True,
                        'API服务器正常运行'
                    )
                else:
                    self.add_result(
                        'API Endpoints',
                        'Health Check',
                        False,
                        f'API服务器响应异常: {response.status_code}'
                    )
            except requests.exceptions.RequestException:
                self.add_result('API Endpoints', 'Health Check', False, '无法连接到API服务器')
                return

            # 测试提醒规则API
            try:
                response = requests.get(f"{self.api_url}/api/v1/reminders/rules", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        rules_count = len(data.get('data', {}).get('rules', []))
                        self.add_result(
                            'API Endpoints',
                            'Reminder Rules API',
                            True,
                            f'成功获取 {rules_count} 个提醒规则'
                        )
                    else:
                        self.add_result('API Endpoints', 'Reminder Rules API', False, 'API返回错误')
                else:
                    self.add_result(
                        'API Endpoints',
                        'Reminder Rules API',
                        False,
                        f'API响应异常: {response.status_code}'
                    )
            except requests.exceptions.RequestException as e:
                self.add_result('API Endpoints', 'Reminder Rules API', False, f'请求失败: {str(e)}')

            # 测试提醒仪表板API
            try:
                response = requests.get(f"{self.api_url}/api/v1/reminders/dashboard", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        dashboard_data = data.get('data', {})
                        self.add_result(
                            'API Endpoints',
                            'Reminder Dashboard API',
                            True,
                            '仪表板数据获取成功',
                            {
                                'total_rules': dashboard_data.get('rules', {}).get('total_rules', 0),
                                'active_rules': dashboard_data.get('rules', {}).get('active_rules', 0)
                            }
                        )
                    else:
                        self.add_result('API Endpoints', 'Reminder Dashboard API', False, 'API返回错误')
                else:
                    self.add_result(
                        'API Endpoints',
                        'Reminder Dashboard API',
                        False,
                        f'API响应异常: {response.status_code}'
                    )
            except requests.exceptions.RequestException as e:
                self.add_result('API Endpoints', 'Reminder Dashboard API', False, f'请求失败: {str(e)}')

        except Exception as e:
            self.add_result('API Endpoints', 'General', False, f'验证失败: {str(e)}')

    def verify_reminder_script(self):
        """验证提醒检查脚本"""
        try:
            # 检查脚本文件是否存在
            script_path = Path("./check_reminders.py")
            if script_path.exists():
                self.add_result(
                    'Reminder Script',
                    'File Existence',
                    True,
                    '提醒检查脚本文件存在'
                )

                # 检查脚本是否可执行（通过导入测试）
                try:
                    import sys
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("check_reminders", script_path)
                    module = importlib.util.module_from_spec(spec)
                    # 只检查语法，不执行
                    with open(script_path, 'r', encoding='utf-8') as f:
                        compile(f.read(), script_path, 'exec')
                    
                    self.add_result(
                        'Reminder Script',
                        'Syntax Check',
                        True,
                        '脚本语法正确'
                    )
                except SyntaxError as e:
                    self.add_result('Reminder Script', 'Syntax Check', False, f'语法错误: {str(e)}')
                except Exception as e:
                    self.add_result('Reminder Script', 'Syntax Check', False, f'检查失败: {str(e)}')
            else:
                self.add_result('Reminder Script', 'File Existence', False, '提醒检查脚本文件不存在')

        except Exception as e:
            self.add_result('Reminder Script', 'General', False, f'验证失败: {str(e)}')

    def verify_system_integration(self):
        """验证系统集成"""
        try:
            # 检查日志目录
            logs_dir = Path("./logs")
            if logs_dir.exists():
                self.add_result(
                    'System Integration',
                    'Logs Directory',
                    True,
                    '日志目录存在'
                )
            else:
                self.add_result('System Integration', 'Logs Directory', False, '日志目录不存在')

            # 检查数据目录
            data_dir = Path("./data")
            if data_dir.exists():
                self.add_result(
                    'System Integration',
                    'Data Directory',
                    True,
                    '数据目录存在'
                )
                
                # 检查数据库文件
                db_file = data_dir / "db.sqlite"
                if db_file.exists():
                    self.add_result(
                        'System Integration',
                        'Database File',
                        True,
                        '数据库文件存在'
                    )
                else:
                    self.add_result('System Integration', 'Database File', False, '数据库文件不存在')
            else:
                self.add_result('System Integration', 'Data Directory', False, '数据目录不存在')

        except Exception as e:
            self.add_result('System Integration', 'General', False, f'验证失败: {str(e)}')

    def run_verification(self):
        """运行完整验证"""
        logger.info("🚀 开始提醒系统验证...")

        # 运行各项验证
        self.verify_database_tables()
        self.verify_reminder_rules()
        self.verify_api_endpoints()
        self.verify_reminder_script()
        self.verify_system_integration()

        # 计算总体评分
        total_tests = len(self.verification_results)
        passed_tests = sum(1 for result in self.verification_results if result['success'])
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        # 生成验证报告
        report = {
            'verification_time': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': round(success_rate, 1)
            },
            'results': self.verification_results,
            'status': 'PASS' if success_rate >= 80 else 'FAIL',
            'recommendations': self.generate_recommendations()
        }

        return report

    def generate_recommendations(self):
        """生成改进建议"""
        recommendations = []

        failed_tests = [result for result in self.verification_results if not result['success']]
        
        for test in failed_tests:
            component = test['component']
            test_name = test['test_name']
            
            if 'Database' in component and 'Table' in test_name:
                recommendations.append("检查数据库表创建脚本，确保所有必需的表都已创建")
            elif 'API Endpoints' in component and 'Health Check' in test_name:
                recommendations.append("启动API服务器：python3 api_server_mock.py")
            elif 'Reminder Script' in component and 'File Existence' in test_name:
                recommendations.append("确保提醒检查脚本文件存在：check_reminders.py")
            elif 'Reminder Rules' in component and 'Active Rules' in test_name:
                recommendations.append("配置提醒规则：运行 setup_reminder_system.py")

        if not recommendations:
            recommendations.append("系统运行良好，建议定期执行验证以保持系统健康")

        return recommendations

    def print_report(self, report: dict):
        """打印验证报告"""
        print("\n" + "="*60)
        print("🔍 提醒系统验证报告")
        print("="*60)

        # 摘要信息
        summary = report['summary']
        print(f"\n📊 验证摘要:")
        print(f"   总测试数: {summary['total_tests']}")
        print(f"   通过测试: {summary['passed_tests']}")
        print(f"   失败测试: {summary['failed_tests']}")
        print(f"   成功率: {summary['success_rate']}%")
        
        status = report['status']
        status_icon = "✅" if status == 'PASS' else "❌"
        print(f"   总体状态: {status_icon} {status}")

        # 详细结果
        print(f"\n📋 详细结果:")
        for result in report['results']:
            status_icon = "✅" if result['success'] else "❌"
            print(f"   {status_icon} {result['component']} - {result['test_name']}: {result['message']}")

        # 改进建议
        if report['recommendations']:
            print(f"\n💡 改进建议:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"   {i}. {rec}")

        print("\n" + "="*60)

def main():
    """主函数"""
    try:
        verifier = ReminderSystemVerifier()
        report = verifier.run_verification()
        verifier.print_report(report)

        # 保存报告到文件
        report_file = f"reminder_verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 详细报告已保存到: {report_file}")

        return report['status'] == 'PASS'

    except Exception as e:
        logger.error(f"❌ 验证过程失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
