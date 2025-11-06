#!/usr/bin/env python3
"""
增强版提醒检查脚本
连接数据库，加载活跃规则，应用触发条件逻辑，生成提醒记录
"""

import sqlite3
import logging
import json
import smtplib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/check_reminders.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ReminderChecker')

class EnhancedReminderChecker:
    """增强版提醒检查器"""

    def __init__(self, db_path: str = "./data/db.sqlite"):
        self.db_path = db_path
        self.settings = {}
        self.load_settings()

    def load_settings(self):
        """加载系统配置"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT setting_key, setting_value FROM reminder_settings")
            for key, value in cursor.fetchall():
                self.settings[key] = value
            conn.close()
            logger.info("✅ 系统配置加载成功")
        except Exception as e:
            logger.error(f"❌ 加载系统配置失败: {e}")

    def get_active_rules(self) -> List[Dict]:
        """获取活跃的提醒规则"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, rule_id, name, description, priority, category,
                       trigger_config, schedule_config, notification_config,
                       is_active, last_triggered, trigger_count
                FROM reminder_rules
                WHERE is_active = 1
                ORDER BY priority ASC, last_triggered ASC
            """)

            columns = [desc[0] for desc in cursor.description]
            rules = []
            for row in cursor.fetchall():
                rule = dict(zip(columns, row))
                # 解析JSON配置
                for config_field in ['trigger_config', 'schedule_config', 'notification_config']:
                    if rule[config_field]:
                        try:
                            rule[config_field] = json.loads(rule[config_field])
                        except json.JSONDecodeError:
                            rule[config_field] = {}
                rules.append(rule)

            conn.close()
            logger.info(f"✅ 加载 {len(rules)} 个活跃提醒规则")
            return rules

        except Exception as e:
            logger.error(f"❌ 获取提醒规则失败: {e}")
            return []

    def should_check_rule(self, rule: Dict) -> bool:
        """检查是否应该执行该规则"""
        try:
            schedule_config = rule.get('schedule_config', {})
            frequency = schedule_config.get('frequency', 'daily')

            # 检查最后执行时间
            last_triggered = rule.get('last_triggered')
            if last_triggered:
                try:
                    last_time = datetime.strptime(last_triggered, '%Y-%m-%d %H:%M:%S')
                    now = datetime.now()

                    if frequency == 'hourly' and (now - last_time).total_seconds() < 3600:
                        return False
                    elif frequency == 'daily' and (now - last_time).days < 1:
                        return False
                    elif frequency == 'weekly' and (now - last_time).days < 7:
                        return False
                    elif frequency == 'monthly' and (now - last_time).days < 30:
                        return False
                except ValueError:
                    logger.warning(f"规则 {rule['rule_id']} 的 last_triggered 格式错误")

            # 检查具体时间配置
            if 'time' in schedule_config:
                target_time = schedule_config['time']
                current_time = datetime.now().strftime('%H:%M')
                if current_time != target_time:
                    return False

            # 检查星期配置
            if 'day' in schedule_config:
                target_day = schedule_config['day']
                current_day = datetime.now().strftime('%A').lower()
                if current_day != target_day:
                    return False

            return True

        except Exception as e:
            logger.error(f"❌ 检查规则执行条件失败: {e}")
            return False

    def evaluate_rule(self, rule: Dict) -> List[Dict]:
        """评估规则，返回触发的提醒记录"""
        try:
            trigger_config = rule.get('trigger_config', {})
            trigger_type = trigger_config.get('type', 'sql_query')

            if trigger_type == 'sql_query':
                return self.evaluate_sql_rule(rule)
            elif trigger_type == 'schedule':
                return self.evaluate_schedule_rule(rule)
            else:
                logger.warning(f"未知的触发类型: {trigger_type}")
                return []

        except Exception as e:
            logger.error(f"❌ 评估规则 {rule['name']} 失败: {e}")
            return []

    def evaluate_sql_rule(self, rule: Dict) -> List[Dict]:
        """评估SQL规则"""
        try:
            trigger_config = rule.get('trigger_config', {})
            query = trigger_config.get('query', '')

            if not query:
                logger.warning(f"规则 {rule['rule_id']} 缺少查询语句")
                return []

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            try:
                cursor.execute(query)
                results = cursor.fetchall()

                if not results:
                    logger.debug(f"规则 {rule['rule_id']} 查询结果为空")
                    return []

                # 获取列名
                columns = [desc[0] for desc in cursor.description]

                # 转换为提醒记录
                reminders = []
                for result in results:
                    row_data = dict(zip(columns, result))

                    # 确定实体类型和ID
                    entity_id = row_data.get('id')
                    entity_type = self.determine_entity_type(rule['rule_id'], row_data)

                    reminders.append({
                        'rule_id': rule['rule_id'],
                        'execution_id': self.generate_execution_id(rule['rule_id']),
                        'business_entity_type': entity_type,
                        'business_entity_id': entity_id,
                        'trigger_data': json.dumps({
                            'rule_name': rule['name'],
                            'query_result': row_data,
                            'triggered_at': datetime.now().isoformat()
                        }),
                        'trigger_reason': rule['description'],
                        'business_data': json.dumps(row_data, ensure_ascii=False)
                    })

                logger.info(f"📋 规则 '{rule['name']}' 触发了 {len(reminders)} 个提醒")
                return reminders

            finally:
                conn.close()

        except Exception as e:
            logger.error(f"❌ 评估SQL规则失败: {e}")
            return []

    def determine_entity_type(self, rule_id: str, row_data: Dict) -> str:
        """根据规则ID和数据确定实体类型"""
        if 'QUOTATION' in rule_id:
            return 'factory_quote'
        elif 'INQUIRY' in rule_id or 'CUSTOMER' in rule_id:
            return 'customer'
        elif 'DRAWING' in rule_id:
            return 'drawing'
        elif 'SYSTEM' in rule_id or 'BACKUP' in rule_id:
            return 'system'
        else:
            return 'unknown'

    def evaluate_schedule_rule(self, rule: Dict) -> List[Dict]:
        """评估定时规则"""
        try:
            # 定时规则总是生成一个提醒
            reminders = [{
                'rule_id': rule['rule_id'],
                'execution_id': self.generate_execution_id(rule['rule_id']),
                'business_entity_type': 'system',
                'business_entity_id': None,
                'trigger_data': json.dumps({
                    'rule_name': rule['name'],
                    'type': 'scheduled',
                    'scheduled_time': datetime.now().isoformat()
                }),
                'trigger_reason': rule['description'],
                'business_data': json.dumps({
                    'scheduled_at': datetime.now().isoformat(),
                    'rule_category': rule.get('category', 'system')
                })
            }]

            logger.info(f"⏰ 定时规则 '{rule['name']}' 触发")
            return reminders

        except Exception as e:
            logger.error(f"❌ 评估定时规则失败: {e}")
            return []

    def generate_execution_id(self, rule_id: str) -> str:
        """生成执行ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{rule_id}_{timestamp}"

    def create_reminder_records(self, reminders: List[Dict]) -> int:
        """创建提醒记录，返回实际创建的记录数"""
        try:
            if not reminders:
                return 0

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            created_count = 0
            for reminder in reminders:
                try:
                    # 检查是否已存在相同的待处理记录
                    cursor.execute("""
                        SELECT COUNT(*) FROM reminder_records
                        WHERE rule_id = ? AND business_entity_id = ? AND status = 'pending'
                    """, (reminder['rule_id'], reminder['business_entity_id']))

                    if cursor.fetchone()[0] == 0:
                        cursor.execute('''
                            INSERT INTO reminder_records
                            (rule_id, execution_id, triggered_at, trigger_data, trigger_reason,
                             business_entity_type, business_entity_id, business_data, status)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            reminder['rule_id'],
                            reminder['execution_id'],
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            reminder['trigger_data'],
                            reminder['trigger_reason'],
                            reminder['business_entity_type'],
                            reminder['business_entity_id'],
                            reminder['business_data'],
                            'pending'
                        ))
                        created_count += 1

                except Exception as e:
                    logger.error(f"❌ 创建单个提醒记录失败: {e}")

            conn.commit()
            conn.close()

            if created_count > 0:
                logger.info(f"✅ 创建 {created_count} 个新的提醒记录")
            else:
                logger.info("📭 所有触发条件均已存在待处理记录")

            return created_count

        except Exception as e:
            logger.error(f"❌ 创建提醒记录失败: {e}")
            return 0

    def update_rule_last_triggered(self, rule_id: int):
        """更新规则最后触发时间"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE reminder_rules
                SET last_triggered = ?, trigger_count = trigger_count + 1
                WHERE id = ?
            ''', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), rule_id))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"❌ 更新规则触发时间失败: {e}")

    def send_notifications(self, reminders: List[Dict]):
        """发送通知"""
        try:
            notification_count = 0
            for reminder in reminders:
                # 这里可以添加实际的通知发送逻辑
                logger.info(f"🔔 准备发送通知: {reminder.get('trigger_reason', 'Unknown')}")
                notification_count += 1

            if notification_count > 0:
                logger.info(f"✅ 准备发送 {notification_count} 个通知")
        except Exception as e:
            logger.error(f"❌ 发送通知失败: {e}")

    def check_daily_limits(self) -> bool:
        """检查每日限制"""
        try:
            max_daily = int(self.settings.get('max_daily_reminders', '100'))

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM reminder_records
                WHERE DATE(triggered_at) = DATE('now')
            """)
            today_count = cursor.fetchone()[0]
            conn.close()

            if today_count >= max_daily:
                logger.warning(f"今日提醒数量已达上限 ({max_daily})")
                return False

            return True

        except Exception as e:
            logger.error(f"❌ 检查每日限制失败: {e}")
            return True  # 出错时允许继续

    def check_all_reminders(self):
        """检查所有提醒规则"""
        try:
            logger.info("🚀 开始检查提醒规则...")

            # 检查每日限制
            if not self.check_daily_limits():
                return

            rules = self.get_active_rules()
            if not rules:
                logger.info("📭 没有活跃的提醒规则")
                return

            total_created = 0
            processed_rules = 0

            for rule in rules:
                try:
                    if self.should_check_rule(rule):
                        logger.debug(f"检查规则: {rule['name']} ({rule['rule_id']})")

                        reminders = self.evaluate_rule(rule)
                        if reminders:
                            created_count = self.create_reminder_records(reminders)
                            total_created += created_count

                            if created_count > 0:
                                self.send_notifications(reminders)

                        # 更新规则最后触发时间
                        self.update_rule_last_triggered(rule['id'])
                        processed_rules += 1
                    else:
                        logger.debug(f"跳过规则 {rule['name']} - 未到执行时间")

                except Exception as e:
                    logger.error(f"❌ 处理规则 {rule['name']} 失败: {e}")

            # 记录总结
            logger.info(f"🎉 提醒检查完成")
            logger.info(f"📊 处理规则数: {processed_rules}/{len(rules)}")
            logger.info(f"📝 生成新提醒: {total_created} 个")

            # 获取统计信息
            stats = self.get_reminder_statistics()
            if stats:
                logger.info(f"📈 今日提醒统计: 总数 {stats['today_count']}，待处理 {stats['pending_count']}")

        except Exception as e:
            logger.error(f"❌ 检查提醒规则失败: {e}")

    def get_reminder_statistics(self) -> Dict:
        """获取提醒统计信息"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 今日提醒数量
            cursor.execute("""
                SELECT COUNT(*) FROM reminder_records
                WHERE DATE(triggered_at) = DATE('now')
            """)
            today_count = cursor.fetchone()[0]

            # 待处理提醒数量
            cursor.execute("""
                SELECT COUNT(*) FROM reminder_records
                WHERE status = 'pending'
            """)
            pending_count = cursor.fetchone()[0]

            # 本周提醒数量
            cursor.execute("""
                SELECT COUNT(*) FROM reminder_records
                WHERE DATE(triggered_at) >= DATE('now', '-7 days')
            """)
            week_count = cursor.fetchone()[0]

            # 成功处理的提醒数量
            cursor.execute("""
                SELECT COUNT(*) FROM reminder_records
                WHERE status = 'completed'
            """)
            completed_count = cursor.fetchone()[0]

            conn.close()

            return {
                'today_count': today_count,
                'pending_count': pending_count,
                'week_count': week_count,
                'completed_count': completed_count,
                'check_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            logger.error(f"❌ 获取提醒统计失败: {e}")
            return {}

    def print_summary(self):
        """打印检查总结"""
        try:
            stats = self.get_reminder_statistics()
            if stats:
                print("\n" + "="*60)
                print("📊 提醒检查总结")
                print("="*60)
                print(f"📅 今日提醒: {stats['today_count']} 个")
                print(f"⏳ 待处理: {stats['pending_count']} 个")
                print(f"📈 本周总计: {stats['week_count']} 个")
                print(f"✅ 已完成: {stats['completed_count']} 个")
                print(f"⏰ 检查时间: {stats['check_time']}")
                print("="*60)
        except Exception as e:
            logger.error(f"❌ 打印总结失败: {e}")

def main():
    """主函数"""
    try:
        logger.info("🚀 启动增强版提醒检查系统...")

        checker = EnhancedReminderChecker()

        # 检查所有提醒
        checker.check_all_reminders()

        # 打印总结
        checker.print_summary()

        logger.info("✅ 增强版提醒检查系统运行完成")

    except Exception as e:
        logger.error(f"❌ 提醒检查系统运行失败: {e}")

if __name__ == "__main__":
    main()