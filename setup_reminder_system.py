#!/usr/bin/env python3
"""
提醒系统设置脚本
基于现有表结构设置提醒规则和配置
"""

import sqlite3
import logging
import json
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_reminder_rules(db_path: str = "./data/db.sqlite"):
    """设置提醒规则"""

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 现有表结构的提醒规则
        reminder_rules = [
            {
                'rule_id': 'CUSTOMER_NO_QUOTES',
                'name': '新客户未报价提醒',
                'description': '客户注册超过7天没有收到任何报价',
                'priority': 1,  # 高优先级
                'category': 'customer',
                'trigger_config': json.dumps({
                    'type': 'sql_query',
                    'condition': 'customers.created_at < date("now", "-7 days") AND customers.total_drawings = 0',
                    'check_frequency': 'daily'
                }),
                'schedule_config': json.dumps({
                    'frequency': 'daily',
                    'time': '09:00'
                }),
                'notification_config': json.dumps({
                    'method': 'email',
                    'recipients': ['admin@company.com'],
                    'template': 'customer_no_quotes'
                }),
                'action_config': json.dumps({
                    'type': 'notify',
                    'escalate_after': 48
                })
            },
            {
                'rule_id': 'DRAWING_UNCLASSIFIED',
                'name': '图纸未分类提醒',
                'description': '图纸超过24小时未分类',
                'priority': 1,
                'category': 'drawing',
                'trigger_config': json.dumps({
                    'type': 'sql_query',
                    'condition': 'drawings.product_category IS NULL AND drawings.created_at < date("now", "-1 day")',
                    'check_frequency': 'daily'
                }),
                'schedule_config': json.dumps({
                    'frequency': 'daily',
                    'time': '10:00'
                }),
                'notification_config': json.dumps({
                    'method': 'system',
                    'template': 'drawing_unclassified'
                }),
                'action_config': json.dumps({
                    'type': 'notify',
                    'auto_classify': True
                })
            },
            {
                'rule_id': 'DRAWING_PENDING',
                'name': '图纸状态待处理提醒',
                'description': '图纸状态为pending超过48小时',
                'priority': 2,
                'category': 'drawing',
                'trigger_config': json.dumps({
                    'type': 'sql_query',
                    'condition': 'drawings.status = "pending" AND drawings.created_at < date("now", "-2 days")',
                    'check_frequency': 'daily'
                }),
                'schedule_config': json.dumps({
                    'frequency': 'daily',
                    'time': '11:00'
                }),
                'notification_config': json.dumps({
                    'method': 'system',
                    'template': 'drawing_pending'
                }),
                'action_config': json.dumps({
                    'type': 'notify',
                    'escalate_after': 24
                })
            },
            {
                'rule_id': 'NEW_CUSTOMER_WELCOME',
                'name': '新客户欢迎提醒',
                'description': '新客户注册24小时内发送欢迎消息',
                'priority': 3,
                'category': 'customer',
                'trigger_config': json.dumps({
                    'type': 'sql_query',
                    'condition': 'customers.created_at >= date("now", "-1 day")',
                    'check_frequency': 'hourly'
                }),
                'schedule_config': json.dumps({
                    'frequency': 'hourly'
                }),
                'notification_config': json.dumps({
                    'method': 'email',
                    'template': 'welcome_new_customer'
                }),
                'action_config': json.dumps({
                    'type': 'notify',
                    'send_welcome_pack': True
                })
            },
            {
                'rule_id': 'CUSTOMER_INACTIVE',
                'name': '客户活跃度提醒',
                'description': '客户超过30天没有新图纸',
                'priority': 3,
                'category': 'customer',
                'trigger_config': json.dumps({
                    'type': 'sql_query',
                    'condition': 'customers.updated_at < date("now", "-30 days") AND customers.total_drawings > 0',
                    'check_frequency': 'weekly'
                }),
                'schedule_config': json.dumps({
                    'frequency': 'weekly',
                    'day': 'monday',
                    'time': '09:00'
                }),
                'notification_config': json.dumps({
                    'method': 'email',
                    'template': 'customer_inactive'
                }),
                'action_config': json.dumps({
                    'type': 'notify',
                    'create_follow_up_task': True
                })
            },
            {
                'rule_id': 'WEEKLY_BACKUP',
                'name': '系统数据备份提醒',
                'description': '每周日提醒数据备份',
                'priority': 3,
                'category': 'system',
                'trigger_config': json.dumps({
                    'type': 'schedule',
                    'condition': '1 = 1',
                    'check_frequency': 'weekly'
                }),
                'schedule_config': json.dumps({
                    'frequency': 'weekly',
                    'day': 'sunday',
                    'time': '22:00'
                }),
                'notification_config': json.dumps({
                    'method': 'system',
                    'template': 'backup_reminder'
                }),
                'action_config': json.dumps({
                    'type': 'notify',
                    'start_backup': True
                })
            }
        ]

        # 清空现有规则
        cursor.execute("DELETE FROM reminder_rules")
        logger.info("🗑️ 清空现有提醒规则")

        # 插入新规则
        for rule in reminder_rules:
            cursor.execute('''
            INSERT OR REPLACE INTO reminder_rules
            (rule_id, name, description, priority, category, trigger_config,
             schedule_config, notification_config, action_config, is_active, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule['rule_id'],
                rule['name'],
                rule['description'],
                rule['priority'],
                rule['category'],
                rule['trigger_config'],
                rule['schedule_config'],
                rule['notification_config'],
                rule['action_config'],
                True,  # is_active
                'system'  # created_by
            ))

        conn.commit()
        conn.close()

        logger.info(f"✅ 插入 {len(reminder_rules)} 个提醒规则")
        return True

    except Exception as e:
        logger.error(f"❌ 设置提醒规则失败: {e}")
        return False

def setup_reminder_settings(db_path: str = "./data/db.sqlite"):
    """设置提醒系统配置"""

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 检查 reminder_settings 表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='reminder_settings'")
        if not cursor.fetchone():
            # 创建 reminder_settings 表
            cursor.execute('''
            CREATE TABLE reminder_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                setting_key VARCHAR(100) UNIQUE NOT NULL,
                setting_value TEXT,
                description TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            logger.info("✅ 创建 reminder_settings 表")

        # 默认设置
        settings = [
            ('check_interval_minutes', '60', '提醒检查间隔（分钟）'),
            ('max_daily_reminders', '50', '每日最大提醒数量'),
            ('email_enabled', 'true', '是否启用邮件通知'),
            ('system_notifications_enabled', 'true', '是否启用系统通知'),
            ('auto_escalation_enabled', 'true', '是否启用自动升级'),
            ('retention_days', '90', '提醒记录保留天数'),
            ('default_timezone', 'Asia/Shanghai', '默认时区'),
            ('notification_batch_size', '10', '通知批次大小')
        ]

        for key, value, description in settings:
            cursor.execute('''
            INSERT OR REPLACE INTO reminder_settings
            (setting_key, setting_value, description)
            VALUES (?, ?, ?)
            ''', (key, value, description))

        conn.commit()
        conn.close()

        logger.info(f"✅ 设置 {len(settings)} 个提醒系统配置")
        return True

    except Exception as e:
        logger.error(f"❌ 设置提醒配置失败: {e}")
        return False

def main():
    """主函数"""
    try:
        logger.info("🚀 开始设置提醒系统...")

        # 设置提醒规则
        rules_success = setup_reminder_rules()

        # 设置系统配置
        settings_success = setup_reminder_settings()

        if rules_success and settings_success:
            logger.info("🎉 提醒系统设置完成!")
            print("✅ 提醒系统设置成功")
            print("📋 提醒规则已配置")
            print("⚙️ 系统配置已更新")
            print("🔧 可以开始使用提醒功能")
        else:
            logger.error("❌ 提醒系统设置失败")

    except Exception as e:
        logger.error(f"❌ 设置提醒系统时出错: {e}")

if __name__ == "__main__":
    main()