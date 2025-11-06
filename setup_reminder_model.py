#!/usr/bin/env python3
"""
提醒系统数据库模型设置
使用SQLAlchemy定义提醒规则、记录和通知历史表
"""

import sqlite3
import logging
from datetime import datetime
from typing import Optional, Dict, Any

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReminderSystemModel:
    """提醒系统数据库模型管理器"""

    def __init__(self, db_path: str = "./data/db.sqlite"):
        self.db_path = db_path

    def create_reminder_tables(self):
        """创建提醒系统相关表"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 1. 提醒规则表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS reminder_rules (
                rule_id TEXT PRIMARY KEY,
                rule_name TEXT NOT NULL,
                trigger_condition_json TEXT NOT NULL,
                check_frequency TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 3,
                notification_method_json TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT,
                category TEXT DEFAULT 'general',
                trigger_count INTEGER DEFAULT 0,
                last_triggered TIMESTAMP,
                created_by TEXT DEFAULT 'system'
            )
            ''')

            # 2. 提醒记录表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS reminder_records (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_id TEXT NOT NULL,
                triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                entity_type TEXT NOT NULL,
                entity_id TEXT,
                details_json TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                handled_at TIMESTAMP,
                handled_by TEXT,
                notification_sent BOOLEAN DEFAULT 0,
                notification_sent_at TIMESTAMP,
                notification_method TEXT,
                FOREIGN KEY (rule_id) REFERENCES reminder_rules(rule_id)
            )
            ''')

            # 3. 通知历史表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS notification_history (
                notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id INTEGER,
                sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notification_type TEXT NOT NULL,
                status TEXT NOT NULL,
                details_json TEXT,
                recipient TEXT,
                subject TEXT,
                content TEXT,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                FOREIGN KEY (record_id) REFERENCES reminder_records(record_id)
            )
            ''')

            logger.info("✅ 提醒系统表创建成功")

            # 创建索引
            self.create_indexes(cursor)

            # 插入默认规则
            self.insert_default_rules(cursor)

            # 插入默认配置
            self.insert_default_settings(cursor)

            conn.commit()
            conn.close()

            logger.info("🎉 提醒系统数据库模型初始化完成")
            return True

        except Exception as e:
            logger.error(f"❌ 创建提醒系统表失败: {e}")
            return False

    def create_indexes(self, cursor):
        """创建数据库索引"""
        indexes = [
            # reminder_records 表索引
            "CREATE INDEX IF NOT EXISTS idx_reminder_records_triggered_at ON reminder_records(triggered_at)",
            "CREATE INDEX IF NOT EXISTS idx_reminder_records_status ON reminder_records(status)",
            "CREATE INDEX IF NOT EXISTS idx_reminder_records_rule_id ON reminder_records(rule_id)",
            "CREATE INDEX IF NOT EXISTS idx_reminder_records_business_entity ON reminder_records(business_entity_type, business_entity_id)",
            "CREATE INDEX IF NOT EXISTS idx_reminder_records_execution_id ON reminder_records(execution_id)",

            # reminder_rules 表索引
            "CREATE INDEX IF NOT EXISTS idx_reminder_rules_active ON reminder_rules(is_active)",
            "CREATE INDEX IF NOT EXISTS idx_reminder_rules_priority ON reminder_rules(priority)",
            "CREATE INDEX IF NOT EXISTS idx_reminder_rules_category ON reminder_rules(category)",
            "CREATE INDEX IF NOT EXISTS idx_reminder_rules_last_triggered ON reminder_rules(last_triggered)",

            # notification_history 表索引
            "CREATE INDEX IF NOT EXISTS idx_notification_history_sent_at ON notification_history(sent_at)",
            "CREATE INDEX IF NOT EXISTS idx_notification_history_record_id ON notification_history(reminder_record_id)",
            "CREATE INDEX IF NOT EXISTS idx_notification_history_type ON notification_history(notification_type)",
            "CREATE INDEX IF NOT EXISTS idx_notification_history_status ON notification_history(status)"
        ]

        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except Exception as e:
                logger.warning(f"创建索引失败: {e}")

        logger.info(f"✅ 创建 {len(indexes)} 个数据库索引")

    def insert_default_rules(self, cursor):
        """插入默认提醒规则"""
        import json

        default_rules = [
            {
                'rule_id': 'QUOTATION_NO_REPLY_14D',
                'name': '报价14天未回复提醒',
                'trigger_config': json.dumps({
                    'type': 'sql_query',
                    'query': '''
                        SELECT fq.id, fq.quote_date, f.factory_name, c.company_name
                        FROM factory_quotes fq
                        JOIN factories f ON fq.factory_id = f.id
                        LEFT JOIN customers c ON fq.customer_id = c.id
                        WHERE fq.quote_date < date('now', '-14 days')
                        AND fq.id NOT IN (
                            SELECT rr.business_entity_id FROM reminder_records rr
                            WHERE rr.rule_id = 'QUOTATION_NO_REPLY_14D'
                            AND rr.status = 'pending'
                        )
                    '''
                }),
                'schedule_config': json.dumps({
                    'frequency': 'daily',
                    'time': '09:00'
                }),
                'notification_config': json.dumps({
                    'type': 'email',
                    'template': 'quotation_followup',
                    'recipients': ['sales@company.com']
                }),
                'priority': 2,
                'description': '报价超过14天未收到客户回复时提醒跟进',
                'category': 'sales'
            },
            {
                'rule_id': 'INQUIRY_NO_RESPONSE_3D',
                'name': '询盘3天未响应提醒',
                'trigger_config': json.dumps({
                    'type': 'sql_query',
                    'query': '''
                        SELECT c.id, c.company_name, c.first_contact_date, c.contact_email
                        FROM customers c
                        WHERE c.first_contact_date < date('now', '-3 days')
                        AND c.total_drawings = 0
                        AND c.id NOT IN (
                            SELECT rr.business_entity_id FROM reminder_records rr
                            WHERE rr.rule_id = 'INQUIRY_NO_RESPONSE_3D'
                            AND rr.status = 'pending'
                        )
                    '''
                }),
                'schedule_config': json.dumps({
                    'frequency': 'daily',
                    'time': '10:00'
                }),
                'notification_config': json.dumps({
                    'type': 'email',
                    'template': 'inquiry_followup',
                    'recipients': ['sales@company.com', 'manager@company.com']
                }),
                'priority': 1,
                'description': '客户询盘超过3天未响应时提醒跟进',
                'category': 'sales'
            },
            {
                'rule_id': 'DRAWING_CLASSIFICATION_OVERDUE',
                'name': '图纸分类超时提醒',
                'trigger_config': json.dumps({
                    'type': 'sql_query',
                    'query': '''
                        SELECT d.id, d.drawing_name, d.created_at, c.company_name
                        FROM drawings d
                        LEFT JOIN customers c ON d.customer_id = c.id
                        WHERE d.product_category IS NULL
                        AND d.created_at < date('now', '-2 days')
                        AND d.id NOT IN (
                            SELECT rr.business_entity_id FROM reminder_records rr
                            WHERE rr.rule_id = 'DRAWING_CLASSIFICATION_OVERDUE'
                            AND rr.status = 'pending'
                        )
                    '''
                }),
                'schedule_config': json.dumps({
                    'frequency': 'daily',
                    'time': '11:00'
                }),
                'notification_config': json.dumps({
                    'type': 'system',
                    'template': 'drawing_classification_overdue',
                    'recipients': ['engineering@company.com']
                }),
                'priority': 2,
                'description': '图纸超过48小时未分类时提醒',
                'category': 'engineering'
            },
            {
                'rule_id': 'NEW_CUSTOMER_NO_DRAWINGS',
                'name': '新客户无图纸提醒',
                'trigger_config': json.dumps({
                    'type': 'sql_query',
                    'query': '''
                        SELECT c.id, c.company_name, c.created_at, c.contact_email
                        FROM customers c
                        WHERE c.total_drawings = 0
                        AND c.created_at < date('now', '-7 days')
                        AND c.id NOT IN (
                            SELECT rr.business_entity_id FROM reminder_records rr
                            WHERE rr.rule_id = 'NEW_CUSTOMER_NO_DRAWINGS'
                            AND rr.status = 'pending'
                        )
                    '''
                }),
                'schedule_config': json.dumps({
                    'frequency': 'daily',
                    'time': '14:00'
                }),
                'notification_config': json.dumps({
                    'type': 'email',
                    'template': 'new_customer_no_drawings',
                    'recipients': ['sales@company.com']
                }),
                'priority': 2,
                'description': '新客户注册7天仍无图纸时提醒',
                'category': 'sales'
            },
            {
                'rule_id': 'WEEKLY_SYSTEM_BACKUP',
                'name': '周日系统备份提醒',
                'trigger_config': json.dumps({
                    'type': 'schedule',
                    'condition': "strftime('%w', 'now') = '0'"  # Sunday
                }),
                'schedule_config': json.dumps({
                    'frequency': 'weekly',
                    'day': 'sunday',
                    'time': '22:00'
                }),
                'notification_config': json.dumps({
                    'type': 'system',
                    'template': 'weekly_backup',
                    'recipients': ['admin@company.com']
                }),
                'priority': 3,
                'description': '每周日提醒进行系统备份',
                'category': 'admin'
            },
            {
                'rule_id': 'DRAWING_PENDING_STATUS',
                'name': '图纸状态待处理提醒',
                'trigger_config': json.dumps({
                    'type': 'sql_query',
                    'query': '''
                        SELECT d.id, d.drawing_name, d.status, d.created_at, c.company_name
                        FROM drawings d
                        LEFT JOIN customers c ON d.customer_id = c.id
                        WHERE d.status = 'pending'
                        AND d.created_at < date('now', '-1 day')
                        AND d.id NOT IN (
                            SELECT rr.business_entity_id FROM reminder_records rr
                            WHERE rr.rule_id = 'DRAWING_PENDING_STATUS'
                            AND rr.status = 'pending'
                        )
                    '''
                }),
                'schedule_config': json.dumps({
                    'frequency': 'daily',
                    'time': '15:00'
                }),
                'notification_config': json.dumps({
                    'type': 'system',
                    'template': 'drawing_pending_status',
                    'recipients': ['engineering@company.com']
                }),
                'priority': 2,
                'description': '图纸状态为pending超过24小时提醒',
                'category': 'engineering'
            }
        ]

        for rule in default_rules:
            cursor.execute('''
                INSERT OR REPLACE INTO reminder_rules
                (rule_id, name, trigger_config, schedule_config, notification_config,
                 priority, description, category, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule['rule_id'],
                rule['name'],
                rule['trigger_config'],
                rule['schedule_config'],
                rule['notification_config'],
                rule['priority'],
                rule['description'],
                rule['category'],
                'system'
            ))

        logger.info(f"✅ 插入 {len(default_rules)} 个默认提醒规则")

    def insert_default_settings(self, cursor):
        """插入默认系统配置"""
        settings = [
            ('reminder_check_interval_minutes', '60', '提醒检查间隔（分钟）'),
            ('max_daily_reminders', '100', '每日最大提醒数量'),
            ('email_notifications_enabled', 'true', '是否启用邮件通知'),
            ('system_notifications_enabled', 'true', '是否启用系统通知'),
            ('auto_escalation_enabled', 'true', '是否启用自动升级'),
            ('escalation_hours', '24', '升级时间（小时）'),
            ('reminder_retention_days', '90', '提醒记录保留天数'),
            ('notification_retry_attempts', '3', '通知重试次数'),
            ('default_timezone', 'Asia/Shanghai', '默认时区'),
            ('batch_notification_size', '10', '批量通知大小')
        ]

        for key, value, description in settings:
            cursor.execute('''
                INSERT OR REPLACE INTO reminder_settings
                (setting_key, setting_value, description)
                VALUES (?, ?, ?)
            ''', (key, value, description))

        logger.info(f"✅ 插入 {len(settings)} 个默认系统配置")

    def drop_old_tables(self):
        """删除旧的表（如果需要重新创建）"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            old_tables = [
                'reminder_rules_old',
                'reminder_records_old',
                'notification_history_old'
            ]

            for table in old_tables:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")

            conn.commit()
            conn.close()
            logger.info("🗑️ 清理旧表完成")

        except Exception as e:
            logger.error(f"❌ 清理旧表失败: {e}")

def main():
    """主函数"""
    try:
        logger.info("🚀 开始设置提醒系统数据库模型...")

        model = ReminderSystemModel()

        # 可选：清理旧表
        # model.drop_old_tables()

        # 创建表和初始数据
        success = model.create_reminder_tables()

        if success:
            logger.info("🎉 提醒系统数据库模型设置完成!")
            print("✅ 提醒系统数据库表已创建")
            print("📋 默认提醒规则已配置")
            print("⚙️ 系统配置已初始化")
            print("🔧 数据库索引已创建")
            print("📊 系统已准备就绪")
        else:
            logger.error("❌ 提醒系统数据库模型设置失败")

    except Exception as e:
        logger.error(f"❌ 设置数据库模型时出错: {e}")

if __name__ == "__main__":
    main()