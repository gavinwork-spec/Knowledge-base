#!/usr/bin/env python3
"""
提醒系统数据库创建脚本
创建提醒相关的数据库表和基础数据
"""

import sqlite3
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_reminder_tables(db_path: str = "./data/db.sqlite"):
    """创建提醒相关表"""

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 1. 提醒规则表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS reminder_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(100) NOT NULL,
            description TEXT,
            trigger_condition TEXT NOT NULL,
            check_frequency VARCHAR(20) NOT NULL,
            priority VARCHAR(10) NOT NULL,
            notification_method VARCHAR(20) NOT NULL,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # 2. 提醒记录表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS reminder_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_id INTEGER NOT NULL,
            trigger_entity_id INTEGER,
            trigger_entity_type VARCHAR(50),
            trigger_data TEXT,
            notification_sent BOOLEAN DEFAULT 0,
            notification_sent_at TIMESTAMP,
            notification_method VARCHAR(20),
            is_acknowledged BOOLEAN DEFAULT 0,
            acknowledged_at TIMESTAMP,
            acknowledged_by VARCHAR(100),
            status VARCHAR(20) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (rule_id) REFERENCES reminder_rules(id)
        )
        ''')

        # 3. 提醒配置表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS reminder_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_key VARCHAR(100) UNIQUE NOT NULL,
            config_value TEXT,
            description TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        logger.info("✅ 提醒系统数据库表创建成功")

        # 创建索引
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_reminder_records_rule_id ON reminder_records(rule_id)",
            "CREATE INDEX IF NOT EXISTS idx_reminder_records_status ON reminder_records(status)",
            "CREATE INDEX IF NOT EXISTS idx_reminder_records_created_at ON reminder_records(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_reminder_rules_active ON reminder_rules(is_active)",
            "CREATE INDEX IF NOT EXISTS idx_reminder_records_entity ON reminder_records(trigger_entity_type, trigger_entity_id)"
        ]

        for index_sql in indexes:
            cursor.execute(index_sql)

        logger.info("✅ 提醒系统索引创建成功")

        # 插入默认提醒规则
        insert_default_reminder_rules(cursor)

        # 插入默认配置
        insert_default_configs(cursor)

        conn.commit()
        conn.close()

        logger.info("✅ 提醒系统初始化完成")
        return True

    except Exception as e:
        logger.error(f"❌ 创建提醒系统失败: {e}")
        return False

def insert_default_reminder_rules(cursor):
    """插入默认提醒规则"""

    default_rules = [
        {
            'name': '新客户未报价提醒',
            'description': '客户注册超过7天没有收到任何报价',
            'trigger_condition': 'customers.created_at < date("now", "-7 days") AND customers.total_drawings = 0',
            'check_frequency': 'daily',
            'priority': 'high',
            'notification_method': 'email'
        },
        {
            'name': '图纸分类待确认提醒',
            'description': '自动分类的图纸超过48小时未确认',
            'trigger_condition': 'drawings.is_classified = 1 AND drawings.status = "pending" AND drawings.classification_date < date("now", "-2 days")',
            'check_frequency': 'hourly',
            'priority': 'high',
            'notification_method': 'system'
        },
        {
            'name': '图纸未分类提醒',
            'description': '图纸超过24小时未分类',
            'trigger_condition': 'drawings.product_category IS NULL AND drawings.created_at < date("now", "-1 day")',
            'check_frequency': 'daily',
            'priority': 'high',
            'notification_method': 'system'
        },
        {
            'name': '长期未更新图纸提醒',
            'description': '图纸超过30天未更新',
            'trigger_condition': 'drawings.updated_at < date("now", "-30 days")',
            'check_frequency': 'weekly',
            'priority': 'low',
            'notification_method': 'email'
        },
        {
            'name': '新客户欢迎提醒',
            'description': '新客户注册24小时内发送欢迎消息',
            'trigger_condition': 'customers.created_at >= date("now", "-1 day")',
            'check_frequency': 'hourly',
            'priority': 'medium',
            'notification_method': 'email'
        },
        {
            'name': '系统数据备份提醒',
            'description': '每周日提醒数据备份',
            'trigger_condition': '1',  # 简化条件，通过频率控制
            'check_frequency': 'weekly',
            'priority': 'low',
            'notification_method': 'system'
        },
        {
            'name': '图纸状态待处理提醒',
            'description': '图纸状态为pending超过48小时',
            'trigger_condition': 'drawings.status = "pending" AND drawings.created_at < date("now", "-2 days")',
            'check_frequency': 'daily',
            'priority': 'medium',
            'notification_method': 'system'
        },
        {
            'name': '客户活跃度提醒',
            'description': '客户超过30天没有新图纸',
            'trigger_condition': 'customers.updated_at < date("now", "-30 days") AND customers.total_drawings > 0',
            'check_frequency': 'weekly',
            'priority': 'low',
            'notification_method': 'email'
        }
    ]

    for rule in default_rules:
        cursor.execute('''
        INSERT OR IGNORE INTO reminder_rules
        (name, description, trigger_condition, check_frequency, priority, notification_method)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            rule['name'],
            rule['description'],
            rule['trigger_condition'],
            rule['check_frequency'],
            rule['priority'],
            rule['notification_method']
        ))

    logger.info(f"✅ 插入 {len(default_rules)} 个默认提醒规则")

def insert_default_configs(cursor):
    """插入默认配置"""

    default_configs = [
        ('reminder_check_interval', '60', '提醒检查间隔（分钟）'),
        ('max_daily_reminders', '50', '每日最大提醒数量'),
        ('email_notification_enabled', 'true', '是否启用邮件通知'),
        ('system_notification_enabled', 'true', '是否启用系统通知'),
        ('reminder_retention_days', '90', '提醒记录保留天数'),
        ('auto_acknowledge_days', '30', '自动确认天数'),
        ('escalation_enabled', 'true', '是否启用升级机制'),
        ('escalation_hours', '24', '升级时间（小时）')
    ]

    for config_key, config_value, description in default_configs:
        cursor.execute('''
        INSERT OR IGNORE INTO reminder_configs
        (config_key, config_value, description)
        VALUES (?, ?, ?)
        ''', (config_key, config_value, description))

    logger.info(f"✅ 插入 {len(default_configs)} 个默认配置")

def main():
    """主函数"""
    try:
        logger.info("🚀 开始创建提醒系统...")

        success = create_reminder_tables()

        if success:
            logger.info("🎉 提醒系统创建成功!")
            print("✅ 提醒系统数据库表已创建")
            print("📋 默认提醒规则已配置")
            print("⚙️ 系统配置已初始化")
            print("🔧 可以开始使用提醒功能")
        else:
            logger.error("❌ 提醒系统创建失败")

    except Exception as e:
        logger.error(f"❌ 创建提醒系统时出错: {e}")

if __name__ == "__main__":
    main()