#!/usr/bin/env python3
"""
插入提醒规则脚本
仅插入默认提醒规则，不创建表
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

def insert_default_reminder_rules(db_path: str = "./data/db.sqlite"):
    """插入默认提醒规则"""

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

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

        # 清空现有规则
        cursor.execute("DELETE FROM reminder_rules")

        # 插入新规则
        for rule in default_rules:
            cursor.execute('''
            INSERT INTO reminder_rules
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

        conn.commit()
        conn.close()

        logger.info(f"✅ 插入 {len(default_rules)} 个默认提醒规则")
        return True

    except Exception as e:
        logger.error(f"❌ 插入提醒规则失败: {e}")
        return False

def main():
    """主函数"""
    try:
        logger.info("🚀 开始插入提醒规则...")

        success = insert_default_reminder_rules()

        if success:
            logger.info("🎉 提醒规则插入成功!")
        else:
            logger.error("❌ 提醒规则插入失败")

    except Exception as e:
        logger.error(f"❌ 插入提醒规则时出错: {e}")

if __name__ == "__main__":
    main()