#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update Reminder Database Schema
更新提醒数据库架构

This script updates the existing reminder database to match the new comprehensive schema.
"""

import sqlite3
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def update_reminder_schema():
    """更新提醒系统数据库架构"""

    db_path = "knowledge_base.db"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        logger.info("Starting reminder database schema update...")

        # 检查现有表结构
        cursor.execute("PRAGMA table_info(reminder_records)")
        existing_columns = [row[1] for row in cursor.fetchall()]
        logger.info(f"Existing columns: {existing_columns}")

        # 如果需要添加新列
        if 'related_entity_type' not in existing_columns and 'business_entity_type' in existing_columns:
            logger.info("Table exists with different column names. Using existing structure.")
            # 使用现有的 business_entity_type 和 business_entity_id
            pass

        elif 'related_entity_type' not in existing_columns:
            logger.info("Adding new columns to reminder_records table...")

            # 添加新列
            alter_queries = [
                "ALTER TABLE reminder_records ADD COLUMN escalation_level INTEGER DEFAULT 1",
                "ALTER TABLE reminder_records ADD COLUMN parent_reminder_id INTEGER",
                "ALTER TABLE reminder_records ADD COLUMN related_entity_type VARCHAR(50)",
                "ALTER TABLE reminder_records ADD COLUMN related_entity_id INTEGER",
                "ALTER TABLE reminder_records ADD COLUMN metadata JSON"
            ]

            for query in alter_queries:
                try:
                    cursor.execute(query)
                    logger.info(f"Executed: {query}")
                except sqlite3.Error as e:
                    if "duplicate column name" in str(e):
                        logger.info(f"Column already exists: {query}")
                    else:
                        logger.error(f"Error executing {query}: {e}")

        # 创建其他缺失的表
        create_tables_sql = """

        -- 提醒规则配置表
        CREATE TABLE IF NOT EXISTS reminder_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_id VARCHAR(10) UNIQUE NOT NULL,
            rule_name VARCHAR(100) NOT NULL,
            description TEXT,
            priority VARCHAR(10) NOT NULL,
            check_frequency VARCHAR(20) NOT NULL,
            trigger_condition TEXT NOT NULL,
            notification_methods VARCHAR(100),
            auto_process BOOLEAN DEFAULT FALSE,
            escalation_enabled BOOLEAN DEFAULT FALSE,
            escalation_delay_hours INTEGER DEFAULT 24,
            max_escalation_level INTEGER DEFAULT 3,
            is_active BOOLEAN DEFAULT TRUE,
            config_params JSON,
            created_by VARCHAR(50),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_by VARCHAR(50)
        );

        -- 通知记录表
        CREATE TABLE IF NOT EXISTS notification_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reminder_id INTEGER NOT NULL,
            notification_type VARCHAR(20) NOT NULL,
            recipient VARCHAR(100) NOT NULL,
            recipient_type VARCHAR(20) DEFAULT 'user',
            subject VARCHAR(200),
            content TEXT NOT NULL,
            send_status VARCHAR(20) DEFAULT 'pending',
            sent_time DATETIME,
            error_message TEXT,
            retry_count INTEGER DEFAULT 0,
            external_id VARCHAR(100),
            response_received BOOLEAN DEFAULT FALSE,
            response_time DATETIME,
            metadata JSON,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        -- 提醒处理记录表
        CREATE TABLE IF NOT EXISTS reminder_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reminder_id INTEGER NOT NULL,
            action_type VARCHAR(50) NOT NULL,
            action_description TEXT,
            action_result TEXT,
            performed_by VARCHAR(50),
            performed_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            duration_ms INTEGER,
            success BOOLEAN DEFAULT TRUE,
            error_message TEXT,
            before_state JSON,
            after_state JSON,
            metadata JSON,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        -- 提醒统计表
        CREATE TABLE IF NOT EXISTS reminder_statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stat_date DATE NOT NULL,
            rule_id VARCHAR(10) NOT NULL,
            total_triggered INTEGER DEFAULT 0,
            total_completed INTEGER DEFAULT 0,
            total_failed INTEGER DEFAULT 0,
            avg_processing_time_seconds REAL DEFAULT 0,
            auto_processed_count INTEGER DEFAULT 0,
            manual_processed_count INTEGER DEFAULT 0,
            escalation_count INTEGER DEFAULT 0,
            notification_sent_count INTEGER DEFAULT 0,
            unique_users_involved INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(rule_id, stat_date)
        );

        -- 提醒模板表
        CREATE TABLE IF NOT EXISTS reminder_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            template_name VARCHAR(50) UNIQUE NOT NULL,
            rule_id VARCHAR(10),
            template_type VARCHAR(20) NOT NULL,
            subject_template VARCHAR(200),
            content_template TEXT NOT NULL,
            variables JSON,
            is_active BOOLEAN DEFAULT TRUE,
            created_by VARCHAR(50),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_by VARCHAR(50)
        );

        -- 提醒用户偏好表
        CREATE TABLE IF NOT EXISTS reminder_user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id VARCHAR(50) NOT NULL,
            rule_id VARCHAR(10),
            notification_methods VARCHAR(100),
            quiet_hours_start TIME,
            quiet_hours_end TIME,
            timezone VARCHAR(50) DEFAULT 'Asia/Shanghai',
            max_daily_notifications INTEGER DEFAULT 50,
            escalation_enabled BOOLEAN DEFAULT TRUE,
            weekend_notifications BOOLEAN DEFAULT FALSE,
            is_active BOOLEAN DEFAULT TRUE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, rule_id)
        );

        -- 提醒系统配置表
        CREATE TABLE IF NOT EXISTS reminder_system_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_key VARCHAR(50) UNIQUE NOT NULL,
            config_value TEXT NOT NULL,
            config_type VARCHAR(20) DEFAULT 'string',
            description TEXT,
            is_encrypted BOOLEAN DEFAULT FALSE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_by VARCHAR(50)
        );
        """

        cursor.executescript(create_tables_sql)
        logger.info("Additional tables created successfully")

        # 创建索引 (分别执行以避免错误)
        try:
            # reminder_records 表索引 (使用现有列名)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reminder_business_entity_new ON reminder_records(business_entity_type, business_entity_id)")

            # notification_records 表索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_notification_records_reminder_id ON notification_records(reminder_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_notification_records_type ON notification_records(notification_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_notification_records_status ON notification_records(send_status)")

            # reminder_actions 表索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reminder_actions_reminder_id ON reminder_actions(reminder_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reminder_actions_type ON reminder_actions(action_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reminder_actions_performed_by ON reminder_actions(performed_by)")

            # reminder_statistics 表索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reminder_statistics_stat_date ON reminder_statistics(stat_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reminder_statistics_rule_id ON reminder_statistics(rule_id)")

            # reminder_templates 表索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reminder_templates_rule_id ON reminder_templates(rule_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reminder_templates_type ON reminder_templates(template_type)")

            # reminder_user_preferences 表索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reminder_user_preferences_user_id ON reminder_user_preferences(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reminder_user_preferences_rule_id ON reminder_user_preferences(rule_id)")

            # reminder_system_config 表索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reminder_system_config_key ON reminder_system_config(config_key)")

            logger.info("Indexes created successfully")
        except sqlite3.Error as e:
            logger.error(f"Error creating indexes: {e}")
            # 继续执行，索引创建失败不影响主要功能

  
        # 插入默认数据
        insert_default_data_sql = """

        -- 插入提醒规则
        INSERT OR REPLACE INTO reminder_rules (
            rule_id, rule_name, description, priority, check_frequency,
            trigger_condition, notification_methods, auto_process,
            escalation_enabled, is_active, config_params
        ) VALUES
        ('R001', '客户图纸状态变更提醒', '监控图纸状态变更并及时通知', '高', '每小时',
         'SELECT d.id FROM drawings d WHERE d.updated_at > datetime(''now'', ''-1 hour'')',
         'email,system', TRUE, TRUE, TRUE,
         '{"notification_users": ["engineering@company.com"], "auto_classify": true}'),

        ('R002', '报价超时预警', '监控报价处理时间，超时自动预警', '高', '每15分钟',
         'SELECT fq.id FROM factory_quotes fq WHERE fq.status = ''processing'' AND fq.created_at < datetime(''now'', ''-24 hours'')',
         'email,sms', FALSE, TRUE, TRUE,
         '{"timeout_hours": 24, "escalation_users": ["manager@company.com"]}'),

        ('R003', '质量异常报警', '监控质量指标，异常时立即报警', '高', '实时',
         'SELECT qi.id FROM quality_inspections qi WHERE qi.score < 70 AND qi.created_at > datetime(''now'', ''-1 hour'')',
         'email,sms,system', TRUE, TRUE, TRUE,
         '{"quality_threshold": 70, "auto_stop_production": true}'),

        ('R004', '交付超期预警', '监控交付进度，提前预警延期风险', '高', '每天上午9点',
         'SELECT po.id FROM production_orders po WHERE po.expected_delivery_date BETWEEN date(''now'') AND date(''now'', ''+7 days'')',
         'email,system', FALSE, TRUE, TRUE,
         '{"warning_days": 7, "progress_threshold": 20}'),

        ('R005', '客户投诉提醒', '监控客户投诉，确保及时处理', '高', '实时',
         'SELECT comp.id FROM complaints comp WHERE comp.created_at > datetime(''now'', ''-1 hour'')',
         'email,system', TRUE, TRUE, TRUE,
         '{"response_timeout_hours": 2, "auto_assign": true}'),

        ('R006', '批次生产计划提醒', '提醒生产计划安排和执行', '中', '每小时',
         'SELECT pb.id FROM production_batches pb WHERE pb.created_at > datetime(''now'', ''-1 hour'')',
         'email,system', FALSE, TRUE, TRUE,
         '{"advance_hours": 24, "notification_roles": ["production_manager"]}'),

        ('R007', '技术参数更新提醒', '监控技术参数变更', '中', '每天上午10点',
         'SELECT tp.id FROM technical_parameters tp WHERE tp.updated_at > datetime(''now'', ''-24 hours'')',
         'system', TRUE, FALSE, TRUE,
         '{"auto_distribute": true, "notification_channels": ["system"]}'),

        ('R008', '客户跟进提醒', '提醒销售人员定期跟进客户', '中', '每天上午11点',
         'SELECT c.id FROM customers c WHERE c.last_contact_date < date(''now'', ''-7 days'')',
         'system', FALSE, FALSE, TRUE,
         '{"followup_days": 7, "assign_to_sales": true}'),

        ('R009', '报价分析报告提醒', '生成报价分析报告', '中', '每天下午2点',
         'SELECT COUNT(*) FROM factory_quotes WHERE updated_at > date(''now'', ''-1 day'')',
         'email', TRUE, FALSE, TRUE,
         '{"auto_generate": true, "recipients": ["management@company.com"]}'),

        ('R010', '生产进度更新提醒', '监控生产进度变更', '中', '每30分钟',
         'SELECT po.id FROM production_orders po WHERE po.updated_at > datetime(''now'', ''-30 minutes'')',
         'system', TRUE, FALSE, TRUE,
         '{"real_time_update": true, "notification_team": ["production"]}'),

        ('R011', '数据备份提醒', '执行数据备份任务', '低', '每天晚上8点',
         'SELECT 1', 'system', TRUE, FALSE, TRUE,
         '{"backup_path": "/backup", "retention_days": 30}'),

        ('R012', '月度统计报告提醒', '生成月度统计报告', '低', '每月1号上午9点',
         'SELECT 1 WHERE date(''now'') = date(''now'', ''start of month'')',
         'email', TRUE, FALSE, TRUE,
         '{"report_type": "monthly", "recipients": ["management@company.com"]}'),

        ('R013', '员工生日提醒', '员工生日祝福', '低', '每天上午8点',
         'SELECT e.id FROM employees e WHERE e.birthday = date(''now'')',
         'system', TRUE, FALSE, TRUE,
         '{"auto_greeting": true, "notification_type": "birthday"}'),

        ('R014', '合同到期提醒', '合同到期前提醒', '低', '每周一上午10点',
         'SELECT c.id FROM contracts c WHERE c.end_date BETWEEN date(''now'') AND date(''now'', ''+30 days'')',
         'email', FALSE, FALSE, TRUE,
         '{"warning_days": 30, "notify_legal": true}'),

        ('R015', '库存预警提醒', '库存水平监控', '低', '每天下午3点',
         'SELECT i.id FROM inventory i WHERE i.quantity < i.safety_stock',
         'system', FALSE, FALSE, TRUE,
         '{"auto_suggest_reorder": true, "notify_purchasing": true}'),

        ('R016', '供应商评估提醒', '供应商绩效评估', '低', '每季度最后一天',
         'SELECT 1 WHERE date(''now'', ''start of month'', ''+2 month'') = date(''now'', ''start of month'', ''+3 month'', ''-1 day'')',
         'email', TRUE, FALSE, TRUE,
         '{"evaluation_period": "quarterly", "auto_generate": true}'),

        ('R017', '设备维护提醒', '设备预防性维护', '低', '每周日上午9点',
         'SELECT eq.id FROM equipment eq WHERE eq.next_maintenance_date BETWEEN date(''now'') AND date(''now'', ''+7 days'')',
         'email,system', FALSE, TRUE, TRUE,
         '{"advance_days": 7, "assign_maintenance": true}');

        -- 插入系统配置
        INSERT OR REPLACE INTO reminder_system_config (
            config_key, config_value, config_type, description
        ) VALUES
        ('system.enabled', 'true', 'boolean', '提醒系统是否启用'),
        ('max_daily_notifications', '100', 'integer', '每日最大通知数'),
        ('notification_rate_limit', '10', 'integer', '每小时最大通知数'),
        ('default_timezone', 'Asia/Shanghai', 'string', '默认时区'),
        ('log_retention_days', '90', 'integer', '日志保留天数'),
        ('email.smtp_server', 'smtp.company.com', 'string', 'SMTP服务器'),
        ('email.smtp_port', '587', 'integer', 'SMTP端口'),
        ('email.use_tls', 'true', 'boolean', '是否使用TLS'),
        ('email.sender', 'system@company.com', 'string', '发件人邮箱'),
        ('sms.provider', 'aliyun', 'string', '短信服务商');
        """

        cursor.executescript(insert_default_data_sql)
        logger.info("Default data inserted successfully")

        # 提交更改
        conn.commit()

        # 验证更新
        cursor.execute("SELECT COUNT(*) FROM reminder_rules")
        rules_count = cursor.fetchone()[0]

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'reminder_%'")
        tables = [row[0] for row in cursor.fetchall()]

        print("\n" + "="*60)
        print("✅ REMINDER DATABASE SCHEMA UPDATE COMPLETED")
        print("="*60)
        print(f"📋 Total reminder rules: {rules_count}")
        print(f"📊 Total tables: {len(tables)}")
        print(f"📅 Tables created: {', '.join(tables)}")
        print("🚀 Reminder system is ready for automation!")
        print("="*60)

        return True

    except Exception as e:
        logger.error(f"Failed to update reminder schema: {e}")
        if conn:
            conn.rollback()
        return False

    finally:
        if conn:
            conn.close()

def main():
    """主函数"""
    success = update_reminder_schema()

    if success:
        logger.info("✅ Reminder database schema updated successfully!")
    else:
        logger.error("❌ Failed to update reminder database schema!")

if __name__ == "__main__":
    main()