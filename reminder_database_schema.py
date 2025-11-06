#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reminder System Database Schema
提醒系统数据库表模型定义

This script defines the complete database schema for the reminder system,
including tables for reminders, notifications, rules, and audit logs.
"""

import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReminderDatabaseSchema:
    """提醒系统数据库模型管理类"""

    def __init__(self, db_path: str = "knowledge_base.db"):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        """连接数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.execute("PRAGMA foreign_keys = ON")
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def execute_schema(self, schema_sql: str):
        """执行数据库架构脚本"""
        try:
            if not self.conn:
                self.connect()

            cursor = self.conn.cursor()
            cursor.executescript(schema_sql)
            self.conn.commit()
            logger.info("Database schema executed successfully")

        except sqlite3.Error as e:
            logger.error(f"Failed to execute schema: {e}")
            if self.conn:
                self.conn.rollback()
            raise

    def create_reminder_tables(self):
        """创建提醒系统相关表"""

        schema_sql = """
        -- ==================== 提醒记录表 ====================
        CREATE TABLE IF NOT EXISTS reminder_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_id VARCHAR(10) NOT NULL,                    -- 规则ID (R001-R017)
            rule_name VARCHAR(100) NOT NULL,                 -- 规则名称
            trigger_time DATETIME NOT NULL,                  -- 触发时间
            trigger_condition TEXT NOT NULL,                 -- 触发条件详情
            priority VARCHAR(10) NOT NULL,                   -- 优先级 (高/中/低)
            status VARCHAR(20) DEFAULT 'pending',            -- 状态 (pending/processing/completed/failed/ignored)
            assigned_to VARCHAR(50),                         -- 分配给谁
            due_time DATETIME,                               -- 截止时间
            completed_time DATETIME,                         -- 完成时间
            notification_methods VARCHAR(100),               -- 通知方式 (email/sms/system)
            auto_processed BOOLEAN DEFAULT FALSE,            -- 是否自动处理
            processing_result TEXT,                          -- 处理结果
            error_message TEXT,                              -- 错误信息
            retry_count INTEGER DEFAULT 0,                   -- 重试次数
            escalation_level INTEGER DEFAULT 1,              -- 升级级别
            parent_reminder_id INTEGER,                      -- 父提醒ID (用于升级)
            related_entity_type VARCHAR(50),                 -- 关联实体类型
            related_entity_id INTEGER,                       -- 关联实体ID
            metadata JSON,                                   -- 扩展数据
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (parent_reminder_id) REFERENCES reminder_records(id)
        );

        -- ==================== 提醒规则配置表 ====================
        CREATE TABLE IF NOT EXISTS reminder_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_id VARCHAR(10) UNIQUE NOT NULL,             -- 规则ID
            rule_name VARCHAR(100) NOT NULL,                 -- 规则名称
            description TEXT,                                -- 规则描述
            priority VARCHAR(10) NOT NULL,                   -- 优先级
            check_frequency VARCHAR(20) NOT NULL,            -- 检查频率
            trigger_condition TEXT NOT NULL,                 -- 触发条件SQL
            notification_methods VARCHAR(100),               -- 通知方式
            auto_process BOOLEAN DEFAULT FALSE,              -- 是否自动处理
            escalation_enabled BOOLEAN DEFAULT FALSE,        -- 是否启用升级
            escalation_delay_hours INTEGER DEFAULT 24,       -- 升级延迟小时数
            max_escalation_level INTEGER DEFAULT 3,          -- 最大升级级别
            is_active BOOLEAN DEFAULT TRUE,                  -- 是否启用
            config_params JSON,                              -- 配置参数
            created_by VARCHAR(50),                          -- 创建人
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_by VARCHAR(50)                           -- 更新人
        );

        -- ==================== 通知记录表 ====================
        CREATE TABLE IF NOT EXISTS notification_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reminder_id INTEGER NOT NULL,                    -- 提醒ID
            notification_type VARCHAR(20) NOT NULL,          -- 通知类型 (email/sms/system/webhook)
            recipient VARCHAR(100) NOT NULL,                 -- 接收人
            recipient_type VARCHAR(20) DEFAULT 'user',       -- 接收人类型 (user/group/role)
            subject VARCHAR(200),                            -- 通知主题
            content TEXT NOT NULL,                           -- 通知内容
            send_status VARCHAR(20) DEFAULT 'pending',       -- 发送状态 (pending/sent/failed/retry)
            sent_time DATETIME,                              -- 发送时间
            error_message TEXT,                              -- 错误信息
            retry_count INTEGER DEFAULT 0,                   -- 重试次数
            external_id VARCHAR(100),                        -- 外部系统ID
            response_received BOOLEAN DEFAULT FALSE,         -- 是否收到响应
            response_time DATETIME,                          -- 响应时间
            metadata JSON,                                   -- 扩展数据
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (reminder_id) REFERENCES reminder_records(id)
        );

        -- ==================== 提醒处理记录表 ====================
        CREATE TABLE IF NOT EXISTS reminder_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reminder_id INTEGER NOT NULL,                    -- 提醒ID
            action_type VARCHAR(50) NOT NULL,                -- 操作类型
            action_description TEXT,                         -- 操作描述
            action_result TEXT,                              -- 操作结果
            performed_by VARCHAR(50),                        -- 执行人
            performed_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            duration_ms INTEGER,                             -- 执行时长(毫秒)
            success BOOLEAN DEFAULT TRUE,                    -- 是否成功
            error_message TEXT,                              -- 错误信息
            before_state JSON,                               -- 操作前状态
            after_state JSON,                                -- 操作后状态
            metadata JSON,                                   -- 扩展数据
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (reminder_id) REFERENCES reminder_records(id)
        );

        -- ==================== 提醒统计表 ====================
        CREATE TABLE IF NOT EXISTS reminder_statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stat_date DATE NOT NULL,                         -- 统计日期
            rule_id VARCHAR(10) NOT NULL,                    -- 规则ID
            total_triggered INTEGER DEFAULT 0,               -- 总触发次数
            total_completed INTEGER DEFAULT 0,               -- 总完成次数
            total_failed INTEGER DEFAULT 0,                  -- 总失败次数
            avg_processing_time_seconds REAL DEFAULT 0,      -- 平均处理时间(秒)
            auto_processed_count INTEGER DEFAULT 0,          -- 自动处理数量
            manual_processed_count INTEGER DEFAULT 0,        -- 人工处理数量
            escalation_count INTEGER DEFAULT 0,              -- 升级次数
            notification_sent_count INTEGER DEFAULT 0,       -- 通知发送数量
            unique_users_involved INTEGER DEFAULT 0,         -- 涉及用户数
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(rule_id, stat_date)
        );

        -- ==================== 提醒模板表 ====================
        CREATE TABLE IF NOT EXISTS reminder_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            template_name VARCHAR(50) UNIQUE NOT NULL,       -- 模板名称
            rule_id VARCHAR(10),                             -- 适用规则ID
            template_type VARCHAR(20) NOT NULL,              -- 模板类型 (email/sms/system)
            subject_template VARCHAR(200),                   -- 主题模板
            content_template TEXT NOT NULL,                  -- 内容模板
            variables JSON,                                  -- 模板变量
            is_active BOOLEAN DEFAULT TRUE,                  -- 是否启用
            created_by VARCHAR(50),                          -- 创建人
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_by VARCHAR(50),                          -- 更新人

            FOREIGN KEY (rule_id) REFERENCES reminder_rules(rule_id)
        );

        -- ==================== 提醒用户偏好表 ====================
        CREATE TABLE IF NOT EXISTS reminder_user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id VARCHAR(50) NOT NULL,                    -- 用户ID
            rule_id VARCHAR(10),                             -- 规则ID (NULL表示全局设置)
            notification_methods VARCHAR(100),               -- 偏好的通知方式
            quiet_hours_start TIME,                          -- 免打扰开始时间
            quiet_hours_end TIME,                            -- 免打扰结束时间
            timezone VARCHAR(50) DEFAULT 'Asia/Shanghai',   -- 时区
            max_daily_notifications INTEGER DEFAULT 50,     -- 每日最大通知数
            escalation_enabled BOOLEAN DEFAULT TRUE,         -- 是否接受升级通知
            weekend_notifications BOOLEAN DEFAULT FALSE,    -- 周末通知
            is_active BOOLEAN DEFAULT TRUE,                  -- 是否启用
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(user_id, rule_id)
        );

        -- ==================== 提醒系统配置表 ====================
        CREATE TABLE IF NOT EXISTS reminder_system_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_key VARCHAR(50) UNIQUE NOT NULL,          -- 配置键
            config_value TEXT NOT NULL,                      -- 配置值
            config_type VARCHAR(20) DEFAULT 'string',        -- 配置类型
            description TEXT,                                -- 配置描述
            is_encrypted BOOLEAN DEFAULT FALSE,              -- 是否加密
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_by VARCHAR(50)                           -- 更新人
        );
        """

        self.execute_schema(schema_sql)
        logger.info("Reminder tables created successfully")

    def create_indexes(self):
        """创建索引以提高查询性能"""

        index_sql = """
        -- reminder_records 表索引
        CREATE INDEX IF NOT EXISTS idx_reminder_records_rule_id ON reminder_records(rule_id);
        CREATE INDEX IF NOT EXISTS idx_reminder_records_status ON reminder_records(status);
        CREATE INDEX IF NOT EXISTS idx_reminder_records_priority ON reminder_records(priority);
        CREATE INDEX IF NOT EXISTS idx_reminder_records_trigger_time ON reminder_records(trigger_time);
        CREATE INDEX IF NOT EXISTS idx_reminder_records_assigned_to ON reminder_records(assigned_to);
        CREATE INDEX IF NOT EXISTS idx_reminder_records_due_time ON reminder_records(due_time);
        CREATE INDEX IF NOT EXISTS idx_reminder_records_created_at ON reminder_records(created_at);
        CREATE INDEX IF NOT EXISTS idx_reminder_records_entity ON reminder_records(related_entity_type, related_entity_id);

        -- reminder_rules 表索引
        CREATE INDEX IF NOT EXISTS idx_reminder_rules_rule_id ON reminder_rules(rule_id);
        CREATE INDEX IF NOT EXISTS idx_reminder_rules_priority ON reminder_rules(priority);
        CREATE INDEX IF NOT EXISTS idx_reminder_rules_is_active ON reminder_rules(is_active);

        -- notification_records 表索引
        CREATE INDEX IF NOT EXISTS idx_notification_records_reminder_id ON notification_records(reminder_id);
        CREATE INDEX IF NOT EXISTS idx_notification_records_type ON notification_records(notification_type);
        CREATE INDEX IF NOT EXISTS idx_notification_records_status ON notification_records(send_status);
        CREATE INDEX IF NOT EXISTS idx_notification_records_recipient ON notification_records(recipient);
        CREATE INDEX IF NOT EXISTS idx_notification_records_sent_time ON notification_records(sent_time);

        -- reminder_actions 表索引
        CREATE INDEX IF NOT EXISTS idx_reminder_actions_reminder_id ON reminder_actions(reminder_id);
        CREATE INDEX IF NOT EXISTS idx_reminder_actions_type ON reminder_actions(action_type);
        CREATE INDEX IF NOT EXISTS idx_reminder_actions_performed_by ON reminder_actions(performed_by);
        CREATE INDEX IF NOT EXISTS idx_reminder_actions_performed_time ON reminder_actions(performed_time);

        -- reminder_statistics 表索引
        CREATE INDEX IF NOT EXISTS idx_reminder_statistics_date ON reminder_statistics(stat_date);
        CREATE INDEX IF NOT EXISTS idx_reminder_statistics_rule_id ON reminder_statistics(rule_id);

        -- reminder_templates 表索引
        CREATE INDEX IF NOT EXISTS idx_reminder_templates_rule_id ON reminder_templates(rule_id);
        CREATE INDEX IF NOT EXISTS idx_reminder_templates_type ON reminder_templates(template_type);

        -- reminder_user_preferences 表索引
        CREATE INDEX IF NOT EXISTS idx_reminder_user_preferences_user_id ON reminder_user_preferences(user_id);
        CREATE INDEX IF NOT EXISTS idx_reminder_user_preferences_rule_id ON reminder_user_preferences(rule_id);

        -- reminder_system_config 表索引
        CREATE INDEX IF NOT EXISTS idx_reminder_system_config_key ON reminder_system_config(config_key);
        """

        self.execute_schema(index_sql)
        logger.info("Reminder indexes created successfully")

    def insert_default_data(self):
        """插入默认数据"""

        default_data_sql = """
        -- 插入默认提醒规则
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

        -- 插入默认系统配置
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
        ('sms.provider', 'aliyun', 'string', '短信服务商'),
        ('webhook.slack_enabled', 'false', 'boolean', 'Slack集成是否启用'),
        ('webhook.dingtalk_enabled', 'false', 'boolean', '钉钉集成是否启用');

        -- 插入默认通知模板
        INSERT OR REPLACE INTO reminder_templates (
            template_name, rule_id, template_type, subject_template, content_template, variables
        ) VALUES
        ('drawing_status_change_email', 'R001', 'email',
         '【重要】图纸状态变更通知 - {drawing_name}',
         '您好，\n\n图纸 {drawing_name} 状态已变更为 {new_status}。\n\n变更时间：{change_time}\n操作人：{operator}\n\n请及时查看并处理相关事宜。\n\n系统自动发送',
         '{"drawing_name": "string", "new_status": "string", "change_time": "datetime", "operator": "string"}'),

        ('quotation_timeout_email', 'R002', 'email',
         '【紧急】报价超时预警 - {customer_name}',
         '您好，\n\n客户 {customer_name} 的报价已超时 {timeout_hours} 小时。\n\n询价单号：{inquiry_id}\n超时时间：{timeout_time}\n\n请立即处理并升级给主管。\n\n系统自动发送',
         '{"customer_name": "string", "timeout_hours": "integer", "inquiry_id": "string", "timeout_time": "datetime"}'),

        ('quality_alert_sms', 'R003', 'sms',
         '质量异常报警',
         '质量异常：{product_name} 质检评分 {score}，低于阈值。请立即处理！',
         '{"product_name": "string", "score": "integer"}');

        -- 插入默认用户偏好设置
        INSERT OR REPLACE INTO reminder_user_preferences (
            user_id, rule_id, notification_methods, max_daily_notifications,
            escalation_enabled, weekend_notifications
        ) VALUES
        ('admin@company.com', NULL, 'email,system', 100, TRUE, FALSE),
        ('manager@company.com', NULL, 'email,sms,system', 200, TRUE, TRUE),
        ('engineering@company.com', 'R001', 'email,system', 50, TRUE, FALSE),
        ('quality@company.com', 'R003', 'email,sms,system', 100, TRUE, FALSE),
        ('sales@company.com', 'R008', 'system', 30, FALSE, FALSE);
        """

        self.execute_schema(default_data_sql)
        logger.info("Default reminder data inserted successfully")

    def create_triggers(self):
        """创建数据库触发器"""

        trigger_sql = """
        -- 更新时间触发器
        CREATE TRIGGER IF NOT EXISTS update_reminder_records_updated_at
            AFTER UPDATE ON reminder_records
            FOR EACH ROW
            BEGIN
                UPDATE reminder_records SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;

        CREATE TRIGGER IF NOT EXISTS update_reminder_rules_updated_at
            AFTER UPDATE ON reminder_rules
            FOR EACH ROW
            BEGIN
                UPDATE reminder_rules SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;

        CREATE TRIGGER IF NOT EXISTS update_reminder_templates_updated_at
            AFTER UPDATE ON reminder_templates
            FOR EACH ROW
            BEGIN
                UPDATE reminder_templates SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;

        CREATE TRIGGER IF NOT EXISTS update_reminder_user_preferences_updated_at
            AFTER UPDATE ON reminder_user_preferences
            FOR EACH ROW
            BEGIN
                UPDATE reminder_user_preferences SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;

        CREATE TRIGGER IF NOT EXISTS update_reminder_system_config_updated_at
            AFTER UPDATE ON reminder_system_config
            FOR EACH ROW
            BEGIN
                UPDATE reminder_system_config SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;

        -- 状态变更触发器
        CREATE TRIGGER IF NOT EXISTS reminder_status_change_log
            AFTER UPDATE OF status ON reminder_records
            WHEN OLD.status != NEW.status
            BEGIN
                INSERT INTO reminder_actions (reminder_id, action_type, action_description, performed_by, performed_time)
                VALUES (NEW.id, 'status_change',
                       CONCAT('状态从 ', OLD.status, ' 变更为 ', NEW.status),
                       'system', CURRENT_TIMESTAMP);
            END;
        """

        self.execute_schema(trigger_sql)
        logger.info("Reminder triggers created successfully")

    def initialize_reminder_system(self):
        """初始化完整的提醒系统数据库"""
        try:
            logger.info("Starting reminder system database initialization...")

            # 创建表
            self.create_reminder_tables()

            # 创建索引
            self.create_indexes()

            # 创建触发器
            self.create_triggers()

            # 插入默认数据
            self.insert_default_data()

            logger.info("✅ Reminder system database initialized successfully!")

        except Exception as e:
            logger.error(f"❌ Failed to initialize reminder system: {e}")
            raise

    def get_schema_info(self) -> Dict:
        """获取数据库架构信息"""
        try:
            if not self.conn:
                self.connect()

            cursor = self.conn.cursor()

            # 获取所有表
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name LIKE 'reminder_%'
                ORDER BY name
            """)
            tables = [row[0] for row in cursor.fetchall()]

            # 获取每个表的字段信息
            schema_info = {}
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                schema_info[table] = {
                    'columns': [{'name': col[1], 'type': col[2], 'nullable': not col[3], 'default': col[4]} for col in columns],
                    'column_count': len(columns)
                }

            # 获取索引信息
            cursor.execute("""
                SELECT name, tbl_name, sql FROM sqlite_master
                WHERE type='index' AND tbl_name LIKE 'reminder_%' AND sql IS NOT NULL
                ORDER BY tbl_name, name
            """)
            indexes = cursor.fetchall()

            schema_info['summary'] = {
                'total_tables': len(tables),
                'total_indexes': len(indexes),
                'tables': tables
            }

            return schema_info

        except Exception as e:
            logger.error(f"Failed to get schema info: {e}")
            return {}

def main():
    """主函数"""
    db_schema = ReminderDatabaseSchema()

    try:
        # 初始化提醒系统数据库
        db_schema.initialize_reminder_system()

        # 获取架构信息
        schema_info = db_schema.get_schema_info()

        print("\n" + "="*60)
        print("📊 REMINDER SYSTEM DATABASE SCHEMA SUMMARY")
        print("="*60)
        print(f"📋 Total Tables: {schema_info.get('summary', {}).get('total_tables', 0)}")
        print(f"🔑 Total Indexes: {schema_info.get('summary', {}).get('total_indexes', 0)}")
        print(f"📅 Database Path: {db_schema.db_path}")
        print(f"✅ Initialization Status: SUCCESS")

        print("\n📋 Tables Created:")
        for table in schema_info.get('summary', {}).get('tables', []):
            column_count = schema_info.get(table, {}).get('column_count', 0)
            print(f"   • {table} ({column_count} columns)")

        print("\n🚀 Reminder system is ready for automation!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.error(f"Database initialization failed: {e}")

    finally:
        db_schema.close()

if __name__ == "__main__":
    main()