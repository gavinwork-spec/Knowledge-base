#!/usr/bin/env python3
"""
提醒系统数据库表创建脚本
创建提醒规则、提醒记录、通知历史等相关表结构
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = "./data/db.sqlite"

def create_reminder_tables():
    """创建提醒系统相关表"""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        print("🔧 开始创建提醒系统数据库表...")

        # 1. 提醒规则表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reminder_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_id VARCHAR(20) UNIQUE NOT NULL,  -- 规则编号 R001-R017
                name VARCHAR(200) NOT NULL,          -- 规则名称
                description TEXT,                     -- 规则描述
                priority INTEGER DEFAULT 3,          -- 优先级 1-高 2-中 3-低
                category VARCHAR(50),                 -- 规则分类

                -- 触发条件配置 (JSON)
                trigger_config TEXT NOT NULL,         -- 触发条件JSON

                -- 调度配置 (JSON)
                schedule_config TEXT NOT NULL,        -- 调度配置JSON

                -- 通知配置 (JSON)
                notification_config TEXT,             -- 通知配置JSON

                -- 动作配置 (JSON)
                action_config TEXT,                   -- 动作配置JSON

                -- 状态字段
                is_active BOOLEAN DEFAULT TRUE,       -- 是否激活
                last_triggered TIMESTAMP,             -- 最后触发时间
                trigger_count INTEGER DEFAULT 0,      -- 触发次数

                -- 审计字段
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by VARCHAR(100),

                CHECK (priority IN (1, 2, 3)),
                CHECK (trigger_count >= 0)
            )
        """)

        # 2. 提醒记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reminder_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_id VARCHAR(20) NOT NULL,
                execution_id VARCHAR(50) UNIQUE NOT NULL,  -- 执行批次ID

                -- 触发信息
                triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                trigger_data TEXT,                         -- 触发时数据JSON
                trigger_reason TEXT,                        -- 触发原因

                -- 业务数据
                business_entity_type VARCHAR(50),           -- 业务实体类型
                business_entity_id INTEGER,                -- 业务实体ID
                business_data TEXT,                         -- 相关业务数据JSON

                -- 处理状态
                status VARCHAR(20) DEFAULT 'pending',      -- pending/processing/completed/failed
                processing_started_at TIMESTAMP,           -- 开始处理时间
                processing_completed_at TIMESTAMP,         -- 完成处理时间

                -- 结果信息
                result_data TEXT,                          -- 处理结果JSON
                error_message TEXT,                        -- 错误信息

                -- 执行统计
                execution_time_ms INTEGER,                 -- 执行耗时(毫秒)

                -- 通知状态
                notifications_sent BOOLEAN DEFAULT FALSE,  -- 是否已发送通知
                notification_count INTEGER DEFAULT 0,      -- 通知发送数量

                FOREIGN KEY (rule_id) REFERENCES reminder_rules(rule_id),
                CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
                CHECK (execution_time_ms >= 0),
                CHECK (notification_count >= 0)
            )
        """)

        # 3. 通知历史表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notification_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reminder_record_id INTEGER NOT NULL,

                -- 通知信息
                notification_type VARCHAR(50) NOT NULL,   -- email/sms/dashboard/system_message
                recipient VARCHAR(200) NOT NULL,          -- 接收者
                subject VARCHAR(500),                     -- 通知主题
                content TEXT NOT NULL,                    -- 通知内容

                -- 发送状态
                status VARCHAR(20) DEFAULT 'pending',     -- pending/sent/failed/retry
                sent_at TIMESTAMP,                         -- 发送时间
                delivery_status VARCHAR(20),               -- delivery_status

                -- 渠道信息
                channel_config TEXT,                      -- 渠道配置JSON
                external_id VARCHAR(100),                 -- 外部系统ID

                -- 反馈信息
                read_at TIMESTAMP,                         -- 阅读时间
                response_data TEXT,                       -- 响应数据JSON

                -- 重试信息
                retry_count INTEGER DEFAULT 0,           -- 重试次数
                next_retry_at TIMESTAMP,                  -- 下次重试时间

                FOREIGN KEY (reminder_record_id) REFERENCES reminder_records(id),
                CHECK (status IN ('pending', 'sent', 'failed', 'retry')),
                CHECK (retry_count >= 0)
            )
        """)

        # 4. 提醒模板表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reminder_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                template_id VARCHAR(50) UNIQUE NOT NULL,
                name VARCHAR(200) NOT NULL,
                description TEXT,

                -- 模板配置
                template_type VARCHAR(50) NOT NULL,       -- email/sms/dashboard
                language VARCHAR(10) DEFAULT 'zh_CN',    -- 语言
                subject_template TEXT,                    -- 主题模板
                content_template TEXT NOT NULL,           -- 内容模板

                -- 变量定义
                variables TEXT,                           -- 变量定义JSON

                -- 样式配置
                html_template TEXT,                       -- HTML模板
                css_style TEXT,                          -- CSS样式

                -- 状态
                is_active BOOLEAN DEFAULT TRUE,
                version INTEGER DEFAULT 1,

                -- 审计
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by VARCHAR(100),

                CHECK (template_type IN ('email', 'sms', 'dashboard', 'system_message')),
                CHECK (version > 0)
            )
        """)

        # 5. 提醒统计表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reminder_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stat_date DATE NOT NULL,
                rule_id VARCHAR(20),

                -- 触发统计
                trigger_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                failed_count INTEGER DEFAULT 0,

                -- 性能统计
                avg_execution_time_ms INTEGER DEFAULT 0,
                max_execution_time_ms INTEGER DEFAULT 0,
                min_execution_time_ms INTEGER DEFAULT 0,

                -- 通知统计
                notification_count INTEGER DEFAULT 0,
                delivered_count INTEGER DEFAULT 0,
                read_count INTEGER DEFAULT 0,

                -- 业务影响统计
                issues_resolved INTEGER DEFAULT 0,
                business_impact_score REAL DEFAULT 0.0,

                FOREIGN KEY (rule_id) REFERENCES reminder_rules(rule_id),
                UNIQUE(stat_date, rule_id),
                CHECK (trigger_count >= 0),
                CHECK (success_count >= 0),
                CHECK (failed_count >= 0),
                CHECK (avg_execution_time_ms >= 0),
                CHECK (business_impact_score >= 0.0)
            )
        """)

        # 6. 提醒配置表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reminder_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                setting_key VARCHAR(100) UNIQUE NOT NULL,
                setting_value TEXT,
                setting_type VARCHAR(20) DEFAULT 'string', -- string/integer/boolean/json
                description TEXT,

                -- 配置分类
                category VARCHAR(50),
                is_system BOOLEAN DEFAULT FALSE,
                is_encrypted BOOLEAN DEFAULT FALSE,

                -- 审计
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_by VARCHAR(100),

                CHECK (setting_type IN ('string', 'integer', 'boolean', 'json'))
            )
        """)

        print("✅ 创建表结构完成")

        # 创建索引
        create_reminder_indexes(cursor)

        # 初始化基础数据
        initialize_reminder_data(cursor)

        conn.commit()
        print("💾 提醒系统数据库表创建完成")

    except Exception as e:
        conn.rollback()
        print(f"❌ 创建表失败: {e}")
        raise
    finally:
        conn.close()

def create_reminder_indexes(cursor):
    """创建提醒系统索引"""

    print("🔍 创建提醒系统索引...")

    indexes = [
        # reminder_rules表索引
        "CREATE INDEX IF NOT EXISTS idx_reminder_rules_active ON reminder_rules(is_active)",
        "CREATE INDEX IF NOT EXISTS idx_reminder_rules_priority ON reminder_rules(priority)",
        "CREATE INDEX IF NOT EXISTS idx_reminder_rules_category ON reminder_rules(category)",
        "CREATE INDEX IF NOT EXISTS idx_reminder_rules_last_triggered ON reminder_rules(last_triggered)",

        # reminder_records表索引
        "CREATE INDEX IF NOT EXISTS idx_reminder_records_rule_id ON reminder_records(rule_id)",
        "CREATE INDEX IF NOT EXISTS idx_reminder_records_triggered_at ON reminder_records(triggered_at)",
        "CREATE INDEX IF NOT EXISTS idx_reminder_records_status ON reminder_records(status)",
        "CREATE INDEX IF NOT EXISTS idx_reminder_records_execution_id ON reminder_records(execution_id)",
        "CREATE INDEX IF NOT EXISTS idx_reminder_records_business_entity ON reminder_records(business_entity_type, business_entity_id)",

        # notification_history表索引
        "CREATE INDEX IF NOT EXISTS idx_notification_history_record_id ON notification_history(reminder_record_id)",
        "CREATE INDEX IF NOT EXISTS idx_notification_history_type ON notification_history(notification_type)",
        "CREATE INDEX IF NOT EXISTS idx_notification_history_status ON notification_history(status)",
        "CREATE INDEX IF NOT EXISTS idx_notification_history_sent_at ON notification_history(sent_at)",
        "CREATE INDEX IF NOT EXISTS idx_notification_history_recipient ON notification_history(recipient)",

        # reminder_statistics表索引
        "CREATE INDEX IF NOT EXISTS idx_reminder_statistics_date ON reminder_statistics(stat_date)",
        "CREATE INDEX IF NOT EXISTS idx_reminder_statistics_rule_id ON reminder_statistics(rule_id)",

        # reminder_templates表索引
        "CREATE INDEX IF NOT EXISTS idx_reminder_templates_type ON reminder_templates(template_type)",
        "CREATE INDEX IF NOT EXISTS idx_reminder_templates_active ON reminder_templates(is_active)",

        # reminder_settings表索引
        "CREATE INDEX IF NOT EXISTS idx_reminder_settings_category ON reminder_settings(category)",
        "CREATE INDEX IF NOT EXISTS idx_reminder_settings_key ON reminder_settings(setting_key)"
    ]

    for index_sql in indexes:
        cursor.execute(index_sql)

    print(f"✅ 创建了 {len(indexes)} 个索引")

def initialize_reminder_data(cursor):
    """初始化提醒系统基础数据"""

    print("📝 初始化提醒系统基础数据...")

    # 初始化17个核心提醒规则
    rules = [
        {
            'rule_id': 'R001',
            'name': '数据质量下降预警',
            'description': '当数据质量评分低于80分时发送预警',
            'priority': 1,
            'category': 'data_quality',
            'trigger_config': json.dumps({
                'condition': 'data_quality_score < 80',
                'metric': 'quality_score',
                'threshold': 80,
                'operator': 'less_than'
            }),
            'schedule_config': json.dumps({
                'frequency': 'daily',
                'time': '09:00'
            }),
            'notification_config': json.dumps([
                {'type': 'email', 'recipients': ['admin@company.com'], 'template': 'data_quality_alert'},
                {'type': 'dashboard', 'level': 'warning'}
            ]),
            'action_config': json.dumps([
                {'type': 'log', 'message': '数据质量评分下降至 {score} 分'},
                {'type': 'create_task', 'assignee': 'data_team', 'priority': 'high'}
            ])
        },
        {
            'rule_id': 'R002',
            'name': '图纸分类覆盖率提醒',
            'description': '提醒需要人工分类的图纸积压',
            'priority': 2,
            'category': 'data_processing',
            'trigger_config': json.dumps({
                'condition': 'unclassified_drawings > 50',
                'metric': 'unclassified_count',
                'threshold': 50,
                'operator': 'greater_than'
            }),
            'schedule_config': json.dumps({
                'frequency': 'daily',
                'time': '10:00'
            }),
            'notification_config': json.dumps([
                {'type': 'dashboard', 'level': 'info'}
            ]),
            'action_config': json.dumps([
                {'type': 'log', 'message': '未分类图纸积压 {count} 个'}
            ])
        },
        {
            'rule_id': 'R003',
            'name': '新客户跟进提醒',
            'description': '提醒销售团队跟进新注册客户',
            'priority': 1,
            'category': 'customer_management',
            'trigger_config': json.dumps({
                'condition': 'customer_created_days_ago = 3',
                'metric': 'customer_age_days',
                'threshold': 3,
                'operator': 'equals'
            }),
            'schedule_config': json.dumps({
                'frequency': 'daily',
                'time': '09:30'
            }),
            'notification_config': json.dumps([
                {'type': 'email', 'recipients': ['sales@company.com'], 'template': 'new_customer_followup'},
                {'type': 'system_message', 'level': 'info'}
            ]),
            'action_config': json.dumps([
                {'type': 'create_task', 'assignee': 'sales_team', 'priority': 'high'}
            ])
        },
        {
            'rule_id': 'R004',
            'name': '报价异常监控',
            'description': '监控工厂报价异常波动',
            'priority': 1,
            'category': 'price_monitoring',
            'trigger_config': json.dumps({
                'condition': 'price_change_percentage > 30',
                'metric': 'price_volatility',
                'threshold': 30,
                'operator': 'greater_than'
            }),
            'schedule_config': json.dumps({
                'frequency': 'realtime'
            }),
            'notification_config': json.dumps([
                {'type': 'email', 'recipients': ['procurement@company.com'], 'template': 'price_anomaly_alert'},
                {'type': 'instant_notification', 'channels': ['slack', 'wechat']}
            ]),
            'action_config': json.dumps([
                {'type': 'log', 'message': '发现价格异常：{product_category} 价格波动 {change}%'},
                {'type': 'escalate', 'condition': 'price_change_percentage > 50', 'to': 'senior_management'}
            ])
        },
        {
            'rule_id': 'R005',
            'name': '月度业绩报告',
            'description': '自动生成上月业绩分析报告',
            'priority': 2,
            'category': 'reporting',
            'trigger_config': json.dumps({
                'condition': 'scheduled_task',
                'schedule': 'monthly'
            }),
            'schedule_config': json.dumps({
                'frequency': 'monthly',
                'day': 1,
                'time': '09:00'
            }),
            'notification_config': json.dumps([
                {'type': 'email', 'recipients': ['management@company.com'], 'template': 'monthly_performance_report'}
            ]),
            'action_config': json.dumps([
                {'type': 'generate_report', 'report_type': 'monthly_performance', 'format': 'pdf'},
                {'type': 'send_email', 'subject': '{month} 月度业绩报告'}
            ])
        }
    ]

    # 插入规则数据
    for rule in rules:
        cursor.execute("""
            INSERT OR REPLACE INTO reminder_rules
            (rule_id, name, description, priority, category, trigger_config,
             schedule_config, notification_config, action_config, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rule['rule_id'], rule['name'], rule['description'],
            rule['priority'], rule['category'], rule['trigger_config'],
            rule['schedule_config'], rule['notification_config'],
            rule['action_config'], True
        ))

    # 初始化通知模板
    templates = [
        {
            'template_id': 'data_quality_alert',
            'name': '数据质量预警模板',
            'description': '数据质量下降预警通知模板',
            'template_type': 'email',
            'subject_template': '【预警】数据质量评分下降',
            'content_template': '''数据质量评分已下降至 {score} 分，低于阈值 {threshold} 分。

问题详情：
- 客户数据完整性：{customer_completeness}%
- 图纸分类覆盖率：{classification_coverage}%
- 工厂数据准确性：{factory_accuracy}%

建议措施：
1. 检查数据导入流程
2. 清理重复和无效数据
3. 完善数据验证规则

查看详情：{dashboard_link}''',
            'variables': json.dumps(['score', 'threshold', 'customer_completeness', 'classification_coverage', 'factory_accuracy', 'dashboard_link'])
        },
        {
            'template_id': 'price_anomaly_alert',
            'name': '价格异常预警模板',
            'description': '价格异常波动预警通知模板',
            'template_type': 'email',
            'subject_template': '【紧急】价格异常预警',
            'content_template': '''检测到 {product_category} 价格异常波动：
- 工厂：{factory_name}
- 当前价格：{current_price}
- 预期范围：{expected_range}
- 波动幅度：{price_change}%

需要立即关注并评估影响。

查看详情：{dashboard_link}''',
            'variables': json.dumps(['product_category', 'factory_name', 'current_price', 'expected_range', 'price_change', 'dashboard_link'])
        }
    ]

    # 插入模板数据
    for template in templates:
        cursor.execute("""
            INSERT OR REPLACE INTO reminder_templates
            (template_id, name, description, template_type, subject_template,
             content_template, variables, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            template['template_id'], template['name'], template['description'],
            template['template_type'], template['subject_template'],
            template['content_template'], template['variables'], True
        ))

    # 初始化系统配置
    settings = [
        ('reminder_enabled', 'true', 'boolean', '是否启用提醒系统', 'system', True, False),
        ('max_retry_count', '3', 'integer', '最大重试次数', 'notification', False, False),
        ('notification_batch_size', '50', 'integer', '通知批量发送大小', 'notification', False, False),
        ('default_timezone', 'Asia/Shanghai', 'string', '默认时区', 'system', False, False),
        ('email_from_address', 'noreply@company.com', 'string', '邮件发送地址', 'email', False, False),
        ('dashboard_base_url', 'http://localhost:3000', 'string', '仪表板基础URL', 'system', False, False)
    ]

    # 插入配置数据
    for setting in settings:
        cursor.execute("""
            INSERT OR REPLACE INTO reminder_settings
            (setting_key, setting_value, setting_type, description,
             category, is_system, is_encrypted)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, setting)

    print(f"✅ 初始化了 {len(rules)} 个提醒规则")
    print(f"✅ 初始化了 {len(templates)} 个通知模板")
    print(f"✅ 初始化了 {len(settings)} 个系统配置")

if __name__ == "__main__":
    print("🚀 开始创建提醒系统数据库表...")
    create_reminder_tables()
    print("🎉 提醒系统数据库表创建完成！")