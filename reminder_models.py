#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提醒系统数据库模型
创建支持17条提醒规则的完整数据库表结构
"""

import sqlite3
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/reminder_models.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ReminderDatabaseModel:
    """提醒系统数据库模型类"""

    def __init__(self, db_path: str = 'knowledge_base.db'):
        self.db_path = db_path
        self.conn = None
        self._initialize_connection()

    def _initialize_connection(self):
        """初始化数据库连接"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # 启用字典式访问
            logger.info(f"数据库连接已建立: {self.db_path}")
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise

    def create_reminder_tables(self) -> bool:
        """创建提醒系统相关的所有表"""
        try:
            cursor = self.conn.cursor()

            # 1. 提醒记录主表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reminder_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id VARCHAR(10) NOT NULL,
                    rule_name VARCHAR(100) NOT NULL,
                    business_entity_type VARCHAR(50) NOT NULL,  -- 实体类型：quote, drawing, customer, etc.
                    business_entity_id INTEGER NOT NULL,      -- 实体ID
                    trigger_time DATETIME NOT NULL,
                    trigger_condition TEXT NOT NULL,
                    priority VARCHAR(10) NOT NULL CHECK (priority IN ('高', '中', '低')),
                    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'cancelled', 'escalated')),
                    assigned_to VARCHAR(50),
                    due_time DATETIME,
                    completed_time DATETIME,
                    notification_methods VARCHAR(100),  -- JSON格式存储通知方式
                    auto_processed BOOLEAN DEFAULT FALSE,
                    processing_result TEXT,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    metadata TEXT,  -- JSON格式存储额外数据
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 2. 提醒规则配置表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reminder_rules (
                    id VARCHAR(10) PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    description TEXT,
                    priority VARCHAR(10) NOT NULL CHECK (priority IN ('高', '中', '低')),
                    check_frequency VARCHAR(20) NOT NULL,  -- 实时,每小时,每天,每周,每月
                    notification_methods TEXT NOT NULL,  -- JSON格式
                    processing_type VARCHAR(20) NOT NULL CHECK (processing_type IN ('完全自动化', '半自动化', '手动')),
                    trigger_conditions TEXT NOT NULL,  -- SQL查询条件
                    auto_action TEXT,  -- 自动执行的动作
                    escalation_rules TEXT,  -- 升级规则JSON
                    enabled BOOLEAN DEFAULT TRUE,
                    config_parameters TEXT,  -- 配置参数JSON
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 3. 提醒通知记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reminder_notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reminder_id INTEGER NOT NULL,
                    notification_type VARCHAR(20) NOT NULL CHECK (notification_type IN ('email', 'sms', 'system', 'webhook')),
                    recipient VARCHAR(100) NOT NULL,
                    subject VARCHAR(200),
                    content TEXT NOT NULL,
                    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'failed', 'cancelled')),
                    sent_time DATETIME,
                    error_message TEXT,
                    external_id VARCHAR(50),  -- 外部系统返回的ID
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (reminder_id) REFERENCES reminder_records (id) ON DELETE CASCADE
                )
            ''')

            # 4. 提醒处理历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reminder_processing_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reminder_id INTEGER NOT NULL,
                    action VARCHAR(50) NOT NULL,
                    processor VARCHAR(50),
                    processing_time DATETIME NOT NULL,
                    result VARCHAR(20) NOT NULL CHECK (result IN ('success', 'failed', 'partial')),
                    details TEXT,
                    next_action_time DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (reminder_id) REFERENCES reminder_records (id) ON DELETE CASCADE
                )
            ''')

            # 5. 提醒统计表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reminder_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    rule_id VARCHAR(10) NOT NULL,
                    total_triggered INTEGER DEFAULT 0,
                    total_processed INTEGER DEFAULT 0,
                    total_completed INTEGER DEFAULT 0,
                    total_failed INTEGER DEFAULT 0,
                    avg_processing_time REAL DEFAULT 0,  -- 平均处理时间（分钟）
                    success_rate REAL DEFAULT 0,  -- 成功率
                    escalation_count INTEGER DEFAULT 0,
                    auto_processed_count INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (rule_id) REFERENCES reminder_rules (id) ON DELETE CASCADE,
                    UNIQUE(date, rule_id)
                )
            ''')

            # 6. 质量检验表（支持R003质量异常报警）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS quality_inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    drawing_id INTEGER NOT NULL,
                    product_id INTEGER,
                    inspection_date DATETIME NOT NULL,
                    score REAL NOT NULL CHECK (score >= 0 AND score <= 100),
                    result VARCHAR(20) NOT NULL CHECK (result IN ('PASS', 'FAIL', 'REWORK')),
                    inspector VARCHAR(50),
                    inspection_type VARCHAR(30),
                    defect_details TEXT,
                    corrective_actions TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (drawing_id) REFERENCES drawings (id) ON DELETE CASCADE
                )
            ''')

            # 7. 生产订单表（支持R004交付超期预警）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS production_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_number VARCHAR(50) UNIQUE NOT NULL,
                    customer_id INTEGER NOT NULL,
                    product_id INTEGER,
                    quantity INTEGER NOT NULL,
                    expected_delivery_date DATETIME NOT NULL,
                    actual_delivery_date DATETIME,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'confirmed', 'in_production', 'ready', 'shipped', 'delivered', 'cancelled')),
                    progress_percentage REAL DEFAULT 0 CHECK (progress_percentage >= 0 AND progress_percentage <= 100),
                    planned_progress REAL DEFAULT 0,
                    urgent_flag BOOLEAN DEFAULT FALSE,
                    total_amount DECIMAL(15,2),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (customer_id) REFERENCES customers (id) ON DELETE CASCADE
                )
            ''')

            # 8. 生产排程表（支持R010生产进度更新提醒）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS production_schedule (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id INTEGER NOT NULL,
                    process_name VARCHAR(50) NOT NULL,
                    process_sequence INTEGER NOT NULL,
                    planned_completion DATETIME,
                    actual_completion DATETIME,
                    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'delayed')),
                    is_critical BOOLEAN DEFAULT FALSE,
                    assigned_to VARCHAR(50),
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (order_id) REFERENCES production_orders (id) ON DELETE CASCADE
                )
            ''')

            # 9. 客户投诉表（支持R005客户投诉提醒）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS complaints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id INTEGER NOT NULL,
                    complaint_date DATETIME NOT NULL,
                    type VARCHAR(30) NOT NULL CHECK (type IN ('QUALITY', 'DELIVERY', 'SERVICE', 'PRICE', 'OTHER')),
                    description TEXT NOT NULL,
                    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'investigating', 'resolved', 'escalated', 'cancelled')),
                    priority VARCHAR(10) DEFAULT '中' CHECK (priority IN ('高', '中', '低')),
                    assigned_to VARCHAR(50),
                    resolution_deadline DATETIME,
                    resolution_details TEXT,
                    escalation_level INTEGER DEFAULT 1,
                    escalation_date DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (customer_id) REFERENCES customers (id) ON DELETE CASCADE
                )
            ''')

            # 10. 技术参数表（支持R007技术参数更新提醒）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS technical_parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id INTEGER NOT NULL,
                    parameter_name VARCHAR(100) NOT NULL,
                    parameter_value TEXT NOT NULL,
                    parameter_type VARCHAR(30),
                    unit VARCHAR(20),
                    min_value REAL,
                    max_value REAL,
                    tolerance REAL,
                    updated_by VARCHAR(50),
                    version INTEGER DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (product_id) REFERENCES products (id) ON DELETE CASCADE
                )
            ''')

            # 11. 生产批次表（支持R006批次生产计划提醒）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS production_batches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_number VARCHAR(50) UNIQUE NOT NULL,
                    product_id INTEGER NOT NULL,
                    order_id INTEGER,
                    quantity INTEGER NOT NULL,
                    status VARCHAR(20) DEFAULT 'planned' CHECK (status IN ('planned', 'in_progress', 'completed', 'cancelled')),
                    planned_start_date DATETIME,
                    actual_start_date DATETIME,
                    planned_completion_date DATETIME,
                    actual_completion_date DATETIME,
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (product_id) REFERENCES products (id) ON DELETE CASCADE,
                    FOREIGN KEY (order_id) REFERENCES production_orders (id) ON DELETE SET NULL
                )
            ''')

            self.conn.commit()
            logger.info("提醒系统数据库表创建成功")

            # 创建索引
            self._create_indexes()

            # 插入默认规则配置
            self._insert_default_rules()

            return True

        except Exception as e:
            logger.error(f"创建提醒系统表失败: {e}")
            self.conn.rollback()
            return False

    def _create_indexes(self):
        """创建数据库索引以优化查询性能"""
        try:
            cursor = self.conn.cursor()

            # reminder_records表索引
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_reminder_rule_id ON reminder_records(rule_id)",
                "CREATE INDEX IF NOT EXISTS idx_reminder_status ON reminder_records(status)",
                "CREATE INDEX IF NOT EXISTS idx_reminder_priority ON reminder_records(priority)",
                "CREATE INDEX IF NOT EXISTS idx_reminder_trigger_time ON reminder_records(trigger_time)",
                "CREATE INDEX IF NOT EXISTS idx_reminder_assigned_to ON reminder_records(assigned_to)",
                "CREATE INDEX IF NOT EXISTS idx_reminder_business_entity ON reminder_records(business_entity_type, business_entity_id)",
                "CREATE INDEX IF NOT EXISTS idx_reminder_due_time ON reminder_records(due_time)",
                "CREATE INDEX IF NOT EXISTS idx_reminder_created_at ON reminder_records(created_at)",

                # reminder_notifications表索引
                "CREATE INDEX IF NOT EXISTS idx_notification_reminder_id ON reminder_notifications(reminder_id)",
                "CREATE INDEX IF NOT EXISTS idx_notification_status ON reminder_notifications(status)",
                "CREATE INDEX IF NOT EXISTS idx_notification_type ON reminder_notifications(notification_type)",
                "CREATE INDEX IF NOT EXISTS idx_notification_created_at ON reminder_notifications(created_at)",

                # reminder_statistics表索引
                "CREATE INDEX IF NOT EXISTS idx_stats_date_rule ON reminder_statistics(date, rule_id)",
                "CREATE INDEX IF NOT EXISTS idx_stats_date ON reminder_statistics(date)",

                # 业务表索引
                "CREATE INDEX IF NOT EXISTS idx_quality_inspections_drawing ON quality_inspections(drawing_id)",
                "CREATE INDEX IF NOT EXISTS idx_quality_inspections_date ON quality_inspections(inspection_date)",
                "CREATE INDEX IF NOT EXISTS idx_quality_inspections_score ON quality_inspections(score)",
                "CREATE INDEX IF NOT EXISTS idx_production_orders_customer ON production_orders(customer_id)",
                "CREATE INDEX IF NOT EXISTS idx_production_orders_status ON production_orders(status)",
                "CREATE INDEX IF NOT EXISTS idx_production_orders_delivery ON production_orders(expected_delivery_date)",
                "CREATE INDEX IF NOT EXISTS idx_production_schedule_order ON production_schedule(order_id)",
                "CREATE INDEX IF NOT EXISTS idx_production_schedule_status ON production_schedule(status)",
                "CREATE INDEX IF NOT EXISTS idx_complaints_customer ON complaints(customer_id)",
                "CREATE INDEX IF NOT EXISTS idx_complaints_status ON complaints(status)",
                "CREATE INDEX IF NOT EXISTS idx_complaints_type ON complaints(type)",
                "CREATE INDEX IF NOT EXISTS idx_technical_params_product ON technical_parameters(product_id)",
                "CREATE INDEX IF NOT EXISTS idx_production_batches_product ON production_batches(product_id)",
                "CREATE INDEX IF NOT EXISTS idx_production_batches_status ON production_batches(status)"
            ]

            for index_sql in indexes:
                cursor.execute(index_sql)

            self.conn.commit()
            logger.info("提醒系统数据库索引创建成功")

        except Exception as e:
            logger.error(f"创建数据库索引失败: {e}")
            raise

    def _insert_default_rules(self):
        """插入默认的提醒规则配置"""
        try:
            cursor = self.conn.cursor()

            # 17条默认规则配置
            default_rules = [
                {
                    'id': 'R001',
                    'name': '客户图纸状态变更',
                    'description': '新图纸上传、状态变更时立即通知相关人员',
                    'priority': '高',
                    'check_frequency': '每小时',
                    'notification_methods': json.dumps(['email', 'system']),
                    'processing_type': '完全自动化',
                    'trigger_conditions': '新图纸上传、图纸审核状态变更、图纸版本更新',
                    'auto_action': '自动标记和分类',
                    'escalation_rules': json.dumps({'enabled': True, 'delay_hours': 24, 'escalate_to': 'manager'})
                },
                {
                    'id': 'R002',
                    'name': '报价超时预警',
                    'description': '报价时间超过设定阈值时预警',
                    'priority': '高',
                    'check_frequency': '每15分钟',
                    'notification_methods': json.dumps(['email', 'sms']),
                    'processing_type': '半自动化',
                    'trigger_conditions': '报价状态为处理中且超过设定时间阈值',
                    'auto_action': '升级到主管',
                    'escalation_rules': json.dumps({'enabled': True, 'normal_timeout': 24, 'urgent_timeout': 12})
                },
                {
                    'id': 'R003',
                    'name': '质量异常报警',
                    'description': '质量评分低于阈值时实时报警',
                    'priority': '高',
                    'check_frequency': '实时',
                    'notification_methods': json.dumps(['email', 'sms', 'system']),
                    'processing_type': '完全自动化',
                    'trigger_conditions': '质量评分低于70分或连续3次不合格',
                    'auto_action': '立即处理和质量分析',
                    'escalation_rules': json.dumps({'enabled': True, 'threshold': 70, 'consecutive_failures': 3})
                },
                {
                    'id': 'R004',
                    'name': '交付超期预警',
                    'description': '预计交付日期临近时预警',
                    'priority': '高',
                    'check_frequency': '每天上午9点',
                    'notification_methods': json.dumps(['email', 'system']),
                    'processing_type': '半自动化',
                    'trigger_conditions': '预计交付日期提前7天或进度落后超过20%',
                    'auto_action': '协调生产计划',
                    'escalation_rules': json.dumps({'enabled': True, 'advance_days': 7, 'delay_threshold': 20})
                },
                {
                    'id': 'R005',
                    'name': '客户投诉提醒',
                    'description': '新投诉记录或状态更新时提醒',
                    'priority': '高',
                    'check_frequency': '实时',
                    'notification_methods': json.dumps(['email', 'system']),
                    'processing_type': '完全自动化',
                    'trigger_conditions': '新投诉创建、状态更新、升级处理',
                    'auto_action': '转交客服团队',
                    'escalation_rules': json.dumps({'enabled': True, 'response_timeout': 2})
                },
                {
                    'id': 'R006',
                    'name': '批次生产计划提醒',
                    'description': '新批次创建或计划变更时提醒',
                    'priority': '中',
                    'check_frequency': '每小时',
                    'notification_methods': json.dumps(['email', 'system']),
                    'processing_type': '半自动化',
                    'trigger_conditions': '新批次创建或计划状态变更',
                    'auto_action': '更新生产排期',
                    'escalation_rules': json.dumps({'enabled': False})
                },
                {
                    'id': 'R007',
                    'name': '技术参数更新提醒',
                    'description': '技术参数文件更新时提醒',
                    'priority': '中',
                    'check_frequency': '每天上午10点',
                    'notification_methods': json.dumps(['system']),
                    'processing_type': '完全自动化',
                    'trigger_conditions': '技术参数文件更新',
                    'auto_action': '自动分发通知',
                    'escalation_rules': json.dumps({'enabled': False})
                },
                {
                    'id': 'R008',
                    'name': '客户跟进提醒',
                    'description': '超过7天未跟进的客户提醒',
                    'priority': '中',
                    'check_frequency': '每天上午11点',
                    'notification_methods': json.dumps(['system']),
                    'processing_type': '半自动化',
                    'trigger_conditions': '客户最后联系时间超过7天',
                    'auto_action': '分配给销售人员',
                    'escalation_rules': json.dumps({'enabled': False})
                },
                {
                    'id': 'R009',
                    'name': '报价分析报告提醒',
                    'description': '报价数据更新完成时提醒',
                    'priority': '中',
                    'check_frequency': '每天下午2点',
                    'notification_methods': json.dumps(['email']),
                    'processing_type': '完全自动化',
                    'trigger_conditions': '报价数据更新完成',
                    'auto_action': '自动生成分析报告',
                    'escalation_rules': json.dumps({'enabled': False})
                },
                {
                    'id': 'R010',
                    'name': '生产进度更新提醒',
                    'description': '生产状态变更时提醒',
                    'priority': '中',
                    'check_frequency': '每30分钟',
                    'notification_methods': json.dumps(['system']),
                    'processing_type': '完全自动化',
                    'trigger_conditions': '生产状态变更',
                    'auto_action': '自动更新进度',
                    'escalation_rules': json.dumps({'enabled': False})
                },
                {
                    'id': 'R011',
                    'name': '数据备份提醒',
                    'description': '数据备份完成或失败时提醒',
                    'priority': '低',
                    'check_frequency': '每天晚上8点',
                    'notification_methods': json.dumps(['system']),
                    'processing_type': '完全自动化',
                    'trigger_conditions': '数据备份任务执行',
                    'auto_action': '记录备份日志',
                    'escalation_rules': json.dumps({'enabled': False})
                },
                {
                    'id': 'R012',
                    'name': '月度统计报告提醒',
                    'description': '月度统计完成时提醒',
                    'priority': '低',
                    'check_frequency': '每月1号上午9点',
                    'notification_methods': json.dumps(['email']),
                    'processing_type': '完全自动化',
                    'trigger_conditions': '月度统计数据生成完成',
                    'auto_action': '自动发送报告',
                    'escalation_rules': json.dumps({'enabled': False})
                },
                {
                    'id': 'R013',
                    'name': '员工生日提醒',
                    'description': '员工生日当天提醒',
                    'priority': '低',
                    'check_frequency': '每天上午8点',
                    'notification_methods': json.dumps(['system']),
                    'processing_type': '完全自动化',
                    'trigger_conditions': '员工生日当天',
                    'auto_action': '自动发送祝福',
                    'escalation_rules': json.dumps({'enabled': False})
                },
                {
                    'id': 'R014',
                    'name': '合同到期提醒',
                    'description': '合同到期前30天提醒',
                    'priority': '低',
                    'check_frequency': '每周一上午10点',
                    'notification_methods': json.dumps(['email']),
                    'processing_type': '完全自动化',
                    'trigger_conditions': '合同到期前30天',
                    'auto_action': '通知法务部门',
                    'escalation_rules': json.dumps({'enabled': False})
                },
                {
                    'id': 'R015',
                    'name': '库存预警提醒',
                    'description': '库存低于安全库存时提醒',
                    'priority': '低',
                    'check_frequency': '每天下午3点',
                    'notification_methods': json.dumps(['system']),
                    'processing_type': '完全自动化',
                    'trigger_conditions': '库存数量低于安全库存阈值',
                    'auto_action': '建议补货计划',
                    'escalation_rules': json.dumps({'enabled': False})
                },
                {
                    'id': 'R016',
                    'name': '供应商评估提醒',
                    'description': '供应商评估完成时提醒',
                    'priority': '低',
                    'check_frequency': '每季度最后一天',
                    'notification_methods': json.dumps(['email']),
                    'processing_type': '完全自动化',
                    'trigger_conditions': '供应商评估周期结束',
                    'auto_action': '更新评估结果',
                    'escalation_rules': json.dumps({'enabled': False})
                },
                {
                    'id': 'R017',
                    'name': '设备维护提醒',
                    'description': '设备维护计划到期时提醒',
                    'priority': '低',
                    'check_frequency': '每周日上午9点',
                    'notification_methods': json.dumps(['email', 'system']),
                    'processing_type': '半自动化',
                    'trigger_conditions': '设备维护计划到期',
                    'auto_action': '安排维护计划',
                    'escalation_rules': json.dumps({'enabled': True, 'delay_days': 7})
                }
            ]

            for rule in default_rules:
                cursor.execute('''
                    INSERT OR REPLACE INTO reminder_rules
                    (id, name, description, priority, check_frequency, notification_methods,
                     processing_type, trigger_conditions, auto_action, escalation_rules, enabled)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    rule['id'], rule['name'], rule['description'], rule['priority'],
                    rule['check_frequency'], rule['notification_methods'], rule['processing_type'],
                    rule['trigger_conditions'], rule['auto_action'], rule['escalation_rules'], True
                ))

            self.conn.commit()
            logger.info("默认提醒规则配置插入成功")

        except Exception as e:
            logger.error(f"插入默认规则配置失败: {e}")
            self.conn.rollback()
            raise

    def get_database_schema(self) -> Dict:
        """获取数据库架构信息"""
        try:
            cursor = self.conn.cursor()

            # 获取所有表名
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%reminder%' OR name IN ('quality_inspections', 'production_orders', 'production_schedule', 'complaints', 'technical_parameters', 'production_batches')")
            tables = [row[0] for row in cursor.fetchall()]

            schema = {'tables': {}, 'indexes': []}

            for table_name in tables:
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()

                schema['tables'][table_name] = {
                    'columns': [
                        {
                            'name': col[1],
                            'type': col[2],
                            'not_null': bool(col[3]),
                            'default_value': col[4],
                            'primary_key': bool(col[5])
                        }
                        for col in columns
                    ]
                }

            # 获取索引信息
            cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='index' AND name LIKE '%reminder%' OR name LIKE '%idx_%'")
            indexes = cursor.fetchall()

            schema['indexes'] = [
                {'name': idx[0], 'sql': idx[1]}
                for idx in indexes if idx[1]  # 排除自动创建的主键索引
            ]

            return schema

        except Exception as e:
            logger.error(f"获取数据库架构失败: {e}")
            return {}

    def validate_database_structure(self) -> Dict:
        """验证数据库结构的完整性"""
        try:
            expected_tables = [
                'reminder_records', 'reminder_rules', 'reminder_notifications',
                'reminder_processing_history', 'reminder_statistics',
                'quality_inspections', 'production_orders', 'production_schedule',
                'complaints', 'technical_parameters', 'production_batches'
            ]

            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = [row[0] for row in cursor.fetchall()]

            validation_result = {
                'status': 'success',
                'missing_tables': [],
                'extra_tables': [],
                'table_count': len(existing_tables),
                'expected_count': len(expected_tables),
                'details': {}
            }

            # 检查缺失的表
            for table in expected_tables:
                if table not in existing_tables:
                    validation_result['missing_tables'].append(table)
                    validation_result['status'] = 'error'

            # 检查每个表的记录数
            for table in existing_tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    validation_result['details'][table] = {'record_count': count}
                except Exception as e:
                    validation_result['details'][table] = {'error': str(e)}
                    validation_result['status'] = 'warning'

            # 检查规则配置
            if 'reminder_rules' in existing_tables:
                cursor.execute("SELECT COUNT(*) FROM reminder_rules WHERE enabled = TRUE")
                enabled_rules = cursor.fetchone()[0]
                validation_result['details']['enabled_rules_count'] = enabled_rules

                if enabled_rules < 17:
                    validation_result['status'] = 'warning'
                    validation_result['missing_rules'] = 17 - enabled_rules

            return validation_result

        except Exception as e:
            logger.error(f"验证数据库结构失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logger.info("数据库连接已关闭")

def main():
    """主函数"""
    try:
        # 创建数据库模型实例
        db_model = ReminderDatabaseModel()

        # 创建提醒系统表
        print("🔧 正在创建提醒系统数据库表...")
        success = db_model.create_reminder_tables()

        if success:
            print("✅ 提醒系统数据库表创建成功")

            # 获取数据库架构
            schema = db_model.get_database_schema()
            print(f"📊 数据库包含 {len(schema['tables'])} 个表")

            # 验证数据库结构
            validation = db_model.validate_database_structure()
            print(f"🔍 数据库验证状态: {validation['status']}")

            if validation['status'] == 'success':
                print("✅ 数据库结构验证通过")
                print(f"📋 启用的规则数量: {validation['details'].get('enabled_rules_count', 0)}")
            else:
                print("⚠️ 数据库结构验证发现问题")
                if validation.get('missing_tables'):
                    print(f"❌ 缺失表: {validation['missing_tables']}")

            # 生成架构报告
            report = {
                'creation_time': datetime.now().isoformat(),
                'database_schema': schema,
                'validation_result': validation,
                'rules_configured': 17,
                'tables_created': len(schema['tables']),
                'indexes_created': len(schema['indexes'])
            }

            # 保存架构报告
            with open('logs/reminder_database_schema.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            print("📄 数据库架构报告已保存到 logs/reminder_database_schema.json")

        else:
            print("❌ 提醒系统数据库表创建失败")

        db_model.close()

    except Exception as e:
        logger.error(f"主程序执行失败: {e}")
        print(f"❌ 执行失败: {e}")

if __name__ == "__main__":
    main()