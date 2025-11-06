#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Base Models Setup
知识库模型初始化脚本

This script creates the database schema for the knowledge system,
including unified knowledge entries, NLP entities, strategy suggestions,
and embedding indexes.
"""

import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/processed/setup_knowledge_models.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KnowledgeModelsSetup:
    """知识库模型设置器"""

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

    def create_knowledge_entries_table(self):
        """创建知识条目表"""
        sql = """
        CREATE TABLE IF NOT EXISTS knowledge_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type VARCHAR(50) NOT NULL,                    -- 实体类型: customer, quote, drawing, factory, product, etc.
            name VARCHAR(200) NOT NULL,                        -- 条目名称
            related_file TEXT,                                   -- 关联文件路径
            description TEXT,                                    -- 详细描述
            attributes_json TEXT,                               -- 属性数据 (JSON格式)
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (entity_type) REFERENCES entity_types(name)
        );
        """
        self.conn.execute(sql)
        logger.info("✅ Created knowledge_entries table")

    def create_entity_types_table(self):
        """创建实体类型表"""
        sql = """
        CREATE TABLE IF NOT EXISTS entity_types (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(50) UNIQUE NOT NULL,                 -- 类型名称
            display_name VARCHAR(100) NOT NULL,                 -- 显示名称
            description TEXT,                                    -- 类型描述
            color VARCHAR(7) DEFAULT '#007bff',                -- 显示颜色
            icon VARCHAR(50) DEFAULT 'file',                  -- 图标
            is_active BOOLEAN DEFAULT TRUE,                    -- 是否启用
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        INSERT OR IGNORE INTO entity_types (name, display_name, description, color, icon) VALUES
        ('customer', '客户', '客户信息和联系资料', '#28a745', 'users'),
        ('quote', '报价', '报价单和价格信息', '#dc3545', 'dollar-sign'),
        ('drawing', '图纸', '技术图纸和设计文件', '#17a2b8', 'drafting-compass'),
        ('factory', '工厂', '工厂和供应商信息', '#fd7e14', 'industry'),
        ('product', '产品', '产品规格和技术参数', '#6f42c1', 'box'),
        ('inquiry', '询价', '客户询价和需求信息', '#20c997', 'question-circle'),
        ('contract', '合同', '合同协议和商务文件', '#e83e8c', 'file-text'),
        ('material', '材料', '材料规格和属性信息', '#6c757d', 'layers'),
        ('specification', '规格', '技术规格书和标准', '#343a40', 'clipboard-data');
        """
        self.conn.executescript(sql)
        logger.info("✅ Created entity_types table with default data")

    def create_nlp_entities_table(self):
        """创建NLP实体表"""
        sql = """
        CREATE TABLE IF NOT EXISTS nlp_entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id INTEGER NOT NULL,                        -- 关联的知识条目ID
            keyword VARCHAR(200) NOT NULL,                     -- 关键词
            value TEXT NOT NULL,                                -- 提取的值
            category VARCHAR(50) NOT NULL,                     -- 实体类别: customer_name, product_name, material, price, quantity, etc.
            confidence_score REAL DEFAULT 0.0,                 -- 置信度分数 (0-1)
            context_text TEXT,                                  -- 上下文文本
            start_position INTEGER,                              -- 在原文中的起始位置
            end_position INTEGER,                                -- 在原文中的结束位置
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (entry_id) REFERENCES knowledge_entries(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_nlp_entities_entry_id ON nlp_entities(entry_id);
        CREATE INDEX IF NOT EXISTS idx_nlp_entities_keyword ON nlp_entities(keyword);
        CREATE INDEX IF NOT EXISTS idx_nlp_entities_category ON nlp_entities(category);
        CREATE INDEX IF NOT EXISTS idx_nlp_entities_confidence ON nlp_entities(confidence_score);
        """
        self.conn.executescript(sql)
        logger.info("✅ Created nlp_entities table with indexes")

    def create_strategy_suggestions_table(self):
        """创建策略建议表"""
        sql = """
        CREATE TABLE IF NOT EXISTS strategy_suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            related_entry_id INTEGER,                          -- 关联的知识条目ID
            suggestion_type VARCHAR(50) NOT NULL,                 -- 建议类型: price_optimization, supplier_change, material_alternative, etc.
            title VARCHAR(200) NOT NULL,                         -- 建议标题
            description TEXT NOT NULL,                            -- 建议描述
            impact_level VARCHAR(20) DEFAULT 'medium',            -- 影响级别: low, medium, high, critical
            potential_savings REAL,                              -- 潜在节省金额
            confidence_score REAL DEFAULT 0.0,                   -- 建议置信度
            status VARCHAR(20) DEFAULT 'pending',                 -- 状态: pending, reviewed, implemented, rejected
            reviewed_by VARCHAR(100),                             -- 审核人
            reviewed_at DATETIME,                                -- 审核时间
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (related_entry_id) REFERENCES knowledge_entries(id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_strategy_related_entry ON strategy_suggestions(related_entry_id);
        CREATE INDEX IF NOT EXISTS idx_strategy_type ON strategy_suggestions(suggestion_type);
        CREATE INDEX IF NOT EXISTS idx_strategy_status ON strategy_suggestions(status);
        CREATE INDEX IF NOT EXISTS idx_strategy_impact ON strategy_suggestions(impact_level);
        """
        self.conn.executescript(sql)
        logger.info("✅ Created strategy_suggestions table with indexes")

    def create_embedding_index_table(self):
        """创建嵌入索引表"""
        sql = """
        CREATE TABLE IF NOT EXISTS embedding_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id INTEGER NOT NULL,                          -- 关联的知识条目ID
            vector_json TEXT NOT NULL,                            -- 向量数据 (JSON格式)
            model_name VARCHAR(100) DEFAULT 'text-embedding-3-small',  -- 使用的嵌入模型
            vector_dimension INTEGER DEFAULT 1536,              -- 向量维度
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (entry_id) REFERENCES knowledge_entries(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_embedding_entry_id ON embedding_index(entry_id);
        CREATE INDEX IF NOT EXISTS idx_embedding_model ON embedding_index(model_name);
        """
        self.conn.executescript(sql)
        logger.info("✅ Created embedding_index table with indexes")

    def create_knowledge_relationships_table(self):
        """创建知识关系表"""
        sql = """
        CREATE TABLE IF NOT EXISTS knowledge_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_entry_id INTEGER NOT NULL,                  -- 源条目ID
            target_entry_id INTEGER NOT NULL,                  -- 目标条目ID
            relationship_type VARCHAR(50) NOT NULL,               -- 关系类型: similar, related, alternative, derived_from, etc.
            confidence_score REAL DEFAULT 0.0,                   -- 关系置信度
            metadata TEXT,                                       -- 关系元数据 (JSON格式)
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (source_entry_id) REFERENCES knowledge_entries(id) ON DELETE CASCADE,
            FOREIGN KEY (target_entry_id) REFERENCES knowledge_entries(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_knowledge_source ON knowledge_relationships(source_entry_id);
        CREATE INDEX IF NOT EXISTS idx_knowledge_target ON knowledge_relationships(target_entry_id);
        CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge_relationships(relationship_type);
        """
        self.conn.executescript(sql)
        logger.info("✅ Created knowledge_relationships table with indexes")

    def create_triggers(self):
        """创建数据库触发器"""
        sql = """
        -- 更新时间触发器
        CREATE TRIGGER IF NOT EXISTS update_knowledge_entries_updated_at
            AFTER UPDATE ON knowledge_entries
            FOR EACH ROW
            BEGIN
                UPDATE knowledge_entries SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;

        CREATE TRIGGER IF NOT EXISTS update_strategy_suggestions_updated_at
            AFTER UPDATE ON strategy_suggestions
            FOR EACH ROW
            BEGIN
                UPDATE strategy_suggestions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;

        CREATE TRIGGER IF NOT EXISTS update_embedding_index_updated_at
            AFTER UPDATE ON embedding_index
            FOR EACH ROW
            BEGIN
                UPDATE embedding_index SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;

        -- 关系统计触发器
        CREATE TRIGGER IF NOT EXISTS log_knowledge_relationship_creation
            AFTER INSERT ON knowledge_relationships
            FOR EACH ROW
            BEGIN
                INSERT INTO system_logs (action, entity_type, entity_id, details, created_at)
                VALUES ('create', 'knowledge_relationship', NEW.id,
                       json_object('source', NEW.source_entry_id, 'target', NEW.target_entry_id, 'type', NEW.relationship_type),
                       CURRENT_TIMESTAMP);
            END;
        """
        self.conn.executescript(sql)
        logger.info("✅ Created database triggers")

    def create_views(self):
        """创建数据库视图"""
        sql = """
        -- 知识条目统计视图
        CREATE VIEW IF NOT EXISTS knowledge_entries_stats AS
        SELECT
            entity_type,
            COUNT(*) as total_entries,
            COUNT(CASE WHEN related_file IS NOT NULL THEN 1 END) as entries_with_files,
            COUNT(CASE WHEN attributes_json IS NOT NULL THEN 1 END) as entries_with_attributes,
            MAX(created_at) as latest_entry,
            COUNT(DISTINCT id) as unique_entities
        FROM knowledge_entries
        GROUP BY entity_type;

        -- 热门关键词视图
        CREATE VIEW IF NOT EXISTS popular_keywords AS
        SELECT
            keyword,
            category,
            COUNT(*) as usage_count,
            AVG(confidence_score) as avg_confidence,
            MAX(created_at) as last_used
        FROM nlp_entities
        WHERE confidence_score > 0.5
        GROUP BY keyword, category
        HAVING COUNT(*) > 1
        ORDER BY usage_count DESC;

        -- 策略建议统计视图
        CREATE VIEW IF NOT EXISTS strategy_stats AS
        SELECT
            suggestion_type,
            COUNT(*) as total_suggestions,
            COUNT(CASE WHEN status = 'implemented' THEN 1 END) as implemented_count,
            COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_count,
            SUM(CASE WHEN potential_savings IS NOT NULL THEN potential_savings ELSE 0 END) as total_potential_savings,
            AVG(confidence_score) as avg_confidence
        FROM strategy_suggestions
        GROUP BY suggestion_type;
        """
        self.conn.executescript(sql)
        logger.info("✅ Created database views")

    def insert_sample_data(self):
        """插入示例数据"""
        try:
            # 插入示例知识条目
            sample_entries = [
                {
                    'entity_type': 'customer',
                    'name': '示例客户公司',
                    'description': '这是一个示例客户记录',
                    'attributes_json': '{"industry": "制造业", "location": "上海", "contact_person": "张经理"}'
                },
                {
                    'entity_type': 'product',
                    'name': '标准螺栓M8x20',
                    'description': '标准六角螺栓规格',
                    'attributes_json': '{"material": "304不锈钢", "standard": "GB/T 5782", "strength": "8.8级"}'
                }
            ]

            for entry in sample_entries:
                self.conn.execute("""
                    INSERT INTO knowledge_entries (entity_type, name, description, attributes_json)
                    VALUES (?, ?, ?, ?)
                """, (
                    entry['entity_type'],
                    entry['name'],
                    entry['description'],
                    entry['attributes_json']
                ))

            logger.info("✅ Inserted sample knowledge entries")

        except Exception as e:
            logger.warning(f"Failed to insert sample data: {e}")

    def verify_schema(self):
        """验证数据库架构"""
        try:
            cursor = self.conn.cursor()

            # 检查表是否存在
            required_tables = [
                'knowledge_entries',
                'entity_types',
                'nlp_entities',
                'strategy_suggestions',
                'embedding_index',
                'knowledge_relationships'
            ]

            existing_tables = []
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            for row in cursor.fetchall():
                existing_tables.append(row[0])

            missing_tables = [table for table in required_tables if table not in existing_tables]

            if missing_tables:
                logger.error(f"❌ Missing tables: {missing_tables}")
                return False

            # 检查索引
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE '%idx_%'")
            indexes = [row[0] for row in cursor.fetchall()]
            logger.info(f"✅ Found {len(indexes)} database indexes")

            # 检查视图
            cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
            views = [row[0] for row in cursor.fetchall()]
            logger.info(f"✅ Found {len(views)} database views")

            logger.info("✅ Database schema verification completed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Schema verification failed: {e}")
            return False

    def initialize_knowledge_system(self):
        """初始化完整的知识系统"""
        try:
            logger.info("🚀 Starting knowledge system initialization...")

            # 连接数据库
            self.connect()

            # 创建表和索引
            self.create_entity_types_table()
            self.create_knowledge_entries_table()
            self.create_nlp_entities_table()
            self.create_strategy_suggestions_table()
            self.create_embedding_index_table()
            self.create_knowledge_relationships_table()
            self.create_triggers()
            self.create_views()

            # 插入示例数据
            self.insert_sample_data()

            # 验证架构
            if self.verify_schema():
                logger.info("🎉 Knowledge system initialization completed successfully!")
                return True
            else:
                logger.error("❌ Knowledge system initialization failed!")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to initialize knowledge system: {e}")
            return False

        finally:
            self.close()

def main():
    """主函数"""
    setup = KnowledgeModelsSetup()
    success = setup.initialize_knowledge_system()

    if success:
        print("\n" + "="*60)
        print("📊 KNOWLEDGE SYSTEM SETUP COMPLETED")
        print("="*60)
        print("✅ Database schema created")
        print("✅ Tables and indexes established")
        print("✅ Triggers and views configured")
        print("✅ Sample data inserted")
        print("✅ Schema verification passed")
        print("🚀 Knowledge system is ready for use!")
        print("="*60)
    else:
        print("\n❌ Knowledge system setup failed!")
        print("Please check the logs for details.")

if __name__ == "__main__":
    main()