"""
数据库迁移脚本 - 从SQLite迁移到PostgreSQL
支持向量存储和微服务架构
"""

import asyncio
import sqlite3
import asyncpg
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MigrationConfig:
    """数据库迁移配置"""
    sqlite_path: str = "knowledge_base.db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "knowledge_base"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    vector_dimension: int = 384
    batch_size: int = 1000


class DatabaseMigrator:
    """数据库迁移器"""

    def __init__(self, config: MigrationConfig):
        self.config = config
        self.sqlite_conn = None
        self.postgres_pool = None

    async def connect(self):
        """连接到数据库"""
        # 连接SQLite
        try:
            self.sqlite_conn = sqlite3.connect(self.config.sqlite_path)
            self.sqlite_conn.row_factory = sqlite3.Row
            logger.info(f"Connected to SQLite database: {self.config.sqlite_path}")
        except Exception as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            raise

        # 连接PostgreSQL
        try:
            self.postgres_pool = await asyncpg.create_pool(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_database,
                user=self.config.postgres_user,
                password=self.config.postgres_password,
                min_size=2,
                max_size=10
            )
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def disconnect(self):
        """断开数据库连接"""
        if self.sqlite_conn:
            self.sqlite_conn.close()
            logger.info("Disconnected from SQLite")

        if self.postgres_pool:
            await self.postgres_pool.close()
            logger.info("Disconnected from PostgreSQL")

    async def run_migration(self):
        """执行完整迁移"""
        try:
            await self.connect()
            logger.info("Starting database migration...")

            # 1. 创建PostgreSQL数据库结构
            await self.create_postgres_schema()

            # 2. 迁移核心数据
            await self.migrate_knowledge_entries()
            await self.migrate_recommendations()
            await self.migrate_chat_history()

            # 3. 迁移向量数据
            await self.migrate_embeddings()

            # 4. 迁移搜索历史
            await self.migrate_search_history()

            # 5. 创建索引
            await self.create_indexes()

            # 6. 验证迁移结果
            await self.verify_migration()

            logger.info("Database migration completed successfully!")

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
        finally:
            await self.disconnect()

    async def create_postgres_schema(self):
        """创建PostgreSQL数据库结构"""
        logger.info("Creating PostgreSQL schema...")

        schema_sql = """
        -- 创建扩展
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE EXTENSION IF NOT EXISTS "pgvector";

        -- 知识条目表
        CREATE TABLE IF NOT EXISTS knowledge_entries (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name TEXT NOT NULL,
            description TEXT,
            entity_type TEXT NOT NULL,
            attributes_json JSONB,
            embedding_vector vector(%s),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            created_by UUID,
            updated_by UUID,
            version INTEGER DEFAULT 1,
            is_active BOOLEAN DEFAULT true,
            file_path TEXT,
            file_hash TEXT,
            confidence_score DECIMAL(3,2) DEFAULT 0.8,
            source_service TEXT DEFAULT 'legacy'
        );

        -- 向量索引表
        CREATE TABLE IF NOT EXISTS vector_index (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            knowledge_entry_id UUID REFERENCES knowledge_entries(id) ON DELETE CASCADE,
            vector vector(%s),
            embedding_model TEXT DEFAULT 'all-MiniLM-L6-v2',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- 事件表
        CREATE TABLE IF NOT EXISTS events (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            event_type VARCHAR(100) NOT NULL,
            source_service VARCHAR(100) NOT NULL,
            correlation_id UUID,
            causation_id UUID,
            user_id UUID,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            data JSONB,
            priority INTEGER DEFAULT 5,
            processed BOOLEAN DEFAULT FALSE,
            retry_count INTEGER DEFAULT 0,
            max_retries INTEGER DEFAULT 3,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- 推荐表
        CREATE TABLE IF NOT EXISTS recommendations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            inquiry_id UUID,
            recommended_products JSONB,
            recommended_suppliers JSONB,
            recommended_price_range NUMRANGE,
            confidence_score DECIMAL(5,2),
            recommendation_type TEXT,
            recommendation_reason TEXT,
            expires_at TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            created_by UUID REFERENCES knowledge_entries(id),
            algorithm_version TEXT DEFAULT 'v1.0'
        );

        -- 聊天历史表
        CREATE TABLE IF NOT EXISTS chat_history (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id VARCHAR(100) NOT NULL,
            user_query TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            query_type VARCHAR(50),
            context_used JSONB,
            feedback_score INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            session_started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- 搜索历史表
        CREATE TABLE IF NOT EXISTS search_history (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            query TEXT NOT NULL,
            query_type VARCHAR(50),
            results_count INTEGER DEFAULT 0,
            search_time_ms INTEGER,
            clicked_result_id UUID,
            user_id UUID,
            session_id VARCHAR(100),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- 用户表
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            username VARCHAR(100) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE,
            full_name TEXT,
            is_active BOOLEAN DEFAULT true,
            last_login TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            preferences_json JSONB DEFAULT '{}'
        );

        -- 工作流状态表
        CREATE TABLE IF NOT EXISTS workflow_states (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workflow_type VARCHAR(100) NOT NULL,
            workflow_id VARCHAR(100) NOT NULL,
            current_step VARCHAR(100) NOT NULL,
            status VARCHAR(50) DEFAULT 'running',
            data JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            completed_at TIMESTAMP WITH TIME ZONE
        );

        -- 文档处理任务表
        CREATE TABLE IF NOT EXISTS document_tasks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            file_path TEXT NOT NULL,
            file_hash TEXT,
            status VARCHAR(50) DEFAULT 'pending',
            processing_started_at TIMESTAMP WITH TIME ZONE,
            processing_completed_at TIMESTAMP WITH TIME ZONE,
            error_message TEXT,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- 系统配置表
        CREATE TABLE IF NOT EXISTS system_config (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            key VARCHAR(200) UNIQUE NOT NULL,
            value JSONB,
            description TEXT,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_by UUID REFERENCES users(id)
        );
        """ % (self.config.vector_dimension, self.config.vector_dimension)

        async with self.postgres_pool.acquire() as conn:
            await conn.execute(schema_sql)

        logger.info("PostgreSQL schema created successfully")

    async def migrate_knowledge_entries(self):
        """迁移知识条目数据"""
        logger.info("Migrating knowledge entries...")

        # 从SQLite读取数据
        cursor = self.sqlite_conn.cursor()
        cursor.execute("SELECT * FROM knowledge_entries")
        rows = cursor.fetchall()

        total_rows = len(rows)
        logger.info(f"Found {total_rows} knowledge entries to migrate")

        async with self.postgres_pool.acquire() as conn:
            batch = []
            for i, row in enumerate(rows):
                # 解析属性
                attributes = {}
                if row['attributes_json']:
                    try:
                        attributes = json.loads(row['attributes_json'])
                    except json.JSONDecodeError:
                        attributes = {}

                # 处理向量数据
                embedding_vector = None
                if row['embedding']:
                    try:
                        embedding_list = json.loads(row['embedding'])
                        embedding_vector = embedding_list
                    except json.JSONDecodeError:
                        pass

                entry_data = {
                    'id': str(uuid.uuid4()),  # 生成新的UUID
                    'name': row['name'],
                    'description': row['description'],
                    'entity_type': row['entity_type'],
                    'attributes_json': json.dumps(attributes) if attributes else None,
                    'embedding_vector': embedding_vector,
                    'created_at': datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.utcnow(),
                    'updated_at': datetime.fromisoformat(row['updated_at']) if row['updated_at'] else datetime.utcnow(),
                    'version': row.get('version', 1),
                    'is_active': row.get('is_active', True),
                    'file_path': attributes.get('file_path'),
                    'file_hash': attributes.get('file_hash'),
                    'confidence_score': attributes.get('confidence_score', 0.8),
                    'source_service': 'legacy_migration'
                }

                batch.append(entry_data)

                # 批量插入
                if len(batch) >= self.config.batch_size or i == total_rows - 1:
                    await self._batch_insert_knowledge_entries(conn, batch)
                    batch = []
                    logger.info(f"Migrated {i + 1}/{total_rows} knowledge entries")

        cursor.close()
        logger.info("Knowledge entries migration completed")

    async def _batch_insert_knowledge_entries(self, conn, batch: List[Dict]):
        """批量插入知识条目"""
        query = """
        INSERT INTO knowledge_entries (
            id, name, description, entity_type, attributes_json,
            embedding_vector, created_at, updated_at, version,
            is_active, file_path, file_hash, confidence_score, source_service
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
        )
        """

        for entry in batch:
            await conn.execute(
                query,
                entry['id'],
                entry['name'],
                entry['description'],
                entry['entity_type'],
                entry['attributes_json'],
                entry['embedding_vector'],
                entry['created_at'],
                entry['updated_at'],
                entry['version'],
                entry['is_active'],
                entry['file_path'],
                entry['file_hash'],
                entry['confidence_score'],
                entry['source_service']
            )

            # 如果有向量数据，同时插入到向量索引表
            if entry['embedding_vector']:
                await conn.execute(
                    """
                    INSERT INTO vector_index (knowledge_entry_id, vector, embedding_model, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    entry['id'],
                    entry['embedding_vector'],
                    'all-MiniLM-L6-v2',  # 默认模型
                    entry['created_at'],
                    entry['updated_at']
                )

    async def migrate_recommendations(self):
        """迁移推荐数据"""
        logger.info("Migrating recommendations...")

        cursor = self.sqlite_conn.cursor()
        cursor.execute("SELECT * FROM recommendations")
        rows = cursor.fetchall()

        if not rows:
            logger.info("No recommendations found to migrate")
            return

        async with self.postgres_pool.acquire() as conn:
            for row in rows:
                try:
                    await conn.execute(
                        """
                        INSERT INTO recommendations (
                            id, recommended_products, recommended_suppliers,
                            recommended_price_range, confidence_score, recommendation_type,
                            recommendation_reason, created_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """,
                        str(uuid.uuid4()),
                        json.loads(row['recommended_products']) if row['recommended_products'] else None,
                        json.loads(row['recommended_suppliers']) if row['recommended_suppliers'] else None,
                        row['recommended_price_range'],
                        row['confidence_score'],
                        row['recommendation_type'],
                        row['recommendation_reason'],
                        datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.utcnow()
                    )
                except Exception as e:
                    logger.warning(f"Failed to migrate recommendation {row.get('id')}: {e}")

        cursor.close()
        logger.info("Recommendations migration completed")

    async def migrate_chat_history(self):
        """迁移聊天历史"""
        logger.info("Migrating chat history...")

        cursor = self.sqlite_conn.cursor()
        cursor.execute("SELECT * FROM chat_history")
        rows = cursor.fetchall()

        if not rows:
            logger.info("No chat history found to migrate")
            return

        async with self.postgres_pool.acquire() as conn:
            for row in rows:
                try:
                    await conn.execute(
                        """
                        INSERT INTO chat_history (
                            id, session_id, user_query, bot_response, query_type,
                            context_used, feedback_score, created_at, session_started_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        """,
                        str(uuid.uuid4()),
                        row['session_id'],
                        row['user_query'],
                        row['bot_response'],
                        row['query_type'],
                        json.loads(row['context_used']) if row['context_used'] else None,
                        row['feedback_score'],
                        datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.utcnow(),
                        datetime.fromisoformat(row['session_started_at']) if row['session_started_at'] else datetime.utcnow()
                    )
                except Exception as e:
                    logger.warning(f"Failed to migrate chat history entry {row.get('id')}: {e}")

        cursor.close()
        logger.info("Chat history migration completed")

    async def migrate_embeddings(self):
        """迁移向量嵌入数据"""
        logger.info("Migrating embeddings...")

        cursor = self.sqlite_conn.cursor()
        cursor.execute("SELECT * FROM embedding_index")
        rows = cursor.fetchall()

        if not rows:
            logger.info("No embeddings found to migrate")
            return

        async with self.postgres_pool.acquire() as conn:
            for row in rows:
                try:
                    vector_data = json.loads(row['vector_data'])
                    await conn.execute(
                        """
                        INSERT INTO vector_index (knowledge_entry_id, vector, embedding_model, created_at, updated_at)
                        VALUES ($1, $2, $3, $4, $5)
                        """,
                        row['knowledge_entry_id'],
                        vector_data,
                        row['model_name'] or 'legacy',
                        datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.utcnow(),
                        datetime.utcnow()
                    )
                except Exception as e:
                    logger.warning(f"Failed to migrate embedding {row.get('id')}: {e}")

        cursor.close()
        logger.info("Embeddings migration completed")

    async def migrate_search_history(self):
        """迁移搜索历史"""
        logger.info("Migrating search history...")

        cursor = self.sqlite_conn.cursor()
        cursor.execute("SELECT * FROM search_history")
        rows = cursor.fetchall()

        if not rows:
            logger.info("No search history found to migrate")
            return

        async with self.postgres_pool.acquire() as conn:
            for row in rows:
                try:
                    await conn.execute(
                        """
                        INSERT INTO search_history (
                            id, query, query_type, results_count, search_time_ms,
                            clicked_result_id, user_id, session_id, created_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        """,
                        str(uuid.uuid4()),
                        row['query'],
                        row['query_type'],
                        row['results_count'],
                        row['search_time_ms'],
                        row['clicked_result_id'],
                        row['user_id'],
                        row['session_id'],
                        datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.utcnow()
                    )
                except Exception as e:
                    logger.warning(f"Failed to migrate search history entry {row.get('id')}: {e}")

        cursor.close()
        logger.info("Search history migration completed")

    async def create_indexes(self):
        """创建数据库索引"""
        logger.info("Creating database indexes...")

        index_queries = [
            # 知识条目索引
            "CREATE INDEX IF NOT EXISTS idx_knowledge_entries_type ON knowledge_entries(entity_type);",
            "CREATE INDEX IF NOT EXISTS idx_knowledge_entries_created ON knowledge_entries(created_at);",
            "CREATE INDEX IF NOT EXISTS idx_knowledge_entries_active ON knowledge_entries(is_active) WHERE is_active = true;",
            "CREATE INDEX IF NOT EXISTS idx_knowledge_entries_source ON knowledge_entries(source_service);",

            # 向量索引
            "CREATE INDEX IF NOT EXISTS idx_vector_index_vector ON vector_index USING ivfflat (vector vector_cosine_ops);",
            "CREATE INDEX IF NOT EXISTS idx_vector_index_entry ON vector_index(knowledge_entry_id);",

            # 事件索引
            "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);",
            "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_events_processed ON events(processed) WHERE processed = false;",
            "CREATE INDEX IF NOT EXISTS idx_events_source ON events(source_service);",
            "CREATE INDEX IF NOT EXISTS idx_events_correlation ON events(correlation_id);",

            # 推荐索引
            "CREATE INDEX IF NOT EXISTS idx_recommendations_inquiry ON recommendations(inquiry_id);",
            "CREATE INDEX IF NOT EXISTS idx_recommendations_created ON recommendations(created_at);",
            "CREATE INDEX IF NOT EXISTS idx_recommendations_type ON recommendations(recommendation_type);",

            # 聊天历史索引
            "CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history(session_id);",
            "CREATE INDEX IF NOT EXISTS idx_chat_history_created ON chat_history(created_at);",
            "CREATE INDEX IF NOT EXISTS idx_chat_history_query_type ON chat_history(query_type);",

            # 搜索历史索引
            "CREATE INDEX IF NOT EXISTS idx_search_history_query ON search_history(query);",
            "CREATE INDEX IF NOT EXISTS idx_search_history_created ON search_history(created_at);",
            "CREATE INDEX IF NOT EXISTS idx_search_history_session ON search_history(session_id);",

            # 用户索引
            "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);",
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);",
            "CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active) WHERE is_active = true;",

            # 工作流索引
            "CREATE INDEX IF NOT EXISTS idx_workflow_states_type ON workflow_states(workflow_type);",
            "CREATE INDEX IF NOT EXISTS idx_workflow_states_status ON workflow_states(status);",
            "CREATE INDEX IF NOT EXISTS idx_workflow_states_updated ON workflow_states(updated_at);",

            # 文档任务索引
            "CREATE INDEX IF NOT EXISTS idx_document_tasks_status ON document_tasks(status);",
            "CREATE INDEX IF NOT EXISTS idx_document_tasks_hash ON document_tasks(file_hash);",
            "CREATE INDEX IF NOT EXISTS idx_document_tasks_created ON document_tasks(created_at);"
        ]

        async with self.postgres_pool.acquire() as conn:
            for query in index_queries:
                try:
                    await conn.execute(query)
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")

        logger.info("Database indexes created successfully")

    async def verify_migration(self):
        """验证迁移结果"""
        logger.info("Verifying migration results...")

        verification_queries = [
            ("SELECT COUNT(*) as count FROM knowledge_entries", "Knowledge entries"),
            ("SELECT COUNT(*) as count FROM vector_index", "Vector indexes"),
            ("SELECT COUNT(*) as count FROM events", "Events"),
            ("SELECT COUNT(*) as count FROM recommendations", "Recommendations"),
            ("SELECT COUNT(*) as count FROM chat_history", "Chat history"),
            ("SELECT COUNT(*) as count FROM search_history", "Search history")
        ]

        async with self.postgres_pool.acquire() as conn:
            for query, description in verification_queries:
                try:
                    result = await conn.fetchrow(query)
                    logger.info(f"{description}: {result['count']} records")
                except Exception as e:
                    logger.error(f"Failed to verify {description}: {e}")

        # 测试向量搜索功能
        await self._test_vector_search()

        logger.info("Migration verification completed")

    async def _test_vector_search(self):
        """测试向量搜索功能"""
        logger.info("Testing vector search functionality...")

        async with self.postgres_pool.acquire() as conn:
            try:
                # 获取一个向量样本
                result = await conn.fetchrow(
                    "SELECT id, vector FROM vector_index LIMIT 1"
                )

                if result:
                    # 执行相似性搜索
                    search_results = await conn.fetch(
                        """
                        SELECT ke.id, ke.name, ke.entity_type,
                        1 - (vi.vector <=> $1) as similarity
                        FROM vector_index vi
                        JOIN knowledge_entries ke ON vi.knowledge_entry_id = ke.id
                        WHERE vi.vector IS NOT NULL
                        ORDER BY vi.vector <=> $1
                        LIMIT 5
                        """,
                        result['vector']
                    )

                    logger.info(f"Vector search test successful. Found {len(search_results)} similar items")
                    for item in search_results:
                        logger.debug(f"  - {item['name']} (similarity: {item['similarity']:.4f})")
                else:
                    logger.warning("No vectors found for search test")

            except Exception as e:
                logger.error(f"Vector search test failed: {e}")


async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate SQLite database to PostgreSQL")
    parser.add_argument("--sqlite-path", default="knowledge_base.db", help="SQLite database path")
    parser.add_argument("--postgres-host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--postgres-port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--postgres-database", default="knowledge_base", help="PostgreSQL database name")
    parser.add_argument("--postgres-user", default="postgres", help="PostgreSQL user")
    parser.add_argument("--postgres-password", default="postgres", help="PostgreSQL password")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for inserts")

    args = parser.parse_args()

    config = MigrationConfig(
        sqlite_path=args.sqlite_path,
        postgres_host=args.postgres_host,
        postgres_port=args.postgres_port,
        postgres_database=args.postgres_database,
        postgres_user=args.postgres_user,
        postgres_password=args.postgres_password,
        batch_size=args.batch_size
    )

    migrator = DatabaseMigrator(config)
    await migrator.run_migration()


if __name__ == "__main__":
    asyncio.run(main())