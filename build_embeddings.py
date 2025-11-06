#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Base Embedding Index Builder
知识库嵌入索引构建器

This script reads all entries from knowledge_entries, generates embeddings using
text-embedding models, stores them to embedding_index, and provides similarity search.
"""

import sqlite3
import json
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import os
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/processed/build_embeddings.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmbeddingIndexBuilder:
    """知识库嵌入索引构建器"""

    def __init__(self, db_path: str = "knowledge_base.db"):
        self.db_path = db_path
        self.conn = None
        self.embedding_model = None
        self.embedding_dimension = 1536  # Default for text-embedding-3-small

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

    def init_embedding_model(self):
        """初始化嵌入模型"""
        try:
            # 尝试导入 OpenAI
            import openai
            from openai import OpenAI

            # 检查 API key
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
                self.model_name = "text-embedding-3-small"
                self.embedding_dimension = 1536
                logger.info("✅ Initialized OpenAI embedding model: text-embedding-3-small")
                return

        except ImportError:
            logger.warning("OpenAI library not available")

        try:
            # 尝试使用 sentence-transformers (本地模型)
            from sentence_transformers import SentenceTransformer

            self.model_name = "all-MiniLM-L6-v2"
            self.embedding_model = SentenceTransformer(self.model_name)
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"✅ Initialized local embedding model: {self.model_name}")
            logger.info(f"✅ Embedding dimension: {self.embedding_dimension}")
            return

        except ImportError:
            logger.warning("sentence-transformers library not available")

        # 如果都不可用，使用简单的词向量
        logger.warning("⚠️ No embedding models available, using simple TF-IDF fallback")
        self.model_name = "tfidf_fallback"
        self.embedding_dimension = 1000

    def generate_text_for_embedding(self, entry_data: Dict) -> str:
        """为知识条目生成用于嵌入的文本"""
        text_parts = []

        # 添加名称
        if entry_data.get('name'):
            text_parts.append(entry_data['name'])

        # 添加描述
        if entry_data.get('description'):
            text_parts.append(entry_data['description'])

        # 添加属性数据
        if entry_data.get('attributes_json'):
            try:
                attributes = json.loads(entry_data['attributes_json'])
                for key, value in attributes.items():
                    if value:
                        text_parts.append(f"{key}: {value}")
            except json.JSONDecodeError:
                pass

        # 添加实体类型
        if entry_data.get('entity_type'):
            text_parts.append(f"类型: {entry_data['entity_type']}")

        return " ".join(text_parts)

    def generate_embedding_openai(self, text: str) -> Optional[List[float]]:
        """使用OpenAI生成嵌入"""
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}")
            return None

    def generate_embedding_local(self, text: str) -> Optional[List[float]]:
        """使用本地模型生成嵌入"""
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Local embedding generation failed: {e}")
            return None

    def generate_embedding_fallback(self, text: str) -> List[float]:
        """使用简单的TF-IDF作为后备方案"""
        # 简单的词频向量
        words = text.lower().split()
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # 创建固定长度的向量
        vector = [0.0] * self.embedding_dimension

        # 使用单词哈希映射到向量维度
        for word, freq in word_freq.items():
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = hash_val % self.embedding_dimension
            vector[idx] += freq

        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.tolist()  # 确保返回Python列表而不是numpy数组

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """生成文本嵌入"""
        if self.model_name == "text-embedding-3-small":
            return self.generate_embedding_openai(text)
        elif self.model_name == "tfidf_fallback":
            return self.generate_embedding_fallback(text)
        else:
            return self.generate_embedding_local(text)

    def create_embedding_index_table(self):
        """创建嵌入索引表"""
        sql = """
        CREATE TABLE IF NOT EXISTS embedding_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id INTEGER NOT NULL,
            vector_json TEXT NOT NULL,
            model_name VARCHAR(100) DEFAULT 'text-embedding-3-small',
            vector_dimension INTEGER DEFAULT 1536,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (entry_id) REFERENCES knowledge_entries(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_embedding_entry_id ON embedding_index(entry_id);
        CREATE INDEX IF NOT EXISTS idx_embedding_model ON embedding_index(model_name);
        """
        self.conn.executescript(sql)
        logger.info("✅ Created embedding_index table")

    def get_knowledge_entries(self) -> List[Dict]:
        """获取所有知识条目"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, entity_type, name, description, attributes_json, created_at
            FROM knowledge_entries
            ORDER BY created_at DESC
        """)

        columns = [desc[0] for desc in cursor.description]
        entries = []

        for row in cursor.fetchall():
            entry = dict(zip(columns, row))
            entries.append(entry)

        logger.info(f"📚 Found {len(entries)} knowledge entries")
        return entries

    def embedding_exists(self, entry_id: int) -> bool:
        """检查嵌入是否已存在"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM embedding_index WHERE entry_id = ?", (entry_id,))
        return cursor.fetchone() is not None

    def save_embedding(self, entry_id: int, embedding: List[float]):
        """保存嵌入到数据库"""
        vector_json = json.dumps(embedding)

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO embedding_index
            (entry_id, vector_json, model_name, vector_dimension, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            entry_id,
            vector_json,
            self.model_name,
            self.embedding_dimension
        ))

        self.conn.commit()

    def build_embeddings(self, batch_size: int = 10, force_rebuild: bool = False):
        """构建所有知识条目的嵌入索引"""
        try:
            logger.info("🚀 Starting embedding index building...")

            # 连接数据库
            self.connect()

            # 创建嵌入索引表
            self.create_embedding_index_table()

            # 初始化嵌入模型
            self.init_embedding_model()

            # 获取所有知识条目
            entries = self.get_knowledge_entries()

            if not entries:
                logger.warning("⚠️ No knowledge entries found")
                return

            # 处理每个条目
            processed_count = 0
            skipped_count = 0

            for i, entry in enumerate(entries):
                entry_id = entry['id']

                # 检查是否需要重新构建
                if not force_rebuild and self.embedding_exists(entry_id):
                    skipped_count += 1
                    continue

                # 生成嵌入文本
                text = self.generate_text_for_embedding(entry)
                if not text.strip():
                    logger.warning(f"⚠️ No text content for entry {entry_id}")
                    continue

                # 生成嵌入
                embedding = self.generate_embedding(text)
                if embedding is None:
                    logger.error(f"❌ Failed to generate embedding for entry {entry_id}")
                    continue

                # 保存嵌入
                self.save_embedding(entry_id, embedding)
                processed_count += 1

                # 显示进度
                if (i + 1) % batch_size == 0:
                    logger.info(f"📊 Processed {i + 1}/{len(entries)} entries")

            logger.info(f"✅ Embedding index building completed!")
            logger.info(f"📈 Processed: {processed_count} entries")
            logger.info(f"⏭️ Skipped: {skipped_count} entries")

        except Exception as e:
            logger.error(f"❌ Failed to build embedding index: {e}")
            raise
        finally:
            self.close()

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        try:
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)

            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0

    def find_similar_entries(self, query: str, top_k: int = 5, min_similarity: float = 0.0) -> List[Dict]:
        """查找相似的知识条目"""
        try:
            # 连接数据库
            self.connect()

            # 初始化嵌入模型
            if not hasattr(self, 'model_name'):
                self.init_embedding_model()

            # 生成查询嵌入
            query_embedding = self.generate_embedding(query)
            if query_embedding is None:
                logger.error("Failed to generate query embedding")
                return []

            # 获取所有嵌入
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT ei.entry_id, ei.vector_json, ke.name, ke.description, ke.entity_type
                FROM embedding_index ei
                JOIN knowledge_entries ke ON ei.entry_id = ke.id
                WHERE ei.model_name = ?
            """, (self.model_name,))

            results = []

            for row in cursor.fetchall():
                entry_id, vector_json, name, description, entity_type = row

                try:
                    stored_embedding = json.loads(vector_json)

                    # 计算相似度
                    similarity = self.cosine_similarity(query_embedding, stored_embedding)

                    # Debug logging
                    logger.debug(f"Entry {entry_id} ({name}): similarity = {similarity:.6f}")

                    if similarity >= min_similarity:
                        results.append({
                            'entry_id': entry_id,
                            'name': name,
                            'description': description,
                            'entity_type': entity_type,
                            'similarity': similarity
                        })
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON for entry {entry_id}")
                    continue

            # 按相似度排序
            results.sort(key=lambda x: x['similarity'], reverse=True)

            # 返回前k个结果
            return results[:top_k]

        except Exception as e:
            logger.error(f"Failed to find similar entries: {e}")
            return []
        finally:
            self.close()

    def get_embedding_stats(self) -> Dict:
        """获取嵌入索引统计信息"""
        try:
            self.connect()

            cursor = self.conn.cursor()

            # 总条目数
            cursor.execute("SELECT COUNT(*) FROM embedding_index")
            total_embeddings = cursor.fetchone()[0]

            # 按模型统计
            cursor.execute("""
                SELECT model_name, COUNT(*)
                FROM embedding_index
                GROUP BY model_name
            """)
            model_stats = dict(cursor.fetchall())

            # 按实体类型统计
            cursor.execute("""
                SELECT ke.entity_type, COUNT(*)
                FROM embedding_index ei
                JOIN knowledge_entries ke ON ei.entry_id = ke.id
                GROUP BY ke.entity_type
            """)
            type_stats = dict(cursor.fetchall())

            return {
                'total_embeddings': total_embeddings,
                'model_stats': model_stats,
                'type_stats': type_stats,
                'model_name': getattr(self, 'model_name', 'unknown'),
                'embedding_dimension': getattr(self, 'embedding_dimension', 0)
            }

        except Exception as e:
            logger.error(f"Failed to get embedding stats: {e}")
            return {}
        finally:
            self.close()

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Build embedding index for knowledge base')
    parser.add_argument('--build', action='store_true', help='Build embedding index')
    parser.add_argument('--search', type=str, help='Search similar entries')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to return')
    parser.add_argument('--stats', action='store_true', help='Show embedding statistics')
    parser.add_argument('--force', action='store_true', help='Force rebuild existing embeddings')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing')

    args = parser.parse_args()

    builder = EmbeddingIndexBuilder()

    if args.build:
        builder.build_embeddings(batch_size=args.batch_size, force_rebuild=args.force)
    elif args.search:
        results = builder.find_similar_entries(args.search, top_k=args.top_k)

        print(f"\n🔍 Search Results for: '{args.search}'")
        print("=" * 60)

        if results:
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['name']} ({result['entity_type']})")
                print(f"   Similarity: {result['similarity']:.3f}")
                if result['description']:
                    print(f"   Description: {result['description'][:100]}...")
        else:
            print("No similar entries found.")

    elif args.stats:
        stats = builder.get_embedding_stats()

        print("\n📊 Embedding Index Statistics")
        print("=" * 40)
        print(f"Total embeddings: {stats.get('total_embeddings', 0)}")
        print(f"Current model: {stats.get('model_name', 'unknown')}")
        print(f"Embedding dimension: {stats.get('embedding_dimension', 0)}")

        print(f"\nBy model:")
        for model, count in stats.get('model_stats', {}).items():
            print(f"  {model}: {count}")

        print(f"\nBy entity type:")
        for entity_type, count in stats.get('type_stats', {}).items():
            print(f"  {entity_type}: {count}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()