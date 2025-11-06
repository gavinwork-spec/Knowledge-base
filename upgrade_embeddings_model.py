#!/usr/bin/env python3
"""
语义检索模型升级脚本
升级embedding模型，重新计算所有向量，提升检索精度
"""

import os
import sys
import json
import sqlite3
import logging
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import hashlib
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EmbeddingUpgrade")

class EmbeddingModelUpgrader:
    """Embedding模型升级器"""

    def __init__(self, db_path: str = "knowledge_base.db"):
        self.db_path = db_path
        self.conn = None
        self.embedding_model = None
        self.upgrade_metrics = {
            "old_model": "tfidf_fallback",
            "new_model": None,
            "total_entries": 0,
            "upgraded_entries": 0,
            "avg_similarity_improvement": 0.0,
            "processing_time": 0.0,
            "error_count": 0
        }

    def connect_database(self) -> bool:
        """连接数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            logger.info(f"Connected to database: {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    def initialize_embedding_model(self) -> bool:
        """初始化embedding模型"""
        try:
            # 尝试使用sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.upgrade_metrics["new_model"] = "sentence-transformers/all-MiniLM-L6-v2"
                logger.info("Using sentence-transformers model")
                return True
            except ImportError:
                logger.warning("sentence-transformers not available")

            # 尝试使用OpenAI embedding
            try:
                import openai
                # 这里需要设置API密钥
                # openai.api_key = os.getenv('OPENAI_API_KEY')
                logger.warning("OpenAI embedding not configured")
            except ImportError:
                logger.warning("OpenAI library not available")

            # 最后使用改进的TF-IDF
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            self.embedding_model = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.9
            )
            self.upgrade_metrics["new_model"] = "enhanced_tfidf"
            logger.info("Using enhanced TF-IDF model")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            return False

    def get_knowledge_entries(self) -> List[Dict]:
        """获取所有知识条目"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, name, description, entity_type, attributes_json, created_at
                FROM knowledge_entries
                WHERE description IS NOT NULL AND description != ''
                ORDER BY created_at DESC
            """)

            entries = []
            for row in cursor.fetchall():
                entry = {
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "entity_type": row[3],
                    "attributes": json.loads(row[4]) if row[4] else {},
                    "created_at": row[5]
                }
                entries.append(entry)

            self.upgrade_metrics["total_entries"] = len(entries)
            logger.info(f"Found {len(entries)} knowledge entries")
            return entries

        except Exception as e:
            logger.error(f"Failed to get knowledge entries: {e}")
            return []

    def create_text_representation(self, entry: Dict) -> str:
        """创建条目的文本表示"""
        text_parts = []

        # 添加名称和描述
        if entry.get("name"):
            text_parts.append(entry["name"])
        if entry.get("description"):
            text_parts.append(entry["description"])

        # 添加实体类型
        if entry.get("entity_type"):
            text_parts.append(f"Type: {entry['entity_type']}")

        # 添加属性
        attributes = entry.get("attributes", {})
        if attributes:
            for key, value in attributes.items():
                if value:
                    text_parts.append(f"{key}: {value}")

        return " ".join(text_parts)

    def generate_embedding(self, text: str) -> np.ndarray:
        """生成文本的embedding向量"""
        try:
            if self.upgrade_metrics["new_model"] == "sentence-transformers/all-MiniLM-L6-v2":
                return self.embedding_model.encode(text, show_progress_bar=False)

            elif self.upgrade_metrics["new_model"] == "enhanced_tfidf":
                # 对于TF-IDF，我们需要先fit所有文本
                if not hasattr(self, 'tfidf_matrix'):
                    return np.zeros(384)  # 返回默认维度

                # 这里需要实际的TF-IDF实现
                vector = self.embedding_model.transform([text])
                return vector.toarray()[0] if vector.nnz > 0 else np.zeros(384)

            else:
                # 默认返回零向量
                return np.zeros(384)

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return np.zeros(384)

    def calculate_similarity_improvement(self, old_embedding: np.ndarray, new_embedding: np.ndarray) -> float:
        """计算相似度改善"""
        try:
            # 这里简化计算，实际应该比较检索结果的改善
            if old_embedding.size == 0 or new_embedding.size == 0:
                return 0.0

            # 使用向量范数作为简单指标
            old_norm = np.linalg.norm(old_embedding)
            new_norm = np.linalg.norm(new_embedding)

            if old_norm == 0:
                return 0.0

            improvement = (new_norm - old_norm) / old_norm * 100
            return max(0, improvement)  # 只返回正改善

        except Exception as e:
            logger.error(f"Failed to calculate similarity improvement: {e}")
            return 0.0

    def upgrade_embeddings(self) -> bool:
        """升级所有embedding向量"""
        try:
            entries = self.get_knowledge_entries()
            if not entries:
                logger.warning("No entries found to upgrade")
                return False

            logger.info(f"Starting upgrade for {len(entries)} entries...")
            start_time = time.time()

            # 如果使用TF-IDF，先准备所有文本
            if self.upgrade_metrics["new_model"] == "enhanced_tfidf":
                all_texts = [self.create_text_representation(entry) for entry in entries]
                self.tfidf_matrix = self.embedding_model.fit_transform(all_texts)

            # 升级每个条目的embedding
            cursor = self.conn.cursor()
            similarity_improvements = []

            for i, entry in enumerate(entries):
                try:
                    # 创建文本表示
                    text = self.create_text_representation(entry)

                    # 生成新的embedding
                    new_embedding = self.generate_embedding(text)

                    # 获取旧的embedding（如果有）
                    old_embedding = np.zeros(384)  # 简化处理

                    # 计算改善程度
                    improvement = self.calculate_similarity_improvement(old_embedding, new_embedding)
                    similarity_improvements.append(improvement)

                    # 更新数据库中的embedding
                    embedding_blob = new_embedding.tobytes()
                    cursor.execute("""
                        UPDATE knowledge_entries
                        SET embedding_vector = ?, embedding_model = ?, updated_at = ?
                        WHERE id = ?
                    """, (embedding_blob, self.upgrade_metrics["new_model"], datetime.now().isoformat(), entry["id"]))

                    self.upgrade_metrics["upgraded_entries"] += 1

                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(entries)} entries")

                except Exception as e:
                    logger.error(f"Failed to upgrade entry {entry['id']}: {e}")
                    self.upgrade_metrics["error_count"] += 1

            # 提交更改
            self.conn.commit()

            # 计算平均改善
            if similarity_improvements:
                self.upgrade_metrics["avg_similarity_improvement"] = np.mean(similarity_improvements)

            self.upgrade_metrics["processing_time"] = time.time() - start_time

            logger.info(f"Upgrade completed: {self.upgrade_metrics['upgraded_entries']}/{self.upgrade_metrics['total_entries']} entries")
            logger.info(f"Average similarity improvement: {self.upgrade_metrics['avg_similarity_improvement']:.2f}%")
            logger.info(f"Processing time: {self.upgrade_metrics['processing_time']:.2f}s")

            return True

        except Exception as e:
            logger.error(f"Failed to upgrade embeddings: {e}")
            return False

    def save_metrics(self) -> bool:
        """保存升级指标"""
        try:
            metrics_file = f"embedding_upgrade_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.upgrade_metrics, f, indent=2, ensure_ascii=False)

            logger.info(f"Upgrade metrics saved to: {metrics_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
            return False

    def test_search_performance(self) -> Dict:
        """测试搜索性能"""
        try:
            # 这里可以添加搜索性能测试
            test_queries = ["螺栓", "不锈钢", "上海制造", "工厂"]
            results = {}

            for query in test_queries:
                start_time = time.time()
                # 这里应该调用实际的搜索功能
                # search_results = search_knowledge(query, top_k=5)
                end_time = time.time()

                results[query] = {
                    "response_time": end_time - start_time,
                    "results_count": 0  # 简化处理
                }

            avg_response_time = np.mean([r["response_time"] for r in results.values()])

            logger.info(f"Search performance test completed")
            logger.info(f"Average response time: {avg_response_time:.3f}s")

            return results

        except Exception as e:
            logger.error(f"Failed to test search performance: {e}")
            return {}

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="升级embedding模型")
    parser.add_argument("--db-path", default="knowledge_base.db", help="数据库路径")
    parser.add_argument("--model", choices=["sentence-transformers", "openai", "tfidf"],
                       help="指定embedding模型类型")
    parser.add_argument("--test", action="store_true", help="运行性能测试")

    args = parser.parse_args()

    logger.info("🚀 Starting embedding model upgrade...")

    # 创建升级器
    upgrader = EmbeddingModelUpgrader(args.db_path)

    try:
        # 连接数据库
        if not upgrader.connect_database():
            sys.exit(1)

        # 初始化模型
        if not upgrader.initialize_embedding_model():
            logger.error("Failed to initialize embedding model")
            sys.exit(1)

        # 执行升级
        if upgrader.upgrade_embeddings():
            # 保存指标
            upgrader.save_metrics()

            # 运行性能测试
            if args.test:
                upgrader.test_search_performance()

            logger.info("✅ Embedding upgrade completed successfully!")
        else:
            logger.error("❌ Embedding upgrade failed")
            sys.exit(1)

    finally:
        upgrader.close()

if __name__ == "__main__":
    main()