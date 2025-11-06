#!/usr/bin/env python3
"""
智能推荐生成脚本
基于历史数据和市场分析，为询盘提供最优产品建议和价格区间推荐
"""

import os
import sys
import json
import sqlite3
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RecommendationGenerator")

@dataclass
class Recommendation:
    """推荐数据类"""
    inquiry_id: int
    recommended_products: List[Dict]
    recommended_suppliers: List[Dict]
    recommended_price_range: Tuple[float, float]
    confidence_score: float
    recommendation_type: str
    recommendation_reason: str
    expires_at: datetime
    created_at: datetime

@dataclass
class ProductRecommendation:
    """产品推荐数据类"""
    product_id: int
    product_name: str
    similarity_score: float
    price_range: Tuple[float, float]
    recommended_supplier: str
    confidence_score: float
    reasons: List[str]

class RecommendationGenerator:
    """智能推荐生成器"""

    def __init__(self, db_path: str = "knowledge_base.db"):
        self.db_path = db_path
        self.conn = None
        self.recommendation_stats = {
            "total_inquiries_processed": 0,
            "total_recommendations_generated": 0,
            "avg_confidence_score": 0.0,
            "processing_time": 0.0,
            "error_count": 0
        }

    def connect_database(self) -> bool:
        """连接数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            logger.info(f"Connected to database: {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    def get_recent_inquiries(self, lookback_days: int = 90) -> List[Dict]:
        """获取最近的询盘数据"""
        try:
            cursor = self.conn.cursor()
            cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()

            cursor.execute("""
                SELECT id, name, description, attributes_json, created_at
                FROM knowledge_entries
                WHERE entity_type = 'inquiry' AND created_at > ?
                ORDER BY created_at DESC
            """, (cutoff_date,))

            inquiries = []
            for row in cursor.fetchall():
                inquiry = dict(row)
                if inquiry['attributes_json']:
                    inquiry['attributes'] = json.loads(inquiry['attributes_json'])
                else:
                    inquiry['attributes'] = {}
                inquiries.append(inquiry)

            logger.info(f"Found {len(inquiries)} recent inquiries")
            return inquiries

        except Exception as e:
            logger.error(f"Failed to get recent inquiries: {e}")
            return []

    def get_knowledge_base_for_recommendation(self) -> Dict[str, List[Dict]]:
        """获取推荐所需的知识库数据"""
        try:
            cursor = self.conn.cursor()

            # 获取产品数据
            cursor.execute("""
                SELECT id, name, description, attributes_json, created_at
                FROM knowledge_entries
                WHERE entity_type IN ('product', 'specification')
                ORDER BY created_at DESC
            """)
            products = []
            for row in cursor.fetchall():
                product = dict(row)
                if product['attributes_json']:
                    product['attributes'] = json.loads(product['attributes_json'])
                else:
                    product['attributes'] = {}
                products.append(product)

            # 获取报价数据
            cursor.execute("""
                SELECT id, name, description, attributes_json, created_at
                FROM knowledge_entries
                WHERE entity_type = 'quote'
                ORDER BY created_at DESC
            """)
            quotes = []
            for row in cursor.fetchall():
                quote = dict(row)
                if quote['attributes_json']:
                    quote['attributes'] = json.loads(quote['attributes_json'])
                else:
                    quote['attributes'] = {}
                quotes.append(quote)

            # 获取工厂数据
            cursor.execute("""
                SELECT id, name, description, attributes_json, created_at
                FROM knowledge_entries
                WHERE entity_type = 'factory'
                ORDER BY created_at DESC
            """)
            factories = []
            for row in cursor.fetchall():
                factory = dict(row)
                if factory['attributes_json']:
                    factory['attributes'] = json.loads(factory['attributes_json'])
                else:
                    factory['attributes'] = {}
                factories.append(factory)

            return {
                "products": products,
                "quotes": quotes,
                "factories": factories
            }

        except Exception as e:
            logger.error(f"Failed to get knowledge base data: {e}")
            return {"products": [], "quotes": [], "factories": []}

    def extract_text_features(self, text: str) -> Dict[str, str]:
        """从文本中提取特征"""
        features = {
            "material": "",
            "specification": "",
            "application": "",
            "quantity": "",
            "budget": ""
        }

        # 材料关键词
        material_patterns = [
            r'(不锈钢|碳钢|合金钢|铜|铝|塑料|尼龙)',
            r'(304|316|45#|Q235|HC420)',
            r'(SUS|A2|A4|B7|B8)'
        ]

        # 规格关键词
        spec_patterns = [
            r'M(\d+)[xX×](\d+)',  # M螺栓规格
            r'Ø(\d+)',            # 直径
            r'(\d+)#(\d+)',       # 英寸规格
            r'(\d+)\s*mm'         # 毫米
        ]

        # 应用场景关键词
        app_patterns = [
            r'(汽车|机械|建筑|电子|航空|船舶)',
            r'(发动机|变速箱|底盘|车身)',
            r'(建筑机械|工程机械|农业机械)'
        ]

        # 数量关键词
        quantity_patterns = [
            r'(\d+)[个件只支套]',
            r'(\d+)[kK](?:[个件只支套])?',
            r'批量|大批量|小批量'
        ]

        # 预算关键词
        budget_patterns = [
            r'(¥|￥|RMB|USD)\s*(\d+(?:\.\d+)?)',
            r'预算\s*[:：]\s*(\d+(?:\.\d+)?)',
            r'价格\s*[:：]\s*(\d+(?:\.\d+)?)'
        ]

        # 提取材料
        for pattern in material_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                features["material"] = match.group(1)
                break

        # 提取规格
        for pattern in spec_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                features["specification"] = match.group(0)
                break

        # 提取应用
        for pattern in app_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                features["application"] = match.group(1)
                break

        # 提取数量
        for pattern in quantity_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                features["quantity"] = match.group(1)
                break

        # 提取预算
        for pattern in budget_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                features["budget"] = match.group(2) if len(match.groups()) > 1 else match.group(1)
                break

        return features

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        try:
            # 简单的关键词重叠相似度
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            if not words1 or not words2:
                return 0.0

            intersection = words1.intersection(words2)
            union = words1.union(words2)

            similarity = len(intersection) / len(union)
            return similarity

        except Exception as e:
            logger.error(f"Failed to calculate text similarity: {e}")
            return 0.0

    def find_similar_products(self, inquiry: Dict, products: List[Dict]) -> List[Tuple[Dict, float]]:
        """查找相似产品"""
        similar_products = []

        inquiry_text = f"{inquiry.get('name', '')} {inquiry.get('description', '')}"
        inquiry_features = self.extract_text_features(inquiry_text)

        for product in products:
            product_text = f"{product.get('name', '')} {product.get('description', '')}"
            product_features = product.get('attributes', {})

            # 计算基础文本相似度
            text_similarity = self.calculate_text_similarity(inquiry_text, product_text)

            # 计算特征匹配度
            feature_score = 0.0
            feature_count = 0

            # 材料匹配
            if inquiry_features["material"] and product_features.get("material"):
                if inquiry_features["material"] in product_features["material"]:
                    feature_score += 1.0
                feature_count += 1

            # 规格匹配
            if inquiry_features["specification"] and product_features.get("specification"):
                if inquiry_features["specification"] in product_features["specification"]:
                    feature_score += 1.0
                feature_count += 1

            # 应用匹配
            if inquiry_features["application"] and product_features.get("application"):
                if inquiry_features["application"] in product_features["application"]:
                    feature_score += 1.0
                feature_count += 1

            # 综合相似度
            feature_similarity = feature_score / max(feature_count, 1)
            overall_similarity = 0.7 * text_similarity + 0.3 * feature_similarity

            if overall_similarity > 0.3:  # 相似度阈值
                similar_products.append((product, overall_similarity))

        # 按相似度排序
        similar_products.sort(key=lambda x: x[1], reverse=True)
        return similar_products[:5]  # 返回前5个最相似的产品

    def get_price_range_for_product(self, product: Dict, quotes: List[Dict]) -> Tuple[float, float]:
        """获取产品的价格区间"""
        product_name = product.get('name', '')
        product_attributes = product.get('attributes', {})

        relevant_prices = []

        for quote in quotes:
            quote_text = f"{quote.get('name', '')} {quote.get('description', '')}"
            quote_attributes = quote.get('attributes', {})

            # 检查是否是相关产品
            text_similarity = self.calculate_text_similarity(product_name, quote_text)
            if text_similarity < 0.3:
                continue

            # 提取价格
            price = quote_attributes.get('price') or quote_attributes.get('total_amount')
            if price:
                try:
                    price_value = float(re.sub(r'[^0-9.]', '', str(price)))
                    if price_value > 0:
                        relevant_prices.append(price_value)
                except (ValueError, TypeError):
                    continue

        if not relevant_prices:
            # 如果没有找到相关价格，使用默认估算
            return (0.0, 0.0)

        # 计算价格区间（去掉异常值）
        relevant_prices.sort()
        if len(relevant_prices) >= 3:
            # 去掉最高和最低的20%
            trim_count = max(1, len(relevant_prices) // 5)
            trimmed_prices = relevant_prices[trim_count:-trim_count]
        else:
            trimmed_prices = relevant_prices

        if trimmed_prices:
            min_price = min(trimmed_prices)
            max_price = max(trimmed_prices)
        else:
            min_price = min(relevant_prices)
            max_price = max(relevant_prices)

        return (min_price, max_price)

    def get_recommended_supplier(self, product: Dict, factories: List[Dict]) -> str:
        """获取推荐供应商"""
        product_attributes = product.get('attributes', {})

        # 寻找有相关认证的工厂
        suitable_factories = []

        for factory in factories:
            factory_attributes = factory.get('attributes', {})

            # 检查认证匹配
            product_cert = product_attributes.get('certification', '')
            factory_cert = factory_attributes.get('certification', '')

            if product_cert and factory_cert:
                if product_cert in factory_cert or factory_cert in product_cert:
                    suitable_factories.append(factory)
                    continue

            # 检查专业领域匹配
            product_specialty = product_attributes.get('specialty', '')
            factory_specialty = factory_attributes.get('specialty', '')

            if product_specialty and factory_specialty:
                if product_specialty in factory_specialty or factory_specialty in product_specialty:
                    suitable_factories.append(factory)
                    continue

            # 如果没有特殊匹配，加入所有工厂
            if not suitable_factories:
                suitable_factories.append(factory)

        if suitable_factories:
            # 选择经验最丰富的工厂
            best_factory = max(suitable_factories, key=lambda f:
                int(f.get('attributes', {}).get('experience', '0').replace('年', '').replace('years', '').strip() or '0'))
            return best_factory.get('name', '待定供应商')

        return '待定供应商'

    def generate_recommendation_for_inquiry(self, inquiry: Dict, kb_data: Dict) -> Optional[Recommendation]:
        """为单个询盘生成推荐"""
        try:
            # 查找相似产品
            similar_products = self.find_similar_products(inquiry, kb_data["products"])

            if not similar_products:
                logger.warning(f"No similar products found for inquiry {inquiry['id']}")
                return None

            # 生成产品推荐
            product_recommendations = []
            for product, similarity_score in similar_products[:3]:  # 推荐3个产品
                price_range = self.get_price_range_for_product(product, kb_data["quotes"])
                recommended_supplier = self.get_recommended_supplier(product, kb_data["factories"])

                product_rec = ProductRecommendation(
                    product_id=product['id'],
                    product_name=product['name'],
                    similarity_score=similarity_score,
                    price_range=price_range,
                    recommended_supplier=recommended_supplier,
                    confidence_score=similarity_score,
                    reasons=[f"相似度: {similarity_score:.2f}"]
                )

                product_recommendations.append(asdict(product_rec))

            # 计算整体价格区间
            all_prices = []
            for product_rec in product_recommendations:
                min_price, max_price = product_rec['price_range']
                if min_price > 0 and max_price > 0:
                    all_prices.extend([min_price, max_price])

            if all_prices:
                overall_min_price = min(all_prices)
                overall_max_price = max(all_prices)
            else:
                overall_min_price = overall_max_price = 0.0

            # 计算置信度
            avg_similarity = np.mean([rec['similarity_score'] for rec in product_recommendations])
            confidence_score = min(avg_similarity, 0.95)  # 最大置信度0.95

            # 生成推荐原因
            reasons = []
            inquiry_features = self.extract_text_features(
                f"{inquiry.get('name', '')} {inquiry.get('description', '')}"
            )

            if inquiry_features["material"]:
                reasons.append(f"基于材料需求: {inquiry_features['material']}")

            if inquiry_features["application"]:
                reasons.append(f"针对应用场景: {inquiry_features['application']}")

            if len(product_recommendations) >= 2:
                reasons.append(f"提供{len(product_recommendations)}个相似产品选择")

            recommendation = Recommendation(
                inquiry_id=inquiry['id'],
                recommended_products=product_recommendations,
                recommended_suppliers=[
                    {'name': rec['recommended_supplier'], 'confidence': rec['confidence_score']}
                    for rec in product_recommendations
                ],
                recommended_price_range=(overall_min_price, overall_max_price),
                confidence_score=confidence_score,
                recommendation_type="product_recommendation",
                recommendation_reason="; ".join(reasons),
                expires_at=datetime.now() + timedelta(days=30),
                created_at=datetime.now()
            )

            return recommendation

        except Exception as e:
            logger.error(f"Failed to generate recommendation for inquiry {inquiry.get('id')}: {e}")
            return None

    def save_recommendation(self, recommendation: Recommendation) -> bool:
        """保存推荐到数据库"""
        try:
            cursor = self.conn.cursor()

            # 创建推荐表（如果不存在）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inquiry_id INTEGER,
                    recommended_products TEXT,
                    recommended_suppliers TEXT,
                    price_min REAL,
                    price_max REAL,
                    confidence_score REAL,
                    recommendation_type TEXT,
                    recommendation_reason TEXT,
                    expires_at TEXT,
                    created_at TEXT,
                    status TEXT DEFAULT 'active'
                )
            """)

            # 插入推荐记录
            cursor.execute("""
                INSERT INTO recommendations (
                    inquiry_id, recommended_products, recommended_suppliers,
                    price_min, price_max, confidence_score, recommendation_type,
                    recommendation_reason, expires_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                recommendation.inquiry_id,
                json.dumps(recommendation.recommended_products, ensure_ascii=False),
                json.dumps(recommendation.recommended_suppliers, ensure_ascii=False),
                recommendation.recommended_price_range[0],
                recommendation.recommended_price_range[1],
                recommendation.confidence_score,
                recommendation.recommendation_type,
                recommendation.recommendation_reason,
                recommendation.expires_at.isoformat(),
                recommendation.created_at.isoformat()
            ))

            self.conn.commit()
            logger.info(f"Saved recommendation for inquiry {recommendation.inquiry_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save recommendation: {e}")
            return False

    def generate_recommendations(self, mode: str = "daily") -> Dict:
        """生成推荐"""
        start_time = datetime.now()

        try:
            logger.info(f"Starting recommendation generation ({mode} mode)")

            # 获取数据
            inquiries = self.get_recent_inquiries()
            kb_data = self.get_knowledge_base_for_recommendation()

            if not inquiries:
                logger.warning("No inquiries found for recommendation")
                return {"status": "no_inquiries"}

            if not kb_data["products"]:
                logger.warning("No products found in knowledge base")
                return {"status": "no_products"}

            # 生成推荐
            recommendations = []
            for inquiry in inquiries:
                recommendation = self.generate_recommendation_for_inquiry(inquiry, kb_data)
                if recommendation:
                    self.save_recommendation(recommendation)
                    recommendations.append(recommendation)

            # 更新统计信息
            self.recommendation_stats["total_inquiries_processed"] = len(inquiries)
            self.recommendation_stats["total_recommendations_generated"] = len(recommendations)

            if recommendations:
                avg_confidence = np.mean([r.confidence_score for r in recommendations])
                self.recommendation_stats["avg_confidence_score"] = avg_confidence

            self.recommendation_stats["processing_time"] = (datetime.now() - start_time).total_seconds()

            # 保存统计信息
            self.save_stats()

            logger.info(f"Generated {len(recommendations)} recommendations")
            logger.info(f"Average confidence score: {self.recommendation_stats['avg_confidence_score']:.2f}")

            return {
                "status": "success",
                "recommendations_count": len(recommendations),
                "inquiries_processed": len(inquiries),
                "avg_confidence": self.recommendation_stats["avg_confidence_score"],
                "processing_time": self.recommendation_stats["processing_time"]
            }

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            self.recommendation_stats["error_count"] += 1
            return {"status": "error", "message": str(e)}

    def save_stats(self) -> bool:
        """保存统计信息"""
        try:
            stats_file = f"recommendation_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.recommendation_stats, f, indent=2, ensure_ascii=False)

            logger.info(f"Recommendation stats saved to: {stats_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save stats: {e}")
            return False

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="生成智能推荐")
    parser.add_argument("--mode", choices=["daily", "weekly"], default="daily",
                       help="推荐生成模式")
    parser.add_argument("--db-path", default="knowledge_base.db", help="数据库路径")

    args = parser.parse_args()

    logger.info("🚀 Starting recommendation generation...")

    # 创建推荐生成器
    generator = RecommendationGenerator(args.db_path)

    try:
        # 连接数据库
        if not generator.connect_database():
            sys.exit(1)

        # 生成推荐
        result = generator.generate_recommendations(args.mode)

        if result["status"] == "success":
            logger.info("✅ Recommendation generation completed successfully!")
            logger.info(f"Generated {result['recommendations_count']} recommendations")
            logger.info(f"Average confidence: {result['avg_confidence']:.2f}")
            logger.info(f"Processing time: {result['processing_time']:.2f}s")
        else:
            logger.error(f"❌ Recommendation generation failed: {result}")

    finally:
        generator.close()

if __name__ == "__main__":
    main()