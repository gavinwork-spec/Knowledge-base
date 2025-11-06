#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Chat Interface - 聊天接口集成
通过自然语言查询客户、产品、报价历史等知识库信息

This script provides a Flask API extension for natural language querying
of the knowledge base using semantic search and intelligent response generation.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np

# Import our knowledge modules
from build_embeddings import EmbeddingIndexBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/processed/chat_interface.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChatInterface:
    """聊天接口处理器"""

    def __init__(self, db_path: str = "knowledge_base.db"):
        self.db_path = db_path
        self.conn = None
        self.embedding_builder = None

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

    def init_embedding_search(self):
        """初始化嵌入搜索"""
        try:
            self.embedding_builder = EmbeddingIndexBuilder()
            self.embedding_builder.connect()
            logger.info("Embedding search initialized")
        except Exception as e:
            logger.error(f"Failed to initialize embedding search: {e}")

    def parse_user_query(self, query: str) -> Dict[str, Any]:
        """解析用户查询意图"""
        query_lower = query.lower()
        parsed_query = {
            'original_query': query,
            'intent': 'general_search',
            'entities': [],
            'filters': {},
            'keywords': []
        }

        # 识别查询意图
        if any(keyword in query_lower for keyword in ['客户', 'customer', '公司']):
            parsed_query['intent'] = 'customer_search'
            parsed_query['filters']['entity_types'] = ['customer']

        elif any(keyword in query_lower for keyword in ['报价', 'quote', '价格', 'price']):
            parsed_query['intent'] = 'quote_search'
            parsed_query['filters']['entity_types'] = ['quote']

        elif any(keyword in query_lower for keyword in ['产品', 'product', '规格', 'specification']):
            parsed_query['intent'] = 'product_search'
            parsed_query['filters']['entity_types'] = ['product', 'specification']

        elif any(keyword in query_lower for keyword in ['工厂', 'factory', '供应商', 'supplier']):
            parsed_query['intent'] = 'factory_search'
            parsed_query['filters']['entity_types'] = ['factory']

        elif any(keyword in query_lower for keyword in ['询价', 'inquiry', '需求']):
            parsed_query['intent'] = 'inquiry_search'
            parsed_query['filters']['entity_types'] = ['inquiry']

        # 提取关键词
        keywords = self._extract_keywords(query)
        parsed_query['keywords'] = keywords

        # 提取实体（如产品规格、材料等）
        entities = self._extract_entities(query)
        parsed_query['entities'] = entities

        return parsed_query

    def _extract_keywords(self, query: str) -> List[str]:
        """提取查询关键词"""
        # 移除停用词并提取重要关键词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}

        words = re.findall(r'[\w\u4e00-\u9fff]+', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 1]

        return keywords[:10]  # 限制关键词数量

    def _extract_entities(self, query: str) -> List[Dict[str, str]]:
        """提取实体信息"""
        entities = []

        # 提取规格信息
        spec_patterns = [
            (r'm\d+\.?\d*', 'specification'),
            (r'φ\d+\.?\d*', 'specification'),
            (r'\d+\.?\d*mm', 'specification'),
            (r'gb/t\s*\d+', 'standard'),
            (r'iso\s*\d+', 'standard'),
        ]

        for pattern, entity_type in spec_patterns:
            matches = re.findall(pattern, query.lower())
            for match in matches:
                entities.append({
                    'type': entity_type,
                    'value': match,
                    'confidence': 0.9
                })

        # 提取材料信息
        materials = ['不锈钢', '碳钢', '合金钢', '铜', '铝', '304', '316', '45#']
        for material in materials:
            if material in query.lower():
                entities.append({
                    'type': 'material',
                    'value': material,
                    'confidence': 0.8
                })

        return entities

    def search_knowledge_base(self, parsed_query: Dict) -> List[Dict]:
        """搜索知识库"""
        try:
            # 使用嵌入搜索
            if self.embedding_builder:
                semantic_results = self.embedding_builder.find_similar_entries(
                    parsed_query['original_query'],
                    top_k=10,
                    min_similarity=0.01
                )

                # 应用过滤器
                if parsed_query['filters'].get('entity_types'):
                    semantic_results = [
                        result for result in semantic_results
                        if result['entity_type'] in parsed_query['filters']['entity_types']
                    ]

                # 获取详细信息
                detailed_results = []
                for result in semantic_results:
                    entry_details = self._get_knowledge_entry_details(result['entry_id'])
                    if entry_details:
                        entry_details['similarity_score'] = result['similarity']
                        detailed_results.append(entry_details)

                return detailed_results

            # 如果没有嵌入搜索，使用传统搜索
            return self._traditional_search(parsed_query)

        except Exception as e:
            logger.error(f"Failed to search knowledge base: {e}")
            return []

    def _traditional_search(self, parsed_query: Dict) -> List[Dict]:
        """传统文本搜索"""
        try:
            cursor = self.conn.cursor()

            # 构建搜索查询
            query_conditions = []
            params = []

            # 实体类型过滤
            if parsed_query['filters'].get('entity_types'):
                placeholders = ','.join(['?'] * len(parsed_query['filters']['entity_types']))
                query_conditions.append(f"ke.entity_type IN ({placeholders})")
                params.extend(parsed_query['filters']['entity_types'])

            # 关键词搜索
            if parsed_query['keywords']:
                keyword_conditions = []
                for keyword in parsed_query['keywords']:
                    keyword_conditions.append("(ke.name LIKE ? OR ke.description LIKE ?)")
                    params.extend([f"%{keyword}%", f"%{keyword}%"])
                if keyword_conditions:
                    query_conditions.append(f"({' OR '.join(keyword_conditions)})")

            # 构建完整查询
            if query_conditions:
                where_clause = " AND ".join(query_conditions)
                query = f"""
                    SELECT ke.id, ke.entity_type, ke.name, ke.description, ke.attributes_json,
                           ke.created_at, et.display_name, et.color, et.icon
                    FROM knowledge_entries ke
                    JOIN entity_types et ON ke.entity_type = et.name
                    WHERE {where_clause}
                    ORDER BY ke.created_at DESC
                    LIMIT 20
                """
            else:
                query = """
                    SELECT ke.id, ke.entity_type, ke.name, ke.description, ke.attributes_json,
                           ke.created_at, et.display_name, et.color, et.icon
                    FROM knowledge_entries ke
                    JOIN entity_types et ON ke.entity_type = et.name
                    ORDER BY ke.created_at DESC
                    LIMIT 10
                """

            cursor.execute(query, params)
            results = []

            for row in cursor.fetchall():
                entry = {
                    'id': row[0],
                    'entity_type': row[1],
                    'name': row[2],
                    'description': row[3],
                    'attributes': json.loads(row[4]) if row[4] else {},
                    'created_at': row[5],
                    'entity_type_display': row[6],
                    'entity_color': row[7],
                    'entity_icon': row[8],
                    'similarity_score': 0.5  # 传统搜索固定分数
                }
                results.append(entry)

            return results

        except Exception as e:
            logger.error(f"Failed to perform traditional search: {e}")
            return []

    def _get_knowledge_entry_details(self, entry_id: int) -> Optional[Dict]:
        """获取知识条目详细信息"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT ke.id, ke.entity_type, ke.name, ke.description, ke.attributes_json,
                       ke.created_at, et.display_name, et.color, et.icon
                FROM knowledge_entries ke
                JOIN entity_types et ON ke.entity_type = et.name
                WHERE ke.id = ?
            """, (entry_id,))

            row = cursor.fetchone()
            if not row:
                return None

            entry = {
                'id': row[0],
                'entity_type': row[1],
                'name': row[2],
                'description': row[3],
                'attributes': json.loads(row[4]) if row[4] else {},
                'created_at': row[5],
                'entity_type_display': row[6],
                'entity_color': row[7],
                'entity_icon': row[8]
            }

            # 获取相关的NLP实体
            cursor.execute("""
                SELECT keyword, value, category, confidence_score
                FROM nlp_entities
                WHERE entry_id = ?
                ORDER BY confidence_score DESC
                LIMIT 5
            """, (entry_id,))
            entry['nlp_entities'] = [dict(zip(['keyword', 'value', 'category', 'confidence'], row))
                                 for row in cursor.fetchall()]

            # 获取相关的策略建议
            cursor.execute("""
                SELECT title, description, confidence_score, potential_savings
                FROM strategy_suggestions
                WHERE related_entry_id = ?
                ORDER BY created_at DESC
                LIMIT 3
            """, (entry_id,))
            entry['strategy_suggestions'] = [dict(zip(['title', 'description', 'confidence_score', 'potential_savings'], row))
                                            for row in cursor.fetchall()]

            return entry

        except Exception as e:
            logger.error(f"Failed to get knowledge entry details: {e}")
            return None

    def generate_response(self, parsed_query: Dict, search_results: List[Dict]) -> Dict[str, Any]:
        """生成响应"""
        try:
            if not search_results:
                return {
                    'status': 'no_results',
                    'message': f"很抱歉，没有找到关于'{parsed_query['original_query']}'的相关信息。",
                    'suggestions': [
                        "尝试使用不同的关键词",
                        "检查拼写是否正确",
                        "提供更具体的描述"
                    ],
                    'results': []
                }

            # 根据查询意图生成结构化响应
            response = {
                'status': 'success',
                'query_intent': parsed_query['intent'],
                'query_entities': parsed_query['entities'],
                'total_results': len(search_results),
                'results': search_results,
                'summary': self._generate_summary(parsed_query, search_results),
                'related_topics': self._find_related_topics(search_results)
            }

            return response

        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return {
                'status': 'error',
                'message': "生成响应时出现错误，请稍后重试。",
                'results': []
            }

    def _generate_summary(self, parsed_query: Dict, search_results: List[Dict]) -> str:
        """生成结果摘要"""
        try:
            if not search_results:
                return "没有找到相关信息"

            # 按实体类型分组
            type_counts = {}
            for result in search_results:
                entity_type = result['entity_type_display']
                type_counts[entity_type] = type_counts.get(entity_type, 0) + 1

            # 构建摘要
            summary_parts = []
            summary_parts.append(f"找到 {len(search_results)} 条相关信息")

            for entity_type, count in type_counts.items():
                summary_parts.append(f"{count} 个{entity_type}")

            if len(search_results) > 0:
                top_result = search_results[0]
                if top_result['similarity_score'] > 0.7:
                    summary_parts.append(f"最相关的结果：{top_result['name']}")

            return "，".join(summary_parts) + "。"

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return f"找到 {len(search_results)} 条相关信息。"

    def _find_related_topics(self, search_results: List[Dict]) -> List[str]:
        """查找相关话题"""
        try:
            topics = set()

            for result in search_results:
                # 从实体类型生成话题
                topics.add(result['entity_type_display'])

                # 从属性生成话题
                if result.get('attributes'):
                    for key, value in result['attributes'].items():
                        if key in ['materials', 'specifications', 'industry']:
                            if isinstance(value, list):
                                topics.update(value)
                            elif value:
                                topics.add(str(value))

            return list(topics)[:8]  # 限制话题数量

        except Exception as e:
            logger.error(f"Failed to find related topics: {e}")
            return []

    def process_query(self, query: str) -> Dict[str, Any]:
        """处理用户查询"""
        try:
            logger.info(f"Processing query: {query}")

            # 解析查询
            parsed_query = self.parse_user_query(query)

            # 搜索知识库
            search_results = self.search_knowledge_base(parsed_query)

            # 生成响应
            response = self.generate_response(parsed_query, search_results)

            # 记录查询日志
            self._log_query(query, parsed_query, response)

            return response

        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            return {
                'status': 'error',
                'message': "处理查询时出现错误，请稍后重试。",
                'results': []
            }

    def _log_query(self, query: str, parsed_query: Dict, response: Dict):
        """记录查询日志"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'intent': parsed_query['intent'],
                'keywords': parsed_query['keywords'],
                'result_count': response.get('total_results', 0),
                'status': response.get('status', 'unknown')
            }

            log_file = Path("data/processed/chat_query_log.json")
            log_file.parent.mkdir(parents=True, exist_ok=True)

            # 读取现有日志
            existing_logs = []
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        existing_logs = json.load(f)
                except:
                    existing_logs = []

            # 添加新日志（保留最近1000条）
            existing_logs.append(log_entry)
            if len(existing_logs) > 1000:
                existing_logs = existing_logs[-1000:]

            # 保存日志
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_logs, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Failed to log query: {e}")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create global chat interface instance
chat_interface = ChatInterface()

def initialize():
    """初始化应用"""
    chat_interface.connect()
    chat_interface.init_embedding_search()
    logger.info("🚀 Chat Interface API started successfully!")

@app.teardown_appcontext
def teardown_db(error):
    """清理数据库连接"""
    pass

# Chat endpoint
@app.route('/api/v1/chat/query', methods=['POST'])
def chat_query():
    """处理聊天查询"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': 'Query is required'
            }), 400

        query = data['query'].strip()
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400

        # 处理查询
        response = chat_interface.process_query(query)

        return jsonify({
            'success': True,
            'data': response
        })

    except Exception as e:
        logger.error(f"Chat query error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Health check endpoint
@app.route('/api/v1/chat/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'service': 'chat-interface',
        'timestamp': datetime.now().isoformat()
    })

# Query statistics endpoint
@app.route('/api/v1/chat/stats', methods=['GET'])
def get_chat_stats():
    """获取聊天统计"""
    try:
        log_file = Path("data/processed/chat_query_log.json")
        if not log_file.exists():
            return jsonify({
                'success': True,
                'data': {
                    'total_queries': 0,
                    'daily_queries': 0,
                    'popular_intents': {},
                    'success_rate': 0
                }
            })

        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)

        # 计算统计数据
        total_queries = len(logs)
        today = datetime.now().date()
        daily_queries = len([log for log in logs if log['timestamp'].split('T')[0] == str(today)])

        # 统计意图分布
        intent_counts = {}
        successful_queries = 0
        for log in logs:
            intent = log.get('intent', 'unknown')
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            if log.get('status') == 'success':
                successful_queries += 1

        success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0

        return jsonify({
            'success': True,
            'data': {
                'total_queries': total_queries,
                'daily_queries': daily_queries,
                'popular_intents': intent_counts,
                'success_rate': round(success_rate, 2)
            }
        })

    except Exception as e:
        logger.error(f"Failed to get chat stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Chat Interface API')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8002, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    # 初始化应用
    initialize()

    logger.info(f"🚀 Starting Chat Interface API on {args.host}:{args.port}")

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )

if __name__ == "__main__":
    main()