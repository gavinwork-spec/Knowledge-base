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

# Root endpoint - Return chat interface HTML
@app.route('/', methods=['GET'])
def root():
    chat_html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XAgent 制造业智能聊天界面</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 1rem 2rem;
            box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
            z-index: 100;
        }

        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1.5rem;
            font-weight: bold;
            color: #2563eb;
        }

        .status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: #10b981;
            color: white;
            border-radius: 20px;
            font-size: 0.875rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: white;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .main-container {
            flex: 1;
            display: flex;
            max-width: 1200px;
            width: 100%;
            margin: 2rem auto;
            gap: 2rem;
            padding: 0 2rem;
        }

        .sidebar {
            width: 300px;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            height: fit-content;
        }

        .sidebar h3 {
            color: #1f2937;
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }

        .agent-list {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .agent-item {
            padding: 0.75rem;
            background: #f3f4f6;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            border-left: 3px solid transparent;
        }

        .agent-item:hover {
            background: #e5e7eb;
            transform: translateX(5px);
        }

        .agent-item.active {
            background: #dbeafe;
            border-left-color: #2563eb;
        }

        .agent-item .agent-name {
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 0.25rem;
        }

        .agent-item .agent-desc {
            font-size: 0.75rem;
            color: #6b7280;
        }

        .chat-container {
            flex: 1;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .chat-header {
            padding: 1.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .chat-header h2 {
            margin-bottom: 0.5rem;
        }

        .chat-header p {
            opacity: 0.9;
            font-size: 0.875rem;
        }

        .chat-messages {
            flex: 1;
            padding: 1.5rem;
            overflow-y: auto;
            background: #f9fafb;
            min-height: 400px;
            max-height: 500px;
        }

        .message {
            margin-bottom: 1rem;
            display: flex;
            gap: 0.75rem;
        }

        .message.user {
            flex-direction: row-reverse;
        }

        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
            flex-shrink: 0;
        }

        .message.user .message-avatar {
            background: #2563eb;
            color: white;
        }

        .message.assistant .message-avatar {
            background: #10b981;
            color: white;
        }

        .message-content {
            max-width: 70%;
            padding: 1rem;
            border-radius: 12px;
            position: relative;
        }

        .message.user .message-content {
            background: #2563eb;
            color: white;
            border-bottom-right-radius: 4px;
        }

        .message.assistant .message-content {
            background: white;
            color: #1f2937;
            border: 1px solid #e5e7eb;
            border-bottom-left-radius: 4px;
        }

        .message-time {
            font-size: 0.75rem;
            opacity: 0.7;
            margin-top: 0.25rem;
        }

        .chat-input {
            padding: 1.5rem;
            background: white;
            border-top: 1px solid #e5e7eb;
        }

        .input-container {
            display: flex;
            gap: 1rem;
            align-items: flex-end;
        }

        .input-wrapper {
            flex: 1;
            position: relative;
        }

        .chat-input textarea {
            width: 100%;
            min-height: 50px;
            max-height: 120px;
            padding: 1rem;
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            resize: none;
            font-family: inherit;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.2s;
        }

        .chat-input textarea:focus {
            border-color: #2563eb;
        }

        .send-button {
            padding: 1rem 2rem;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .send-button:hover {
            background: #1d4ed8;
            transform: translateY(-1px);
        }

        .send-button:disabled {
            background: #9ca3af;
            cursor: not-allowed;
            transform: none;
        }

        .typing-indicator {
            display: none;
            padding: 1rem;
            color: #6b7280;
            font-style: italic;
        }

        .typing-indicator.show {
            display: block;
        }

        .quick-actions {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }

        .quick-action {
            padding: 0.5rem 1rem;
            background: #f3f4f6;
            border: 1px solid #e5e7eb;
            border-radius: 20px;
            font-size: 0.875rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .quick-action:hover {
            background: #e5e7eb;
            transform: translateY(-1px);
        }

        @media (max-width: 768px) {
            .main-container {
                flex-direction: column;
                margin: 1rem;
                padding: 0 1rem;
            }

            .sidebar {
                width: 100%;
                order: 2;
            }

            .chat-container {
                order: 1;
            }

            .message-content {
                max-width: 85%;
            }
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="header-content">
            <div class="logo">
                🏭 XAgent 制造业智能系统
            </div>
            <div class="status">
                <div class="status-dot"></div>
                <span>系统运行中</span>
            </div>
        </div>
    </header>

    <div class="main-container">
        <aside class="sidebar">
            <h3>🤖 选择智能体</h3>
            <div class="agent-list">
                <div class="agent-item active" data-agent="safety">
                    <div class="agent-name">🛡️ 安全检查员</div>
                    <div class="agent-desc">Safety Inspector</div>
                </div>
                <div class="agent-item" data-agent="quality">
                    <div class="agent-name">🎯 质量控制器</div>
                    <div class="agent-desc">Quality Controller</div>
                </div>
                <div class="agent-item" data-agent="maintenance">
                    <div class="agent-name">🔧 维护技术员</div>
                    <div class="agent-desc">Maintenance Technician</div>
                </div>
                <div class="agent-item" data-agent="production">
                    <div class="agent-name">📊 生产经理</div>
                    <div class="agent-desc">Production Manager</div>
                </div>
            </div>
        </aside>

        <div class="chat-container">
            <div class="chat-header">
                <h2 id="current-agent-name">🛡️ 安全检查员</h2>
                <p id="current-agent-desc">制造业安全标准检查与风险评估专家</p>
            </div>

            <div class="chat-messages" id="chat-messages">
                <div class="message assistant">
                    <div class="message-avatar">🛡️</div>
                    <div class="message-content">
                        <div>您好！我是安全检查员智能体。我可以帮助您：</div>
                        <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
                            <li>进行安全标准检查</li>
                            <li>评估工作场所风险</li>
                            <li>提供安全建议</li>
                            <li>分析安全合规性</li>
                        </ul>
                        <div style="margin-top: 0.5rem;">请问有什么可以帮助您的吗？</div>
                        <div class="message-time">刚刚</div>
                    </div>
                </div>
            </div>

            <div class="chat-input">
                <div class="quick-actions">
                    <div class="quick-action">检查设备安全</div>
                    <div class="quick-action">评估风险</div>
                    <div class="quick-action">安全标准咨询</div>
                    <div class="quick-action">应急准备</div>
                </div>
                <div class="input-container">
                    <div class="input-wrapper">
                        <textarea id="message-input" placeholder="请输入您的问题..." rows="1"></textarea>
                    </div>
                    <button class="send-button" id="send-button">
                        <span>发送</span>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="22" y1="2" x2="11" y2="13"></line>
                            <polygon points="22,2 15,22 11,13 2,9 22,2"></polygon>
                        </svg>
                    </button>
                </div>
                <div class="typing-indicator" id="typing-indicator">智能体正在思考...</div>
            </div>
        </div>
    </div>

    <script>
        // 智能体配置
        const agents = {
            safety: {
                name: '🛡️ 安全检查员',
                desc: '制造业安全标准检查与风险评估专家',
                avatar: '🛡️',
                greeting: '您好！我是安全检查员智能体。我可以帮助您：\\n• 进行安全标准检查\\n• 评估工作场所风险\\n• 提供安全建议\\n• 分析安全合规性\\n\\n请问有什么可以帮助您的吗？'
            },
            quality: {
                name: '🎯 质量控制器',
                desc: '产品质量管理与标准控制专家',
                avatar: '🎯',
                greeting: '您好！我是质量控制器智能体。我可以帮助您：\\n• 分析产品质量标准\\n• 提供质量控制方案\\n• 评估质量管理体系\\n• 优化检测流程\\n\\n请问有什么质量方面的问题需要咨询吗？'
            },
            maintenance: {
                name: '🔧 维护技术员',
                desc: '设备维护与故障诊断专家',
                avatar: '🔧',
                greeting: '您好！我是维护技术员智能体。我可以帮助您：\\n• 诊断设备故障\\n• 制定维护计划\\n• 提供维修建议\\n• 优化设备性能\\n\\n请问有什么设备维护方面的问题吗？'
            },
            production: {
                name: '📊 生产经理',
                desc: '生产计划与流程优化专家',
                avatar: '📊',
                greeting: '您好！我是生产经理智能体。我可以帮助您：\\n• 制定生产计划\\n• 优化生产流程\\n• 分析生产数据\\n• 提高生产效率\\n\\n请问有什么生产管理方面的问题需要咨询吗？'
            }
        };

        let currentAgent = 'safety';
        let isTyping = false;

        // DOM元素
        const chatMessages = document.getElementById('chat-messages');
        const messageInput = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');
        const typingIndicator = document.getElementById('typing-indicator');
        const currentAgentName = document.getElementById('current-agent-name');
        const currentAgentDesc = document.getElementById('current-agent-desc');

        // 智能体切换
        document.querySelectorAll('.agent-item').forEach(item => {
            item.addEventListener('click', function() {
                document.querySelectorAll('.agent-item').forEach(el => el.classList.remove('active'));
                this.classList.add('active');

                currentAgent = this.dataset.agent;
                const agent = agents[currentAgent];

                currentAgentName.textContent = agent.name;
                currentAgentDesc.textContent = agent.desc;

                // 添加切换消息
                addMessage('assistant', agent.greeting, agent.avatar);
            });
        });

        // 快速操作
        document.querySelectorAll('.quick-action').forEach(action => {
            action.addEventListener('click', function() {
                messageInput.value = this.textContent;
                messageInput.focus();
            });
        });

        // 发送消息
        function sendMessage() {
            const message = messageInput.value.trim();
            if (!message || isTyping) return;

            addMessage('user', message, '👤');
            messageInput.value = '';
            messageInput.style.height = 'auto';

            // 显示输入指示器
            showTyping();

            // 调用实际的API
            fetch('/api/v1/chat/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: message,
                    agent_type: currentAgent
                })
            })
            .then(response => response.json())
            .then(data => {
                hideTyping();
                if (data.success) {
                    addMessage('assistant', data.response, agents[currentAgent].avatar);
                } else {
                    addMessage('assistant', '抱歉，处理您的请求时遇到了问题。请稍后再试。', agents[currentAgent].avatar);
                }
            })
            .catch(error => {
                hideTyping();
                console.error('Error:', error);
                // 如果API调用失败，使用模拟响应
                const response = generateResponse(message, currentAgent);
                addMessage('assistant', response, agents[currentAgent].avatar);
            });
        }

        // 生成模拟响应
        function generateResponse(message, agentType) {
            const responses = {
                safety: [
                    '根据安全标准检查，我建议您重点关注设备防护措施和员工培训。',
                    '我已经分析了您提到的安全风险，建议立即采取以下防护措施...',
                    '关于安全合规性，请确保符合OSHA标准和行业规范。',
                    '我建议您进行定期的安全审计和风险评估。'
                ],
                quality: [
                    '根据质量管理体系要求，建议您加强过程控制点监控。',
                    '产品质量分析显示需要关注关键参数的稳定性。',
                    '建议采用统计过程控制(SPC)方法来提升质量水平。',
                    '质量改进建议：优化检测流程，加强供应商质量管理。'
                ],
                maintenance: [
                    '根据设备运行数据，建议您制定预防性维护计划。',
                    '故障分析表明需要定期检查关键部件的磨损情况。',
                    '建议采用预测性维护技术来减少停机时间。',
                    '设备维护记录显示需要加强润滑和清洁保养。'
                ],
                production: [
                    '生产效率分析建议优化生产流程和资源配置。',
                    '根据生产数据，建议调整生产计划以提高产能利用率。',
                    '建议采用精益生产方法来减少浪费和提高效率。',
                    '生产计划需要考虑设备维护窗口和物料供应情况。'
                ]
            };

            const agentResponses = responses[agentType] || responses.safety;
            return agentResponses[Math.floor(Math.random() * agentResponses.length)];
        }

        // 添加消息
        function addMessage(type, content, avatar) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;

            const time = new Date().toLocaleTimeString('zh-CN', {
                hour: '2-digit',
                minute: '2-digit'
            });

            messageDiv.innerHTML = `
                <div class="message-avatar">${avatar}</div>
                <div class="message-content">
                    <div>${content.replace(/\\n/g, '<br>')}</div>
                    <div class="message-time">${time}</div>
                </div>
            `;

            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        // 显示/隐藏输入指示器
        function showTyping() {
            isTyping = true;
            typingIndicator.classList.add('show');
            sendButton.disabled = true;
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function hideTyping() {
            isTyping = false;
            typingIndicator.classList.remove('show');
            sendButton.disabled = false;
        }

        // 事件监听
        sendButton.addEventListener('click', sendMessage);

        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // 自动调整文本框高度
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });

        // 页面加载完成后的初始化
        window.addEventListener('load', () => {
            messageInput.focus();
        });
    </script>
</body>
</html>"""

    return chat_html, 200, {'Content-Type': 'text/html; charset=utf-8'}

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({
        'service': 'chat-interface-api',
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'success': True
    })

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