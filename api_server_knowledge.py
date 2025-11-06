#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Base API Server
知识库API服务器

This server provides RESTful API endpoints for accessing the knowledge base,
including search, retrieval, and management of knowledge entries.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import os
import sys

# Import our knowledge modules
from build_embeddings import EmbeddingIndexBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/processed/api_server_knowledge.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
DB_PATH = "knowledge_base.db"
STATIC_FOLDER = "github-frontend"

# Global instances
embedding_builder = None

class KnowledgeAPI:
    """知识库API处理器"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        """连接数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
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

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """执行查询并返回结果"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)

            columns = [desc[0] for desc in cursor.description]
            results = []

            for row in cursor.fetchall():
                result = dict(zip(columns, row))
                results.append(result)

            return results
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []

    def get_knowledge_entries(self, filters: Dict = None, limit: int = 50, offset: int = 0) -> Dict:
        """获取知识条目"""
        try:
            # 构建查询条件
            where_conditions = []
            params = []

            if filters:
                if filters.get('entity_type'):
                    where_conditions.append("ke.entity_type = ?")
                    params.append(filters['entity_type'])

                if filters.get('search'):
                    where_conditions.append("(ke.name LIKE ? OR ke.description LIKE ?)")
                    search_term = f"%{filters['search']}%"
                    params.extend([search_term, search_term])

            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"

            # 主查询
            query = f"""
                SELECT
                    ke.id,
                    ke.entity_type,
                    ke.name,
                    ke.description,
                    ke.attributes_json,
                    ke.created_at,
                    ke.updated_at,
                    et.display_name as entity_type_display,
                    et.color as entity_color,
                    et.icon as entity_icon
                FROM knowledge_entries ke
                JOIN entity_types et ON ke.entity_type = et.name
                WHERE {where_clause}
                ORDER BY ke.updated_at DESC
                LIMIT ? OFFSET ?
            """

            params.extend([limit, offset])
            entries = self.execute_query(query, tuple(params))

            # 获取总数
            count_query = f"""
                SELECT COUNT(*) as total
                FROM knowledge_entries ke
                WHERE {where_clause}
            """
            count_params = params[:-2]  # Remove limit and offset
            count_result = self.execute_query(count_query, tuple(count_params))
            total = count_result[0]['total'] if count_result else 0

            # 解析属性JSON
            for entry in entries:
                if entry['attributes_json']:
                    try:
                        entry['attributes'] = json.loads(entry['attributes_json'])
                    except json.JSONDecodeError:
                        entry['attributes'] = {}
                else:
                    entry['attributes'] = {}

            return {
                'entries': entries,
                'total': total,
                'limit': limit,
                'offset': offset
            }

        except Exception as e:
            logger.error(f"Failed to get knowledge entries: {e}")
            return {'entries': [], 'total': 0, 'limit': limit, 'offset': offset}

    def get_knowledge_entry(self, entry_id: int) -> Optional[Dict]:
        """获取单个知识条目详情"""
        try:
            query = """
                SELECT
                    ke.id,
                    ke.entity_type,
                    ke.name,
                    ke.description,
                    ke.attributes_json,
                    ke.created_at,
                    ke.updated_at,
                    et.display_name as entity_type_display,
                    et.color as entity_color,
                    et.icon as entity_icon
                FROM knowledge_entries ke
                JOIN entity_types et ON ke.entity_type = et.name
                WHERE ke.id = ?
            """

            results = self.execute_query(query, (entry_id,))

            if not results:
                return None

            entry = results[0]

            # 解析属性
            if entry['attributes_json']:
                try:
                    entry['attributes'] = json.loads(entry['attributes_json'])
                except json.JSONDecodeError:
                    entry['attributes'] = {}
            else:
                entry['attributes'] = {}

            # 获取相关NLP实体
            nlp_query = """
                SELECT keyword, value, category, confidence_score, context_text
                FROM nlp_entities
                WHERE entry_id = ?
                ORDER BY confidence_score DESC
            """
            entry['nlp_entities'] = self.execute_query(nlp_query, (entry_id,))

            # 获取策略建议
            strategy_query = """
                SELECT id, suggestion_type, title, description, impact_level,
                       potential_savings, confidence_score, status, created_at
                FROM strategy_suggestions
                WHERE related_entry_id = ?
                ORDER BY created_at DESC
            """
            entry['strategy_suggestions'] = self.execute_query(strategy_query, (entry_id,))

            return entry

        except Exception as e:
            logger.error(f"Failed to get knowledge entry {entry_id}: {e}")
            return None

    def search_knowledge(self, query: str, top_k: int = 10, entity_types: List[str] = None) -> List[Dict]:
        """搜索知识库"""
        try:
            global embedding_builder
            if not embedding_builder:
                embedding_builder = EmbeddingIndexBuilder()

            # 使用嵌入搜索
            semantic_results = embedding_builder.find_similar_entries(query, top_k)

            # 如果指定了实体类型，过滤结果
            if entity_types:
                semantic_results = [
                    result for result in semantic_results
                    if result['entity_type'] in entity_types
                ]

            # 获取详细信息
            enhanced_results = []
            for result in semantic_results:
                entry_details = self.get_knowledge_entry(result['entry_id'])
                if entry_details:
                    entry_details['similarity_score'] = result['similarity']
                    enhanced_results.append(entry_details)

            # 如果嵌入搜索结果不足，使用文本搜索补充
            if len(enhanced_results) < top_k:
                text_filters = {'search': query}
                if entity_types:
                    # 使用第一个实体类型进行补充搜索
                    text_filters['entity_type'] = entity_types[0]

                text_results = self.get_knowledge_entries(
                    filters=text_filters,
                    limit=top_k - len(enhanced_results)
                )

                for entry in text_results['entries']:
                    # 避免重复
                    if not any(r['id'] == entry['id'] for r in enhanced_results):
                        entry['similarity_score'] = 0.5  # 文本匹配的基础分数
                        enhanced_results.append(entry)

            return enhanced_results[:top_k]

        except Exception as e:
            logger.error(f"Failed to search knowledge: {e}")
            return []

    def get_entity_types(self) -> List[Dict]:
        """获取所有实体类型"""
        try:
            query = """
                SELECT name, display_name, description, color, icon,
                       (SELECT COUNT(*) FROM knowledge_entries WHERE entity_type = entity_types.name) as entry_count
                FROM entity_types
                WHERE is_active = 1
                ORDER BY display_name
            """
            return self.execute_query(query)
        except Exception as e:
            logger.error(f"Failed to get entity types: {e}")
            return []

    def get_knowledge_stats(self) -> Dict:
        """获取知识库统计信息"""
        try:
            stats = {}

            # 总条目数
            total_query = "SELECT COUNT(*) as total FROM knowledge_entries"
            total_result = self.execute_query(total_query)
            stats['total_entries'] = total_result[0]['total'] if total_result else 0

            # 按类型统计
            type_query = """
                SELECT ke.entity_type, et.display_name, COUNT(*) as count
                FROM knowledge_entries ke
                JOIN entity_types et ON ke.entity_type = et.name
                GROUP BY ke.entity_type, et.display_name
                ORDER BY count DESC
            """
            stats['by_type'] = self.execute_query(type_query)

            # 最近更新
            recent_query = """
                SELECT entity_type, name, updated_at
                FROM knowledge_entries
                ORDER BY updated_at DESC
                LIMIT 5
            """
            stats['recent_updates'] = self.execute_query(recent_query)

            # 热门关键词
            keywords_query = """
                SELECT keyword, category, COUNT(*) as usage_count
                FROM nlp_entities
                WHERE confidence_score > 0.5
                GROUP BY keyword, category
                HAVING COUNT(*) > 1
                ORDER BY usage_count DESC
                LIMIT 10
            """
            stats['popular_keywords'] = self.execute_query(keywords_query)

            return stats

        except Exception as e:
            logger.error(f"Failed to get knowledge stats: {e}")
            return {}

# Create global API instance
knowledge_api = KnowledgeAPI()

def initialize():
    """初始化应用"""
    knowledge_api.connect()
    logger.info("🚀 Knowledge Base API Server started successfully!")

@app.teardown_appcontext
def teardown_db(error):
    """清理数据库连接"""
    pass  # We keep the connection open for performance

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'knowledge-base-api',
        'version': '1.0.0'
    })

# Knowledge entries endpoints
@app.route('/api/knowledge/entries', methods=['GET'])
def get_knowledge_entries():
    """获取知识条目列表"""
    try:
        # 获取查询参数
        filters = {}

        if request.args.get('entity_type'):
            filters['entity_type'] = request.args.get('entity_type')

        if request.args.get('search'):
            filters['search'] = request.args.get('search')

        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))

        result = knowledge_api.get_knowledge_entries(filters, limit, offset)

        return jsonify({
            'success': True,
            'data': result
        })

    except Exception as e:
        logger.error(f"API error in get_knowledge_entries: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/knowledge/entries/<int:entry_id>', methods=['GET'])
def get_knowledge_entry(entry_id):
    """获取单个知识条目详情"""
    try:
        entry = knowledge_api.get_knowledge_entry(entry_id)

        if not entry:
            return jsonify({
                'success': False,
                'error': 'Knowledge entry not found'
            }), 404

        return jsonify({
            'success': True,
            'data': entry
        })

    except Exception as e:
        logger.error(f"API error in get_knowledge_entry: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Search endpoints
@app.route('/api/knowledge/search', methods=['GET'])
def search_knowledge():
    """搜索知识库"""
    try:
        query = request.args.get('q', '').strip()

        if not query:
            return jsonify({
                'success': False,
                'error': 'Search query is required'
            }), 400

        top_k = int(request.args.get('top_k', 10))

        # 获取实体类型过滤
        entity_types = request.args.getlist('entity_type')

        results = knowledge_api.search_knowledge(query, top_k, entity_types)

        return jsonify({
            'success': True,
            'data': {
                'query': query,
                'results': results,
                'total': len(results)
            }
        })

    except Exception as e:
        logger.error(f"API error in search_knowledge: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Entity types endpoint
@app.route('/api/knowledge/entity-types', methods=['GET'])
def get_entity_types():
    """获取所有实体类型"""
    try:
        entity_types = knowledge_api.get_entity_types()

        return jsonify({
            'success': True,
            'data': entity_types
        })

    except Exception as e:
        logger.error(f"API error in get_entity_types: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Statistics endpoint
@app.route('/api/knowledge/stats', methods=['GET'])
def get_knowledge_stats():
    """获取知识库统计信息"""
    try:
        stats = knowledge_api.get_knowledge_stats()

        return jsonify({
            'success': True,
            'data': stats
        })

    except Exception as e:
        logger.error(f"API error in get_knowledge_stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Embedding management endpoints
@app.route('/api/knowledge/embeddings/build', methods=['POST'])
def build_embeddings():
    """重新构建嵌入索引"""
    try:
        global embedding_builder
        if not embedding_builder:
            embedding_builder = EmbeddingIndexBuilder()

        # 在后台线程中构建嵌入
        import threading

        def build_in_background():
            try:
                embedding_builder.build_embeddings(force_rebuild=True)
                logger.info("Embedding index rebuild completed")
            except Exception as e:
                logger.error(f"Embedding index rebuild failed: {e}")

        thread = threading.Thread(target=build_in_background)
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'message': 'Embedding index rebuild started'
        })

    except Exception as e:
        logger.error(f"API error in build_embeddings: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/knowledge/embeddings/stats', methods=['GET'])
def get_embedding_stats():
    """获取嵌入索引统计信息"""
    try:
        global embedding_builder
        if not embedding_builder:
            embedding_builder = EmbeddingIndexBuilder()

        stats = embedding_builder.get_embedding_stats()

        return jsonify({
            'success': True,
            'data': stats
        })

    except Exception as e:
        logger.error(f"API error in get_embedding_stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Serve frontend static files
@app.route('/')
def serve_frontend():
    """提供前端页面"""
    try:
        return send_from_directory(STATIC_FOLDER, 'knowledge.html')
    except:
        return """
        <html>
            <head><title>Knowledge Base API</title></head>
            <body>
                <h1>🧠 Knowledge Base API Server</h1>
                <p>API is running successfully!</p>
                <p><a href="/api/health">Health Check</a></p>
                <p><a href="/api/knowledge/stats">Statistics</a></p>
            </body>
        </html>
        """

@app.route('/<path:filename>')
def serve_static(filename):
    """提供静态文件"""
    try:
        return send_from_directory(STATIC_FOLDER, filename)
    except:
        return "File not found", 404

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

    parser = argparse.ArgumentParser(description='Knowledge Base API Server')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8001, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    # Initialize the application
    initialize()

    logger.info(f"🚀 Starting Knowledge Base API Server on {args.host}:{args.port}")

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )

if __name__ == "__main__":
    main()