"""
跨模态搜索引擎
支持文本、图像、表格、图表的统一搜索和检索
实现多模态内容的语义理解和跨模态匹配

核心功能：
- 统一的多模态搜索接口
- 跨模态语义匹配
- 多模态内容理解
- 智能相关性排序
- 实时搜索建议
- 搜索结果聚合和展示
"""

import asyncio
import logging
import json
import uuid
import math
import re
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from collections import defaultdict
import heapq

import asyncpg
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# 导入多模态内容类
from multimodal_document_processor import (
    MultimodalContent, ContentType, ExtractedText,
    ExtractedImage, ExtractedTable, ExtractedChart
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class SearchType(Enum):
    """搜索类型"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"
    MULTIMODAL = "multimodal"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class SearchMode(Enum):
    """搜索模式"""
    EXACT = "exact"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"
    CROSS_MODAL = "cross_modal"
    HYBRID = "hybrid"


class ContentType(Enum):
    """内容类型"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"


@dataclass
class SearchQuery:
    """搜索查询"""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    image_data: Optional[bytes] = None
    content_types: List[ContentType] = field(default_factory=list)
    search_type: SearchType = SearchType.HYBRID
    search_mode: SearchMode = SearchMode.HYBRID
    filters: Dict[str, Any] = field(default_factory=dict)
    pagination: Dict[str, Any] = field(default_factory=dict)
    ranking_weights: Dict[str, float] = field(default_factory=lambda: {
        'text_similarity': 0.4,
        'semantic_similarity': 0.3,
        'content_relevance': 0.2,
        'recency': 0.1
    })
    boost_factors: Dict[str, float] = field(default_factory=dict)
    user_context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'query_id': self.query_id,
            'text': self.text,
            'content_types': [ct.value for ct in self.content_types],
            'search_type': self.search_type.value,
            'search_mode': self.search_mode.value,
            'filters': self.filters,
            'pagination': self.pagination,
            'ranking_weights': self.ranking_weights,
            'boost_factors': self.boost_factors,
            'user_context': self.user_context,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class SearchResult:
    """搜索结果"""
    content_id: str
    document_id: str
    content_type: ContentType
    title: str
    snippet: str
    confidence: float
    relevance_score: float
    bbox: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    explanation: Optional[str] = None
    highlights: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'content_id': self.content_id,
            'document_id': self.document_id,
            'content_type': self.content_type.value,
            'title': self.title,
            'snippet': self.snippet,
            'confidence': self.confidence,
            'relevance_score': self.relevance_score,
            'bbox': self.bbox,
            'metadata': self.metadata,
            'explanation': self.explanation,
            'highlights': self.highlights
        }


@dataclass
class SearchResponse:
    """搜索响应"""
    query_id: str
    total_results: int
    results: List[SearchResult]
    search_time_ms: int
    facets: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    analytics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'query_id': self.query_id,
            'total_results': self.total_results,
            'results': [result.to_dict() for result in self.results],
            'search_time_ms': self.search_time_ms,
            'facets': self.facets,
            'suggestions': self.suggestions,
            'analytics': self.analytics
        }


class MultimodalVectorizer:
    """多模态向量化器"""

    def __init__(self):
        self.text_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english',
            lowercase=True
        )
        self.is_fitted = False

    async def fit_text_corpus(self, texts: List[str]):
        """训练文本向量化器"""
        try:
            if texts:
                self.text_vectorizer.fit(texts)
                self.is_fitted = True
                logger.info(f"Text vectorizer fitted with {len(texts)} documents")
        except Exception as e:
            logger.error(f"Failed to fit text vectorizer: {e}")

    async def vectorize_text(self, text: str) -> np.ndarray:
        """向量化文本"""
        try:
            if not self.is_fitted:
                # 简单的词频向量化
                words = re.findall(r'\w+', text.lower())
                vector = np.zeros(1000)  # 固定维度
                word_counts = defaultdict(int)
                for word in words:
                    word_counts[word] += 1

                # 简单哈希映射
                for word, count in word_counts.items():
                    idx = hash(word) % 1000
                    vector[idx] = count

                # 归一化
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm

                return vector
            else:
                vector = self.text_vectorizer.transform([text]).toarray()[0]
                return vector

        except Exception as e:
            logger.error(f"Text vectorization failed: {e}")
            return np.zeros(1000)

    async def vectorize_multimodal_content(self, content: MultimodalContent) -> np.ndarray:
        """向量化多模态内容"""
        try:
            vectors = []

            # 文本向量化
            if content.extracted_text:
                text_vector = await self.vectorize_text(content.extracted_text.content)
                vectors.append(text_vector)

            # 表格向量化
            if content.extracted_table:
                table_text = " ".join(content.extracted_table.headers)
                for row in content.extracted_table.rows:
                    table_text += " " + " ".join(row)
                table_vector = await self.vectorize_text(table_text)
                vectors.append(table_vector)

            # 图表向量化
            if content.extracted_chart:
                chart_text = f"{content.extracted_chart.chart_type} {content.extracted_chart.title or ''}"
                chart_vector = await self.vectorize_text(chart_text)
                vectors.append(chart_vector)

            # 图像描述向量化
            if content.extracted_image and content.extracted_image.description:
                image_vector = await self.vectorize_text(content.extracted_image.description)
                vectors.append(image_vector)

            if vectors:
                # 平均合并向量
                combined_vector = np.mean(vectors, axis=0)
                return combined_vector

            return np.zeros(1000)

        except Exception as e:
            logger.error(f"Multimodal vectorization failed: {e}")
            return np.zeros(1000)


class SimilarityCalculator:
    """相似度计算器"""

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        try:
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(np.dot(vec1, vec2) / (norm1 * norm2))

        except Exception as e:
            logger.error(f"Cosine similarity calculation failed: {e}")
            return 0.0

    @staticmethod
    def jaccard_similarity(set1: set, set2: set) -> float:
        """计算Jaccard相似度"""
        try:
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0
        except Exception as e:
            logger.error(f"Jaccard similarity calculation failed: {e}")
            return 0.0

    @staticmethod
    def edit_distance_similarity(str1: str, str2: str) -> float:
        """计算编辑距离相似度"""
        try:
            if not str1 and not str2:
                return 1.0
            if not str1 or not str2:
                return 0.0

            # 简化的编辑距离计算
            max_len = max(len(str1), len(str2))
            distance = sum(1 for a, b in zip(str1, str2) if a != b) + abs(len(str1) - len(str2))
            similarity = 1 - (distance / max_len)

            return similarity

        except Exception as e:
            logger.error(f"Edit distance calculation failed: {e}")
            return 0.0


class ContentIndexer:
    """内容索引器"""

    def __init__(self, redis_client: redis.Redis, db_pool: asyncpg.Pool):
        self.redis = redis_client
        self.db_pool = db_pool
        self.vectorizer = MultimodalVectorizer()
        self.similarity_calculator = SimilarityCalculator()

    async def index_content(self, content: MultimodalContent) -> bool:
        """索引内容"""
        try:
            # 生成向量
            vector = await self.vectorizer.vectorize_multimodal_content(content)

            # 存储到Redis
            content_key = f"content:{content.document_id}:{content.page_number}:{content.content_type.value}"

            content_data = {
                'document_id': content.document_id,
                'page_number': content.page_number,
                'content_type': content.content_type.value,
                'content_data': content.to_dict(),
                'vector': vector.tolist(),
                'indexed_at': datetime.now(timezone.utc).isoformat()
            }

            await self.redis.hset(content_key, mapping={
                'data': json.dumps(content_data, ensure_ascii=False),
                'vector': json.dumps(vector.tolist()),
                'content_type': content.content_type.value,
                'document_id': content.document_id,
                'page_number': str(content.page_number)
            })

            # 设置过期时间
            await self.redis.expire(content_key, 7 * 24 * 3600)  # 7天

            # 创建倒排索引
            await self._create_inverted_index(content)

            logger.info(f"Indexed content: {content.document_id}:{content.page_number}:{content.content_type.value}")
            return True

        except Exception as e:
            logger.error(f"Content indexing failed: {e}")
            return False

    async def _create_inverted_index(self, content: MultimodalContent):
        """创建倒排索引"""
        try:
            # 文本倒排索引
            if content.extracted_text:
                words = re.findall(r'\w+', content.extracted_text.content.lower())
                for word in set(words):
                    await self.redis.sadd(f"idx:text:{word}", f"{content.document_id}:{content.page_number}")

            # 标签倒排索引
            for tag in content.tags:
                await self.redis.sadd(f"idx:tag:{tag}", f"{content.document_id}:{content.page_number}")

            # 内容类型倒排索引
            await self.redis.sadd(f"idx:type:{content.content_type.value}", f"{content.document_id}:{content.page_number}")

        except Exception as e:
            logger.error(f"Inverted index creation failed: {e}")

    async def search_by_vector(self, query_vector: np.ndarray, content_types: List[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """向量搜索"""
        try:
            # 获取所有内容键
            if content_types:
                pattern = "content:*"
            else:
                pattern = f"content:*:*:*"

            keys = []
            for content_type in content_types or ['text', 'image', 'table', 'chart']:
                type_keys = await self.redis.keys(f"content:*:{content_type}")
                keys.extend(type_keys)

            # 计算相似度
            results = []
            for key in keys:
                try:
                    data = await self.redis.hgetall(key)
                    if data and b'vector' in data:
                        stored_vector = np.array(json.loads(data[b'vector'].decode('utf-8')))
                        similarity = self.similarity_calculator.cosine_similarity(query_vector, stored_vector)

                        if similarity > 0.1:  # 相似度阈值
                            content_data = json.loads(data[b'data'].decode('utf-8'))
                            results.append({
                                'key': key.decode('utf-8'),
                                'similarity': similarity,
                                'content_data': content_data
                            })
                except Exception as e:
                    logger.error(f"Error processing key {key}: {e}")
                    continue

            # 按相似度排序
            results.sort(key=lambda x: x['similarity'], reverse=True)

            return results[:limit]

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def search_by_text(self, text: str, content_types: List[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """文本搜索"""
        try:
            # 向量化查询文本
            query_vector = await self.vectorizer.vectorize_text(text)

            # 执行向量搜索
            results = await self.search_by_vector(query_vector, content_types, limit)

            return results

        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []


class CrossModalSearchEngine:
    """跨模态搜索引擎"""

    def __init__(self, redis_client: redis.Redis, db_pool: asyncpg.Pool):
        self.redis = redis_client
        self.db_pool = db_pool
        self.indexer = ContentIndexer(redis_client, db_pool)
        self.is_initialized = False

    async def initialize(self):
        """初始化搜索引擎"""
        try:
            # 加载现有的文本语料库
            await self._load_text_corpus()

            self.is_initialized = True
            logger.info("Cross-modal search engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize search engine: {e}")
            raise

    async def _load_text_corpus(self):
        """加载文本语料库"""
        try:
            # 从数据库加载现有文本内容
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, name, description, attributes_json
                    FROM knowledge_entries
                    WHERE is_active = true
                    LIMIT 10000
                    """
                )

                texts = []
                for row in rows:
                    text_parts = [row['name'], row['description'] or '']

                    # 添加属性文本
                    if row['attributes_json']:
                        try:
                            attrs = json.loads(row['attributes_json'])
                            for value in attrs.values():
                                if isinstance(value, str) and value.strip():
                                    text_parts.append(value.strip())
                        except:
                            pass

                    combined_text = ' '.join(text_parts)
                    if combined_text.strip():
                        texts.append(combined_text)

                # 训练向量化器
                await self.indexer.vectorizer.fit_text_corpus(texts)

                logger.info(f"Loaded {len(texts)} text samples for vectorizer training")

        except Exception as e:
            logger.error(f"Failed to load text corpus: {e}")

    async def search(self, query: SearchQuery) -> SearchResponse:
        """执行搜索"""
        start_time = time.time()

        try:
            # 验证查询
            if not query.text and not query.image_data:
                raise ValueError("Query must contain either text or image data")

            results = []
            total_results = 0

            # 文本搜索
            if query.text:
                text_results = await self._search_text(query)
                results.extend(text_results)

            # 图像搜索
            if query.image_data:
                image_results = await self._search_image(query)
                results.extend(image_results)

            # 结果去重和排序
            unique_results = self._deduplicate_results(results)
            sorted_results = self._rank_results(unique_results, query)

            # 分页
            offset = query.pagination.get('offset', 0)
            limit = query.pagination.get('limit', 20)
            paginated_results = sorted_results[offset:offset + limit]

            # 转换为SearchResult对象
            search_results = []
            for result in paginated_results:
                search_result = SearchResult(
                    content_id=result['key'],
                    document_id=result['content_data']['document_id'],
                    content_type=ContentType(result['content_data']['content_type']),
                    title=self._extract_title(result['content_data']),
                    snippet=self._extract_snippet(result['content_data']),
                    confidence=result.get('similarity', 0.0),
                    relevance_score=result.get('relevance_score', result.get('similarity', 0.0)),
                    metadata=result['content_data'].get('metadata', {}),
                    explanation=result.get('explanation'),
                    highlights=result.get('highlights', [])
                )
                search_results.append(search_result)

            # 生成搜索建议
            suggestions = await self._generate_suggestions(query)

            # 生成分析数据
            analytics = await self._generate_analytics(query, len(search_results))

            search_time = int((time.time() - start_time) * 1000)

            return SearchResponse(
                query_id=query.query_id,
                total_results=len(sorted_results),
                results=search_results,
                search_time_ms=search_time,
                suggestions=suggestions,
                analytics=analytics
            )

        except Exception as e:
            search_time = int((time.time() - start_time) * 1000)
            logger.error(f"Search failed: {e}")

            return SearchResponse(
                query_id=query.query_id,
                total_results=0,
                results=[],
                search_time_ms=search_time,
                error_message=str(e)
            )

    async def _search_text(self, query: SearchQuery) -> List[Dict[str, Any]]:
        """文本搜索"""
        try:
            content_types = [ct.value for ct in query.content_types] if query.content_types else None

            # 根据搜索模式选择搜索策略
            if query.search_mode in [SearchMode.EXACT, SearchMode.FUZZY]:
                results = await self._exact_text_search(query.text, content_types)
            elif query.search_mode in [SearchMode.SEMANTIC, SearchMode.CROSS_MODAL]:
                results = await self.indexer.search_by_text(query.text, content_types)
            else:  # HYBRID
                exact_results = await self._exact_text_search(query.text, content_types)
                semantic_results = await self.indexer.search_by_text(query.text, content_types)

                # 合并结果
                results = self._merge_search_results(exact_results, semantic_results, query.ranking_weights)

            return results

        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []

    async def _search_image(self, query: SearchQuery) -> List[Dict[str, Any]]:
        """图像搜索"""
        try:
            # 向量化图像
            image_vector = await self.indexer.vectorizer.vectorize_text("image content")  # 简化处理

            # 执行向量搜索
            content_types = [ct.value for ct in query.content_types] if query.content_types else ['image']
            results = await self.indexer.search_by_vector(image_vector, content_types)

            return results

        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return []

    async def _exact_text_search(self, text: str, content_types: List[str] = None) -> List[Dict[str, Any]]:
        """精确文本搜索"""
        try:
            words = re.findall(r'\w+', text.lower())
            if not words:
                return []

            # 使用倒排索引查找匹配的内容
            matching_keys = None
            for word in words:
                word_keys = await self.redis.smembers(f"idx:text:{word}")
                if matching_keys is None:
                    matching_keys = word_keys
                else:
                    matching_keys = matching_keys.intersection(word_keys)

            if not matching_keys:
                return []

            # 获取匹配的内容
            results = []
            for key in matching_keys:
                try:
                    data = await self.redis.hgetall(key)
                    if data and b'data' in data:
                        content_data = json.loads(data[b'data'].decode('utf-8'))

                        # 内容类型过滤
                        if content_types and content_data['content_type'] not in content_types:
                            continue

                        # 计算文本匹配度
                        match_score = self._calculate_text_match_score(text, content_data)

                        results.append({
                            'key': key.decode('utf-8'),
                            'similarity': match_score,
                            'content_data': content_data,
                            'match_type': 'exact'
                        })

                except Exception as e:
                    logger.error(f"Error processing key {key}: {e}")
                    continue

            return sorted(results, key=lambda x: x['similarity'], reverse=True)

        except Exception as e:
            logger.error(f"Exact text search failed: {e}")
            return []

    def _calculate_text_match_score(self, query_text: str, content_data: Dict[str, Any]) -> float:
        """计算文本匹配分数"""
        try:
            query_words = set(re.findall(r'\w+', query_text.lower()))

            # 从内容中提取文本
            content_text = ""
            if content_data['content_type'] == 'text' and content_data['content_data'].get('extracted_text'):
                content_text = content_data['content_data']['extracted_text']['content']
            elif content_data['content_type'] == 'table' and content_data['content_data'].get('extracted_table'):
                table = content_data['content_data']['extracted_table']
                content_text = " ".join(table['headers'])
                for row in table['rows']:
                    content_text += " " + " ".join(row)

            content_words = set(re.findall(r'\w+', content_text.lower()))

            # 计算Jaccard相似度
            intersection = len(query_words & content_words)
            union = len(query_words | content_words)

            return intersection / union if union > 0 else 0.0

        except Exception as e:
            logger.error(f"Text match score calculation failed: {e}")
            return 0.0

    def _merge_search_results(self, exact_results: List[Dict], semantic_results: List[Dict], weights: Dict[str, float]) -> List[Dict]:
        """合并搜索结果"""
        try:
            # 创建结果映射
            result_map = {}

            # 处理精确匹配结果
            for result in exact_results:
                key = result['key']
                result_map[key] = result.copy()
                result_map[key]['exact_score'] = result.get('similarity', 0.0)
                result_map[key]['semantic_score'] = 0.0

            # 处理语义搜索结果
            for result in semantic_results:
                key = result['key']
                if key in result_map:
                    result_map[key]['semantic_score'] = result.get('similarity', 0.0)
                else:
                    result_map[key] = result.copy()
                    result_map[key]['exact_score'] = 0.0
                    result_map[key]['semantic_score'] = result.get('similarity', 0.0)

            # 计算综合分数
            for result in result_map.values():
                exact_weight = weights.get('exact_match', 0.6)
                semantic_weight = weights.get('semantic_match', 0.4)

                combined_score = (
                    result['exact_score'] * exact_weight +
                    result['semantic_score'] * semantic_weight
                )
                result['similarity'] = combined_score

            return sorted(result_map.values(), key=lambda x: x['similarity'], reverse=True)

        except Exception as e:
            logger.error(f"Result merging failed: {e}")
            return []

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """去重搜索结果"""
        try:
            seen_keys = set()
            unique_results = []

            for result in results:
                key = result['key']
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique_results.append(result)

            return unique_results

        except Exception as e:
            logger.error(f"Result deduplication failed: {e}")
            return results

    def _rank_results(self, results: List[Dict], query: SearchQuery) -> List[Dict]:
        """排序搜索结果"""
        try:
            # 应用提升因子
            for result in results:
                for factor, boost in query.boost_factors.items():
                    if factor in result.get('metadata', {}):
                        result['similarity'] *= boost

            # 应用时间衰减
            if 'recency' in query.ranking_weights:
                current_time = datetime.now(timezone.utc)
                for result in results:
                    indexed_at = result.get('content_data', {}).get('indexed_at')
                    if indexed_at:
                        index_time = datetime.fromisoformat(indexed_at.replace('Z', '+00:00'))
                        days_old = (current_time - index_time).days
                        time_boost = math.exp(-days_old / 30)  # 30天衰减
                        result['similarity'] *= (1 + time_boost * query.ranking_weights['recency'])

            # 重新排序
            return sorted(results, key=lambda x: x.get('similarity', 0), reverse=True)

        except Exception as e:
            logger.error(f"Result ranking failed: {e}")
            return results

    def _extract_title(self, content_data: Dict[str, Any]) -> str:
        """提取标题"""
        try:
            if content_data['content_type'] == 'text' and content_data['content_data'].get('extracted_text'):
                text = content_data['content_data']['extracted_text']['content']
                # 取前100个字符作为标题
                return text[:100] + "..." if len(text) > 100 else text
            elif content_data['content_type'] == 'table' and content_data['content_data'].get('extracted_table'):
                table = content_data['content_data']['extracted_table']
                return f"Table: {', '.join(table['headers'][:3])}"
            elif content_data['content_type'] == 'chart' and content_data['content_data'].get('extracted_chart'):
                chart = content_data['content_data']['extracted_chart']
                return f"Chart: {chart.get('title', chart.get('chart_type', 'Unknown'))}"
            elif content_data['content_type'] == 'image' and content_data['content_data'].get('extracted_image'):
                img = content_data['content_data']['extracted_image']
                return f"Image: {img.get('description', 'Unknown content')}"
            else:
                return f"{content_data['content_type'].title()} content"

        except Exception as e:
            logger.error(f"Title extraction failed: {e}")
            return "Unknown content"

    def _extract_snippet(self, content_data: Dict[str, Any]) -> str:
        """提取摘要"""
        try:
            if content_data['content_type'] == 'text' and content_data['content_data'].get('extracted_text'):
                text = content_data['content_data']['extracted_text']['content']
                return text[:200] + "..." if len(text) > 200 else text
            elif content_data['content_type'] == 'table' and content_data['content_data'].get('extracted_table'):
                table = content_data['content_data']['extracted_table']
                snippet = f"Table with {len(table['headers'])} columns"
                if table['rows']:
                    snippet += f" and {len(table['rows'])} rows"
                return snippet
            elif content_data['content_type'] == 'chart' and content_data['content_data'].get('extracted_chart'):
                chart = content_data['content_data']['extracted_chart']
                return f"{chart.get('chart_type', 'Chart')} with {len(chart.get('data_points', []))} data points"
            else:
                return self._extract_title(content_data)

        except Exception as e:
            logger.error(f"Snippet extraction failed: {e}")
            return "No snippet available"

    async def _generate_suggestions(self, query: SearchQuery) -> List[str]:
        """生成搜索建议"""
        try:
            suggestions = []

            # 基于查询文本生成建议
            if query.text:
                # 简单的建议生成逻辑
                words = query.text.split()
                if len(words) > 1:
                    # 建议相关词汇
                    suggestions.extend([
                        f"{words[0]} analysis",
                        f"{words[-1]} overview",
                        f"{words[0]} {words[1]} examples"
                    ])

            return suggestions[:5]  # 限制建议数量

        except Exception as e:
            logger.error(f"Suggestion generation failed: {e}")
            return []

    async def _generate_analytics(self, query: SearchQuery, result_count: int) -> Dict[str, Any]:
        """生成分析数据"""
        try:
            return {
                'query_length': len(query.text),
                'has_image': query.image_data is not None,
                'content_types_searched': [ct.value for ct in query.content_types],
                'search_type': query.search_type.value,
                'search_mode': query.search_mode.value,
                'result_count': result_count,
                'filters_applied': len(query.filters),
                'search_timestamp': query.timestamp.isoformat()
            }

        except Exception as e:
            logger.error(f"Analytics generation failed: {e}")
            return {}


# FastAPI应用
app = FastAPI(
    title="Cross-Modal Search Engine",
    description="多模态跨媒体搜索引擎API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局搜索引擎实例
search_engine = None


@app.on_event("startup")
async def startup_event():
    """启动事件"""
    global search_engine
    redis_client = redis.from_url("redis://localhost:6379", decode_responses=False)
    db_pool = await asyncpg.create_pool("postgresql://postgres:postgres@localhost:5432/knowledge_base")

    search_engine = CrossModalSearchEngine(redis_client, db_pool)
    await search_engine.initialize()


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "cross-modal-search-engine",
        "initialized": search_engine.is_initialized if search_engine else False,
        "timestamp": datetime.utcnow().isoformat()
    }


class SearchRequest(BaseModel):
    """搜索请求"""
    text: str = ""
    content_types: List[str] = []
    search_type: str = "hybrid"
    search_mode: str = "hybrid"
    filters: Dict[str, Any] = {}
    pagination: Dict[str, Any] = {}
    ranking_weights: Dict[str, float] = {}
    boost_factors: Dict[str, float] = {}
    user_context: Dict[str, Any] = {}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """执行搜索"""
    try:
        # 构建搜索查询
        query = SearchQuery(
            text=request.text,
            content_types=[ContentType(ct) for ct in request.content_types],
            search_type=SearchType(request.search_type),
            search_mode=SearchMode(request.search_mode),
            filters=request.filters,
            pagination=request.pagination,
            ranking_weights=request.ranking_weights or {
                'text_similarity': 0.4,
                'semantic_similarity': 0.3,
                'content_relevance': 0.2,
                'recency': 0.1
            },
            boost_factors=request.boost_factors,
            user_context=request.user_context
        )

        # 执行搜索
        response = await search_engine.search(query)
        return response

    except Exception as e:
        logger.error(f"Search API failed: {e}")
        return SearchResponse(
            query_id="",
            total_results=0,
            results=[],
            search_time_ms=0,
            error_message=str(e)
        )


@app.get("/suggest")
async def get_suggestions(q: str = Query(...), limit: int = Query(5)):
    """获取搜索建议"""
    try:
        query = SearchQuery(text=q)
        suggestions = await search_engine._generate_suggestions(query)
        return {
            "query": q,
            "suggestions": suggestions[:limit]
        }

    except Exception as e:
        logger.error(f"Suggestions API failed: {e}")
        return {"query": q, "suggestions": []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)