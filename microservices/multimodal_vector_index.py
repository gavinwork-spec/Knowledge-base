"""
多模态向量索引系统
高效存储和检索多模态内容的向量嵌入
支持大规模向量搜索和实时索引更新

核心功能：
- 高效向量存储 (Redis + PostgreSQL)
- 实时向量索引更新
- 大规模向量搜索
- 向量压缩和优化
- 分布式索引分片
- 索引性能监控
"""

import asyncio
import logging
import json
import uuid
import pickle
import gzip
import hashlib
import numpy as np
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import math

import asyncpg
import redis.asyncio as redis
from redis.asyncio import Redis
import faiss
import psycopg2
from psycopg2.extras import execute_values

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class IndexType(Enum):
    """索引类型"""
    FLAT = "flat"
    IVF_FLAT = "ivf_flat"
    IVF_PQ = "ivf_pq"
    HNSW = "hnsw"


class StorageBackend(Enum):
    """存储后端"""
    REDIS = "redis"
    POSTGRESQL = "postgresql"
    HYBRID = "hybrid"


class CompressionType(Enum):
    """压缩类型"""
    NONE = "none"
    GZIP = "gzip"
    PICKLE = "pickle"
    QUANTIZED = "quantized"


@dataclass
class VectorIndexConfig:
    """向量索引配置"""
    dimension: int = 384
    index_type: IndexType = IndexType.IVF_FLAT
    nlist: int = 100  # IVF聚类数量
    m: int = 16  # HNSW的连接数
    efConstruction: int = 200  # HNSW构建参数
    efSearch: int = 50  # HNSW搜索参数
    compression: CompressionType = CompressionType.GZIP
    batch_size: int = 1000
    max_memory_mb: int = 2048
    storage_backend: StorageBackend = StorageBackend.HYBRID
    shard_count: int = 1
    cache_size: int = 10000


@dataclass
class VectorMetadata:
    """向量元数据"""
    vector_id: str
    content_id: str
    content_type: str
    document_id: str
    page_number: int
    created_at: datetime
    updated_at: datetime
    tags: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'vector_id': self.vector_id,
            'content_id': self.content_id,
            'content_type': self.content_type,
            'document_id': self.document_id,
            'page_number': self.page_number,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'tags': self.tags,
            'attributes': self.attributes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorMetadata':
        return cls(
            vector_id=data['vector_id'],
            content_id=data['content_id'],
            content_type=data['content_type'],
            document_id=data['document_id'],
            page_number=data['page_number'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            tags=data.get('tags', []),
            attributes=data.get('attributes', {})
        )


@dataclass
class SearchResult:
    """搜索结果"""
    vector_id: str
    metadata: VectorMetadata
    distance: float
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'vector_id': self.vector_id,
            'metadata': self.metadata.to_dict(),
            'distance': self.distance,
            'score': self.score
        }


class VectorCompressor:
    """向量压缩器"""

    def __init__(self, compression_type: CompressionType = CompressionType.GZIP):
        self.compression_type = compression_type

    def compress(self, vector: np.ndarray) -> bytes:
        """压缩向量"""
        try:
            if self.compression_type == CompressionType.NONE:
                return vector.tobytes()
            elif self.compression_type == CompressionType.GZIP:
                return gzip.compress(vector.tobytes())
            elif self.compression_type == CompressionType.PICKLE:
                return pickle.dumps(vector)
            elif self.compression_type == CompressionType.QUANTIZED:
                # 简单的量化压缩
                quantized = (vector * 255).astype(np.uint8)
                return quantized.tobytes()
            else:
                return vector.tobytes()

        except Exception as e:
            logger.error(f"Vector compression failed: {e}")
            return vector.tobytes()

    def decompress(self, data: bytes, dtype: np.dtype = np.float32, shape: Tuple[int, ...] = (384,)) -> np.ndarray:
        """解压缩向量"""
        try:
            if self.compression_type == CompressionType.NONE:
                return np.frombuffer(data, dtype=dtype).reshape(shape)
            elif self.compression_type == CompressionType.GZIP:
                decompressed = gzip.decompress(data)
                return np.frombuffer(decompressed, dtype=dtype).reshape(shape)
            elif self.compression_type == CompressionType.PICKLE:
                return pickle.loads(data)
            elif self.compression_type == CompressionType.QUANTIZED:
                quantized = np.frombuffer(data, dtype=np.uint8).reshape(shape)
                return quantized.astype(dtype) / 255.0
            else:
                return np.frombuffer(data, dtype=dtype).reshape(shape)

        except Exception as e:
            logger.error(f"Vector decompression failed: {e}")
            return np.zeros(shape, dtype=dtype)


class FaissIndexManager:
    """FAISS索引管理器"""

    def __init__(self, config: VectorIndexConfig):
        self.config = config
        self.index = None
        self.is_built = False
        self.vector_mapping: Dict[int, str] = {}  # 索引ID到向量ID的映射
        self.metadata_cache: Dict[str, VectorMetadata] = {}

    def create_index(self, dimension: int) -> bool:
        """创建FAISS索引"""
        try:
            if self.config.index_type == IndexType.FLAT:
                self.index = faiss.IndexFlatL2(dimension)
            elif self.config.index_type == IndexType.IVF_FLAT:
                quantizer = faiss.IndexFlatL2(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, self.config.nlist)
            elif self.config.index_type == IndexType.IVF_PQ:
                # 简化的IVF_PQ实现
                quantizer = faiss.IndexFlatL2(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, self.config.nlist)
            elif self.config.index_type == IndexType.HNSW:
                # HNSW索引需要特殊的库，这里简化为IVF
                quantizer = faiss.IndexFlatL2(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, self.config.nlist)

            self.is_built = True
            logger.info(f"Created FAISS index: {self.config.index_type.value}, dimension: {dimension}")
            return True

        except Exception as e:
            logger.error(f"FAISS index creation failed: {e}")
            return False

    def add_vectors(self, vectors: np.ndarray, vector_ids: List[str]) -> bool:
        """添加向量到索引"""
        try:
            if not self.is_built:
                if not self.create_index(vectors.shape[1]):
                    return False

            # 训练索引（如果是IVF类型）
            if hasattr(self.index, 'train') and not self.index.is_trained:
                if len(vectors) >= self.config.nlist:
                    self.index.train(vectors)
                    logger.info("Trained IVF index")
                else:
                    logger.warning("Insufficient vectors for training IVF index")

            # 添加向量
            start_idx = len(self.vector_mapping)
            self.index.add(vectors)

            # 更新映射
            for i, vector_id in enumerate(vector_ids):
                idx = start_idx + i
                self.vector_mapping[idx] = vector_id

            return True

        except Exception as e:
            logger.error(f"Vector addition failed: {e}")
            return False

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[SearchResult]:
        """搜索向量"""
        try:
            if not self.is_built:
                return []

            # 设置搜索参数
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = min(self.config.nlist, 10)

            distances, indices = self.index.search(query_vector.reshape(1, -1), k)

            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.vector_mapping):
                    vector_id = self.vector_mapping[idx]
                    metadata = self.metadata_cache.get(vector_id)

                    if metadata:
                        result = SearchResult(
                            vector_id=vector_id,
                            metadata=metadata,
                            distance=float(distance),
                            score=1.0 / (1.0 + float(distance))  # 转换为相似度分数
                        )
                        results.append(result)

            return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def save_index(self, file_path: str) -> bool:
        """保存索引到文件"""
        try:
            faiss.write_index(self.index, file_path)
            # 保存映射
            mapping_data = {
                'vector_mapping': self.vector_mapping,
                'metadata_cache': {k: v.to_dict() for k, v in self.metadata_cache.items()},
                'config': {
                    'dimension': self.config.dimension,
                    'index_type': self.config.index_type.value,
                    'nlist': self.config.nlist
                }
            }

            mapping_file = file_path.replace('.index', '_mapping.json')
            with open(mapping_file, 'w') as f:
                json.dump(mapping_data, f)

            logger.info(f"Saved index to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Index saving failed: {e}")
            return False

    def load_index(self, file_path: str) -> bool:
        """从文件加载索引"""
        try:
            # 加载索引
            self.index = faiss.read_index(file_path)

            # 加载映射
            mapping_file = file_path.replace('.index', '_mapping.json')
            with open(mapping_file, 'r') as f:
                mapping_data = json.load(f)

            self.vector_mapping = {int(k): v for k, v in mapping_data['vector_mapping'].items()}
            self.metadata_cache = {
                k: VectorMetadata.from_dict(v) for k, v in mapping_data['metadata_cache'].items()
            }

            self.is_built = True
            logger.info(f"Loaded index from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Index loading failed: {e}")
            return False


class VectorStorage:
    """向量存储后端"""

    def __init__(self, storage_type: StorageBackend, redis_client: Redis, db_pool: asyncpg.Pool):
        self.storage_type = storage_type
        self.redis = redis_client
        self.db_pool = db_pool
        self.compressor = VectorCompressor()

    async def store_vector(self, vector_id: str, vector: np.ndarray, metadata: VectorMetadata) -> bool:
        """存储向量"""
        try:
            if self.storage_type in [StorageBackend.REDIS, StorageBackend.HYBRID]:
                # Redis存储
                compressed_vector = self.compressor.compress(vector)

                await self.redis.hset(
                    f"vector:{vector_id}",
                    mapping={
                        'data': compressed_vector,
                        'metadata': json.dumps(metadata.to_dict(), ensure_ascii=False),
                        'dimension': str(vector.shape[0]),
                        'compression': self.compressor.compression_type.value,
                        'created_at': metadata.created_at.isoformat()
                    }
                )

                # 设置过期时间（30天）
                await self.redis.expire(f"vector:{vector_id}", 30 * 24 * 3600)

            if self.storage_type in [StorageBackend.POSTGRESQL, StorageBackend.HYBRID]:
                # PostgreSQL存储
                await self._store_to_postgresql(vector_id, vector, metadata)

            return True

        except Exception as e:
            logger.error(f"Vector storage failed: {e}")
            return False

    async def _store_to_postgresql(self, vector_id: str, vector: np.ndarray, metadata: VectorMetadata):
        """存储到PostgreSQL"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO vector_index (
                        id, knowledge_entry_id, vector, embedding_model,
                        created_at, updated_at, metadata_json
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (id) DO UPDATE SET
                        vector = EXCLUDED.vector,
                        updated_at = EXCLUDED.updated_at,
                        metadata_json = EXCLUDED.metadata_json
                    """,
                    vector_id,
                    metadata.content_id,
                    vector.tolist(),
                    'multimodal_v1',
                    metadata.created_at,
                    metadata.updated_at,
                    json.dumps(metadata.to_dict(), ensure_ascii=False)
                )

        except Exception as e:
            logger.error(f"PostgreSQL vector storage failed: {e}")
            raise

    async def retrieve_vector(self, vector_id: str) -> Tuple[Optional[np.ndarray], Optional[VectorMetadata]]:
        """检索向量"""
        try:
            vector = None
            metadata = None

            # 尝试从Redis获取
            if self.storage_type in [StorageBackend.REDIS, StorageBackend.HYBRID]:
                data = await self.redis.hgetall(f"vector:{vector_id}")
                if data and b'data' in data:
                    compressed_vector = data[b'data']
                    dimension = int(data[b'dimension'].decode('utf-8'))

                    vector = self.compressor.decompress(
                        compressed_vector,
                        dtype=np.float32,
                        shape=(dimension,)
                    )

                    if b'metadata' in data:
                        metadata_dict = json.loads(data[b'metadata'].decode('utf-8'))
                        metadata = VectorMetadata.from_dict(metadata_dict)

            # 如果Redis没有，尝试PostgreSQL
            if vector is None and self.storage_type in [StorageBackend.POSTGRESQL, StorageBackend.HYBRID]:
                vector, metadata = await self._retrieve_from_postgresql(vector_id)

            return vector, metadata

        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            return None, None

    async def _retrieve_from_postgresql(self, vector_id: str) -> Tuple[Optional[np.ndarray], Optional[VectorMetadata]]:
        """从PostgreSQL检索向量"""
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT vector, metadata_json, embedding_model
                    FROM vector_index
                    WHERE id = $1
                    """,
                    vector_id
                )

                if row:
                    vector = np.array(row['vector'], dtype=np.float32)
                    metadata = VectorMetadata.from_dict(json.loads(row['metadata_json']))

                    return vector, metadata

            return None, None

        except Exception as e:
            logger.error(f"PostgreSQL vector retrieval failed: {e}")
            return None, None

    async def delete_vector(self, vector_id: str) -> bool:
        """删除向量"""
        try:
            if self.storage_type in [StorageBackend.REDIS, StorageBackend.HYBRID]:
                await self.redis.delete(f"vector:{vector_id}")

            if self.storage_type in [StorageBackend.POSTGRESQL, StorageBackend.HYBRID]:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("DELETE FROM vector_index WHERE id = $1", vector_id)

            return True

        except Exception as e:
            logger.error(f"Vector deletion failed: {e}")
            return False


class MultimodalVectorIndex:
    """多模态向量索引系统"""

    def __init__(self, config: VectorIndexConfig):
        self.config = config
        self.faiss_manager = FaissIndexManager(config)
        self.storage = None
        self.is_initialized = False
        self.index_cache = {}  # LRU缓存
        self.cache_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.metrics = {
            'total_vectors': 0,
            'search_count': 0,
            'index_updates': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    async def initialize(self, redis_client: Redis, db_pool: asyncpg.Pool):
        """初始化索引系统"""
        try:
            self.storage = VectorStorage(
                storage_type=self.config.storage_backend,
                redis_client=redis_client,
                db_pool=db_pool
            )

            # 尝试加载现有索引
            await self._load_existing_index()

            self.is_initialized = True
            logger.info("Multimodal vector index initialized successfully")

        except Exception as e:
            logger.error(f"Vector index initialization failed: {e}")
            raise

    async def _load_existing_index(self):
        """加载现有索引"""
        try:
            # 这里可以尝试从文件系统加载预训练的索引
            index_file = "multimodal_vector.index"
            if os.path.exists(index_file):
                self.faiss_manager.load_index(index_file)
                logger.info("Loaded existing vector index")

        except Exception as e:
            logger.info(f"No existing index to load: {e}")

    async def add_vector(self, vector_id: str, vector: np.ndarray, metadata: VectorMetadata) -> bool:
        """添加向量到索引"""
        try:
            # 存储到后端
            if not await self.storage.store_vector(vector_id, vector, metadata):
                return False

            # 更新缓存
            with self.cache_lock:
                self.index_cache[vector_id] = (vector, metadata)

            # 添加到FAISS索引
            if self.faiss_manager.is_built:
                vectors = vector.reshape(1, -1)
                success = self.faiss_manager.add_vectors(vectors, [vector_id])
                if success:
                    self.faiss_manager.metadata_cache[vector_id] = metadata
                    self.metrics['index_updates'] += 1

            self.metrics['total_vectors'] += 1
            return True

        except Exception as e:
            logger.error(f"Vector addition failed: {e}")
            return False

    async def batch_add_vectors(self, vectors_data: List[Tuple[str, np.ndarray, VectorMetadata]]) -> int:
        """批量添加向量"""
        try:
            success_count = 0

            # 分批处理
            batch_size = self.config.batch_size
            for i in range(0, len(vectors_data), batch_size):
                batch = vectors_data[i:i + batch_size]

                # 存储到后端
                storage_tasks = []
                for vector_id, vector, metadata in batch:
                    task = self.storage.store_vector(vector_id, vector, metadata)
                    storage_tasks.append(task)

                storage_results = await asyncio.gather(*storage_tasks)

                # 添加到FAISS索引
                vectors = np.array([v[1] for v in batch])
                vector_ids = [v[0] for v in batch]

                if self.faiss_manager.add_vectors(vectors, vector_ids):
                    # 更新FAISS缓存
                    for vector_id, _, metadata in batch:
                        self.faiss_manager.metadata_cache[vector_id] = metadata

                success_count += sum(1 for result in storage_results if result)

                self.metrics['total_vectors'] += len(batch)

            logger.info(f"Batch added {success_count}/{len(vectors_data)} vectors")
            return success_count

        except Exception as e:
            logger.error(f"Batch vector addition failed: {e}")
            return 0

    async def search(self, query_vector: np.ndarray, k: int = 10, filters: Dict[str, Any] = None) -> List[SearchResult]:
        """搜索向量"""
        try:
            self.metrics['search_count'] += 1

            # 检查缓存
            cache_key = self._generate_cache_key(query_vector, k, filters)
            cached_results = await self._get_cached_results(cache_key)

            if cached_results:
                self.metrics['cache_hits'] += 1
                return cached_results

            self.metrics['cache_misses'] += 1

            # 从FAISS索引搜索
            results = self.faiss_manager.search(query_vector, k)

            # 应用过滤器
            if filters:
                filtered_results = []
                for result in results:
                    if self._apply_filters(result.metadata, filters):
                        filtered_results.append(result)
                results = filtered_results

            # 缓存结果
            await self._cache_results(cache_key, results)

            return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def _get_cached_results(self, cache_key: str) -> Optional[List[SearchResult]]:
        """获取缓存结果"""
        try:
            cached_data = await self.redis.get(f"search_cache:{cache_key}")
            if cached_data:
                results_data = json.loads(cached_data.decode('utf-8'))
                return [SearchResult(**result) for result in results_data]
            return None

        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
            return None

    async def _cache_results(self, cache_key: str, results: List[SearchResult]):
        """缓存搜索结果"""
        try:
            # 缓存5分钟
            ttl = 300
            results_data = [result.to_dict() for result in results]
            await self.redis.setex(
                f"search_cache:{cache_key}",
                ttl,
                json.dumps(results_data, ensure_ascii=False)
            )

        except Exception as e:
            logger.error(f"Result caching failed: {e}")

    def _generate_cache_key(self, query_vector: np.ndarray, k: int, filters: Dict[str, Any]) -> str:
        """生成缓存键"""
        vector_hash = hashlib.md5(query_vector.tobytes()).hexdigest()
        filters_str = json.dumps(filters, sort_keys=True) if filters else ""
        filters_hash = hashlib.md5(filters_str.encode()).hexdigest()
        return f"{vector_hash}:{k}:{filters_hash}"

    def _apply_filters(self, metadata: VectorMetadata, filters: Dict[str, Any]) -> bool:
        """应用过滤器"""
        try:
            # 内容类型过滤
            if 'content_types' in filters:
                if metadata.content_type not in filters['content_types']:
                    return False

            # 标签过滤
            if 'tags' in filters:
                required_tags = set(filters['tags'])
                if not required_tags.intersection(set(metadata.tags)):
                    return False

            # 时间范围过滤
            if 'date_range' in filters:
                start_date = filters['date_range'].get('start')
                end_date = filters['date_range'].get('end')
                if start_date and metadata.created_at < start_date:
                    return False
                if end_date and metadata.created_at > end_date:
                    return False

            # 文档ID过滤
            if 'document_ids' in filters:
                if metadata.document_id not in filters['document_ids']:
                    return False

            return True

        except Exception as e:
            logger.error(f"Filter application failed: {e}")
            return True

    async def rebuild_index(self) -> bool:
        """重建索引"""
        try:
            logger.info("Rebuilding vector index...")

            # 从存储加载所有向量
            # 这里简化处理，实际实现需要从数据库加载所有向量

            # 创建新索引
            self.faiss_manager = FaissIndexManager(self.config)
            self.faiss_manager.create_index(self.config.dimension)

            # 重新添加向量
            # 这里需要从存储后端加载所有向量并重新索引

            logger.info("Vector index rebuilt successfully")
            return True

        except Exception as e:
            logger.error(f"Index rebuild failed: {e}")
            return False

    async def get_metrics(self) -> Dict[str, Any]:
        """获取索引指标"""
        try:
            index_size = len(self.faiss_manager.vector_mapping) if self.faiss_manager.is_built else 0
            cache_size = len(self.index_cache)

            return {
                'total_vectors': self.metrics['total_vectors'],
                'index_vectors': index_size,
                'cache_size': cache_size,
                'search_count': self.metrics['search_count'],
                'index_updates': self.metrics['index_updates'],
                'cache_hits': self.metrics['cache_hits'],
                'cache_misses': self.metrics['cache_misses'],
                'cache_hit_rate': (
                    self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses'])
                    if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0
                ),
                'index_config': {
                    'dimension': self.config.dimension,
                    'index_type': self.config.index_type.value,
                    'compression': self.config.compression.value,
                    'storage_backend': self.config.storage_backend.value
                }
            }

        except Exception as e:
            logger.error(f"Metrics retrieval failed: {e}")
            return {}

    async def optimize_index(self):
        """优化索引"""
        try:
            logger.info("Optimizing vector index...")

            # 清理缓存
            with self.cache_lock:
                old_size = len(self.index_cache)
                self.index_cache.clear()
                logger.info(f"Cleared {old_size} items from index cache")

            # 重新构建索引（如果需要）
            if self.metrics['index_updates'] > 1000:  # 如果更新次数超过阈值
                await self.rebuild_index()

            logger.info("Vector index optimization completed")

        except Exception as e:
            logger.error(f"Index optimization failed: {e}")


# API接口定义
class VectorIndexRequest(BaseModel):
    """向量索引请求"""
    vector_id: str
    vector: List[float]
    metadata: Dict[str, Any]


class VectorSearchRequest(BaseModel):
    """向量搜索请求"""
    query_vector: List[float]
    k: int = 10
    filters: Dict[str, Any] = {}


# FastAPI应用
app = FastAPI(
    title="Multimodal Vector Index",
    description="多模态向量索引和搜索API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局索引实例
vector_index = None


@app.on_event("startup")
async def startup_event():
    """启动事件"""
    global vector_index
    config = VectorIndexConfig(
        dimension=384,
        index_type=IndexType.IVF_FLAT,
        storage_backend=StorageBackend.HYBRID
    )

    redis_client = redis.from_url("redis://localhost:6379", decode_responses=False)
    db_pool = await asyncpg.create_pool("postgresql://postgres:postgres@localhost:5432/knowledge_base")

    vector_index = MultimodalVectorIndex(config)
    await vector_index.initialize(redis_client, db_pool)


@app.get("/health")
async def health_check():
    """健康检查"""
    metrics = await vector_index.get_metrics() if vector_index else {}
    return {
        "status": "healthy",
        "service": "multimodal-vector-index",
        "initialized": vector_index.is_initialized if vector_index else False,
        "metrics": metrics,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/metrics")
async def get_metrics():
    """获取指标"""
    if not vector_index:
        raise HTTPException(status_code=500, detail="Vector index not initialized")

    metrics = await vector_index.get_metrics()
    return metrics


@app.post("/index", response_model=Dict[str, Any])
async def add_vector(request: VectorIndexRequest):
    """添加向量到索引"""
    if not vector_index:
        raise HTTPException(status_code=500, detail="Vector index not initialized")

    try:
        vector = np.array(request.vector, dtype=np.float32)
        metadata = VectorMetadata(
            vector_id=request.vector_id,
            content_id=request.metadata.get('content_id', request.vector_id),
            content_type=request.metadata.get('content_type', 'unknown'),
            document_id=request.metadata.get('document_id', ''),
            page_number=request.metadata.get('page_number', 1),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tags=request.metadata.get('tags', []),
            attributes=request.metadata.get('attributes', {})
        )

        success = await vector_index.add_vector(request.vector_id, vector, metadata)

        return {
            "success": success,
            "vector_id": request.vector_id,
            "message": "Vector added successfully" if success else "Failed to add vector"
        }

    except Exception as e:
        logger.error(f"Vector addition API failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=List[Dict[str, Any]])
async def search_vectors(request: VectorSearchRequest):
    """搜索向量"""
    if not vector_index:
        raise HTTPException(status_code=500, detail="Vector index not initialized")

    try:
        query_vector = np.array(request.query_vector, dtype=np.float32)

        results = await vector_index.search(
            query_vector=query_vector,
            k=request.k,
            filters=request.filters
        )

        return [result.to_dict() for result in results]

    except Exception as e:
        logger.error(f"Vector search API failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebuild")
async def rebuild_index():
    """重建索引"""
    if not vector_index:
        raise HTTPException(status_code=500, detail="Vector index not initialized")

    try:
        success = await vector_index.rebuild_index()

        return {
            "success": success,
            "message": "Index rebuilt successfully" if success else "Failed to rebuild index"
        }

    except Exception as e:
        logger.error(f"Index rebuild API failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize")
async def optimize_index():
    """优化索引"""
    if not vector_index:
        raise HTTPException(status_code=500, detail="Vector index not initialized")

    try:
        await vector_index.optimize_index()

        return {
            "success": True,
            "message": "Index optimization completed"
        }

    except Exception as e:
        logger.error(f"Index optimization API failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)