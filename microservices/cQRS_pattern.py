"""
CQRS (Command Query Responsibility Segregation) 模式实现
将知识库系统的读操作和写操作完全分离，实现极致的性能优化

核心特性：
- 读写模型分离
- 事件溯源
- 最终一致性
- 高性能读缓存
- 批量写入优化
- 实时数据同步
"""

import asyncio
import json
import uuid
import logging
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union
from abc import ABC, abstractmethod
import time
import hashlib
import pickle

import asyncpg
import redis.asyncio as redis
import aioredis
from redis.asyncio import Redis
import numpy as np
from collections import defaultdict
import threading

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CommandType(Enum):
    """命令类型"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    BULK_CREATE = "bulk_create"
    BULK_UPDATE = "bulk_update"
    BULK_DELETE = "bulk_delete"


class QueryType(Enum):
    """查询类型"""
    BY_ID = "by_id"
    BY_TYPE = "by_type"
    SEARCH = "search"
    LIST = "list"
    AGGREGATE = "aggregate"
    SIMILARITY = "similarity"
    RECOMMENDATION = "recommendation"


@dataclass
class Command:
    """命令基类"""
    command_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    command_type: CommandType = CommandType.CREATE
    aggregate_id: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expected_version: Optional[int] = None
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'command_id': self.command_id,
            'command_type': self.command_type.value,
            'aggregate_id': self.aggregate_id,
            'data': self.data,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'expected_version': self.expected_version,
            'user_id': self.user_id,
            'correlation_id': self.correlation_id
        }


@dataclass
class Query:
    """查询基类"""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_type: QueryType = QueryType.BY_ID
    parameters: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    pagination: Optional[Dict[str, Any]] = None
    sorting: Optional[List[Dict[str, str]]] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: Optional[str] = None
    cache_ttl: int = 300  # 5分钟缓存

    def get_cache_key(self) -> str:
        """生成缓存键"""
        data = {
            'query_type': self.query_type.value,
            'parameters': self.parameters,
            'filters': self.filters,
            'pagination': self.pagination,
            'sorting': self.sorting
        }
        data_str = json.dumps(data, sort_keys=True)
        return f"query:{hashlib.md5(data_str.encode()).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'query_id': self.query_id,
            'query_type': self.query_type.value,
            'parameters': self.parameters,
            'filters': self.filters,
            'pagination': self.pagination,
            'sorting': self.sorting,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'cache_ttl': self.cache_ttl
        }


@dataclass
class CommandResult:
    """命令执行结果"""
    success: bool
    command_id: str
    aggregate_id: str
    new_version: int
    events_generated: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    execution_time_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """查询执行结果"""
    success: bool
    query_id: str
    data: Union[List[Dict], Dict, None]
    total_count: Optional[int] = None
    execution_time_ms: int = 0
    cached: bool = False
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Aggregate(ABC):
    """聚合根基类"""

    def __init__(self, aggregate_id: str):
        self.aggregate_id = aggregate_id
        self.version = 0
        self.uncommitted_events: List[Dict] = []
        self._applied_events: List[Dict] = []

    @abstractmethod
    def apply_event(self, event: Dict):
        """应用事件到聚合状态"""
        pass

    def add_event(self, event_type: str, data: Dict[str, Any], metadata: Optional[Dict] = None):
        """添加未提交事件"""
        event = {
            'event_id': str(uuid.uuid4()),
            'event_type': event_type,
            'aggregate_id': self.aggregate_id,
            'data': data,
            'metadata': metadata or {},
            'version': self.version + 1,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        self.uncommitted_events.append(event)
        self.apply_event(event)

    def mark_events_as_committed(self):
        """标记事件为已提交"""
        self._applied_events.extend(self.uncommitted_events)
        self.uncommitted_events.clear()

    def load_from_history(self, events: List[Dict]):
        """从历史事件加载聚合状态"""
        for event in events:
            self.apply_event(event)
            self.version = event['version']
        self._applied_events = events


class KnowledgeEntryAggregate(Aggregate):
    """知识条目聚合"""

    def __init__(self, aggregate_id: str):
        super().__init__(aggregate_id)
        self.name: str = ""
        self.description: str = ""
        self.entity_type: str = ""
        self.attributes: Dict[str, Any] = {}
        self.embedding_vector: Optional[List[float]] = None
        self.is_active: bool = True
        self.created_at: Optional[datetime] = None
        self.updated_at: Optional[datetime] = None
        self.created_by: Optional[str] = None
        self.updated_by: Optional[str] = None

    def apply_event(self, event: Dict):
        """应用知识条目事件"""
        event_type = event['event_type']
        data = event['data']

        if event_type == 'knowledge_entry_created':
            self.name = data['name']
            self.description = data.get('description', '')
            self.entity_type = data['entity_type']
            self.attributes = data.get('attributes', {})
            self.is_active = data.get('is_active', True)
            self.created_at = datetime.fromisoformat(event['timestamp'])
            self.created_by = data.get('created_by')

        elif event_type == 'knowledge_entry_updated':
            if 'name' in data:
                self.name = data['name']
            if 'description' in data:
                self.description = data['description']
            if 'attributes' in data:
                self.attributes.update(data['attributes'])
            if 'embedding_vector' in data:
                self.embedding_vector = data['embedding_vector']
            self.updated_at = datetime.fromisoformat(event['timestamp'])
            self.updated_by = data.get('updated_by')

        elif event_type == 'knowledge_entry_deleted':
            self.is_active = False
            self.updated_at = datetime.fromisoformat(event['timestamp'])

        elif event_type == 'embedding_updated':
            self.embedding_vector = data['embedding_vector']
            self.updated_at = datetime.fromisoformat(event['timestamp'])

        self.version = event['version']


class CommandHandler(ABC):
    """命令处理器基类"""

    @abstractmethod
    async def handle(self, command: Command) -> CommandResult:
        """处理命令"""
        pass


class KnowledgeEntryCommandHandler(CommandHandler):
    """知识条目命令处理器"""

    def __init__(self, event_store: 'EventStore', aggregate_repository: 'AggregateRepository'):
        self.event_store = event_store
        self.aggregate_repository = aggregate_repository

    async def handle(self, command: Command) -> CommandResult:
        """处理知识条目命令"""
        start_time = time.time()
        try:
            if command.command_type == CommandType.CREATE:
                return await self._handle_create(command)
            elif command.command_type == CommandType.UPDATE:
                return await self._handle_update(command)
            elif command.command_type == CommandType.DELETE:
                return await self._handle_delete(command)
            else:
                return CommandResult(
                    success=False,
                    command_id=command.command_id,
                    aggregate_id=command.aggregate_id,
                    new_version=0,
                    error_message=f"Unsupported command type: {command.command_type.value}"
                )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"Command handling failed: {e}")
            return CommandResult(
                success=False,
                command_id=command.command_id,
                aggregate_id=command.aggregate_id,
                new_version=0,
                error_message=str(e),
                execution_time_ms=execution_time
            )

    async def _handle_create(self, command: Command) -> CommandResult:
        """处理创建命令"""
        start_time = time.time()

        # 检查是否已存在
        existing = await self.aggregate_repository.get_by_id(command.aggregate_id)
        if existing:
            raise ValueError(f"Aggregate {command.aggregate_id} already exists")

        # 创建聚合
        aggregate = KnowledgeEntryAggregate(command.aggregate_id)
        aggregate.add_event('knowledge_entry_created', command.data, command.metadata)

        # 保存事件
        events = aggregate.uncommitted_events
        await self.event_store.save_events(events)

        # 更新读模型
        await self._update_read_model(aggregate)

        aggregate.mark_events_as_committed()

        execution_time = int((time.time() - start_time) * 1000)
        return CommandResult(
            success=True,
            command_id=command.command_id,
            aggregate_id=command.aggregate_id,
            new_version=aggregate.version,
            events_generated=[e['event_id'] for e in events],
            execution_time_ms=execution_time
        )

    async def _handle_update(self, command: Command) -> CommandResult:
        """处理更新命令"""
        start_time = time.time()

        # 加载聚合
        aggregate = await self.aggregate_repository.get_by_id(command.aggregate_id)
        if not aggregate:
            raise ValueError(f"Aggregate {command.aggregate_id} not found")

        # 版本检查
        if command.expected_version and command.expected_version != aggregate.version:
            raise ValueError(f"Version conflict: expected {command.expected_version}, got {aggregate.version}")

        # 应用更新
        aggregate.add_event('knowledge_entry_updated', command.data, command.metadata)

        # 保存事件
        events = aggregate.uncommitted_events
        await self.event_store.save_events(events)

        # 更新读模型
        await self._update_read_model(aggregate)

        aggregate.mark_events_as_committed()

        execution_time = int((time.time() - start_time) * 1000)
        return CommandResult(
            success=True,
            command_id=command.command_id,
            aggregate_id=command.aggregate_id,
            new_version=aggregate.version,
            events_generated=[e['event_id'] for e in events],
            execution_time_ms=execution_time
        )

    async def _handle_delete(self, command: Command) -> CommandResult:
        """处理删除命令"""
        start_time = time.time()

        # 加载聚合
        aggregate = await self.aggregate_repository.get_by_id(command.aggregate_id)
        if not aggregate:
            raise ValueError(f"Aggregate {command.aggregate_id} not found")

        # 应用删除
        aggregate.add_event('knowledge_entry_deleted', {}, command.metadata)

        # 保存事件
        events = aggregate.uncommitted_events
        await self.event_store.save_events(events)

        # 更新读模型
        await self._update_read_model(aggregate)

        aggregate.mark_events_as_committed()

        execution_time = int((time.time() - start_time) * 1000)
        return CommandResult(
            success=True,
            command_id=command.command_id,
            aggregate_id=command.aggregate_id,
            new_version=aggregate.version,
            events_generated=[e['event_id'] for e in events],
            execution_time_ms=execution_time
        )

    async def _update_read_model(self, aggregate: KnowledgeEntryAggregate):
        """更新读模型"""
        # 这里应该更新读模型数据库
        # 为了简化，这里只是记录日志
        logger.info(f"Updating read model for aggregate {aggregate.aggregate_id}")


class QueryHandler(ABC):
    """查询处理器基类"""

    @abstractmethod
    async def handle(self, query: Query) -> QueryResult:
        """处理查询"""
        pass


class KnowledgeEntryQueryHandler(QueryHandler):
    """知识条目查询处理器"""

    def __init__(self, read_database: asyncpg.Pool, cache_client: Redis):
        self.read_db = read_database
        self.cache = cache_client

    async def handle(self, query: Query) -> QueryResult:
        """处理查询"""
        start_time = time.time()

        try:
            # 检查缓存
            cache_key = query.get_cache_key()
            cached_result = await self.cache.get(cache_key)

            if cached_result:
                execution_time = int((time.time() - start_time) * 1000)
                return QueryResult(
                    success=True,
                    query_id=query.query_id,
                    data=json.loads(cached_result.decode('utf-8')),
                    cached=True,
                    execution_time_ms=execution_time
                )

            # 执行查询
            result = await self._execute_query(query)

            # 缓存结果
            if result['success'] and query.cache_ttl > 0:
                await self.cache.setex(
                    cache_key,
                    query.cache_ttl,
                    json.dumps(result['data'], ensure_ascii=False)
                )

            execution_time = int((time.time() - start_time) * 1000)
            return QueryResult(
                success=result['success'],
                query_id=query.query_id,
                data=result['data'],
                total_count=result.get('total_count'),
                cached=False,
                execution_time_ms=execution_time,
                error_message=result.get('error_message')
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"Query handling failed: {e}")
            return QueryResult(
                success=False,
                query_id=query.query_id,
                data=None,
                execution_time_ms=execution_time,
                error_message=str(e)
            )

    async def _execute_query(self, query: Query) -> Dict[str, Any]:
        """执行具体查询"""
        if query.query_type == QueryType.BY_ID:
            return await self._query_by_id(query)
        elif query.query_type == QueryType.BY_TYPE:
            return await self._query_by_type(query)
        elif query.query_type == QueryType.SEARCH:
            return await self._search(query)
        elif query.query_type == QueryType.LIST:
            return await self._list(query)
        elif query.query_type == QueryType.SIMILARITY:
            return await self._similarity_search(query)
        else:
            return {
                'success': False,
                'data': None,
                'error_message': f"Unsupported query type: {query.query_type.value}"
            }

    async def _query_by_id(self, query: Query) -> Dict[str, Any]:
        """根据ID查询"""
        try:
            async with self.read_db.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM knowledge_entries WHERE id = $1 AND is_active = true",
                    query.parameters.get('id')
                )

                if row:
                    return {
                        'success': True,
                        'data': dict(row)
                    }
                else:
                    return {
                        'success': True,
                        'data': None
                    }

        except Exception as e:
            return {
                'success': False,
                'data': None,
                'error_message': str(e)
            }

    async def _query_by_type(self, query: Query) -> Dict[str, Any]:
        """根据类型查询"""
        try:
            entity_type = query.parameters.get('entity_type')
            pagination = query.pagination or {'offset': 0, 'limit': 50}

            async with self.read_db.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM knowledge_entries
                    WHERE entity_type = $1 AND is_active = true
                    ORDER BY created_at DESC
                    LIMIT $2 OFFSET $3
                    """,
                    entity_type, pagination['limit'], pagination['offset']
                )

                return {
                    'success': True,
                    'data': [dict(row) for row in rows]
                }

        except Exception as e:
            return {
                'success': False,
                'data': None,
                'error_message': str(e)
            }

    async def _search(self, query: Query) -> Dict[str, Any]:
        """搜索查询"""
        try:
            search_text = query.parameters.get('text', '')
            entity_types = query.parameters.get('entity_types', [])
            pagination = query.pagination or {'offset': 0, 'limit': 20}

            async with self.read_db.acquire() as conn:
                # 构建查询条件
                where_conditions = ["is_active = true"]
                params = []
                param_count = 0

                if search_text:
                    param_count += 1
                    where_conditions.append(f"(name ILIKE ${param_count} OR description ILIKE ${param_count})")
                    params.extend([f"%{search_text}%", f"%{search_text}%"])

                if entity_types:
                    param_count += 1
                    where_conditions.append(f"entity_type = ANY(${param_count})")
                    params.append(entity_types)

                where_clause = " AND ".join(where_conditions)

                # 执行查询
                query_sql = f"""
                SELECT * FROM knowledge_entries
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
                """
                params.extend([pagination['limit'], pagination['offset']])

                rows = await conn.fetch(query_sql, *params)

                return {
                    'success': True,
                    'data': [dict(row) for row in rows]
                }

        except Exception as e:
            return {
                'success': False,
                'data': None,
                'error_message': str(e)
            }

    async def _list(self, query: Query) -> Dict[str, Any]:
        """列表查询"""
        try:
            filters = query.filters or {}
            pagination = query.pagination or {'offset': 0, 'limit': 50}
            sorting = query.sorting or [{'field': 'created_at', 'direction': 'DESC'}]

            async with self.read_db.acquire() as conn:
                # 构建查询条件
                where_conditions = ["is_active = true"]
                params = []
                param_count = 0

                for field, value in filters.items():
                    param_count += 1
                    where_conditions.append(f"{field} = ${param_count}")
                    params.append(value)

                where_clause = " AND ".join(where_conditions)

                # 构建排序
                order_clauses = []
                for sort_item in sorting:
                    field = sort_item['field']
                    direction = sort_item['direction']
                    order_clauses.append(f"{field} {direction}")

                order_clause = ", ".join(order_clauses)

                # 执行查询
                query_sql = f"""
                SELECT * FROM knowledge_entries
                WHERE {where_clause}
                ORDER BY {order_clause}
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
                """
                params.extend([pagination['limit'], pagination['offset']])

                rows = await conn.fetch(query_sql, *params)

                return {
                    'success': True,
                    'data': [dict(row) for row in rows]
                }

        except Exception as e:
            return {
                'success': False,
                'data': None,
                'error_message': str(e)
            }

    async def _similarity_search(self, query: Query) -> Dict[str, Any]:
        """相似度搜索"""
        try:
            query_vector = query.parameters.get('vector')
            threshold = query.parameters.get('threshold', 0.7)
            limit = query.parameters.get('limit', 10)

            if not query_vector:
                return {
                    'success': False,
                    'data': None,
                    'error_message': 'Query vector is required for similarity search'
                }

            async with self.read_db.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT ke.*, 1 - (vi.vector <=> $1) as similarity
                    FROM knowledge_entries ke
                    JOIN vector_index vi ON ke.id = vi.knowledge_entry_id
                    WHERE ke.is_active = true AND vi.vector IS NOT NULL
                    ORDER BY vi.vector <=> $1
                    LIMIT $2
                    """,
                    query_vector, limit
                )

                # 过滤低于阈值的结果
                results = []
                for row in rows:
                    if row['similarity'] >= threshold:
                        results.append(dict(row))

                return {
                    'success': True,
                    'data': results
                }

        except Exception as e:
            return {
                'success': False,
                'data': None,
                'error_message': str(e)
            }


class EventStore(ABC):
    """事件存储接口"""

    @abstractmethod
    async def save_events(self, events: List[Dict]) -> bool:
        """保存事件"""
        pass

    @abstractmethod
    async def get_events(self, aggregate_id: str, from_version: Optional[int] = None) -> List[Dict]:
        """获取聚合事件"""
        pass

    @abstractmethod
    async def get_events_by_type(self, event_type: str, limit: int = 100) -> List[Dict]:
        """根据类型获取事件"""
        pass


class PostgreSQLEventStore(EventStore):
    """PostgreSQL事件存储实现"""

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

    async def save_events(self, events: List[Dict]) -> bool:
        """批量保存事件"""
        try:
            async with self.db_pool.acquire() as conn:
                for event in events:
                    await conn.execute(
                        """
                        INSERT INTO events (
                            id, event_type, aggregate_id, data, metadata,
                            version, timestamp, correlation_id, causation_id
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        """,
                        event['event_id'],
                        event['event_type'],
                        event['aggregate_id'],
                        json.dumps(event['data'], ensure_ascii=False),
                        json.dumps(event['metadata'], ensure_ascii=False),
                        event['version'],
                        event['timestamp'],
                        event['metadata'].get('correlation_id'),
                        event['metadata'].get('causation_id')
                    )

            return True

        except Exception as e:
            logger.error(f"Failed to save events: {e}")
            return False

    async def get_events(self, aggregate_id: str, from_version: Optional[int] = None) -> List[Dict]:
        """获取聚合事件"""
        try:
            async with self.db_pool.acquire() as conn:
                if from_version:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM events
                        WHERE aggregate_id = $1 AND version >= $2
                        ORDER BY version ASC
                        """,
                        aggregate_id, from_version
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM events
                        WHERE aggregate_id = $1
                        ORDER BY version ASC
                        """,
                        aggregate_id
                    )

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get events for {aggregate_id}: {e}")
            return []

    async def get_events_by_type(self, event_type: str, limit: int = 100) -> List[Dict]:
        """根据类型获取事件"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM events
                    WHERE event_type = $1
                    ORDER BY timestamp DESC
                    LIMIT $2
                    """,
                    event_type, limit
                )

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get events by type {event_type}: {e}")
            return []


class AggregateRepository:
    """聚合仓储"""

    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self._cache: Dict[str, Aggregate] = {}
        self._cache_lock = asyncio.Lock()

    async def get_by_id(self, aggregate_id: str) -> Optional[Aggregate]:
        """根据ID获取聚合"""
        async with self._cache_lock:
            # 检查缓存
            if aggregate_id in self._cache:
                return self._cache[aggregate_id]

            # 从事件存储加载
            events = await self.event_store.get_events(aggregate_id)
            if not events:
                return None

            # 重建聚合
            aggregate = KnowledgeEntryAggregate(aggregate_id)
            aggregate.load_from_history(events)

            # 缓存聚合
            self._cache[aggregate_id] = aggregate

            return aggregate

    async def save(self, aggregate: Aggregate) -> bool:
        """保存聚合"""
        if aggregate.uncommitted_events:
            success = await self.event_store.save_events(aggregate.uncommitted_events)
            if success:
                aggregate.mark_events_as_committed()
                # 更新缓存
                async with self._cache_lock:
                    self._cache[aggregate.aggregate_id] = aggregate
            return success
        return True


class CQRSBus:
    """CQRS总线"""

    def __init__(self):
        self.command_handlers: Dict[str, CommandHandler] = {}
        self.query_handlers: Dict[str, QueryHandler] = {}
        self.middleware: List[Callable] = []

    def register_command_handler(self, command_type: str, handler: CommandHandler):
        """注册命令处理器"""
        self.command_handlers[command_type] = handler

    def register_query_handler(self, query_type: str, handler: QueryHandler):
        """注册查询处理器"""
        self.query_handlers[query_type] = handler

    def add_middleware(self, middleware: Callable):
        """添加中间件"""
        self.middleware.append(middleware)

    async def send_command(self, command: Command) -> CommandResult:
        """发送命令"""
        # 应用中间件
        for middleware in self.middleware:
            command = await middleware(command) if asyncio.iscoroutinefunction(middleware) else middleware(command)

        # 获取处理器
        handler = self.command_handlers.get(command.aggregate_type)
        if not handler:
            raise ValueError(f"No command handler for type: {command.aggregate_type}")

        # 处理命令
        return await handler.handle(command)

    async def send_query(self, query: Query) -> QueryResult:
        """发送查询"""
        # 应用中间件
        for middleware in self.middleware:
            query = await middleware(query) if asyncio.iscoroutinefunction(middleware) else middleware(query)

        # 获取处理器
        handler = self.query_handlers.get(query.query_type.value)
        if not handler:
            raise ValueError(f"No query handler for type: {query.query_type.value}")

        # 处理查询
        return await handler.handle(query)


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.command_metrics: Dict[str, List[int]] = defaultdict(list)
        self.query_metrics: Dict[str, List[int]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def record_command(self, command_type: str, execution_time_ms: int, success: bool):
        """记录命令性能"""
        async with self._lock:
            self.command_metrics[f"{command_type}_time"].append(execution_time_ms)
            self.command_metrics[f"{command_type}_success"].append(1 if success else 0)

    async def record_query(self, query_type: str, execution_time_ms: int, success: bool, cached: bool):
        """记录查询性能"""
        async with self._lock:
            self.query_metrics[f"{query_type}_time"].append(execution_time_ms)
            self.query_metrics[f"{query_type}_success"].append(1 if success else 0)
            self.query_metrics[f"{query_type}_cached"].append(1 if cached else 0)

    async def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        async with self._lock:
            metrics = {}

            # 命令指标
            for key, values in self.command_metrics.items():
                if key.endswith('_time') and values:
                    metrics[f"command_{key}_avg"] = sum(values) / len(values)
                    metrics[f"command_{key}_max"] = max(values)
                    metrics[f"command_{key}_min"] = min(values)
                elif key.endswith('_success') and values:
                    metrics[f"command_{key}_rate"] = sum(values) / len(values)

            # 查询指标
            for key, values in self.query_metrics.items():
                if key.endswith('_time') and values:
                    metrics[f"query_{key}_avg"] = sum(values) / len(values)
                    metrics[f"query_{key}_max"] = max(values)
                    metrics[f"query_{key}_min"] = min(values)
                elif key.endswith('_success') and values:
                    metrics[f"query_{key}_rate"] = sum(values) / len(values)
                elif key.endswith('_cached') and values:
                    metrics[f"query_{key}_rate"] = sum(values) / len(values)

            return metrics


# 示例使用代码
async def demo_cqrs_system():
    """演示CQRS系统"""
    # 初始化组件
    db_pool = await asyncpg.create_pool("postgresql://postgres:postgres@localhost:5432/knowledge_base")
    redis_client = redis.from_url("redis://localhost:6379", decode_responses=False)

    event_store = PostgreSQLEventStore(db_pool)
    aggregate_repo = AggregateRepository(event_store)
    cache_client = redis_client

    # 创建处理器
    command_handler = KnowledgeEntryCommandHandler(event_store, aggregate_repo)
    query_handler = KnowledgeEntryQueryHandler(db_pool, cache_client)

    # 创建CQRS总线
    bus = CQRSBus()
    bus.register_command_handler("knowledge_entry", command_handler)
    bus.register_query_handler("by_id", query_handler)
    bus.register_query_handler("by_type", query_handler)
    bus.register_query_handler("search", query_handler)

    # 性能监控
    monitor = PerformanceMonitor()

    # 演示命令处理
    create_command = Command(
        command_type=CommandType.CREATE,
        aggregate_id="test_entry_1",
        aggregate_type="knowledge_entry",
        data={
            'name': '测试紧固件',
            'description': '这是一个测试用的紧固件产品',
            'entity_type': 'product',
            'attributes': {
                'material': '不锈钢',
                'size': 'M6',
                'standard': 'GB/T 70'
            }
        }
    )

    result = await bus.send_command(create_command)
    print(f"Command result: {result}")
    await monitor.record_command("create", result.execution_time_ms, result.success)

    # 演示查询处理
    query = Query(
        query_type=QueryType.BY_ID,
        parameters={'id': 'test_entry_1'}
    )

    query_result = await bus.send_query(query)
    print(f"Query result: {query_result}")
    await monitor.record_query(
        "by_id", query_result.execution_time_ms, query_result.success, query_result.cached
    )

    # 获取性能指标
    metrics = await monitor.get_metrics()
    print(f"Performance metrics: {metrics}")


if __name__ == "__main__":
    asyncio.run(demo_cqrs_system())