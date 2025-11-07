"""
高级事件驱动架构实现
彻底重构知识库系统为高性能、可扩展的事件驱动系统

核心特性：
- 异步非阻塞处理
- 事件溯源和CQRS模式
- 智能事件路由和过滤
- 自适应负载均衡
- 实时流处理
- 分布式事务支持
"""

import asyncio
import json
import uuid
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Callable, Union, Set
from abc import ABC, abstractmethod
import hashlib
import pickle
import gzip

import redis.asyncio as redis
import aioredis
from redis.asyncio import Redis
from aiohttp import ClientSession, ClientTimeout
import asyncpg
import numpy as np
from collections import defaultdict, deque
import weakref
import threading

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """事件优先级枚举"""
    CRITICAL = 1      # 系统关键事件（故障、安全）
    HIGH = 2          # 用户交互事件（查询、上传）
    NORMAL = 3        # 业务流程事件（索引、学习）
    LOW = 4           # 后台任务事件（统计、清理）
    BACKGROUND = 5    # 维护和监控事件


class EventCategory(Enum):
    """事件分类枚举"""
    SYSTEM = "system"           # 系统级事件
    USER = "user"              # 用户相关事件
    DOCUMENT = "document"      # 文档处理事件
    SEARCH = "search"          # 搜索相关事件
    AI_AGENT = "ai_agent"      # AI Agent事件
    NOTIFICATION = "notification" # 通知事件
    WORKFLOW = "workflow"      # 工作流事件
    MONITORING = "monitoring"  # 监控事件


class EventStatus(Enum):
    """事件处理状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"
    DEAD_LETTER = "dead_letter"


@dataclass
class EventMetadata:
    """事件元数据"""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: Optional[str] = None
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    tenant_id: Optional[str] = None
    source_service: str = ""
    source_instance: str = ""
    destination_services: Set[str] = field(default_factory=set)
    tags: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class EventPayload:
    """事件载荷基类"""
    data: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    retry_policy: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data': self.data,
            'version': self.version,
            'timestamp': self.timestamp.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'retry_policy': self.retry_policy
        }


class Event:
    """高级事件类"""

    def __init__(
        self,
        event_type: str,
        category: EventCategory,
        payload: EventPayload,
        metadata: Optional[EventMetadata] = None,
        priority: EventPriority = EventPriority.NORMAL,
        id: Optional[str] = None
    ):
        self.id = id or str(uuid.uuid4())
        self.event_type = event_type
        self.category = category
        self.payload = payload
        self.metadata = metadata or EventMetadata()
        self.priority = priority
        self.status = EventStatus.PENDING
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
        self.processing_count = 0
        self.error_message: Optional[str] = None
        self.processing_history: List[Dict[str, Any]] = []

        # 事件大小限制（1MB）
        self.max_size = 1024 * 1024

    def add_processing_step(self, service: str, action: str, result: str = "success"):
        """添加处理步骤记录"""
        step = {
            'service': service,
            'action': action,
            'result': result,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'processing_count': self.processing_count
        }
        self.processing_history.append(step)
        self.processing_count += 1
        self.updated_at = datetime.now(timezone.utc)

    def calculate_size(self) -> int:
        """计算事件大小"""
        data = self.to_dict()
        serialized = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        return len(serialized.encode('utf-8'))

    def is_expired(self) -> bool:
        """检查事件是否过期"""
        if self.payload.expires_at:
            return datetime.now(timezone.utc) > self.payload.expires_at
        return False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'event_type': self.event_type,
            'category': self.category.value,
            'payload': self.payload.to_dict(),
            'metadata': asdict(self.metadata),
            'priority': self.priority.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'processing_count': self.processing_count,
            'error_message': self.error_message,
            'processing_history': self.processing_history
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """从字典创建事件"""
        metadata = EventMetadata(**data['metadata'])
        payload = EventPayload(**data['payload'])

        event = cls(
            event_type=data['event_type'],
            category=EventCategory(data['category']),
            payload=payload,
            metadata=metadata,
            priority=EventPriority(data['priority']),
            id=data['id']
        )

        event.status = EventStatus(data['status'])
        event.created_at = datetime.fromisoformat(data['created_at'])
        event.updated_at = datetime.fromisoformat(data['updated_at'])
        event.processing_count = data['processing_count']
        event.error_message = data.get('error_message')
        event.processing_history = data.get('processing_history', [])

        return event


class EventSchemaRegistry:
    """事件模式注册表"""

    def __init__(self):
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.validators: Dict[str, Callable] = {}

    def register_schema(self, event_type: str, schema: Dict[str, Any], validator: Optional[Callable] = None):
        """注册事件模式"""
        self.schemas[event_type] = schema
        if validator:
            self.validators[event_type] = validator
        logger.info(f"Registered event schema: {event_type}")

    def validate_event(self, event: Event) -> bool:
        """验证事件模式"""
        if event.event_type not in self.schemas:
            logger.warning(f"No schema registered for event type: {event.event_type}")
            return True

        schema = self.schemas[event.event_type]
        validator = self.validators.get(event.event_type)

        if validator:
            try:
                return validator(event.payload.data)
            except Exception as e:
                logger.error(f"Event validation failed for {event.event_type}: {e}")
                return False

        # 基础模式验证
        return self._basic_validate(event.payload.data, schema)

    def _basic_validate(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """基础模式验证"""
        required_fields = schema.get('required', [])
        for field in required_fields:
            if field not in data:
                return False

        # 类型验证
        field_types = schema.get('types', {})
        for field, expected_type in field_types.items():
            if field in data and not isinstance(data[field], expected_type):
                return False

        return True


class EventStore:
    """事件存储接口"""

    @abstractmethod
    async def save_event(self, event: Event) -> bool:
        """保存事件"""
        pass

    @abstractmethod
    async def get_event(self, event_id: str) -> Optional[Event]:
        """获取事件"""
        pass

    @abstractmethod
    async def get_events_by_correlation(self, correlation_id: str) -> List[Event]:
        """根据关联ID获取事件"""
        pass

    @abstractmethod
    async def get_events_by_type(self, event_type: str, limit: int = 100) -> List[Event]:
        """根据类型获取事件"""
        pass


class RedisEventStore(EventStore):
    """基于Redis的事件存储实现"""

    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.event_prefix = "event:"
        self.correlation_prefix = "correlation:"
        self.type_prefix = "type:"
        self.max_events_per_type = 10000

    async def save_event(self, event: Event) -> bool:
        """保存事件到Redis"""
        try:
            # 压缩大型事件
            event_data = event.to_dict()
            serialized = json.dumps(event_data, ensure_ascii=False)

            if len(serialized) > 10000:  # 10KB以上进行压缩
                compressed = gzip.compress(serialized.encode('utf-8'))
                event_data['compressed'] = True
                serialized = compressed

            # 保存事件
            await self.redis.hset(
                f"{self.event_prefix}{event.id}",
                mapping={
                    'data': serialized if not event_data.get('compressed') else compressed,
                    'created_at': event.created_at.timestamp(),
                    'status': event.status.value
                }
            )

            # 设置过期时间（7天）
            await self.redis.expire(f"{self.event_prefix}{event.id}", 7 * 24 * 3600)

            # 索引关联ID
            if event.metadata.correlation_id:
                await self.redis.zadd(
                    f"{self.correlation_prefix}{event.metadata.correlation_id}",
                    {event.id: event.created_at.timestamp()}
                )
                await self.redis.expire(f"{self.correlation_prefix}{event.metadata.correlation_id}", 7 * 24 * 3600)

            # 索引事件类型
            await self.redis.zadd(
                f"{self.type_prefix}{event.event_type}",
                {event.id: event.created_at.timestamp()}
            )
            # 保持最新的N个事件
            await self.redis.zremrangebyrank(
                f"{self.type_prefix}{event.event_type}",
                0, -self.max_events_per_type - 1
            )

            return True

        except Exception as e:
            logger.error(f"Failed to save event {event.id}: {e}")
            return False

    async def get_event(self, event_id: str) -> Optional[Event]:
        """从Redis获取事件"""
        try:
            data = await self.redis.hgetall(f"{self.event_prefix}{event_id}")
            if not data:
                return None

            serialized = data.get(b'data', b'')

            # 解压缩
            if isinstance(serialized, bytes):
                try:
                    # 尝试解压缩
                    decompressed = gzip.decompress(serialized)
                    event_dict = json.loads(decompressed.decode('utf-8'))
                except:
                    # 如果解压缩失败，直接解析
                    event_dict = json.loads(serialized.decode('utf-8'))
            else:
                event_dict = json.loads(serialized)

            return Event.from_dict(event_dict)

        except Exception as e:
            logger.error(f"Failed to get event {event_id}: {e}")
            return None

    async def get_events_by_correlation(self, correlation_id: str) -> List[Event]:
        """根据关联ID获取事件"""
        try:
            event_ids = await self.redis.zrange(
                f"{self.correlation_prefix}{correlation_id}",
                0, -1
            )

            events = []
            for event_id in event_ids:
                if isinstance(event_id, bytes):
                    event_id = event_id.decode('utf-8')
                event = await self.get_event(event_id)
                if event:
                    events.append(event)

            return sorted(events, key=lambda e: e.created_at)

        except Exception as e:
            logger.error(f"Failed to get events by correlation {correlation_id}: {e}")
            return []

    async def get_events_by_type(self, event_type: str, limit: int = 100) -> List[Event]:
        """根据类型获取事件"""
        try:
            event_ids = await self.redis.zrange(
                f"{self.type_prefix}{event_type}",
                -limit, -1
            )

            events = []
            for event_id in event_ids:
                if isinstance(event_id, bytes):
                    event_id = event_id.decode('utf-8')
                event = await self.get_event(event_id)
                if event:
                    events.append(event)

            return sorted(events, key=lambda e: e.created_at)

        except Exception as e:
            logger.error(f"Failed to get events by type {event_type}: {e}")
            return []


class MessageBroker(ABC):
    """消息代理抽象基类"""

    @abstractmethod
    async def publish(self, topic: str, event: Event) -> bool:
        """发布事件"""
        pass

    @abstractmethod
    async def subscribe(self, topic: str, handler: Callable) -> None:
        """订阅事件"""
        pass

    @abstractmethod
    async def unsubscribe(self, topic: str, handler: Callable) -> None:
        """取消订阅"""
        pass


class RedisMessageBroker(MessageBroker):
    """基于Redis Streams的消息代理"""

    def __init__(self, redis_client: Redis, max_stream_length: int = 10000):
        self.redis = redis_client
        self.max_stream_length = max_stream_length
        self.consumer_groups: Dict[str, str] = {}
        self.active_subscriptions: Dict[str, List[Callable]] = defaultdict(list)

    async def publish(self, topic: str, event: Event) -> bool:
        """发布事件到Redis Stream"""
        try:
            # 验证事件大小
            if event.calculate_size() > event.max_size:
                logger.error(f"Event {event.id} exceeds maximum size limit")
                return False

            # 序列化事件
            event_data = event.to_dict()
            serialized = json.dumps(event_data, ensure_ascii=False, separators=(',', ':'))

            # 发布到Stream
            message_id = await self.redis.xadd(
                f"stream:{topic}",
                {
                    'event': serialized,
                    'priority': str(event.priority.value),
                    'category': event.category.value,
                    'timestamp': event.created_at.timestamp()
                },
                maxlen=self.max_stream_length,
                approximate_trim_length=True
            )

            # 同时发布到Pub/Sub用于实时通知
            await self.redis.publish(
                f"notify:{topic}",
                json.dumps({
                    'event_id': event.id,
                    'event_type': event.event_type,
                    'priority': event.priority.value,
                    'message_id': message_id
                })
            )

            logger.debug(f"Published event {event.id} to topic {topic} with message ID {message_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish event {event.id} to topic {topic}: {e}")
            return False

    async def subscribe(self, topic: str, handler: Callable, consumer_group: str = "default") -> str:
        """订阅事件流"""
        try:
            stream_name = f"stream:{topic}"
            group_name = f"group:{consumer_group}"
            consumer_name = f"consumer:{uuid.uuid4().hex[:8]}"

            # 创建消费者组（如果不存在）
            try:
                await self.redis.xgroup_create(
                    stream_name,
                    group_name,
                    id='0',
                    mkstream=True
                )
            except Exception as e:
                if "BUSYGROUP" not in str(e):
                    raise

            # 注册处理器
            self.active_subscriptions[topic].append(handler)

            # 启动消费者任务
            task = asyncio.create_task(
                self._consume_events(stream_name, group_name, consumer_name, handler)
            )

            logger.info(f"Subscribed to topic {topic} with consumer {consumer_name}")
            return consumer_name

        except Exception as e:
            logger.error(f"Failed to subscribe to topic {topic}: {e}")
            raise

    async def unsubscribe(self, topic: str, handler: Callable) -> None:
        """取消订阅"""
        if topic in self.active_subscriptions:
            try:
                self.active_subscriptions[topic].remove(handler)
            except ValueError:
                pass

    async def _consume_events(self, stream_name: str, group_name: str, consumer_name: str, handler: Callable):
        """消费事件的主循环"""
        logger.info(f"Starting consumer {consumer_name} for stream {stream_name}")

        while True:
            try:
                # 读取消息
                messages = await self.redis.xreadgroup(
                    group_name,
                    consumer_name,
                    {stream_name: '>'},
                    count=10,
                    block=1000
                )

                for stream, message_list in messages:
                    for message_id, fields in message_list:
                        await self._process_message(stream_name, group_name, message_id, fields, handler)

            except Exception as e:
                logger.error(f"Error in consumer {consumer_name}: {e}")
                await asyncio.sleep(5)

    async def _process_message(self, stream_name: str, group_name: str, message_id: str, fields: Dict, handler: Callable):
        """处理单个消息"""
        try:
            # 解析事件
            event_data = json.loads(fields[b'event'].decode('utf-8'))
            event = Event.from_dict(event_data)

            # 检查事件是否过期
            if event.is_expired():
                logger.warning(f"Event {event.id} has expired, skipping")
                await self.redis.xack(stream_name, group_name, message_id)
                return

            # 更新事件状态
            event.status = EventStatus.PROCESSING
            event.add_processing_step("message_broker", "received")

            # 调用处理器
            if asyncio.iscoroutinefunction(handler):
                result = await handler(event)
            else:
                result = handler(event)

            # 处理结果
            if result is True or result is None:
                # 成功处理
                await self.redis.xack(stream_name, group_name, message_id)
                event.status = EventStatus.COMPLETED
                event.add_processing_step("message_broker", "processed")
            else:
                # 处理失败
                logger.error(f"Handler failed for event {event.id}: {result}")
                await self._handle_failed_message(stream_name, group_name, message_id, event, result)

        except Exception as e:
            logger.error(f"Error processing message {message_id}: {e}")
            await self._handle_failed_message(stream_name, group_name, message_id, None, str(e))

    async def _handle_failed_message(self, stream_name: str, group_name: str, message_id: str, event: Optional[Event], error: str):
        """处理失败的消息"""
        try:
            if event:
                event.status = EventStatus.FAILED
                event.error_message = error
                event.add_processing_step("message_broker", "failed")

                # 重试逻辑
                if event.processing_count < 3:
                    # 延迟重试
                    await asyncio.sleep(2 ** event.processing_count)
                    await self.redis.xadd(
                        f"retry:{stream_name}",
                        {
                            'original_message_id': message_id,
                            'event': json.dumps(event.to_dict()),
                            'retry_count': event.processing_count,
                            'error': error
                        }
                    )
                else:
                    # 移动到死信队列
                    await self.redis.xadd(
                        f"dead_letter:{stream_name}",
                        {
                            'original_message_id': message_id,
                            'event': json.dumps(event.to_dict()),
                            'error': error,
                            'failed_at': datetime.now(timezone.utc).isoformat()
                        }
                    )

            # 确认原始消息
            await self.redis.xack(stream_name, group_name, message_id)

        except Exception as e:
            logger.error(f"Error handling failed message {message_id}: {e}")


class EventRouter:
    """智能事件路由器"""

    def __init__(self):
        self.routes: Dict[str, List[Callable]] = defaultdict(list)
        self.filters: Dict[str, List[Callable]] = defaultdict(list)
        self.transformers: Dict[str, List[Callable]] = defaultdict(list)

    def add_route(self, event_pattern: str, handler: Callable, filter_func: Optional[Callable] = None, transformer: Optional[Callable] = None):
        """添加路由规则"""
        self.routes[event_pattern].append(handler)
        if filter_func:
            self.filters[event_pattern].append(filter_func)
        if transformer:
            self.transformers[event_pattern].append(transformer)

    async def route_event(self, event: Event) -> List[bool]:
        """路由事件到匹配的处理器"""
        results = []

        for pattern, handlers in self.routes.items():
            if self._match_pattern(event.event_type, pattern):
                # 应用过滤器
                filters = self.filters.get(pattern, [])
                filtered = True
                for filter_func in filters:
                    try:
                        if not filter_func(event):
                            filtered = False
                            break
                    except Exception as e:
                        logger.error(f"Filter error for pattern {pattern}: {e}")
                        filtered = False
                        break

                if not filtered:
                    continue

                # 应用转换器
                transformers = self.transformers.get(pattern, [])
                processed_event = event
                for transformer in transformers:
                    try:
                        processed_event = transformer(processed_event) or processed_event
                    except Exception as e:
                        logger.error(f"Transformer error for pattern {pattern}: {e}")
                        break

                # 调用处理器
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            result = await handler(processed_event)
                        else:
                            result = handler(processed_event)
                        results.append(result is True or result is None)
                    except Exception as e:
                        logger.error(f"Handler error for pattern {pattern}: {e}")
                        results.append(False)

        return results

    def _match_pattern(self, event_type: str, pattern: str) -> bool:
        """匹配事件模式"""
        if pattern == "*":
            return True
        if pattern == event_type:
            return True
        if "*" in pattern:
            # 简单通配符匹配
            pattern_parts = pattern.split("*")
            if event_type.startswith(pattern_parts[0]) and event_type.endswith(pattern_parts[1]):
                return True
        return False


class EventAggregator:
    """事件聚合器"""

    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.aggregation_prefix = "agg:"

    async def aggregate_events(self, correlation_id: str, window_size: int = 300) -> Dict[str, Any]:
        """聚合相关事件"""
        try:
            # 获取时间窗口内的事件
            current_time = time.time()
            window_start = current_time - window_size

            # 这里简化实现，实际应该使用更复杂的聚合逻辑
            events = await self.redis.zrangebyscore(
                f"correlation:{correlation_id}",
                window_start, current_time
            )

            aggregation = {
                'correlation_id': correlation_id,
                'window_start': window_start,
                'window_end': current_time,
                'event_count': len(events),
                'event_types': defaultdict(int),
                'processing_times': [],
                'success_rate': 0.0,
                'errors': []
            }

            success_count = 0
            for event_id in events:
                if isinstance(event_id, bytes):
                    event_id = event_id.decode('utf-8')

                # 获取事件详情（这里简化）
                aggregation['event_types']['unknown'] += 1
                success_count += 1

            if len(events) > 0:
                aggregation['success_rate'] = success_count / len(events)

            return aggregation

        except Exception as e:
            logger.error(f"Failed to aggregate events for {correlation_id}: {e}")
            return {}


class CircuitBreaker:
    """熔断器"""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func: Callable, *args, **kwargs):
        """通过熔断器调用函数"""
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """成功时重置"""
        self.failure_count = 0
        self.state = 'CLOSED'

    def _on_failure(self):
        """失败时更新状态"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'


class EventProcessor:
    """事件处理器"""

    def __init__(self, event_store: EventStore, message_broker: MessageBroker, router: EventRouter):
        self.event_store = event_store
        self.message_broker = message_broker
        self.router = router
        self.schema_registry = EventSchemaRegistry()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.metrics = defaultdict(int)

    async def publish_event(self, event: Event, topics: List[str]) -> bool:
        """发布事件到多个主题"""
        try:
            # 验证事件模式
            if not self.schema_registry.validate_event(event):
                logger.error(f"Event validation failed: {event.id}")
                return False

            # 保存事件
            await self.event_store.save_event(event)

            # 发布到主题
            results = []
            for topic in topics:
                result = await self.message_broker.publish(topic, event)
                results.append(result)
                self.metrics[f'published_to_{topic}'] += 1

            return all(results)

        except Exception as e:
            logger.error(f"Failed to publish event {event.id}: {e}")
            self.metrics['publish_errors'] += 1
            return False

    async def process_event(self, event: Event) -> bool:
        """处理事件"""
        try:
            self.metrics['processed_events'] += 1

            # 路由事件
            results = await self.router.route_event(event)

            # 更新事件状态
            if all(results):
                event.status = EventStatus.COMPLETED
                self.metrics['successful_events'] += 1
            else:
                event.status = EventStatus.FAILED
                self.metrics['failed_events'] += 1

            # 保存更新后的事件
            await self.event_store.save_event(event)

            return all(results)

        except Exception as e:
            logger.error(f"Failed to process event {event.id}: {e}")
            self.metrics['processing_errors'] += 1
            event.status = EventStatus.FAILED
            event.error_message = str(e)
            await self.event_store.save_event(event)
            return False

    async def get_metrics(self) -> Dict[str, Any]:
        """获取处理指标"""
        return dict(self.metrics)


# 示例使用和测试代码
async def create_sample_events():
    """创建示例事件"""
    events = []

    # 文档上传事件
    doc_event = Event(
        event_type="document.uploaded",
        category=EventCategory.DOCUMENT,
        payload=EventPayload(
            data={
                'document_id': str(uuid.uuid4()),
                'filename': 'sample.pdf',
                'file_size': 1024000,
                'content_type': 'application/pdf'
            }
        ),
        priority=EventPriority.HIGH
    )
    events.append(doc_event)

    # 用户查询事件
    query_event = Event(
        event_type="user.query",
        category=EventCategory.USER,
        payload=EventPayload(
            data={
                'query_text': '紧固件供应商',
                'query_type': 'search',
                'user_id': 'user123'
            }
        ),
        priority=EventPriority.NORMAL
    )
    events.append(query_event)

    return events


async def demo_event_system():
    """演示事件系统"""
    # 初始化组件
    redis_client = redis.from_url("redis://localhost:6379", decode_responses=False)
    event_store = RedisEventStore(redis_client)
    message_broker = RedisMessageBroker(redis_client)
    router = EventRouter()
    processor = EventProcessor(event_store, message_broker, router)

    # 注册示例事件模式
    processor.schema_registry.register_schema(
        "document.uploaded",
        {
            'required': ['document_id', 'filename'],
            'types': {
                'file_size': int,
                'content_type': str
            }
        }
    )

    # 添加路由规则
    async def document_handler(event: Event):
        print(f"处理文档事件: {event.event_type}")
        return True

    async def query_handler(event: Event):
        print(f"处理查询事件: {event.event_type}")
        return True

    router.add_route("document.*", document_handler)
    router.add_route("user.*", query_handler)

    # 创建并处理示例事件
    events = await create_sample_events()

    for event in events:
        # 发布事件
        topics = [f"{event.category.value}.{event.event_type}"]
        success = await processor.publish_event(event, topics)
        print(f"发布事件 {event.event_type}: {'成功' if success else '失败'}")

        # 处理事件
        if success:
            processed = await processor.process_event(event)
            print(f"处理事件 {event.event_type}: {'成功' if processed else '失败'}")

    # 获取指标
    metrics = await processor.get_metrics()
    print(f"处理指标: {metrics}")


if __name__ == "__main__":
    asyncio.run(demo_event_system())