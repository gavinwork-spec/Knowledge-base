"""
事件驱动架构核心模块
基于设计文档实现的事件系统基础设施
"""

import asyncio
import json
import uuid
import logging
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
import redis.asyncio as redis
from contextlib import asynccontextmanager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """事件类型枚举"""
    # 文档相关事件
    DOCUMENT_UPLOADED = "document.uploaded"
    DOCUMENT_PROCESSED = "document.processed"
    DOCUMENT_EXTRACTED = "document.extracted"
    DOCUMENT_INDEXED = "document.indexed"
    DOCUMENT_PROCESSING_FAILED = "document.processing_failed"

    # 搜索相关事件
    SEARCH_QUERY = "search.query"
    SEARCH_COMPLETED = "search.completed"
    SEARCH_FAILED = "search.failed"
    SEARCH_INDEX_UPDATED = "search.index_updated"

    # Agent相关事件
    AGENT_TRIGGERED = "agent.triggered"
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    AGENT_HEARTBEAT = "agent.heartbeat"

    # 用户相关事件
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    USER_QUERY = "user.query"
    USER_FEEDBACK = "user.feedback"
    USER_REGISTERED = "user.registered"

    # 系统相关事件
    SYSTEM_ALERT = "system.alert"
    SYSTEM_MAINTENANCE = "system.maintenance"
    SYSTEM_BACKUP = "system.backup"
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"

    # 业务相关事件
    INQUIRY_RECEIVED = "business.inquiry_received"
    QUOTE_GENERATED = "business.quote_generated"
    STRATEGY_UPDATED = "business.strategy_updated"
    RECOMMENDATION_CREATED = "business.recommendation_created"


@dataclass
class Event:
    """事件数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType = EventType.SYSTEM_ALERT
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    user_id: Optional[str] = None
    priority: int = 5  # 1-10, 1最高优先级
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "id": self.id,
            "type": self.type.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "user_id": self.user_id,
            "priority": self.priority,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Event':
        """从字典创建事件对象"""
        event_type = EventType(data.get("type", EventType.SYSTEM_ALERT.value))
        timestamp = datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat()))

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=event_type,
            source=data.get("source", ""),
            timestamp=timestamp,
            data=data.get("data", {}),
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id"),
            user_id=data.get("user_id"),
            priority=data.get("priority", 5),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3)
        )


class EventPriority(Enum):
    """事件优先级"""
    CRITICAL = 1
    HIGH = 3
    NORMAL = 5
    LOW = 7
    BACKGROUND = 10


class EventBus:
    """事件总线 - 负责事件的发布和订阅"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.consumer_groups: Dict[str, str] = {}
        self.is_connected = False

    async def connect(self):
        """连接到Redis"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=False)
            await self.redis_client.ping()
            self.is_connected = True
            logger.info(f"EventBus connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.is_connected = False
            raise

    async def disconnect(self):
        """断开Redis连接"""
        if self.redis_client:
            await self.redis_client.close()
            self.is_connected = False
            logger.info("EventBus disconnected from Redis")

    @asynccontextmanager
    async def connection(self):
        """连接上下文管理器"""
        await self.connect()
        try:
            yield self
        finally:
            await self.disconnect()

    async def publish_event(self, event: Event):
        """发布事件到事件总线"""
        if not self.is_connected:
            raise RuntimeError("EventBus not connected to Redis")

        try:
            event_data = json.dumps(event.to_dict(), ensure_ascii=False)
            stream_name = f"events:{event.type.value}"

            # 发布到Redis Streams
            message_id = await self.redis_client.xadd(
                stream_name,
                {
                    "event_data": event_data.encode('utf-8'),
                    "source": event.source.encode('utf-8'),
                    "priority": str(event.priority).encode('utf-8')
                },
                maxlen=10000,
                approximate_trim_length=True
            )

            # 发布到Pub/Sub用于实时通知
            await self.redis_client.publish(
                f"notifications:{event.type.value}",
                event_data
            )

            logger.info(f"Event {event.id} ({event.type.value}) published to stream {stream_name}")
            return message_id

        except Exception as e:
            logger.error(f"Failed to publish event {event.id}: {e}")
            raise

    async def create_consumer_group(self, service_name: str, event_types: List[EventType]):
        """为服务创建消费者组"""
        if not self.is_connected:
            raise RuntimeError("EventBus not connected to Redis")

        group_name = f"{service_name}_group"

        for event_type in event_types:
            stream_name = f"events:{event_type.value}"

            try:
                # 创建消费者组（如果不存在）
                await self.redis_client.xgroup_create(
                    stream_name,
                    group_name,
                    mkstream=True,
                    id='0'
                )
                logger.info(f"Created consumer group {group_name} for stream {stream_name}")
            except Exception as e:
                if "BUSYGROUP" not in str(e):
                    logger.error(f"Failed to create consumer group for {stream_name}: {e}")

        self.consumer_groups[service_name] = group_name
        return group_name

    async def consume_events(
        self,
        service_name: str,
        event_types: List[EventType],
        callback: Callable[[Event], None],
        batch_size: int = 10,
        block_timeout: int = 1000
    ):
        """消费事件"""
        if not self.is_connected:
            raise RuntimeError("EventBus not connected to Redis")

        group_name = self.consumer_groups.get(service_name, f"{service_name}_group")
        consumer_name = f"{service_name}_{uuid.uuid4().hex[:8]}"

        logger.info(f"Starting event consumption for {service_name} (consumer: {consumer_name})")

        while True:
            try:
                # 构建要读取的流列表
                streams = {}
                for event_type in event_types:
                    stream_name = f"events:{event_type.value}"
                    streams[stream_name] = '>'

                if not streams:
                    await asyncio.sleep(1)
                    continue

                # 读取事件
                results = await self.redis_client.xreadgroup(
                    group_name,
                    consumer_name,
                    streams,
                    count=batch_size,
                    block=block_timeout
                )

                for stream_name, messages in results:
                    for message_id, fields in messages:
                        try:
                            # 解析事件数据
                            event_data = json.loads(fields[b'event_data'].decode('utf-8'))
                            event = Event.from_dict(event_data)

                            # 调用回调函数处理事件
                            await self._process_event_callback(callback, event)

                            # 确认消息处理完成
                            await self.redis_client.xack(stream_name, group_name, message_id)

                            logger.debug(f"Processed event {event.id} from {stream_name}")

                        except Exception as e:
                            logger.error(f"Error processing message {message_id}: {e}")
                            # 可以选择将消息移动到死信队列或重试
                            await self._handle_failed_message(stream_name, group_name, message_id, event, e)

            except Exception as e:
                logger.error(f"Error in event consumption loop: {e}")
                await asyncio.sleep(5)

    async def _process_event_callback(self, callback: Callable[[Event], None], event: Event):
        """处理事件回调"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)
        except Exception as e:
            logger.error(f"Error in event callback for {event.id}: {e}")
            raise

    async def _handle_failed_message(self, stream_name: str, group_name: str, message_id: str, event: Event, error: Exception):
        """处理失败的消息"""
        try:
            if event.retry_count < event.max_retries:
                # 增加重试计数并重新发布
                event.retry_count += 1
                await asyncio.sleep(2 ** event.retry_count)  # 指数退避
                await self.publish_event(event)
            else:
                # 移动到死信队列
                await self.redis_client.xadd(
                    f"dead_letter:{stream_name}",
                    {
                        "original_message_id": message_id,
                        "error": str(error),
                        "event_data": json.dumps(event.to_dict(), ensure_ascii=False),
                        "failed_at": datetime.utcnow().isoformat()
                    }
                )
                logger.warning(f"Message {message_id} moved to dead letter queue after {event.max_retries} retries")
        except Exception as e:
            logger.error(f"Failed to handle failed message {message_id}: {e}")

    async def get_event_statistics(self) -> Dict[str, Any]:
        """获取事件统计信息"""
        if not self.is_connected:
            raise RuntimeError("EventBus not connected to Redis")

        stats = {
            "total_streams": 0,
            "total_pending_messages": 0,
            "streams": {}
        }

        try:
            # 获取所有流信息
            streams = await self.redis_client.keys("events:*")
            stats["total_streams"] = len(streams)

            for stream_name in streams:
                stream_name = stream_name.decode('utf-8')
                try:
                    info = await self.redis_client.xinfo_stream(stream_name)
                    pending = await self.redis_client.xpending_count(stream_name, 'default_group')

                    stats["streams"][stream_name] = {
                        "length": info.get('length', 0),
                        "pending": pending,
                        "last_id": info.get('last-id', ''),
                        "groups": info.get('groups', 0)
                    }
                    stats["total_pending_messages"] += pending
                except Exception as e:
                    logger.warning(f"Failed to get info for stream {stream_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to get event statistics: {e}")

        return stats


class ServiceCommunicator:
    """服务间通信器"""

    def __init__(self, event_bus: EventBus, service_name: str):
        self.event_bus = event_bus
        self.service_name = service_name
        self.service_channels: Dict[str, str] = {}

    async def send_message_to_service(self, target_service: str, message_type: str, data: Dict[str, Any]):
        """向特定服务发送消息"""
        event = Event(
            type=EventType.SYSTEM_ALERT,
            source=self.service_name,
            data={
                "message_type": message_type,
                "target_service": target_service,
                "payload": data
            }
        )

        await self.event_bus.publish_event(event)
        logger.info(f"Sent {message_type} message to {target_service}")

    async def subscribe_to_service_messages(self, message_types: List[str], callback: Callable):
        """订阅来自其他服务的消息"""
        async def service_message_handler(event: Event):
            data = event.data
            if data.get("target_service") == self.service_name:
                message_type = data.get("message_type")
                if message_type in message_types:
                    await callback(event)

        # 订阅系统消息事件
        await self.event_bus.consume_events(
            self.service_name,
            [EventType.SYSTEM_ALERT],
            service_message_handler
        )


# 工具函数
def create_event(
    event_type: EventType,
    source: str,
    data: Dict[str, Any],
    user_id: Optional[str] = None,
    priority: int = 5,
    correlation_id: Optional[str] = None
) -> Event:
    """创建事件的便捷函数"""
    return Event(
        type=event_type,
        source=source,
        data=data,
        user_id=user_id,
        priority=priority,
        correlation_id=correlation_id
    )


def create_correlated_events(
    base_event: Event,
    new_events: List[Tuple[EventType, Dict[str, Any]]]
) -> List[Event]:
    """创建相关联的事件序列"""
    events = []
    correlation_id = base_event.correlation_id or base_event.id

    for event_type, data in new_events:
        event = Event(
            type=event_type,
            source=base_event.source,
            data=data,
            correlation_id=correlation_id,
            causation_id=base_event.id,
            user_id=base_event.user_id,
            priority=base_event.priority
        )
        events.append(event)

    return events


# 事件装饰器
def event_handler(event_type: EventType):
    """事件处理器装饰器"""
    def decorator(func):
        func.event_type = event_type
        return func
    return decorator


class EventSourcing:
    """事件溯源存储"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def save_event(self, event: Event):
        """保存事件到事件存储"""
        stream_name = "event_store:all"
        await self.redis.xadd(
            stream_name,
            {
                "event_data": json.dumps(event.to_dict(), ensure_ascii=False),
                "event_type": event.type.value,
                "source": event.source,
                "timestamp": event.timestamp.isoformat()
            }
        )

    async def get_events_for_aggregate(self, aggregate_id: str, from_version: int = 0) -> List[Event]:
        """获取聚合对象的所有事件"""
        stream_name = f"event_store:aggregate:{aggregate_id}"
        events = []

        try:
            messages = await self.redis.xrange(stream_name, min=from_version)
            for message_id, fields in messages:
                event_data = json.loads(fields[b'event_data'].decode('utf-8'))
                events.append(Event.from_dict(event_data))
        except Exception as e:
            logger.error(f"Failed to get events for aggregate {aggregate_id}: {e}")

        return events

    async def create_snapshot(self, aggregate_id: str, snapshot_data: Dict[str, Any]):
        """创建聚合快照"""
        stream_name = f"snapshots:{aggregate_id}"
        await self.redis.xadd(
            stream_name,
            {
                "snapshot_data": json.dumps(snapshot_data, ensure_ascii=False),
                "created_at": datetime.utcnow().isoformat(),
                "version": snapshot_data.get("version", 0)
            }
        )