"""
Agent Communication Protocol
XAgent-inspired communication system for multi-agent coordination,
task delegation, and result aggregation with advanced message passing.
"""

import asyncio
import json
import logging
import uuid
import time
import hashlib
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import networkx as nx
import redis
import aiohttp
import websockets
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, PriorityQueue
import pickle
import zlib
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages between agents"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    TASK_DELEGATION = "task_delegation"
    RESULT_AGGREGATION = "result_aggregation"
    STATUS_UPDATE = "status_update"
    HEARTBEAT = "heartbeat"
    DISCOVERY = "discovery"
    REGISTRATION = "registration"
    COLLABORATION_REQUEST = "collaboration_request"
    COLLABORATION_RESPONSE = "collaboration_response"
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_RESPONSE = "resource_response"
    ERROR = "error"
    ACKNOWLEDGMENT = "acknowledgment"


class Priority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class MessageStatus(Enum):
    """Message delivery status"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    PROCESSED = "processed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class AgentMessage:
    """Standard message format for agent communication"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""
    message_type: MessageType = MessageType.TASK_REQUEST
    priority: Priority = Priority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3
    status: MessageStatus = MessageStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.correlation_id is None and self.message_type != MessageType.HEARTBEAT:
            self.correlation_id = self.message_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['message_type'] = self.message_type.value
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create message from dictionary"""
        data['message_type'] = MessageType(data['message_type'])
        data['priority'] = Priority(data['priority'])
        data['status'] = MessageStatus(data['status'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

    def compress(self) -> str:
        """Compress message for efficient transmission"""
        data = json.dumps(self.to_dict()).encode('utf-8')
        compressed = zlib.compress(data)
        return base64.b64encode(compressed).decode('utf-8')

    @classmethod
    def decompress(cls, compressed_data: str) -> 'AgentMessage':
        """Decompress message from transmission format"""
        compressed = base64.b64decode(compressed_data.encode('utf-8'))
        data = zlib.decompress(compressed).decode('utf-8')
        return cls.from_dict(json.loads(data))


@dataclass
class TaskRequest:
    """Task request payload"""
    task_id: str
    task_type: str
    task_description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    requirements: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[datetime] = None
    priority: Priority = Priority.NORMAL
    collaboration_needed: bool = False
    expected_outputs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['priority'] = self.priority.value
        if self.deadline:
            data['deadline'] = self.deadline.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskRequest':
        """Create from dictionary"""
        data['priority'] = Priority(data['priority'])
        if data.get('deadline'):
            data['deadline'] = datetime.fromisoformat(data['deadline'])
        return cls(**data)


@dataclass
class TaskResponse:
    """Task response payload"""
    task_id: str
    success: bool
    result: Any = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    resources_used: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskResponse':
        """Create from dictionary"""
        return cls(**data)


class MessageRouter:
    """Advanced message routing and delivery system"""

    def __init__(self, agent_id: str, redis_url: str = "redis://localhost:6379"):
        self.agent_id = agent_id
        self.redis_client = redis.from_url(redis_url)
        self.message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.pending_messages: Dict[str, AgentMessage] = {}
        self.message_queue = PriorityQueue()
        self.routing_table: Dict[str, str] = {}  # agent_id -> endpoint
        self.subscribed_topics: Set[str] = set()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.running = False
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'messages_failed': 0,
            'average_delivery_time': 0.0
        }

    async def start(self):
        """Start the message router"""
        self.running = True

        # Start message processing loop
        asyncio.create_task(self._process_message_queue())

        # Start heartbeat loop
        asyncio.create_task(self._heartbeat_loop())

        # Start message cleanup loop
        asyncio.create_task(self._cleanup_loop())

        # Subscribe to agent-specific channels
        await self._subscribe_to_channels()

        logger.info(f"Message router started for agent {self.agent_id}")

    async def stop(self):
        """Stop the message router"""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info(f"Message router stopped for agent {self.agent_id}")

    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register a message handler"""
        self.message_handlers[message_type].append(handler)
        logger.info(f"Registered handler for {message_type} messages")

    async def send_message(self, message: AgentMessage) -> bool:
        """Send a message to another agent"""
        try:
            # Set sender
            message.sender_id = self.agent_id

            # Route based on receiver type
            if message.receiver_id == "broadcast":
                success = await self._broadcast_message(message)
            elif message.receiver_id.startswith("group:"):
                success = await self._send_to_group(message)
            else:
                success = await self._send_to_agent(message)

            if success:
                message.status = MessageStatus.SENT
                self.stats['messages_sent'] += 1
                logger.debug(f"Message {message.message_id} sent to {message.receiver_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to send message {message.message_id}: {e}")
            message.status = MessageStatus.FAILED
            self.stats['messages_failed'] += 1
            return False

    async def _send_to_agent(self, message: AgentMessage) -> bool:
        """Send message directly to an agent"""
        try:
            # Get agent endpoint from routing table
            endpoint = self.routing_table.get(message.receiver_id)
            if not endpoint:
                # Fallback to Redis pub/sub
                channel = f"agent:{message.receiver_id}"
                self.redis_client.publish(channel, message.compress())
                return True

            # Send via HTTP/WebSocket if endpoint available
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{endpoint}/message",
                    json=message.to_dict(),
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200

        except Exception as e:
            logger.error(f"Failed to send message to {message.receiver_id}: {e}")
            return False

    async def _broadcast_message(self, message: AgentMessage) -> bool:
        """Broadcast message to all agents"""
        try:
            channel = "broadcast:all"
            self.redis_client.publish(channel, message.compress())
            return True
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
            return False

    async def _send_to_group(self, message: AgentMessage) -> bool:
        """Send message to a group of agents"""
        try:
            group_name = message.receiver_id[6:]  # Remove "group:" prefix
            channel = f"group:{group_name}"
            self.redis_client.publish(channel, message.compress())
            return True
        except Exception as e:
            logger.error(f"Failed to send message to group {group_name}: {e}")
            return False

    async def _subscribe_to_channels(self):
        """Subscribe to relevant Redis channels"""
        pubsub = self.redis_client.pubsub()

        # Subscribe to agent-specific channel
        agent_channel = f"agent:{self.agent_id}"
        pubsub.subscribe(agent_channel)

        # Subscribe to broadcast channel
        pubsub.subscribe("broadcast:all")

        # Subscribe to relevant group channels
        for topic in self.subscribed_topics:
            pubsub.subscribe(f"group:{topic}")

        # Start listening for messages
        asyncio.create_task(self._listen_for_messages(pubsub))

    async def _listen_for_messages(self, pubsub):
        """Listen for incoming messages"""
        while self.running:
            try:
                message = pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    # Decompress and process message
                    agent_message = AgentMessage.decompress(message['data'].decode('utf-8'))
                    await self._handle_incoming_message(agent_message)

            except Exception as e:
                logger.error(f"Error listening for messages: {e}")
                await asyncio.sleep(0.1)

    async def _handle_incoming_message(self, message: AgentMessage):
        """Handle incoming message"""
        try:
            # Update message status
            message.status = MessageStatus.DELIVERED
            self.stats['messages_received'] += 1

            # Check if this is a response to a pending message
            if message.reply_to and message.reply_to in self.pending_messages:
                original_message = self.pending_messages[message.reply_to]
                original_message.status = MessageStatus.PROCESSED
                del self.pending_messages[message.reply_to]

            # Add to processing queue
            priority_value = 5 - message.priority.value  # Invert for min-heap
            self.message_queue.put((priority_value, time.time(), message))

        except Exception as e:
            logger.error(f"Error handling incoming message: {e}")

    async def _process_message_queue(self):
        """Process messages from queue"""
        while self.running:
            try:
                # Get message from queue
                priority, timestamp, message = self.message_queue.get(timeout=1.0)

                # Check for timeout
                if time.time() - timestamp > message.timeout_seconds:
                    message.status = MessageStatus.TIMEOUT
                    continue

                # Process message
                await self._process_message(message)

            except:
                await asyncio.sleep(0.01)

    async def _process_message(self, message: AgentMessage):
        """Process a single message"""
        try:
            # Get handlers for message type
            handlers = self.message_handlers.get(message.message_type, [])

            # Call all handlers
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            self.executor, handler, message
                        )
                except Exception as e:
                    logger.error(f"Handler failed for message {message.message_id}: {e}")

            # Send acknowledgment if requested
            if message.correlation_id and message.reply_to is None:
                ack = AgentMessage(
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    message_type=MessageType.ACKNOWLEDGMENT,
                    reply_to=message.message_id,
                    correlation_id=message.correlation_id,
                    payload={'original_message_id': message.message_id}
                )
                await self.send_message(ack)

            message.status = MessageStatus.PROCESSED

        except Exception as e:
            logger.error(f"Error processing message {message.message_id}: {e}")
            message.status = MessageStatus.FAILED

    async def _heartbeat_loop(self):
        """Send periodic heartbeat messages"""
        while self.running:
            try:
                heartbeat = AgentMessage(
                    sender_id=self.agent_id,
                    receiver_id="broadcast",
                    message_type=MessageType.HEARTBEAT,
                    payload={
                        'timestamp': datetime.now().isoformat(),
                        'status': 'active',
                        'stats': self.stats
                    }
                )
                await self.send_message(heartbeat)
                await asyncio.sleep(30)  # Heartbeat every 30 seconds

            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")
                await asyncio.sleep(30)

    async def _cleanup_loop(self):
        """Clean up old messages and stats"""
        while self.running:
            try:
                # Clean up old pending messages
                current_time = datetime.now()
                expired_messages = [
                    msg_id for msg_id, msg in self.pending_messages.items()
                    if (current_time - msg.timestamp).total_seconds() > msg.timeout_seconds
                ]

                for msg_id in expired_messages:
                    message = self.pending_messages[msg_id]
                    message.status = MessageStatus.TIMEOUT
                    del self.pending_messages[msg_id]

                await asyncio.sleep(60)  # Cleanup every minute

            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
                await asyncio.sleep(60)


class TaskDelegator:
    """Advanced task delegation system with intelligent routing"""

    def __init__(self, message_router: MessageRouter):
        self.message_router = message_router
        self.agent_capabilities: Dict[str, Set[str]] = {}
        self.agent_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.task_queue = PriorityQueue()
        self.delegation_history: List[Dict[str, Any]] = []
        self.collaboration_cache: Dict[str, List[str]] = {}

    async def delegate_task(self, task: TaskRequest, preferred_agents: List[str] = None) -> str:
        """Delegate a task to the most suitable agent(s)"""
        try:
            # Find suitable agents
            suitable_agents = await self._find_suitable_agents(task, preferred_agents)

            if not suitable_agents:
                raise ValueError("No suitable agents found for task")

            # Determine delegation strategy
            if task.collaboration_needed and len(suitable_agents) > 1:
                return await self._delegate_collaborative_task(task, suitable_agents)
            else:
                return await self._delegate_single_task(task, suitable_agents)

        except Exception as e:
            logger.error(f"Failed to delegate task {task.task_id}: {e}")
            raise

    async def _find_suitable_agents(self, task: TaskRequest, preferred_agents: List[str] = None) -> List[str]:
        """Find agents capable of handling the task"""
        suitable_agents = []

        # Check requirements against agent capabilities
        for agent_id, capabilities in self.agent_capabilities.items():
            # Skip if not in preferred agents list
            if preferred_agents and agent_id not in preferred_agents:
                continue

            # Check if agent meets requirements
            if self._agent_meets_requirements(agent_id, task):
                suitable_agents.append(agent_id)

        # Sort by performance score
        suitable_agents.sort(
            key=lambda agent_id: self._calculate_agent_score(agent_id, task),
            reverse=True
        )

        return suitable_agents

    def _agent_meets_requirements(self, agent_id: str, task: TaskRequest) -> bool:
        """Check if agent meets task requirements"""
        capabilities = self.agent_capabilities.get(agent_id, set())
        requirements = task.requirements.get('capabilities', [])

        # Check if agent has required capabilities
        for req in requirements:
            if req not in capabilities:
                return False

        # Check resource constraints
        agent_resources = self.agent_performance.get(agent_id, {})
        required_resources = task.requirements.get('resources', {})

        for resource, min_amount in required_resources.items():
            available = agent_resources.get(resource, 0)
            if available < min_amount:
                return False

        return True

    def _calculate_agent_score(self, agent_id: str, task: TaskRequest) -> float:
        """Calculate score for agent suitability"""
        score = 0.0

        # Base score for having required capabilities
        capabilities = self.agent_capabilities.get(agent_id, set())
        required_caps = task.requirements.get('capabilities', [])
        capability_match = len([cap for cap in required_caps if cap in capabilities])
        score += capability_match * 10.0

        # Performance score
        performance = self.agent_performance.get(agent_id, {})
        score += performance.get('success_rate', 0.5) * 20.0
        score += performance.get('speed_score', 0.5) * 15.0
        score += performance.get('quality_score', 0.5) * 15.0

        # Availability score
        availability = performance.get('availability', 0.5)
        score += availability * 10.0

        # Task-specific experience
        task_type = task.task_type
        experience = performance.get(f'experience_{task_type}', 0.0)
        score += experience * 20.0

        return score

    async def _delegate_single_task(self, task: TaskRequest, agents: List[str]) -> str:
        """Delegate task to a single best agent"""
        best_agent = agents[0]

        message = AgentMessage(
            sender_id=self.message_router.agent_id,
            receiver_id=best_agent,
            message_type=MessageType.TASK_DELEGATION,
            payload={'task_request': task.to_dict()},
            priority=task.priority
        )

        success = await self.message_router.send_message(message)

        if success:
            # Record delegation
            self.delegation_history.append({
                'task_id': task.task_id,
                'agent_id': best_agent,
                'delegated_at': datetime.now().isoformat(),
                'strategy': 'single_agent'
            })

            logger.info(f"Task {task.task_id} delegated to agent {best_agent}")
            return best_agent
        else:
            raise RuntimeError(f"Failed to delegate task to agent {best_agent}")

    async def _delegate_collaborative_task(self, task: TaskRequest, agents: List[str]) -> str:
        """Delegate task for collaborative execution"""
        # Primary agent
        primary_agent = agents[0]
        # Supporting agents
        supporting_agents = agents[1:3]  # Limit to 2 supporting agents

        # Create collaboration request
        collaboration_id = str(uuid.uuid4())

        message = AgentMessage(
            sender_id=self.message_router.agent_id,
            receiver_id=primary_agent,
            message_type=MessageType.COLLABORATION_REQUEST,
            payload={
                'task_request': task.to_dict(),
                'collaboration_id': collaboration_id,
                'supporting_agents': supporting_agents,
                'coordination_required': True
            },
            priority=task.priority
        )

        success = await self.message_router.send_message(message)

        if success:
            # Notify supporting agents
            for agent_id in supporting_agents:
                support_message = AgentMessage(
                    sender_id=self.message_router.agent_id,
                    receiver_id=agent_id,
                    message_type=MessageType.COLLABORATION_REQUEST,
                    payload={
                        'collaboration_id': collaboration_id,
                        'primary_agent': primary_agent,
                        'task_request': task.to_dict(),
                        'role': 'supporting'
                    },
                    priority=task.priority
                )
                await self.message_router.send_message(support_message)

            # Record delegation
            self.delegation_history.append({
                'task_id': task.task_id,
                'primary_agent': primary_agent,
                'supporting_agents': supporting_agents,
                'collaboration_id': collaboration_id,
                'delegated_at': datetime.now().isoformat(),
                'strategy': 'collaborative'
            })

            # Cache collaboration
            self.collaboration_cache[collaboration_id] = [primary_agent] + supporting_agents

            logger.info(f"Collaborative task {task.task_id} delegated to {len(agents)} agents")
            return collaboration_id
        else:
            raise RuntimeError(f"Failed to delegate collaborative task to agent {primary_agent}")

    def update_agent_capabilities(self, agent_id: str, capabilities: Set[str]):
        """Update agent capabilities"""
        self.agent_capabilities[agent_id] = capabilities
        logger.info(f"Updated capabilities for agent {agent_id}: {capabilities}")

    def update_agent_performance(self, agent_id: str, performance_metrics: Dict[str, float]):
        """Update agent performance metrics"""
        self.agent_performance[agent_id].update(performance_metrics)
        logger.info(f"Updated performance for agent {agent_id}: {performance_metrics}")

    def get_delegation_stats(self) -> Dict[str, Any]:
        """Get delegation statistics"""
        recent_delegations = [
            d for d in self.delegation_history
            if (datetime.now() - datetime.fromisoformat(d['delegated_at'])).total_seconds() < 3600
        ]

        return {
            'total_delegations': len(self.delegation_history),
            'recent_delegations': len(recent_delegations),
            'collaborative_tasks': len([d for d in recent_delegations if d.get('strategy') == 'collaborative']),
            'active_collaborations': len(self.collaboration_cache),
            'agent_count': len(self.agent_capabilities)
        }


# Factory functions
def create_message_router(agent_id: str, redis_url: str = "redis://localhost:6379") -> MessageRouter:
    """Create a message router"""
    return MessageRouter(agent_id, redis_url)


def create_task_delegator(message_router: MessageRouter) -> TaskDelegator:
    """Create a task delegator"""
    return TaskDelegator(message_router)


# Usage example
if __name__ == "__main__":
    async def test_communication():
        # Test basic message routing
        router1 = create_message_router("agent1")
        router2 = create_message_router("agent2")

        await router1.start()
        await router2.start()

        # Send test message
        task_request = TaskRequest(
            task_id="test_task_001",
            task_type="document_processing",
            task_description="Process incoming documents",
            parameters={'file_path': '/path/to/file.pdf'}
        )

        message = AgentMessage(
            sender_id="agent1",
            receiver_id="agent2",
            message_type=MessageType.TASK_DELEGATION,
            payload={'task_request': task_request.to_dict()}
        )

        success = await router1.send_message(message)
        print(f"Message sent: {success}")

        await asyncio.sleep(2)

        await router1.stop()
        await router2.stop()

    asyncio.run(test_communication())