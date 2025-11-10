#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAgent Communication Protocols
XAgent通信协议

This module implements secure, efficient communication protocols for XAgent agents,
supporting various message types, priority handling, and secure message passing.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union, Set
from dataclasses import dataclass, asdict, field
from pathlib import Path
import hashlib
import threading
from queue import PriorityQueue, Queue
import redis
import aiofiles
import cryptography.fernet
from cryptography.fernet import Fernet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler('data/processed/xagent_communication.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MessageType(Enum):
    """XAgent message types"""
    TASK_ASSIGNMENT = "task_assignment"
    TASK_UPDATE = "task_update"
    TASK_COMPLETION = "task_completion"
    TASK_FAILURE = "task_failure"
    COLLABORATION_REQUEST = "collaboration_request"
    COLLABORATION_RESPONSE = "collaboration_response"
    KNOWLEDGE_SHARE = "knowledge_share"
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_RELEASE = "resource_release"
    STATUS_UPDATE = "status_update"
    HEARTBEAT = "heartbeat"
    ALERT = "alert"
    COORDINATION = "coordination"
    NEGOTIATION = "negotiation"
    SYNCHRONIZATION = "synchronization"

class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 0    # Safety incidents, system failures
    HIGH = 1       # Task failures, urgent collaboration
    NORMAL = 2     # Regular task communication
    LOW = 3        # Status updates, periodic sync
    BULK = 4       # Large data transfers, analytics

class DeliveryMode(Enum):
    """Message delivery modes"""
    FIRE_AND_FORGET = "fire_and_forget"
    ACKNOWLEDGED = "acknowledged"
    TRANSACTIONAL = "transactional"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"

@dataclass
class XAgentMessage:
    """XAgent message structure"""
    message_id: str
    message_type: MessageType
    priority: MessagePriority
    sender_id: str
    recipient_ids: Union[str, List[str]]
    timestamp: datetime
    payload: Dict[str, Any]
    delivery_mode: DeliveryMode = DeliveryMode.FIRE_AND_FORGET
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    requires_encryption: bool = False
    signature: Optional[str] = None

    def __post_init__(self):
        """Validate message structure"""
        if isinstance(self.recipient_ids, str):
            self.recipient_ids = [self.recipient_ids]

        if self.expires_at is None and self.priority in [MessagePriority.CRITICAL, MessagePriority.HIGH]:
            self.expires_at = datetime.now() + timedelta(hours=1)

class SecureMessageHandler:
    """Handles message encryption and decryption"""

    def __init__(self, encryption_key: Optional[bytes] = None):
        if encryption_key is None:
            encryption_key = Fernet.generate_key()
        self.fernet = Fernet(encryption_key)
        self.encryption_key = encryption_key

    def encrypt_message(self, message: XAgentMessage) -> XAgentMessage:
        """Encrypt message payload"""
        try:
            if not message.requires_encryption:
                return message

            # Convert payload to JSON and encrypt
            payload_json = json.dumps(message.payload, default=str, ensure_ascii=False)
            encrypted_payload = self.fernet.encrypt(payload_json.encode())

            # Create secure message copy
            secure_message = XAgentMessage(
                message_id=message.message_id,
                message_type=message.message_type,
                priority=message.priority,
                sender_id=message.sender_id,
                recipient_ids=message.recipient_ids,
                timestamp=message.timestamp,
                payload={"encrypted_data": encrypted_payload.decode(), "original_keys": list(message.payload.keys())},
                delivery_mode=message.delivery_mode,
                correlation_id=message.correlation_id,
                reply_to=message.reply_to,
                expires_at=message.expires_at,
                retry_count=message.retry_count,
                max_retries=message.max_retries,
                requires_encryption=True,
                signature=self._sign_message(message)
            )

            return secure_message

        except Exception as e:
            logger.error(f"❌ Failed to encrypt message {message.message_id}: {e}")
            return message

    def decrypt_message(self, message: XAgentMessage) -> XAgentMessage:
        """Decrypt message payload"""
        try:
            if not message.requires_encryption or "encrypted_data" not in message.payload:
                return message

            # Decrypt payload
            encrypted_data = message.payload["encrypted_data"]
            decrypted_payload = self.fernet.decrypt(encrypted_data.encode()).decode()

            # Restore original payload
            original_payload = json.loads(decrypted_payload)

            # Create decrypted message copy
            decrypted_message = XAgentMessage(
                message_id=message.message_id,
                message_type=message.message_type,
                priority=message.priority,
                sender_id=message.sender_id,
                recipient_ids=message.recipient_ids,
                timestamp=message.timestamp,
                payload=original_payload,
                delivery_mode=message.delivery_mode,
                correlation_id=message.correlation_id,
                reply_to=message.reply_to,
                expires_at=message.expires_at,
                retry_count=message.retry_count,
                max_retries=message.max_retries,
                requires_encryption=False,
                signature=message.signature
            )

            return decrypted_message

        except Exception as e:
            logger.error(f"❌ Failed to decrypt message {message.message_id}: {e}")
            return message

    def _sign_message(self, message: XAgentMessage) -> str:
        """Create message signature"""
        content = f"{message.message_id}{message.sender_id}{message.timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()

    def verify_signature(self, message: XAgentMessage) -> bool:
        """Verify message signature"""
        if not message.signature:
            return True

        expected_signature = self._sign_message(message)
        return message.signature == expected_signature

class MessageQueue:
    """Priority-based message queue with batching"""

    def __init__(self, max_size: int = 10000, batch_size: int = 50, batch_timeout: float = 0.1):
        self.queue = PriorityQueue(maxsize=max_size)
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.lock = threading.Lock()
        self.total_messages = 0
        self.processed_messages = 0

    def put(self, message: XAgentMessage) -> bool:
        """Add message to queue with priority"""
        try:
            priority_value = message.priority.value
            if message.expires_at and datetime.now() > message.expires_at:
                logger.warning(f"⚠️ Message {message.message_id} expired, skipping")
                return False

            self.queue.put((priority_value, time.time(), message), timeout=1)
            with self.lock:
                self.total_messages += 1

            logger.debug(f"📨 Queued message {message.message_id} with priority {message.priority.name}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to queue message {message.message_id}: {e}")
            return False

    def get(self, timeout: Optional[float] = None) -> Optional[XAgentMessage]:
        """Get message from queue"""
        try:
            _, _, message = self.queue.get(timeout=timeout or 1)
            with self.lock:
                self.processed_messages += 1

            # Check expiration
            if message.expires_at and datetime.now() > message.expires_at:
                logger.warning(f"⚠️ Message {message.message_id} expired during dequeue")
                return None

            return message

        except Exception as e:
            logger.debug(f"Queue get error: {e}")
            return None

    def get_batch(self) -> List[XAgentMessage]:
        """Get batch of messages for processing"""
        messages = []
        start_time = time.time()

        while len(messages) < self.batch_size and (time.time() - start_time) < self.batch_timeout:
            message = self.get(timeout=0.01)
            if message:
                messages.append(message)
            else:
                break

        return messages

    def size(self) -> int:
        """Get queue size"""
        return self.queue.qsize()

    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return self.queue.empty()

class XAgentCommunicationHub:
    """Central communication hub for XAgent agents"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_channels: Dict[str, Queue] = {}
        self.broadcast_channels: Dict[str, List[str]] = {}
        self.message_queue = MessageQueue(
            max_size=config.get("max_queue_size", 10000),
            batch_size=config.get("batch_size", 50),
            batch_timeout=config.get("batch_timeout", 0.1)
        )
        self.secure_handler = SecureMessageHandler()
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        self.delivery_confirmations: Dict[str, Dict] = {}
        self.message_stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_failed": 0,
            "messages_expired": 0
        }
        self.running = False
        self.processor_thread = None

        # Redis connection for distributed communication (optional)
        self.redis_client = None
        if config.get("use_redis", False):
            self.redis_client = redis.Redis(
                host=config.get("redis_host", "localhost"),
                port=config.get("redis_port", 6379),
                db=config.get("redis_db", 0),
                decode_responses=True
            )

    async def start(self):
        """Start the communication hub"""
        if self.running:
            logger.warning("⚠️ Communication hub already running")
            return

        self.running = True
        self.processor_thread = threading.Thread(target=self._process_messages, daemon=True)
        self.processor_thread.start()

        logger.info("🚀 XAgent Communication Hub started")

    async def stop(self):
        """Stop the communication hub"""
        self.running = False
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=5)

        logger.info("🛑 XAgent Communication Hub stopped")

    def register_agent(self, agent_id: str):
        """Register an agent with the communication hub"""
        if agent_id not in self.agent_channels:
            self.agent_channels[agent_id] = Queue()
            logger.info(f"✅ Registered agent: {agent_id}")

    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agent_channels:
            del self.agent_channels[agent_id]
            logger.info(f"📤 Unregistered agent: {agent_id}")

    async def send_message(self, message: XAgentMessage) -> bool:
        """Send a message to agent(s)"""
        try:
            # Validate message
            if not await self._validate_message(message):
                return False

            # Apply security if required
            if message.requires_encryption:
                message = self.secure_handler.encrypt_message(message)

            # Route message based on delivery mode
            success = await self._route_message(message)

            if success:
                self.message_stats["messages_sent"] += 1
                logger.debug(f"📤 Message {message.message_id} sent successfully")
            else:
                self.message_stats["messages_failed"] += 1
                logger.error(f"❌ Failed to send message {message.message_id}")

            return success

        except Exception as e:
            logger.error(f"❌ Error sending message {message.message_id}: {e}")
            self.message_stats["messages_failed"] += 1
            return False

    async def _validate_message(self, message: XAgentMessage) -> bool:
        """Validate message structure and content"""
        # Check required fields
        if not message.message_id or not message.sender_id or not message.recipient_ids:
            logger.error("❌ Message missing required fields")
            return False

        # Verify signature if present
        if message.signature and not self.secure_handler.verify_signature(message):
            logger.error(f"❌ Invalid signature for message {message.message_id}")
            return False

        # Check expiration
        if message.expires_at and datetime.now() > message.expires_at:
            logger.warning(f"⚠️ Message {message.message_id} has expired")
            return False

        # Check recipient registration
        if message.delivery_mode in [DeliveryMode.FIRE_AND_FORGET, DeliveryMode.ACKNOWLEDGED]:
            for recipient_id in message.recipient_ids:
                if recipient_id not in self.agent_channels and recipient_id != "broadcast":
                    logger.warning(f"⚠️ Recipient {recipient_id} not registered")
                    return False

        return True

    async def _route_message(self, message: XAgentMessage) -> bool:
        """Route message based on delivery mode"""
        try:
            if message.delivery_mode == DeliveryMode.BROADCAST:
                return await self._broadcast_message(message)
            elif message.delivery_mode == DeliveryMode.MULTICAST:
                return await self._multicast_message(message)
            else:
                return await self._direct_message(message)

        except Exception as e:
            logger.error(f"❌ Error routing message {message.message_id}: {e}")
            return False

    async def _broadcast_message(self, message: XAgentMessage) -> bool:
        """Broadcast message to all agents"""
        success_count = 0

        for agent_id in self.agent_channels.keys():
            if agent_id != message.sender_id:  # Don't send to self
                agent_message = XAgentMessage(
                    message_id=message.message_id,
                    message_type=message.message_type,
                    priority=message.priority,
                    sender_id=message.sender_id,
                    recipient_ids=[agent_id],
                    timestamp=message.timestamp,
                    payload=message.payload,
                    delivery_mode=DeliveryMode.FIRE_AND_FORGET,
                    correlation_id=message.correlation_id,
                    reply_to=message.reply_to
                )

                if self.message_queue.put(agent_message):
                    success_count += 1

        logger.debug(f"📡 Broadcast message {message.message_id} to {success_count} agents")
        return success_count > 0

    async def _multicast_message(self, message: XAgentMessage) -> bool:
        """Multicast message to specific agent groups"""
        success_count = 0

        for recipient_id in message.recipient_ids:
            if recipient_id in self.agent_channels and recipient_id != message.sender_id:
                agent_message = XAgentMessage(
                    message_id=message.message_id,
                    message_type=message.message_type,
                    priority=message.priority,
                    sender_id=message.sender_id,
                    recipient_ids=[recipient_id],
                    timestamp=message.timestamp,
                    payload=message.payload,
                    delivery_mode=DeliveryMode.FIRE_AND_FORGET,
                    correlation_id=message.correlation_id,
                    reply_to=message.reply_to
                )

                if self.message_queue.put(agent_message):
                    success_count += 1

        logger.debug(f"📨 Multicast message {message.message_id} to {success_count} agents")
        return success_count > 0

    async def _direct_message(self, message: XAgentMessage) -> bool:
        """Send direct message to specific recipients"""
        success_count = 0

        for recipient_id in message.recipient_ids:
            if recipient_id == "broadcast":
                continue

            # Add to message queue for processing
            if self.message_queue.put(message):
                success_count += 1

                # Handle acknowledgments
                if message.delivery_mode == DeliveryMode.ACKNOWLEDGED:
                    self.delivery_confirmations[message.message_id] = {
                        "recipients": message.recipient_ids,
                        "confirmed": [],
                        "timestamp": datetime.now(),
                        "timeout": datetime.now() + timedelta(minutes=5)
                    }

        return success_count > 0

    def _process_messages(self):
        """Process messages from queue (background thread)"""
        logger.info("🔄 Message processor started")

        while self.running:
            try:
                # Get batch of messages
                messages = self.message_queue.get_batch()

                if messages:
                    logger.debug(f"📦 Processing batch of {len(messages)} messages")

                    for message in messages:
                        self._deliver_message(message)

                # Small sleep to prevent busy waiting
                time.sleep(0.001)

            except Exception as e:
                logger.error(f"❌ Error in message processor: {e}")
                time.sleep(0.1)

        logger.info("🛑 Message processor stopped")

    def _deliver_message(self, message: XAgentMessage):
        """Deliver message to recipient channels"""
        try:
            # Decrypt if necessary
            if message.requires_encryption:
                message = self.secure_handler.decrypt_message(message)

            delivered_count = 0
            for recipient_id in message.recipient_ids:
                if recipient_id == "broadcast":
                    # Skip broadcast handled in routing
                    continue

                if recipient_id in self.agent_channels:
                    try:
                        self.agent_channels[recipient_id].put_nowait(message)
                        delivered_count += 1
                        self.message_stats["messages_delivered"] += 1

                        # Handle acknowledgments
                        if message.message_id in self.delivery_confirmations:
                            self.delivery_confirmations[message.message_id]["confirmed"].append(recipient_id)

                    except Exception as e:
                        logger.error(f"❌ Failed to deliver to {recipient_id}: {e}")

            if delivered_count == 0:
                logger.warning(f"⚠️ No recipients found for message {message.message_id}")

        except Exception as e:
            logger.error(f"❌ Error delivering message {message.message_id}: {e}")

    def get_messages_for_agent(self, agent_id: str, timeout: float = 1.0) -> List[XAgentMessage]:
        """Get pending messages for a specific agent"""
        messages = []

        if agent_id in self.agent_channels:
            channel = self.agent_channels[agent_id]

            # Collect all available messages
            while not channel.empty():
                try:
                    message = channel.get_nowait()
                    messages.append(message)
                except:
                    break

        return messages

    async def send_task_assignment(self, sender_id: str, recipient_ids: List[str],
                                 task_data: Dict[str, Any], priority: MessagePriority = MessagePriority.NORMAL) -> str:
        """Send task assignment message"""
        message = XAgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.TASK_ASSIGNMENT,
            priority=priority,
            sender_id=sender_id,
            recipient_ids=recipient_ids,
            timestamp=datetime.now(),
            payload=task_data,
            delivery_mode=DeliveryMode.ACKNOWLEDGED,
            requires_encryption=True
        )

        await self.send_message(message)
        return message.message_id

    async def send_collaboration_request(self, sender_id: str, recipient_ids: List[str],
                                       collaboration_data: Dict[str, Any]) -> str:
        """Send collaboration request message"""
        message = XAgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.COLLABORATION_REQUEST,
            priority=MessagePriority.HIGH,
            sender_id=sender_id,
            recipient_ids=recipient_ids,
            timestamp=datetime.now(),
            payload=collaboration_data,
            delivery_mode=DeliveryMode.TRANSACTIONAL,
            correlation_id=str(uuid.uuid4())
        )

        await self.send_message(message)
        return message.message_id

    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication hub statistics"""
        return {
            "message_stats": self.message_stats,
            "registered_agents": list(self.agent_channels.keys()),
            "queue_size": self.message_queue.size(),
            "delivery_confirmations": len(self.delivery_confirmations),
            "processor_running": self.running,
            "uptime": time.time() - getattr(self, 'start_time', time.time())
        }

# Example usage and testing
async def test_xagent_communication():
    """Test XAgent communication system"""
    logger.info("🧪 Testing XAgent Communication System")

    # Create communication hub
    config = {
        "max_queue_size": 1000,
        "batch_size": 10,
        "use_redis": False
    }

    hub = XAgentCommunicationHub(config)

    # Register test agents
    test_agents = ["agent1", "agent2", "agent3"]
    for agent_id in test_agents:
        hub.register_agent(agent_id)

    await hub.start()

    try:
        # Send test messages
        task_message = XAgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.TASK_ASSIGNMENT,
            priority=MessagePriority.NORMAL,
            sender_id="orchestrator",
            recipient_ids=["agent1"],
            timestamp=datetime.now(),
            payload={"task": "process_quality_data", "priority": "normal"}
        )

        await hub.send_message(task_message)

        # Send collaboration request
        collab_message = XAgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.COLLABORATION_REQUEST,
            priority=MessagePriority.HIGH,
            sender_id="agent1",
            recipient_ids=["agent2", "agent3"],
            timestamp=datetime.now(),
            payload={"request_type": "data_analysis", "requirements": ["statistical_analysis"]},
            delivery_mode=DeliveryMode.MULTICAST
        )

        await hub.send_message(collab_message)

        # Retrieve messages for agents
        for agent_id in test_agents:
            messages = hub.get_messages_for_agent(agent_id)
            logger.info(f"📨 Agent {agent_id} received {len(messages)} messages")
            for message in messages:
                logger.info(f"  - {message.message_type.value}: {message.payload}")

        # Get stats
        stats = hub.get_communication_stats()
        logger.info(f"📊 Communication stats: {json.dumps(stats, indent=2)}")

    finally:
        await hub.stop()

if __name__ == "__main__":
    asyncio.run(test_xagent_communication())