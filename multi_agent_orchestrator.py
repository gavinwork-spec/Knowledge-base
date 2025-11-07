#!/usr/bin/env python3
"""
Multi-Agent Orchestration System
Implements sophisticated multi-agent collaboration with task decomposition,
agent communication protocols, and autonomous problem-solving capabilities.
Inspired by XAgent and leading multi-agent frameworks.
"""

import asyncio
import json
import logging
import uuid
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import networkx as nx
import redis
import asyncpg
from pathlib import Path
import yaml
import pickle
import threading
from queue import PriorityQueue, Empty
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent execution status"""
    IDLE = "idle"
    BUSY = "busy"
    WAITING = "waiting"
    FAILED = "failed"
    COMPLETED = "completed"
    SUSPENDED = "suspended"

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    WAITING_FOR_AGENTS = "waiting_for_agents"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DECOMPOSED = "decomposed"

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

class MessagePriority(Enum):
    """Message priority levels"""
    URGENT = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

class AgentCapability(Enum):
    """Agent capability types"""
    DATA_PROCESSING = "data_processing"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    TEXT_ANALYSIS = "text_analysis"
    DOCUMENT_PROCESSING = "document_processing"
    FILE_SYSTEM_MONITOR = "file_system_monitor"
    DATABASE_MANAGEMENT = "database_management"
    API_INTEGRATION = "api_integration"
    MACHINE_LEARNING = "machine_learning"
    NOTIFICATION = "notification"
    REPORTING = "reporting"
    COORDINATION = "coordination"

@dataclass
class AgentMessage:
    """Message between agents"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""
    message_type: str = "task"
    content: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    reply_to: Optional[str] = None
    correlation_id: Optional[str] = None
    requires_reply: bool = False
    reply_timeout: int = 300  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentTask:
    """Task definition for agent execution"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_task_id: Optional[str] = None
    task_type: str = "general"
    title: str = ""
    description: str = ""
    required_capabilities: List[AgentCapability] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    estimated_duration: Optional[int] = None  # seconds
    max_retry_attempts: int = 3
    retry_count: int = 0
    subtasks: List['AgentTask'] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            'id': self.id,
            'parent_task_id': self.parent_task_id,
            'task_type': self.task_type,
            'title': self.title,
            'description': self.description,
            'required_capabilities': [cap.value for cap in self.required_capabilities],
            'parameters': self.parameters,
            'dependencies': self.dependencies,
            'priority': self.priority.value,
            'status': self.status.value,
            'assigned_agent_id': self.assigned_agent_id,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'estimated_duration': self.estimated_duration,
            'max_retry_attempts': self.max_retry_attempts,
            'retry_count': self.retry_count,
            'subtasks': [subtask.to_dict() for subtask in self.subtasks],
            'result': self.result,
            'error_message': self.error_message,
            'metadata': self.metadata
        }

@dataclass
class AgentProfile:
    """Agent profile and capabilities"""
    id: str = ""
    name: str = ""
    agent_type: str = "general"
    description: str = ""
    capabilities: List[AgentCapability] = field(default_factory=list)
    max_concurrent_tasks: int = 1
    priority_weight: float = 1.0
    reliability_score: float = 1.0
    average_task_duration: float = 300.0  # seconds
    success_rate: float = 1.0
    last_active: Optional[datetime] = None
    workload_capacity: int = 100
    current_workload: int = 0
    specializations: List[str] = field(default_factory=list)
    communication_protocols: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseAgent(ABC):
    """Abstract base class for all agents"""

    def __init__(self, agent_id: str, name: str, orchestrator):
        self.id = agent_id
        self.name = name
        self.orchestrator = orchestrator
        self.status = AgentStatus.IDLE
        self.current_tasks = []
        self.completed_tasks = []
        self.failed_tasks = []
        self.message_queue = asyncio.Queue()
        self.capabilities = []
        self.profile = AgentProfile(id=agent_id, name=name)
        self.start_time = datetime.now()
        self.last_heartbeat = datetime.now()
        self.performance_metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'average_response_time': 0.0,
            'message_count': 0
        }

    @abstractmethod
    async def initialize(self):
        """Initialize the agent"""
        pass

    @abstractmethod
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a specific task"""
        pass

    @abstractmethod
    def get_capabilities(self) -> List[AgentCapability]:
        """Get agent capabilities"""
        pass

    async def start(self):
        """Start the agent main loop"""
        logger.info(f"Starting agent {self.name} ({self.id})")
        await self.initialize()

        # Start message processing loop
        asyncio.create_task(self._message_loop())

        # Start heartbeat
        asyncio.create_task(self._heartbeat_loop())

    async def stop(self):
        """Stop the agent"""
        logger.info(f"Stopping agent {self.name} ({self.id})")
        self.status = AgentStatus.IDLE

    async def _message_loop(self):
        """Process incoming messages"""
        while True:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                await self._handle_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in message loop for agent {self.id}: {e}")

    async def _handle_message(self, message: AgentMessage):
        """Handle incoming message"""
        self.performance_metrics['message_count'] += 1

        try:
            if message.message_type == "task_assignment":
                await self._handle_task_assignment(message)
            elif message.message_type == "task_cancellation":
                await self._handle_task_cancellation(message)
            elif message.message_type == "status_request":
                await self._handle_status_request(message)
            elif message.message_type == "capability_request":
                await self._handle_capability_request(message)
            elif message.message_type == "ping":
                await self._handle_ping(message)
            else:
                logger.warning(f"Unknown message type: {message.message_type}")

            # Send reply if required
            if message.requires_reply:
                reply = AgentMessage(
                    sender_id=self.id,
                    receiver_id=message.sender_id,
                    message_type="reply",
                    content={"status": "received", "message_id": message.id},
                    reply_to=message.id,
                    correlation_id=message.correlation_id
                )
                await self.orchestrator.send_message(reply)

        except Exception as e:
            logger.error(f"Error handling message in agent {self.id}: {e}")
            if message.requires_reply:
                error_reply = AgentMessage(
                    sender_id=self.id,
                    receiver_id=message.sender_id,
                    message_type="error",
                    content={"error": str(e), "message_id": message.id},
                    reply_to=message.id,
                    correlation_id=message.correlation_id
                )
                await self.orchestrator.send_message(error_reply)

    async def _handle_task_assignment(self, message: AgentMessage):
        """Handle task assignment message"""
        task_data = message.content.get('task')
        if not task_data:
            logger.error(f"No task data in assignment message for agent {self.id}")
            return

        task = AgentTask(**task_data)
        await self.assign_task(task)

    async def _handle_task_cancellation(self, message: AgentMessage):
        """Handle task cancellation message"""
        task_id = message.content.get('task_id')
        if task_id:
            await self.cancel_task(task_id)

    async def _handle_status_request(self, message: AgentMessage):
        """Handle status request message"""
        status_info = await self.get_status()
        # Status response will be handled by orchestrator

    async def _handle_capability_request(self, message: AgentMessage):
        """Handle capability request message"""
        capabilities = [cap.value for cap in self.get_capabilities()]
        # Capability response will be handled by orchestrator

    async def _handle_ping(self, message: AgentMessage):
        """Handle ping message"""
        self.last_heartbeat = datetime.now()

    async def assign_task(self, task: AgentTask):
        """Assign a task to this agent"""
        if len(self.current_tasks) >= self.profile.max_concurrent_tasks:
            logger.warning(f"Agent {self.id} is at maximum capacity")
            return False

        logger.info(f"Assigning task {task.id} to agent {self.id}")
        self.current_tasks.append(task)
        self.status = AgentStatus.BUSY
        task.assigned_agent_id = self.id
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()

        # Start task execution
        asyncio.create_task(self._execute_task_wrapper(task))
        return True

    async def cancel_task(self, task_id: str):
        """Cancel a specific task"""
        for i, task in enumerate(self.current_tasks):
            if task.id == task_id:
                task.status = TaskStatus.CANCELLED
                self.current_tasks.pop(i)
                logger.info(f"Cancelled task {task_id} for agent {self.id}")
                return True
        return False

    async def _execute_task_wrapper(self, task: AgentTask):
        """Execute task with error handling and metrics"""
        start_time = time.time()

        try:
            logger.info(f"Agent {self.id} executing task {task.id}")
            result = await self.execute_task(task)

            # Task completed successfully
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()

            self.completed_tasks.append(task)
            self.current_tasks.remove(task)

            # Update metrics
            self.performance_metrics['tasks_completed'] += 1
            execution_time = time.time() - start_time
            self.performance_metrics['total_execution_time'] += execution_time

            # Update success rate
            total_tasks = self.performance_metrics['tasks_completed'] + self.performance_metrics['tasks_failed']
            if total_tasks > 0:
                self.performance_metrics['success_rate'] = self.performance_metrics['tasks_completed'] / total_tasks

            # Notify orchestrator
            await self.orchestrator.notify_task_completion(task.id, result)

            logger.info(f"Agent {self.id} completed task {task.id} in {execution_time:.2f}s")

        except Exception as e:
            # Task failed
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()

            self.failed_tasks.append(task)
            self.current_tasks.remove(task)

            # Update metrics
            self.performance_metrics['tasks_failed'] += 1
            execution_time = time.time() - start_time
            self.performance_metrics['total_execution_time'] += execution_time

            # Update success rate
            total_tasks = self.performance_metrics['tasks_completed'] + self.performance_metrics['tasks_failed']
            if total_tasks > 0:
                self.performance_metrics['success_rate'] = self.performance_metrics['tasks_completed'] / total_tasks

            # Notify orchestrator
            await self.orchestrator.notify_task_failure(task.id, str(e))

            logger.error(f"Agent {self.id} failed task {task.id}: {e}")

        finally:
            # Update agent status
            if len(self.current_tasks) == 0:
                self.status = AgentStatus.IDLE

    async def _heartbeat_loop(self):
        """Send periodic heartbeat to orchestrator"""
        while True:
            try:
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                heartbeat = AgentMessage(
                    sender_id=self.id,
                    receiver_id="orchestrator",
                    message_type="heartbeat",
                    content={
                        'status': self.status.value,
                        'current_tasks': len(self.current_tasks),
                        'completed_tasks': len(self.completed_tasks),
                        'failed_tasks': len(self.failed_tasks),
                        'timestamp': datetime.now().isoformat()
                    }
                )
                await self.orchestrator.send_message(heartbeat)
                self.last_heartbeat = datetime.now()
            except Exception as e:
                logger.error(f"Error sending heartbeat from agent {self.id}: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'id': self.id,
            'name': self.name,
            'status': self.status.value,
            'current_tasks': len(self.current_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'capabilities': [cap.value for cap in self.get_capabilities()],
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'performance_metrics': self.performance_metrics
        }

class TaskDecomposer:
    """Task decomposition engine for complex tasks"""

    def __init__(self):
        self.decomposition_strategies = {
            'sequential': self._sequential_decomposition,
            'parallel': self._parallel_decomposition,
            'hierarchical': self._hierarchical_decomposition,
            'conditional': self._conditional_decomposition,
            'pipeline': self._pipeline_decomposition
        }

    async def decompose_task(self, task: AgentTask) -> List[AgentTask]:
        """Decompose a complex task into subtasks"""
        logger.info(f"Decomposing task {task.id}: {task.title}")

        # Determine decomposition strategy
        strategy = self._determine_decomposition_strategy(task)

        # Apply decomposition
        subtasks = await self.decomposition_strategies[strategy](task)

        # Set up dependencies
        await self._setup_task_dependencies(task, subtasks)

        # Update parent task
        task.subtasks = subtasks
        task.status = TaskStatus.DECOMPOSED

        logger.info(f"Decomposed task {task.id} into {len(subtasks)} subtasks using {strategy} strategy")
        return subtasks

    def _determine_decomposition_strategy(self, task: AgentTask) -> str:
        """Determine the best decomposition strategy for a task"""
        # Analyze task characteristics
        if 'pipeline' in task.title.lower() or 'sequence' in task.title.lower():
            return 'pipeline'
        elif 'parallel' in task.title.lower() or 'concurrent' in task.title.lower():
            return 'parallel'
        elif 'conditional' in task.title.lower() or 'if' in task.title.lower():
            return 'conditional'
        elif task.parameters.get('hierarchical', False):
            return 'hierarchical'
        else:
            return 'sequential'  # Default strategy

    async def _sequential_decomposition(self, task: AgentTask) -> List[AgentTask]:
        """Decompose task into sequential subtasks"""
        subtasks = []

        # Example sequential decomposition based on common patterns
        if 'data_processing' in task.task_type:
            subtasks = await self._create_data_processing_subtasks(task)
        elif 'document_analysis' in task.task_type:
            subtasks = await self._create_document_analysis_subtasks(task)
        elif 'knowledge_extraction' in task.task_type:
            subtasks = await self._create_knowledge_extraction_subtasks(task)
        else:
            # Generic sequential decomposition
            subtasks = await self._create_generic_sequential_subtasks(task)

        return subtasks

    async def _parallel_decomposition(self, task: AgentTask) -> List[AgentTask]:
        """Decompose task into parallel subtasks"""
        subtasks = []

        # Example parallel decomposition
        if 'batch_processing' in task.task_type:
            subtasks = await self._create_batch_processing_subtasks(task)
        elif 'multi_source_analysis' in task.task_type:
            subtasks = await self._create_multi_source_subtasks(task)
        else:
            # Generic parallel decomposition
            subtasks = await self._create_generic_parallel_subtasks(task)

        return subtasks

    async def _hierarchical_decomposition(self, task: AgentTask) -> List[AgentTask]:
        """Decompose task into hierarchical subtasks"""
        subtasks = []

        # Create main categories
        main_categories = task.parameters.get('categories', ['primary', 'secondary'])

        for category in main_categories:
            subtask = AgentTask(
                parent_task_id=task.id,
                task_type=f"{task.task_type}_{category}",
                title=f"{task.title} - {category.title()}",
                description=f"{task.description} - {category} phase",
                required_capabilities=task.required_capabilities,
                parameters={**task.parameters, 'category': category},
                priority=task.priority,
                deadline=task.deadline
            )
            subtasks.append(subtask)

        return subtasks

    async def _conditional_decomposition(self, task: AgentTask) -> List[AgentTask]:
        """Decompose task based on conditions"""
        subtasks = []

        # Get conditions from task parameters
        conditions = task.parameters.get('conditions', [])

        for condition in conditions:
            subtask = AgentTask(
                parent_task_id=task.id,
                task_type=f"{task.task_type}_conditional",
                title=f"{task.title} - Condition: {condition.get('name', 'unnamed')}",
                description=f"Execute if condition met: {condition.get('description', '')}",
                required_capabilities=task.required_capabilities,
                parameters={**task.parameters, 'condition': condition},
                priority=task.priority,
                deadline=task.deadline
            )
            subtasks.append(subtask)

        return subtasks

    async def _pipeline_decomposition(self, task: AgentTask) -> List[AgentTask]:
        """Decompose task into pipeline stages"""
        subtasks = []

        # Define pipeline stages
        stages = task.parameters.get('pipeline_stages', ['input', 'process', 'output'])

        for i, stage in enumerate(stages):
            subtask = AgentTask(
                parent_task_id=task.id,
                task_type=f"{task.task_type}_stage",
                title=f"{task.title} - Stage {i+1}: {stage}",
                description=f"Pipeline stage {i+1}: {stage}",
                required_capabilities=task.required_capabilities,
                parameters={**task.parameters, 'stage': stage, 'stage_number': i + 1},
                priority=task.priority,
                deadline=task.deadline
            )
            subtasks.append(subtask)

        return subtasks

    async def _setup_task_dependencies(self, parent_task: AgentTask, subtasks: List[AgentTask]):
        """Set up dependencies between subtasks"""
        if not subtasks:
            return

        # For sequential and pipeline, create chain dependencies
        if parent_task.task_type in ['sequential', 'pipeline']:
            for i in range(len(subtasks) - 1):
                subtasks[i + 1].dependencies.append(subtasks[i].id)

        # For hierarchical, parent depends on all children
        elif parent_task.task_type == 'hierarchical':
            for subtask in subtasks:
                parent_task.dependencies.append(subtask.id)

        # For conditional, add conditions as dependencies
        elif parent_task.task_type == 'conditional':
            conditions = parent_task.parameters.get('conditions', [])
            for i, condition in enumerate(conditions):
                if i < len(subtasks):
                    subtasks[i].dependencies.append(f"condition_{i}")

    async def _create_data_processing_subtasks(self, task: AgentTask) -> List[AgentTask]:
        """Create subtasks for data processing"""
        return [
            AgentTask(
                parent_task_id=task.id,
                task_type="data_validation",
                title="Validate Input Data",
                description="Validate and clean input data",
                required_capabilities=[AgentCapability.DATA_PROCESSING],
                parameters={'validation_rules': task.parameters.get('validation_rules', {})},
                priority=task.priority
            ),
            AgentTask(
                parent_task_id=task.id,
                task_type="data_transformation",
                title="Transform Data",
                description="Apply transformations to the data",
                required_capabilities=[AgentCapability.DATA_PROCESSING],
                parameters={'transformations': task.parameters.get('transformations', {})},
                priority=task.priority
            ),
            AgentTask(
                parent_task_id=task.id,
                task_type="data_output",
                title="Save Results",
                description="Save processed data to output location",
                required_capabilities=[AgentCapability.DATA_PROCESSING],
                parameters={'output_path': task.parameters.get('output_path', '')},
                priority=task.priority
            )
        ]

    async def _create_document_analysis_subtasks(self, task: AgentTask) -> List[AgentTask]:
        """Create subtasks for document analysis"""
        return [
            AgentTask(
                parent_task_id=task.id,
                task_type="document_parsing",
                title="Parse Documents",
                description="Extract content from documents",
                required_capabilities=[AgentCapability.DOCUMENT_PROCESSING],
                parameters={'documents': task.parameters.get('documents', [])},
                priority=task.priority
            ),
            AgentTask(
                parent_task_id=task.id,
                task_type="content_analysis",
                title="Analyze Content",
                description="Analyze extracted content",
                required_capabilities=[AgentCapability.TEXT_ANALYSIS],
                parameters={'analysis_type': task.parameters.get('analysis_type', 'general')},
                priority=task.priority
            ),
            AgentTask(
                parent_task_id=task.id,
                task_type="knowledge_extraction",
                title="Extract Knowledge",
                description="Extract structured knowledge from content",
                required_capabilities=[AgentCapability.KNOWLEDGE_EXTRACTION],
                parameters={'extraction_rules': task.parameters.get('extraction_rules', {})},
                priority=task.priority
            )
        ]

    async def _create_knowledge_extraction_subtasks(self, task: AgentTask) -> List[AgentTask]:
        """Create subtasks for knowledge extraction"""
        return [
            AgentTask(
                parent_task_id=task.id,
                task_type="entity_extraction",
                title="Extract Entities",
                description="Extract named entities from text",
                required_capabilities=[AgentCapability.TEXT_ANALYSIS],
                parameters={'entity_types': task.parameters.get('entity_types', [])},
                priority=task.priority
            ),
            AgentTask(
                parent_task_id=task.id,
                task_type="relationship_extraction",
                title="Extract Relationships",
                description="Extract relationships between entities",
                required_capabilities=[AgentCapability.TEXT_ANALYSIS],
                parameters={'relationship_types': task.parameters.get('relationship_types', [])},
                priority=task.priority
            ),
            AgentTask(
                parent_task_id=task.id,
                task_type="knowledge_graph_update",
                title="Update Knowledge Graph",
                description="Update knowledge graph with extracted information",
                required_capabilities=[AgentCapability.KNOWLEDGE_EXTRACTION],
                parameters={'graph_config': task.parameters.get('graph_config', {})},
                priority=task.priority
            )
        ]

    async def _create_generic_sequential_subtasks(self, task: AgentTask) -> List[AgentTask]:
        """Create generic sequential subtasks"""
        return [
            AgentTask(
                parent_task_id=task.id,
                task_type="preparation",
                title="Prepare Resources",
                description="Prepare necessary resources and dependencies",
                required_capabilities=task.required_capabilities,
                parameters={},
                priority=task.priority
            ),
            AgentTask(
                parent_task_id=task.id,
                task_type="execution",
                title="Execute Main Task",
                description="Execute the main task logic",
                required_capabilities=task.required_capabilities,
                parameters=task.parameters,
                priority=task.priority
            ),
            AgentTask(
                parent_task_id=task.id,
                task_type="cleanup",
                title="Cleanup Resources",
                description="Clean up resources and finalize results",
                required_capabilities=task.required_capabilities,
                parameters={},
                priority=task.priority
            )
        ]

    async def _create_batch_processing_subtasks(self, task: AgentTask) -> List[AgentTask]:
        """Create parallel batch processing subtasks"""
        batch_items = task.parameters.get('batch_items', [])
        batch_size = task.parameters.get('batch_size', 10)

        subtasks = []
        for i in range(0, len(batch_items), batch_size):
            batch = batch_items[i:i + batch_size]
            subtask = AgentTask(
                parent_task_id=task.id,
                task_type="batch_process",
                title=f"Process Batch {i // batch_size + 1}",
                description=f"Process batch of {len(batch)} items",
                required_capabilities=task.required_capabilities,
                parameters={'batch_items': batch, 'batch_number': i // batch_size + 1},
                priority=task.priority
            )
            subtasks.append(subtask)

        return subtasks

    async def _create_multi_source_subtasks(self, task: AgentTask) -> List[AgentTask]:
        """Create parallel multi-source analysis subtasks"""
        sources = task.parameters.get('sources', [])

        subtasks = []
        for i, source in enumerate(sources):
            subtask = AgentTask(
                parent_task_id=task.id,
                task_type="source_analysis",
                title=f"Analyze Source {i + 1}",
                description=f"Analyze data from source: {source.get('name', 'unnamed')}",
                required_capabilities=task.required_capabilities,
                parameters={'source': source},
                priority=task.priority
            )
            subtasks.append(subtask)

        return subtasks

    async def _create_generic_parallel_subtasks(self, task: AgentTask) -> List[AgentTask]:
        """Create generic parallel subtasks"""
        parallel_count = task.parameters.get('parallel_count', 2)

        subtasks = []
        for i in range(parallel_count):
            subtask = AgentTask(
                parent_task_id=task.id,
                task_type="parallel_execution",
                title=f"Parallel Task {i + 1}",
                description=f"Execute parallel task instance {i + 1}",
                required_capabilities=task.required_capabilities,
                parameters={**task.parameters, 'instance_id': i + 1},
                priority=task.priority
            )
            subtasks.append(subtask)

        return subtasks

class MultiAgentOrchestrator:
    """Main multi-agent orchestration system"""

    def __init__(self, redis_client: redis.Redis = None):
        self.redis_client = redis_client
        self.agents = {}  # agent_id -> agent_instance
        self.agent_profiles = {}  # agent_id -> agent_profile
        self.tasks = {}  # task_id -> task_instance
        self.task_queue = PriorityQueue()
        self.message_queue = asyncio.Queue()
        self.task_dependencies = nx.DiGraph()  # Task dependency graph
        self.decomposer = TaskDecomposer()
        self.running = False
        self.start_time = datetime.now()

        # Performance metrics
        self.metrics = {
            'total_tasks_created': 0,
            'total_tasks_completed': 0,
            'total_tasks_failed': 0,
            'total_agents': 0,
            'average_task_duration': 0.0,
            'system_uptime': 0.0,
            'message_count': 0
        }

        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def start(self):
        """Start the orchestrator"""
        logger.info("Starting Multi-Agent Orchestrator")
        self.running = True

        # Start background tasks
        asyncio.create_task(self._message_loop())
        asyncio.create_task(self._task_scheduling_loop())
        asyncio.create_task(self._task_monitoring_loop())
        asyncio.create_task(self._agent_monitoring_loop())
        asyncio.create_task(self._metrics_collection_loop())

    async def stop(self):
        """Stop the orchestrator"""
        logger.info("Stopping Multi-Agent Orchestrator")
        self.running = False

        # Stop all agents
        for agent in self.agents.values():
            await agent.stop()

        # Shutdown executor
        self.executor.shutdown(wait=True)

    async def register_agent(self, agent: BaseAgent) -> bool:
        """Register a new agent"""
        try:
            # Update agent profile
            agent.profile.capabilities = agent.get_capabilities()
            self.agent_profiles[agent.id] = agent.profile

            # Start agent
            await agent.start()

            # Add to agents dict
            self.agents[agent.id] = agent

            self.metrics['total_agents'] += 1

            logger.info(f"Registered agent {agent.name} ({agent.id})")
            return True

        except Exception as e:
            logger.error(f"Failed to register agent {agent.id}: {e}")
            return False

    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        try:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                await agent.stop()

                del self.agents[agent_id]
                del self.agent_profiles[agent_id]

                self.metrics['total_agents'] -= 1

                logger.info(f"Unregistered agent {agent_id}")
                return True
            else:
                logger.warning(f"Agent {agent_id} not found")
                return False

        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False

    async def submit_task(self, task: AgentTask) -> str:
        """Submit a new task for execution"""
        task.id = str(uuid.uuid4())
        task.created_at = datetime.now()
        task.status = TaskStatus.PENDING

        self.tasks[task.id] = task
        self.metrics['total_tasks_created'] += 1

        # Check if task needs decomposition
        if await self._should_decompose_task(task):
            logger.info(f"Decomposing complex task {task.id}")
            subtasks = await self.decomposer.decompose_task(task)

            # Add subtasks to queue
            for subtask in subtasks:
                self.tasks[subtask.id] = subtask
                await self._enqueue_task(subtask)
        else:
            # Add task to queue directly
            await self._enqueue_task(task)

        logger.info(f"Submitted task {task.id}: {task.title}")
        return task.id

    async def _should_decompose_task(self, task: AgentTask) -> bool:
        """Determine if a task should be decomposed"""
        # Decompose if task is complex or has specific indicators
        complexity_indicators = [
            task.parameters.get('decompose', False),
            'complex' in task.title.lower(),
            len(task.description) > 500,
            task.estimated_duration and task.estimated_duration > 1800,  # 30 minutes
            len(task.required_capabilities) > 2
        ]

        return any(complexity_indicators)

    async def _enqueue_task(self, task: AgentTask):
        """Add task to the priority queue"""
        priority_value = task.priority.value
        self.task_queue.put((priority_value, task.id, task))

    async def send_message(self, message: AgentMessage) -> bool:
        """Send message to an agent"""
        try:
            if message.receiver_id == "orchestrator":
                # Message is for orchestrator
                await self.message_queue.put(message)
            elif message.receiver_id in self.agents:
                # Message is for specific agent
                await self.agents[message.receiver_id].message_queue.put(message)
            else:
                logger.warning(f"Unknown receiver: {message.receiver_id}")
                return False

            self.metrics['message_count'] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    async def _message_loop(self):
        """Process incoming messages"""
        while self.running:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                await self._handle_orchestrator_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in orchestrator message loop: {e}")

    async def _handle_orchestrator_message(self, message: AgentMessage):
        """Handle messages addressed to orchestrator"""
        if message.message_type == "heartbeat":
            # Update agent heartbeat
            if message.sender_id in self.agents:
                self.agents[message.sender_id].last_heartbeat = datetime.now()

        elif message.message_type == "task_completion":
            # Handle task completion notification
            task_id = message.content.get('task_id')
            result = message.content.get('result')
            await self._handle_task_completion(task_id, result)

        elif message.message_type == "task_failure":
            # Handle task failure notification
            task_id = message.content.get('task_id')
            error = message.content.get('error')
            await self._handle_task_failure(task_id, error)

    async def _task_scheduling_loop(self):
        """Schedule tasks to available agents"""
        while self.running:
            try:
                # Get next task from queue
                try:
                    priority, task_id, task = self.task_queue.get(timeout=1.0)
                except Empty:
                    await asyncio.sleep(0.1)
                    continue

                # Check if task dependencies are satisfied
                if not await self._are_dependencies_satisfied(task):
                    # Re-queue task for later
                    await self._enqueue_task(task)
                    continue

                # Find suitable agent
                suitable_agent = await self._find_suitable_agent(task)
                if suitable_agent:
                    # Assign task to agent
                    success = await suitable_agent.assign_task(task)
                    if success:
                        logger.info(f"Assigned task {task_id} to agent {suitable_agent.id}")
                    else:
                        # Re-queue task
                        await self._enqueue_task(task)
                        await asyncio.sleep(1.0)  # Brief delay before retry
                else:
                    # No suitable agent available, re-queue
                    await self._enqueue_task(task)
                    await asyncio.sleep(2.0)  # Longer delay for agent availability

            except Exception as e:
                logger.error(f"Error in task scheduling loop: {e}")

    async def _are_dependencies_satisfied(self, task: AgentTask) -> bool:
        """Check if all task dependencies are satisfied"""
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                dep_task = self.tasks[dep_id]
                if dep_task.status not in [TaskStatus.COMPLETED]:
                    return False
        return True

    async def _find_suitable_agent(self, task: AgentTask) -> Optional[BaseAgent]:
        """Find a suitable agent for the task"""
        suitable_agents = []

        for agent in self.agents.values():
            # Check if agent has required capabilities
            if not set(task.required_capabilities).issubset(set(agent.get_capabilities())):
                continue

            # Check if agent is available
            if agent.status != AgentStatus.IDLE:
                continue

            # Check if agent has capacity
            if len(agent.current_tasks) >= agent.profile.max_concurrent_tasks:
                continue

            suitable_agents.append(agent)

        if not suitable_agents:
            return None

        # Select best agent based on multiple criteria
        best_agent = min(suitable_agents, key=lambda a: (
            len(a.current_tasks),  # Prefer less loaded agents
            -a.profile.reliability_score,  # Prefer more reliable agents
            -a.profile.priority_weight  # Prefer higher priority agents
        ))

        return best_agent

    async def _task_monitoring_loop(self):
        """Monitor task execution and handle timeouts/failures"""
        while self.running:
            try:
                current_time = datetime.now()

                # Check for overdue tasks
                for task in self.tasks.values():
                    if (task.status == TaskStatus.RUNNING and
                        task.deadline and
                        current_time > task.deadline):

                        logger.warning(f"Task {task.id} is overdue")
                        await self._handle_task_timeout(task)

                # Check for stuck tasks (no heartbeat from assigned agent)
                for task in self.tasks.values():
                    if (task.status == TaskStatus.RUNNING and
                        task.assigned_agent_id and
                        task.assigned_agent_id in self.agents):

                        agent = self.agents[task.assigned_agent_id]
                        time_since_heartbeat = current_time - agent.last_heartbeat

                        if time_since_heartbeat > timedelta(minutes=5):
                            logger.warning(f"Agent {agent.id} appears unresponsive")
                            await self._handle_unresponsive_agent(agent, task)

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in task monitoring loop: {e}")

    async def _agent_monitoring_loop(self):
        """Monitor agent health and performance"""
        while self.running:
            try:
                current_time = datetime.now()

                # Check for inactive agents
                for agent in self.agents.values():
                    time_since_heartbeat = current_time - agent.last_heartbeat

                    if time_since_heartbeat > timedelta(minutes=2):
                        logger.warning(f"Agent {agent.id} missed heartbeat")

                        # Consider agent as failed
                        if time_since_heartbeat > timedelta(minutes=5):
                            logger.error(f"Agent {agent.id} considered failed")
                            await self._handle_failed_agent(agent)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in agent monitoring loop: {e}")

    async def _metrics_collection_loop(self):
        """Collect and update system metrics"""
        while self.running:
            try:
                # Update system uptime
                self.metrics['system_uptime'] = (datetime.now() - self.start_time).total_seconds()

                # Calculate average task duration
                completed_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]
                if completed_tasks:
                    durations = []
                    for task in completed_tasks:
                        if task.started_at and task.completed_at:
                            duration = (task.completed_at - task.started_at).total_seconds()
                            durations.append(duration)

                    if durations:
                        self.metrics['average_task_duration'] = sum(durations) / len(durations)

                # Store metrics in Redis if available
                if self.redis_client:
                    metrics_key = f"orchestrator:metrics:{int(time.time())}"
                    self.redis_client.setex(metrics_key, 3600, json.dumps(self.metrics))

                await asyncio.sleep(300)  # Update every 5 minutes

            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")

    async def notify_task_completion(self, task_id: str, result: Dict[str, Any]):
        """Handle task completion notification"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()

            self.metrics['total_tasks_completed'] += 1

            # Check if parent task needs to be updated
            if task.parent_task_id and task.parent_task_id in self.tasks:
                parent_task = self.tasks[task.parent_task_id]
                await self._check_parent_task_completion(parent_task)

            logger.info(f"Task {task_id} completed successfully")

    async def notify_task_failure(self, task_id: str, error: str):
        """Handle task failure notification"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.error_message = error
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()

            # Check if retry is possible
            if task.retry_count < task.max_retry_attempts:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                await self._enqueue_task(task)
                logger.info(f"Retrying task {task_id} (attempt {task.retry_count})")
            else:
                self.metrics['total_tasks_failed'] += 1
                logger.error(f"Task {task_id} failed permanently: {error}")

    async def _handle_task_timeout(self, task: AgentTask):
        """Handle task timeout"""
        logger.warning(f"Task {task.id} timed out")

        # Cancel task if it's still running
        if task.assigned_agent_id and task.assigned_agent_id in self.agents:
            agent = self.agents[task.assigned_agent_id]
            await agent.cancel_task(task.id)

        # Mark as failed
        task.status = TaskStatus.FAILED
        task.error_message = "Task timed out"
        await self.notify_task_failure(task.id, "Task timed out")

    async def _handle_unresponsive_agent(self, agent: BaseAgent, task: AgentTask):
        """Handle unresponsive agent"""
        logger.error(f"Agent {agent.id} is unresponsive for task {task.id}")

        # Mark agent as failed
        agent.status = AgentStatus.FAILED

        # Re-queue task for another agent
        task.status = TaskStatus.PENDING
        task.assigned_agent_id = None
        await self._enqueue_task(task)

    async def _handle_failed_agent(self, agent: BaseAgent):
        """Handle completely failed agent"""
        logger.error(f"Agent {agent.id} has failed")

        # Cancel all tasks assigned to this agent
        tasks_to_requeue = []
        for task in agent.current_tasks:
            tasks_to_requeue.append(task)

        # Unregister agent
        await self.unregister_agent(agent.id)

        # Re-queue tasks
        for task in tasks_to_requeue:
            task.status = TaskStatus.PENDING
            task.assigned_agent_id = None
            await self._enqueue_task(task)

    async def _check_parent_task_completion(self, parent_task: AgentTask):
        """Check if parent task should be marked as completed"""
        if not parent_task.subtasks:
            return

        # Check if all subtasks are completed
        all_completed = all(
            subtask.status == TaskStatus.COMPLETED
            for subtask in parent_task.subtasks
        )

        if all_completed:
            # Aggregate results from subtasks
            aggregated_result = await self._aggregate_subtask_results(parent_task)
            parent_task.result = aggregated_result
            parent_task.status = TaskStatus.COMPLETED
            parent_task.completed_at = datetime.now()

            await self.notify_task_completion(parent_task.id, aggregated_result)

    async def _aggregate_subtask_results(self, parent_task: AgentTask) -> Dict[str, Any]:
        """Aggregate results from subtasks"""
        results = {}

        for subtask in parent_task.subtasks:
            if subtask.result:
                results[subtask.id] = subtask.result

        return {
            'aggregated_from': len(parent_task.subtasks),
            'subtask_results': results,
            'parent_task_id': parent_task.id,
            'aggregation_timestamp': datetime.now().isoformat()
        }

    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            'orchestrator': {
                'status': 'running' if self.running else 'stopped',
                'uptime_seconds': self.metrics['system_uptime'],
                'total_agents': len(self.agents),
                'active_agents': len([a for a in self.agents.values() if a.status != AgentStatus.IDLE]),
                'total_tasks': len(self.tasks),
                'pending_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
                'running_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.RUNNING]),
                'completed_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]),
                'failed_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED]),
                'metrics': self.metrics
            },
            'agents': {
                agent_id: await agent.get_status()
                for agent_id, agent in self.agents.items()
            }
        }

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            return task.to_dict()
        return None

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a specific task"""
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]

        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False

        # Cancel task if it's assigned to an agent
        if task.assigned_agent_id and task.assigned_agent_id in self.agents:
            agent = self.agents[task.assigned_agent_id]
            success = await agent.cancel_task(task_id)
            if success:
                task.status = TaskStatus.CANCELLED
                return True

        # Mark as cancelled if not assigned
        task.status = TaskStatus.CANCELLED
        return True

# Example usage and testing
if __name__ == "__main__":
    async def test_multi_agent_system():
        """Test the multi-agent orchestration system"""

        # Create orchestrator
        orchestrator = MultiAgentOrchestrator()

        # Start orchestrator
        await orchestrator.start()

        print("Multi-Agent Orchestrator started successfully!")

        # Get system status
        status = await orchestrator.get_system_status()
        print(f"System Status: {json.dumps(status, indent=2, default=str)}")

        # Keep running for a while
        await asyncio.sleep(5)

        # Stop orchestrator
        await orchestrator.stop()
        print("Multi-Agent Orchestrator stopped")

    # Run test
    asyncio.run(test_multi_agent_system())