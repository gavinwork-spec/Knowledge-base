"""
Agent Marketplace
XAgent-inspired marketplace system for agent registration, discovery,
matching, and dynamic scaling with intelligent agent management.
"""

import asyncio
import json
import logging
import sqlite3
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import uuid
import time
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor

# Database and serialization
import yaml
import pickle
import aiofiles
import aiohttp

# Security and validation
import jwt
from cryptography.fernet import Fernet
from pydantic import BaseModel, ValidationError

# Analytics and monitoring
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Import agent framework
from multi_agent_orchestrator import BaseAgent, AgentTask, AgentCapability, AgentStatus
from multi_agent_system.protocols.agent_communication import (
    MessageRouter, TaskDelegator, AgentMessage, TaskRequest, TaskResponse,
    MessageType, Priority
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent status in marketplace"""
    REGISTERED = "registered"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    DEPRECATED = "deprecated"
    SUSPENDED = "suspended"


class AgentType(Enum):
    """Agent types in marketplace"""
    SPECIALIZED = "specialized"
    COORDINATOR = "coordinator"
    GENERIC = "generic"
    EXTERNAL = "external"
    HYBRID = "hybrid"


@dataclass
class AgentCapability:
    """Detailed agent capability definition"""
    capability_id: str
    name: str
    description: str
    category: str
    version: str
    complexity: str  # simple, moderate, complex
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    success_rate: float = 0.0
    average_execution_time: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class AgentRegistration:
    """Agent registration information"""
    agent_id: str
    name: str
    description: str
    agent_type: AgentType
    version: str
    author: str
    contact_info: Dict[str, str]
    capabilities: List[AgentCapability]
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    cost_model: Dict[str, Any] = field(default_factory=dict)
    sla_requirements: Dict[str, Any] = field(default_factory=dict)
    security_level: str = "standard"
    endpoints: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    status: AgentStatus = AgentStatus.REGISTERED


@dataclass
class MarketplaceMetrics:
    """Marketplace performance metrics"""
    total_agents: int = 0
    active_agents: int = 0
    total_tasks_processed: int = 0
    average_task_completion_time: float = 0.0
    marketplace_uptime: float = 0.0
    agent_turnover_rate: float = 0.0
    capability_coverage: Dict[str, int] = field(default_factory=dict)
    load_distribution: Dict[str, float] = field(default_factory=dict)


class AgentRegistry:
    """Registry for managing agent registrations and capabilities"""

    def __init__(self, db_path: str = "marketplace.db"):
        self.db_path = Path(db_path)
        self.registered_agents: Dict[str, AgentRegistration] = {}
        self.capability_index: Dict[str, Set[str]] = defaultdict(set)  # capability -> agent_ids
        self.type_index: Dict[AgentType, Set[str]] = defaultdict(set)  # type -> agent_ids
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)  # tag -> agent_ids
        self.registration_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        # Initialize database
        self._initialize_database()

        # Load existing registrations
        self._load_registrations()

    def _initialize_database(self):
        """Initialize marketplace database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Agents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                agent_type TEXT NOT NULL,
                version TEXT NOT NULL,
                author TEXT,
                contact_info TEXT,
                capabilities TEXT,
                resource_limits TEXT,
                cost_model TEXT,
                sla_requirements TEXT,
                security_level TEXT DEFAULT 'standard',
                endpoints TEXT,
                metadata TEXT,
                registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_heartbeat TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'registered'
            )
        ''')

        # Capabilities table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS capabilities (
                capability_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                category TEXT,
                version TEXT,
                complexity TEXT,
                resource_requirements TEXT,
                dependencies TEXT,
                tags TEXT,
                success_rate REAL DEFAULT 0.0,
                average_execution_time REAL DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Task history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS task_history (
                task_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                execution_time REAL,
                success BOOLEAN,
                error_message TEXT,
                metadata TEXT,
                FOREIGN KEY (agent_id) REFERENCES agents (agent_id)
            )
        ''')

        # Marketplace metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS marketplace_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_agents INTEGER,
                active_agents INTEGER,
                total_tasks_processed INTEGER,
                average_task_completion_time REAL,
                marketplace_uptime REAL,
                agent_turnover_rate REAL,
                capability_coverage TEXT,
                load_distribution TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def _load_registrations(self):
        """Load existing agent registrations from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Load agents
            cursor.execute("SELECT * FROM agents")
            for row in cursor.fetchall():
                agent_data = self._deserialize_agent_registration(row)
                if agent_data:
                    self.registered_agents[agent_data.agent_id] = agent_data
                    self._update_indexes(agent_data)

            conn.close()
            logger.info(f"Loaded {len(self.registered_agents)} agent registrations")

        except Exception as e:
            logger.error(f"Error loading registrations: {e}")

    def register_agent(self, registration: AgentRegistration) -> bool:
        """Register a new agent in the marketplace"""
        with self._lock:
            try:
                # Validate registration
                if not self._validate_registration(registration):
                    return False

                # Check if agent already exists
                if registration.agent_id in self.registered_agents:
                    # Update existing registration
                    existing = self.registered_agents[registration.agent_id]
                    registration.registered_at = existing.registered_at

                # Save to database
                self._save_registration_to_db(registration)

                # Update in-memory registry
                self.registered_agents[registration.agent_id] = registration
                self._update_indexes(registration)

                # Record registration history
                self.registration_history.append({
                    'action': 'register' if registration.agent_id not in self.registered_agents else 'update',
                    'agent_id': registration.agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'agent_type': registration.agent_type.value
                })

                logger.info(f"Agent {registration.agent_id} registered successfully")
                return True

            except Exception as e:
                logger.error(f"Error registering agent {registration.agent_id}: {e}")
                return False

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the marketplace"""
        with self._lock:
            try:
                if agent_id not in self.registered_agents:
                    logger.warning(f"Agent {agent_id} not found in registry")
                    return False

                # Remove from database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM agents WHERE agent_id = ?", (agent_id,))
                conn.commit()
                conn.close()

                # Remove from memory
                registration = self.registered_agents[agent_id]
                del self.registered_agents[agent_id]

                # Update indexes
                self._remove_from_indexes(registration)

                # Record registration history
                self.registration_history.append({
                    'action': 'unregister',
                    'agent_id': agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'agent_type': registration.agent_type.value
                })

                logger.info(f"Agent {agent_id} unregistered successfully")
                return True

            except Exception as e:
                logger.error(f"Error unregistering agent {agent_id}: {e}")
                return False

    def discover_agents(self,
                       capabilities: List[str] = None,
                       agent_type: AgentType = None,
                       tags: Set[str] = None,
                       status: AgentStatus = None,
                       min_success_rate: float = None,
                       max_execution_time: float = None) -> List[AgentRegistration]:
        """Discover agents matching specified criteria"""
        candidates = set(self.registered_agents.values())

        # Filter by capabilities
        if capabilities:
            capability_candidates = set()
            for capability in capabilities:
                capability_candidates.update(self.capability_index[capability])
            candidates.intersection_update(capability_candidates)

        # Filter by agent type
        if agent_type:
            type_candidates = self.type_index[agent_type]
            candidates.intersection_update(type_candidates)

        # Filter by tags
        if tags:
            tag_candidates = set()
            for tag in tags:
                tag_candidates.update(self.tag_index[tag])
            candidates.intersection_update(tag_candidates)

        # Filter by status
        if status:
            candidates = {agent for agent in candidates if agent.status == status}
        else:
            # By default, only return active agents
            candidates = {agent for agent in candidates if agent.status == AgentStatus.ACTIVE}

        # Apply performance filters
        if min_success_rate is not None:
            candidates = {agent for agent in candidates
                         if all(cap.success_rate >= min_success_rate for cap in agent.capabilities)}

        if max_execution_time is not None:
            candidates = {agent for agent in candidates
                         if all(cap.average_execution_time <= max_execution_time for cap in agent.capabilities)}

        return list(candidates)

    def find_best_agent(self, task_type: str, requirements: Dict[str, Any]) -> Optional[AgentRegistration]:
        """Find the best agent for a specific task"""
        candidates = self.discover_agents(
            capabilities=requirements.get('required_capabilities', []),
            tags=requirements.get('preferred_tags', set())
        )

        if not candidates:
            return None

        # Score candidates based on multiple factors
        scored_candidates = []
        for agent in candidates:
            score = self._calculate_agent_score(agent, task_type, requirements)
            scored_candidates.append((score, agent))

        # Return highest scoring agent
        if scored_candidates:
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            return scored_candidates[0][1]

        return None

    def _calculate_agent_score(self, agent: AgentRegistration, task_type: str, requirements: Dict[str, Any]) -> float:
        """Calculate score for agent suitability"""
        score = 0.0

        # Capability match score
        required_capabilities = set(requirements.get('required_capabilities', []))
        agent_capabilities = {cap.name for cap in agent.capabilities}

        if required_capabilities:
            capability_match = len(required_capabilities.intersection(agent_capabilities))
            score += (capability_match / len(required_capabilities)) * 40

        # Success rate score
        avg_success_rate = statistics.mean([cap.success_rate for cap in agent.capabilities]) if agent.capabilities else 0
        score += avg_success_rate * 25

        # Performance score (inverse execution time)
        avg_execution_time = statistics.mean([cap.average_execution_time for cap in agent.capabilities]) if agent.capabilities else 1.0
        performance_score = max(0, (1.0 - avg_execution_time / 100.0)) * 20  # Normalize to 0-20
        score += performance_score

        # Experience/reliability score
        if agent.registered_at:
            days_registered = (datetime.now() - agent.registered_at).days
            experience_score = min(days_registered / 365, 1.0) * 10
            score += experience_score

        # Recent activity score
        if agent.last_heartbeat:
            days_since_heartbeat = (datetime.now() - agent.last_heartbeat).days
            activity_score = max(0, (1.0 - days_since_heartbeat / 30)) * 5
            score += activity_score

        return score

    def update_agent_status(self, agent_id: str, status: AgentStatus, heartbeat: bool = True) -> bool:
        """Update agent status"""
        with self._lock:
            try:
                if agent_id not in self.registered_agents:
                    return False

                agent = self.registered_agents[agent_id]
                old_status = agent.status
                agent.status = status

                if heartbeat:
                    agent.last_heartbeat = datetime.now()

                # Update in database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE agents SET status = ?, last_heartbeat = ? WHERE agent_id = ?",
                    (status.value, agent.last_heartbeat.isoformat(), agent_id)
                )
                conn.commit()
                conn.close()

                # Log status change
                if old_status != status:
                    logger.info(f"Agent {agent_id} status changed from {old_status.value} to {status.value}")

                return True

            except Exception as e:
                logger.error(f"Error updating agent status {agent_id}: {e}")
                return False

    def update_capability_metrics(self, agent_id: str, capability_name: str,
                                success: bool, execution_time: float) -> bool:
        """Update capability performance metrics"""
        with self._lock:
            try:
                if agent_id not in self.registered_agents:
                    return False

                agent = self.registered_agents[agent_id]

                # Find the capability
                capability = None
                for cap in agent.capabilities:
                    if cap.name == capability_name:
                        capability = cap
                        break

                if not capability:
                    return False

                # Update metrics
                # Simple exponential moving average
                alpha = 0.1  # Learning rate
                capability.success_rate = capability.success_rate * (1 - alpha) + (1.0 if success else 0.0) * alpha
                capability.average_execution_time = capability.average_execution_time * (1 - alpha) + execution_time * alpha
                capability.last_updated = datetime.now()

                # Update in database
                self._save_capability_to_db(capability)

                return True

            except Exception as e:
                logger.error(f"Error updating capability metrics: {e}")
                return False

    def get_marketplace_metrics(self) -> MarketplaceMetrics:
        """Get marketplace performance metrics"""
        metrics = MarketplaceMetrics()

        # Basic counts
        metrics.total_agents = len(self.registered_agents)
        metrics.active_agents = len([a for a in self.registered_agents.values() if a.status == AgentStatus.ACTIVE])

        # Calculate capability coverage
        for capability_name, agent_ids in self.capability_index.items():
            metrics.capability_coverage[capability_name] = len(agent_ids)

        # Load task history for performance metrics
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Total tasks processed
            cursor.execute("SELECT COUNT(*) FROM task_history")
            metrics.total_tasks_processed = cursor.fetchone()[0]

            # Average completion time
            cursor.execute("SELECT AVG(execution_time) FROM task_history WHERE execution_time IS NOT NULL")
            result = cursor.fetchone()[0]
            metrics.average_task_completion_time = result if result else 0.0

            conn.close()
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")

        # Calculate load distribution
        total_tasks = max(metrics.total_tasks_processed, 1)
        for agent_id, agent in self.registered_agents.items():
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM task_history WHERE agent_id = ?",
                    (agent_id,)
                )
                agent_tasks = cursor.fetchone()[0]
                metrics.load_distribution[agent_id] = agent_tasks / total_tasks
                conn.close()
            except:
                metrics.load_distribution[agent_id] = 0.0

        return metrics

    def _validate_registration(self, registration: AgentRegistration) -> bool:
        """Validate agent registration"""
        # Check required fields
        if not registration.agent_id or not registration.name or not registration.agent_type:
            return False

        # Validate capabilities
        if not registration.capabilities:
            return False

        # Validate endpoints
        if not registration.endpoints:
            return False

        # Validate contact info
        if not registration.contact_info or 'email' not in registration.contact_info:
            return False

        return True

    def _save_registration_to_db(self, registration: AgentRegistration):
        """Save agent registration to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Serialize complex fields
        capabilities_data = json.dumps([asdict(cap) for cap in registration.capabilities])

        cursor.execute('''
            INSERT OR REPLACE INTO agents
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            registration.agent_id,
            registration.name,
            registration.description,
            registration.agent_type.value,
            registration.version,
            registration.author,
            json.dumps(registration.contact_info),
            capabilities_data,
            json.dumps(registration.resource_limits),
            json.dumps(registration.cost_model),
            json.dumps(registration.sla_requirements),
            registration.security_level,
            json.dumps(registration.endpoints),
            json.dumps(registration.metadata),
            registration.registered_at.isoformat(),
            registration.last_heartbeat.isoformat(),
            registration.status.value
        ))

        conn.commit()
        conn.close()

    def _save_capability_to_db(self, capability: AgentCapability):
        """Save capability to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO capabilities
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            capability.capability_id,
            capability.name,
            capability.description,
            capability.category,
            capability.version,
            capability.complexity,
            json.dumps(capability.resource_requirements),
            json.dumps(capability.dependencies),
            json.dumps(list(capability.tags)),
            capability.success_rate,
            capability.average_execution_time,
            capability.last_updated.isoformat()
        ))

        conn.commit()
        conn.close()

    def _deserialize_agent_registration(self, row) -> Optional[AgentRegistration]:
        """Deserialize agent registration from database row"""
        try:
            capabilities_data = json.loads(row[7]) if row[7] else []
            capabilities = []
            for cap_data in capabilities_data:
                capabilities.append(AgentCapability(
                    capability_id=cap_data['capability_id'],
                    name=cap_data['name'],
                    description=cap_data['description'],
                    category=cap_data['category'],
                    version=cap_data['version'],
                    complexity=cap_data['complexity'],
                    resource_requirements=cap_data.get('resource_requirements', {}),
                    dependencies=cap_data.get('dependencies', []),
                    tags=set(cap_data.get('tags', [])),
                    success_rate=cap_data.get('success_rate', 0.0),
                    average_execution_time=cap_data.get('average_execution_time', 0.0),
                    last_updated=datetime.fromisoformat(cap_data.get('last_updated', datetime.now().isoformat()))
                ))

            return AgentRegistration(
                agent_id=row[0],
                name=row[1],
                description=row[2],
                agent_type=AgentType(row[3]),
                version=row[4],
                author=row[5],
                contact_info=json.loads(row[6]) if row[6] else {},
                capabilities=capabilities,
                resource_limits=json.loads(row[8]) if row[8] else {},
                cost_model=json.loads(row[9]) if row[9] else {},
                sla_requirements=json.loads(row[10]) if row[10] else {},
                security_level=row[11],
                endpoints=json.loads(row[12]) if row[12] else {},
                metadata=json.loads(row[13]) if row[13] else {},
                registered_at=datetime.fromisoformat(row[14]) if row[14] else datetime.now(),
                last_heartbeat=datetime.fromisoformat(row[15]) if row[15] else datetime.now(),
                status=AgentStatus(row[16])
            )

        except Exception as e:
            logger.error(f"Error deserializing agent registration: {e}")
            return None

    def _update_indexes(self, registration: AgentRegistration):
        """Update search indexes for fast lookup"""
        # Update capability index
        for capability in registration.capabilities:
            self.capability_index[capability.name].add(registration.agent_id)

        # Update type index
        self.type_index[registration.agent_type].add(registration.agent_id)

        # Update tag index
        for capability in registration.capabilities:
            for tag in capability.tags:
                self.tag_index[tag].add(registration.agent_id)

    def _remove_from_indexes(self, registration: AgentRegistration):
        """Remove agent from search indexes"""
        # Remove from capability index
        for capability in registration.capabilities:
            self.capability_index[capability.name].discard(registration.agent_id)

        # Remove from type index
        self.type_index[registration.agent_type].discard(registration.agent_id)

        # Remove from tag index
        for capability in registration.capabilities:
            for tag in capability.tags:
                self.tag_index[tag].discard(registration.agent_id)


class AgentMarketplace:
    """Main marketplace for agent registration, discovery, and management"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.registry = AgentRegistry(self.config.get('db_path', 'marketplace.db'))
        self.message_router = None
        self.task_delegator = None
        self.running = False
        self.monitoring_enabled = self.config.get('monitoring_enabled', True)
        self.auto_scaling_enabled = self.config.get('auto_scaling_enabled', True)
        self.security_enabled = self.config.get('security_enabled', True)

        # Background tasks
        self.background_tasks = []

    async def start(self):
        """Start the marketplace"""
        self.running = True
        logger.info("Agent Marketplace started")

        # Start background monitoring
        if self.monitoring_enabled:
            await self._start_monitoring()

        # Start auto-scaling
        if self.auto_scaling_enabled:
            await self._start_auto_scaling()

    async def stop(self):
        """Stop the marketplace"""
        self.running = False

        # Stop background tasks
        for task in self.background_tasks:
            task.cancel()

        logger.info("Agent Marketplace stopped")

    async def register_external_agent(self, agent_info: Dict[str, Any]) -> str:
        """Register an external agent via API"""
        try:
            # Validate and create registration
            registration = await self._create_registration_from_info(agent_info)

            # Validate security
            if self.security_enabled:
                if not await self._validate_agent_security(registration):
                    raise ValueError("Security validation failed")

            # Register in marketplace
            success = self.registry.register_agent(registration)

            if success:
                # Send welcome message
                await self._send_welcome_message(registration)
                return registration.agent_id
            else:
                raise ValueError("Registration failed")

        except Exception as e:
            logger.error(f"Error registering external agent: {e}")
            raise

    async def discover_and_match(self, task_request: TaskRequest) -> Optional[str]:
        """Discover and match best agent for a task"""
        try:
            # Extract requirements from task
            requirements = {
                'required_capabilities': task_request.requirements.get('capabilities', []),
                'preferred_tags': set(task_request.requirements.get('tags', [])),
                'min_success_rate': task_request.requirements.get('min_success_rate', 0.5),
                'max_execution_time': task_request.requirements.get('max_execution_time', 3600)
            }

            # Find best agent
            best_agent = self.registry.find_best_agent(task_request.task_type, requirements)

            if best_agent:
                return best_agent.agent_id
            else:
                logger.warning(f"No suitable agent found for task type: {task_request.task_type}")
                return None

        except Exception as e:
            logger.error(f"Error discovering and matching agent: {e}")
            return None

    async def get_marketplace_status(self) -> Dict[str, Any]:
        """Get marketplace status and metrics"""
        metrics = self.registry.get_marketplace_metrics()

        return {
            'status': 'running' if self.running else 'stopped',
            'metrics': asdict(metrics),
            'total_capabilities': len(self.registry.capability_index),
            'agent_types': {agent_type.value: len(agents) for agent_type, agents in self.registry.type_index.items()},
            'recent_registrations': self.registry.registration_history[-10:],
            'health_check': await self._perform_health_check()
        }

    async def _start_monitoring(self):
        """Start background monitoring"""
        task = asyncio.create_task(self._monitoring_loop())
        self.background_tasks.append(task)

    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                # Check agent heartbeats
                await self._check_agent_heartbeats()

                # Update marketplace metrics
                await self._update_marketplace_metrics()

                # Perform health checks
                await self._perform_health_check()

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _check_agent_heartbeats(self):
        """Check agent heartbeats and update status"""
        for agent_id, agent in self.registry.registered_agents.items():
            if agent.last_heartbeat:
                time_since_heartbeat = (datetime.now() - agent.last_heartbeat).total_seconds()

                # Mark as idle if no heartbeat for 5 minutes
                if time_since_heartbeat > 300 and agent.status == AgentStatus.ACTIVE:
                    self.registry.update_agent_status(agent_id, AgentStatus.IDLE, heartbeat=False)

                # Mark as maintenance if no heartbeat for 15 minutes
                elif time_since_heartbeat > 900 and agent.status in [AgentStatus.IDLE, AgentStatus.ACTIVE]:
                    self.registry.update_agent_status(agent_id, AgentStatus.MAINTENANCE, heartbeat=False)

    async def _update_marketplace_metrics(self):
        """Update marketplace metrics in database"""
        metrics = self.registry.get_marketplace_metrics()

        try:
            conn = sqlite3.connect(self.registry.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO marketplace_metrics
                (total_agents, active_agents, total_tasks_processed, average_task_completion_time,
                 marketplace_uptime, agent_turnover_rate, capability_coverage, load_distribution)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.total_agents,
                metrics.active_agents,
                metrics.total_tasks_processed,
                metrics.average_task_completion_time,
                metrics.marketplace_uptime,
                metrics.agent_turnover_rate,
                json.dumps(metrics.capability_coverage),
                json.dumps(metrics.load_distribution)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error updating marketplace metrics: {e}")

    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform marketplace health check"""
        health_status = {
            'overall': 'healthy',
            'checks': {}
        }

        try:
            # Check database connectivity
            conn = sqlite3.connect(self.registry.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM agents")
            agent_count = cursor.fetchone()[0]
            conn.close()

            health_status['checks']['database'] = {
                'status': 'healthy',
                'agent_count': agent_count
            }

            # Check active agent ratio
            total_agents = len(self.registry.registered_agents)
            active_agents = len([a for a in self.registry.registered_agents.values() if a.status == AgentStatus.ACTIVE])

            if total_agents > 0:
                active_ratio = active_agents / total_agents
                if active_ratio < 0.5:
                    health_status['overall'] = 'warning'
                    health_status['checks']['active_agents'] = {
                        'status': 'warning',
                        'message': f'Low active agent ratio: {active_ratio:.2%}'
                    }
                else:
                    health_status['checks']['active_agents'] = {
                        'status': 'healthy',
                        'active_ratio': active_ratio
                    }

        except Exception as e:
            health_status['overall'] = 'error'
            health_status['checks']['error'] = str(e)

        return health_status


# Factory function
def create_agent_marketplace(config: Dict[str, Any] = None) -> AgentMarketplace:
    """Create an agent marketplace"""
    return AgentMarketplace(config)


# Usage example
if __name__ == "__main__":
    async def test_marketplace():
        # Create marketplace
        marketplace = create_agent_marketplace({
            'db_path': 'test_marketplace.db',
            'monitoring_enabled': True,
            'auto_scaling_enabled': True
        })

        await marketplace.start()

        # Test external agent registration
        agent_info = {
            'agent_id': 'external_agent_001',
            'name': 'External Document Processor',
            'description': 'External agent for document processing',
            'agent_type': 'specialized',
            'version': '1.0.0',
            'author': 'External Provider',
            'contact_info': {
                'email': 'contact@provider.com',
                'phone': '+1234567890'
            },
            'capabilities': [
                {
                    'capability_id': 'doc_process_001',
                    'name': 'document_processing',
                    'description': 'Process various document formats',
                    'category': 'processing',
                    'version': '1.0',
                    'complexity': 'moderate',
                    'tags': ['documents', 'processing', 'pdf', 'word']
                }
            ],
            'endpoints': {
                'api': 'https://api.provider.com/v1',
                'websocket': 'wss://api.provider.com/ws'
            }
        }

        agent_id = await marketplace.register_external_agent(agent_info)
        print(f"Registered external agent: {agent_id}")

        # Test agent discovery
        task_request = TaskRequest(
            task_id="marketplace_test_001",
            task_type="document_processing",
            description="Test document processing task",
            requirements={
                'capabilities': ['document_processing']
            }
        )

        matched_agent = await marketplace.discover_and_match(task_request)
        print(f"Matched agent: {matched_agent}")

        # Get marketplace status
        status = await marketplace.get_marketplace_status()
        print(f"Marketplace status: {status['status']}")
        print(f"Total agents: {status['metrics']['total_agents']}")

        await marketplace.stop()

    asyncio.run(test_marketplace())