#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAgent-based Manufacturing Multi-Agent Orchestration System
基于XAgent的制造业多智能体编排系统

This system implements advanced XAgent patterns for manufacturing knowledge management,
featuring hierarchical agent coordination, dynamic task allocation, and intelligent
decision-making capabilities for industrial automation and optimization.
"""

import asyncio
import json
import logging
import uuid
import time
import yaml
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from pathlib import Path
import threading
from queue import PriorityQueue, Empty
import networkx as nx
import numpy as np

# LangChain integration for RAG capabilities
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# LangFuse for observability
try:
    from langfuse import Langfuse
    langfuse_available = True
except ImportError:
    langfuse_available = False

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler('data/processed/xagent_orchestrator.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Manufacturing-specific agent roles"""
    COORDINATOR = "coordinator"
    SAFETY_INSPECTOR = "safety_inspector"
    QUALITY_CONTROLLER = "quality_controller"
    MAINTENANCE_TECHNICIAN = "maintenance_technician"
    PRODUCTION_MANAGER = "production_manager"
    SUPPLY_CHAIN_COORDINATOR = "supply_chain_coordinator"
    COMPLIANCE_AUDITOR = "compliance_auditor"
    KNOWLEDGE_MANAGER = "knowledge_manager"
    PROCESS_OPTIMIZER = "process_optimizer"
    INVENTORY_MANAGER = "inventory_manager"
    RISK_ANALYZER = "risk_analyzer"
    PERFORMANCE_MONITOR = "performance_monitor"

class AgentCapability(Enum):
    """Agent capabilities"""
    DATA_ANALYSIS = "data_analysis"
    DOCUMENT_PROCESSING = "document_processing"
    RAG_QUERY = "rag_query"
    SAFETY_CHECK = "safety_check"
    QUALITY_INSPECTION = "quality_inspection"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    PROCESS_OPTIMIZATION = "process_optimization"
    ANOMALY_DETECTION = "anomaly_detection"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"
    REAL_TIME_MONITORING = "real_time_monitoring"
    COMPLIANCE_CHECKING = "compliance_checking"
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1  # Safety incidents, production stoppage
    HIGH = 2      # Quality issues, maintenance alerts
    NORMAL = 3    # Routine analysis, reporting
    LOW = 4       # Learning updates, data cleanup

class TaskStatus(Enum):
    """Task execution status"""
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    WAITING = "waiting_for_agents"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

@dataclass
class AgentSkill:
    """Agent skill definition"""
    name: str
    capability: AgentCapability
    proficiency: float  # 0.0 to 1.0
    experience: int  # Number of tasks completed
    last_used: datetime = field(default_factory=datetime.now)
    success_rate: float = 1.0

@dataclass
class ManufacturingContext:
    """Manufacturing execution context"""
    facility_id: str
    production_line: Optional[str] = None
    equipment_type: Optional[str] = None
    process_stage: Optional[str] = None
    safety_level: str = "standard"  # low, standard, high, critical
    quality_standards: List[str] = field(default_factory=list)
    compliance_requirements: List[str] = field(default_factory=list)
    current_shift: Optional[str] = None
    environmental_conditions: Dict[str, Any] = field(default_factory=dict)

@dataclass
class XAgentTask:
    """XAgent task definition"""
    task_id: str
    title: str
    description: str
    priority: TaskPriority
    required_capabilities: List[AgentCapability]
    context: ManufacturingContext
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_requirements: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: timedelta = field(default=timedelta(minutes=5))
    max_duration: timedelta = field(default=timedelta(hours=1))
    created_at: datetime = field(default_factory=datetime.now)
    status: TaskStatus = TaskStatus.QUEUED
    assigned_agents: List[str] = field(default_factory=list)
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

class XAgent(ABC):
    """Base XAgent implementation for manufacturing domain"""

    def __init__(
        self,
        agent_id: str,
        name: str,
        role: AgentRole,
        capabilities: List[AgentCapability],
        orchestrator: Optional['XAgentOrchestrator'] = None
    ):
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.capabilities = capabilities
        self.orchestrator = orchestrator
        self.skills: Dict[str, AgentSkill] = {}
        self.status = "idle"
        self.current_task: Optional[XAgentTask] = None
        self.task_queue = deque()
        self.performance_metrics = {
            "tasks_completed": 0,
            "average_task_time": 0.0,
            "success_rate": 1.0,
            "last_activity": datetime.now()
        }
        self.knowledge_base = {}
        self.communication_channels = {}
        self.is_active = True

    async def initialize(self):
        """Initialize the agent"""
        logger.info(f"🤖 Initializing XAgent: {self.name} ({self.role.value})")

        # Initialize skills based on capabilities
        for capability in self.capabilities:
            skill_name = f"{self.role.value}_{capability.value}"
            self.skills[skill_name] = AgentSkill(
                name=skill_name,
                capability=capability,
                proficiency=0.7,  # Start with moderate proficiency
                experience=0
            )

        # Setup communication channels
        await self.setup_communication_channels()

        # Load domain knowledge
        await self.load_knowledge_base()

        self.performance_metrics["last_activity"] = datetime.now()
        logger.info(f"✅ XAgent {self.name} initialized successfully")

    async def setup_communication_channels(self):
        """Setup communication channels for agent collaboration"""
        self.communication_channels = {
            "broadcast": [],
            "direct": {},
            "role_based": defaultdict(list)
        }

    async def load_knowledge_base(self):
        """Load domain-specific knowledge base"""
        # Initialize with basic manufacturing knowledge
        self.knowledge_base = {
            "safety_procedures": [],
            "quality_standards": [],
            "equipment_specs": {},
            "process_parameters": {},
            "compliance_checklists": {}
        }

    @abstractmethod
    async def execute_task(self, task: XAgentTask) -> Dict[str, Any]:
        """Execute a manufacturing-specific task"""
        pass

    async def can_handle_task(self, task: XAgentTask) -> bool:
        """Check if agent can handle the task"""
        required_caps = set(task.required_capabilities)
        agent_caps = set(self.capabilities)
        return required_caps.issubset(agent_caps) and self.is_active

    async def assign_task(self, task: XAgentTask):
        """Assign a task to this agent"""
        self.task_queue.append(task)
        self.status = "busy"
        logger.info(f"📋 Task {task.task_id} assigned to {self.name}")

    async def process_tasks(self):
        """Process queued tasks"""
        while self.task_queue and self.is_active:
            task = self.task_queue.popleft()
            self.current_task = task

            try:
                task.status = TaskStatus.RUNNING
                task.assigned_agents.append(self.agent_id)

                start_time = time.time()
                result = await self.execute_task(task)
                execution_time = time.time() - start_time

                task.result = result
                task.status = TaskStatus.COMPLETED
                task.progress = 1.0

                # Update performance metrics
                self._update_performance_metrics(execution_time, True)

                # Notify orchestrator of completion
                if self.orchestrator:
                    await self.orchestrator.notify_task_completion(task, self.agent_id, result)

                logger.info(f"✅ Task {task.task_id} completed by {self.name} in {execution_time:.2f}s")

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error_message = str(e)
                task.retry_count += 1

                self._update_performance_metrics(0, False)
                logger.error(f"❌ Task {task.task_id} failed for {self.name}: {e}")

                if self.orchestrator:
                    await self.orchestrator.notify_task_failure(task, self.agent_id, str(e))

            finally:
                self.current_task = None
                if not self.task_queue:
                    self.status = "idle"

    def _update_performance_metrics(self, execution_time: float, success: bool):
        """Update agent performance metrics"""
        self.performance_metrics["tasks_completed"] += 1

        # Update average task time
        total_tasks = self.performance_metrics["tasks_completed"]
        current_avg = self.performance_metrics["average_task_time"]
        new_avg = (current_avg * (total_tasks - 1) + execution_time) / total_tasks
        self.performance_metrics["average_task_time"] = new_avg

        # Update success rate
        successes = self.performance_metrics["success_rate"] * (total_tasks - 1)
        new_success_rate = (successes + (1 if success else 0)) / total_tasks
        self.performance_metrics["success_rate"] = new_success_rate

        self.performance_metrics["last_activity"] = datetime.now()

    async def communicate(self, target_agent_id: str, message: Dict[str, Any]):
        """Communicate with another agent"""
        if self.orchestrator:
            await self.orchestrator.route_message(self.agent_id, target_agent_id, message)

    async def learn_from_feedback(self, task: XAgentTask, feedback: Dict[str, Any]):
        """Learn from task execution feedback"""
        # Update skills based on feedback
        for capability in task.required_capabilities:
            skill_name = f"{self.role.value}_{capability.value}"
            if skill_name in self.skills:
                skill = self.skills[skill_name]
                skill.experience += 1

                # Adjust proficiency based on feedback
                if feedback.get("success", True):
                    skill.proficiency = min(1.0, skill.proficiency + 0.05)
                else:
                    skill.proficiency = max(0.1, skill.proficiency - 0.1)

                skill.last_used = datetime.now()

class ManufacturingSafetyInspector(XAgent):
    """Specialized agent for safety inspection and compliance"""

    def __init__(self, agent_id: str, orchestrator: Optional['XAgentOrchestrator'] = None):
        super().__init__(
            agent_id=agent_id,
            name="Manufacturing Safety Inspector",
            role=AgentRole.SAFETY_INSPECTOR,
            capabilities=[
                AgentCapability.SAFETY_CHECK,
                AgentCapability.COMPLIANCE_CHECKING,
                AgentCapability.ANOMALY_DETECTION,
                AgentCapability.REAL_TIME_MONITORING
            ],
            orchestrator=orchestrator
        )

    async def execute_task(self, task: XAgentTask) -> Dict[str, Any]:
        """Execute safety inspection task"""
        result = {
            "inspection_id": task.task_id,
            "timestamp": datetime.now().isoformat(),
            "facility": task.context.facility_id,
            "safety_level": task.context.safety_level,
            "findings": [],
            "recommendations": [],
            "compliance_status": "compliant"
        }

        # Simulate safety inspection process
        if "safety_check" in task.input_data:
            check_type = task.input_data["safety_check"]
            if check_type == "equipment_safety":
                result["findings"] = await self._check_equipment_safety(task.context)
            elif check_type == "procedural_compliance":
                result["findings"] = await self._check_procedural_compliance(task.context)
            elif check_type == "environmental_safety":
                result["findings"] = await self._check_environmental_safety(task.context)

        # Generate safety recommendations
        result["recommendations"] = await self._generate_safety_recommendations(result["findings"])

        return result

    async def _check_equipment_safety(self, context: ManufacturingContext) -> List[Dict]:
        """Check equipment safety"""
        findings = []

        # Simulate equipment safety checks
        if context.equipment_type:
            findings.append({
                "category": "equipment",
                "item": context.equipment_type,
                "status": "safe",
                "last_inspection": datetime.now().isoformat(),
                "next_inspection": (datetime.now() + timedelta(days=30)).isoformat()
            })

        return findings

    async def _check_procedural_compliance(self, context: ManufacturingContext) -> List[Dict]:
        """Check procedural compliance"""
        findings = []

        # Simulate compliance checks
        for standard in context.compliance_requirements:
            findings.append({
                "category": "compliance",
                "standard": standard,
                "status": "compliant",
                "check_date": datetime.now().isoformat()
            })

        return findings

    async def _check_environmental_safety(self, context: ManufacturingContext) -> List[Dict]:
        """Check environmental safety"""
        findings = []

        # Simulate environmental checks
        findings.append({
            "category": "environmental",
            "parameter": "air_quality",
            "status": "acceptable",
            "reading": "safe",
            "timestamp": datetime.now().isoformat()
        })

        return findings

    async def _generate_safety_recommendations(self, findings: List[Dict]) -> List[str]:
        """Generate safety recommendations"""
        recommendations = []

        high_risk_items = [f for f in findings if f.get("status") == "unsafe"]
        if high_risk_items:
            recommendations.append("Immediate action required for unsafe conditions")

        recommendations.extend([
            "Schedule regular safety inspections",
            "Update safety training documentation",
            "Review emergency response procedures"
        ])

        return recommendations

class QualityController(XAgent):
    """Specialized agent for quality control and inspection"""

    def __init__(self, agent_id: str, orchestrator: Optional['XAgentOrchestrator'] = None):
        super().__init__(
            agent_id=agent_id,
            name="Quality Controller",
            role=AgentRole.QUALITY_CONTROLLER,
            capabilities=[
                AgentCapability.QUALITY_INSPECTION,
                AgentCapability.DATA_ANALYSIS,
                AgentCapability.ANOMALY_DETECTION,
                AgentCapability.COMPLIANCE_CHECKING
            ],
            orchestrator=orchestrator
        )

    async def execute_task(self, task: XAgentTask) -> Dict[str, Any]:
        """Execute quality control task"""
        result = {
            "quality_check_id": task.task_id,
            "timestamp": datetime.now().isoformat(),
            "inspection_results": [],
            "quality_metrics": {},
            "compliance_status": "pass",
            "recommendations": []
        }

        # Process quality inspection
        if "quality_inspection" in task.input_data:
            inspection_data = task.input_data["quality_inspection"]
            result["inspection_results"] = await self._perform_quality_inspection(
                inspection_data, task.context
            )

            # Calculate quality metrics
            result["quality_metrics"] = await self._calculate_quality_metrics(
                result["inspection_results"]
            )

            # Generate recommendations
            result["recommendations"] = await self._generate_quality_recommendations(
                result["inspection_results"]
            )

        return result

    async def _perform_quality_inspection(self, inspection_data: Dict, context: ManufacturingContext) -> List[Dict]:
        """Perform detailed quality inspection"""
        results = []

        # Simulate quality measurements
        measurements = inspection_data.get("measurements", [])
        for measurement in measurements:
            result = {
                "parameter": measurement.get("parameter", "unknown"),
                "specified_value": measurement.get("specified", 0),
                "measured_value": measurement.get("measured", 0),
                "tolerance": measurement.get("tolerance", 0),
                "status": "pass",
                "deviation": 0
            }

            # Calculate deviation and status
            deviation = abs(result["measured_value"] - result["specified_value"])
            result["deviation"] = deviation
            result["status"] = "pass" if deviation <= result["tolerance"] else "fail"

            results.append(result)

        return results

    async def _calculate_quality_metrics(self, inspection_results: List[Dict]) -> Dict[str, Any]:
        """Calculate quality metrics"""
        total_checks = len(inspection_results)
        passed_checks = len([r for r in inspection_results if r["status"] == "pass"])

        metrics = {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": total_checks - passed_checks,
            "pass_rate": (passed_checks / total_checks * 100) if total_checks > 0 else 0,
            "average_deviation": np.mean([r["deviation"] for r in inspection_results]) if inspection_results else 0
        }

        return metrics

    async def _generate_quality_recommendations(self, inspection_results: List[Dict]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []

        failed_checks = [r for r in inspection_results if r["status"] == "fail"]
        if failed_checks:
            recommendations.append("Review and adjust process parameters for failed checks")

        high_deviation = [r for r in inspection_results if r["deviation"] > 0]
        if high_deviation:
            recommendations.append("Implement process control measures to reduce variation")

        recommendations.extend([
            "Conduct regular quality training",
            "Update inspection procedures",
            "Implement statistical process control (SPC)"
        ])

        return recommendations

class XAgentOrchestrator:
    """XAgent orchestration system for manufacturing"""

    def __init__(self):
        self.agents: Dict[str, XAgent] = {}
        self.task_queue = PriorityQueue()
        self.active_tasks: Dict[str, XAgentTask] = {}
        self.completed_tasks: List[XAgentTask] = []
        self.agent_graph = nx.DiGraph()
        self.communication_hub = defaultdict(list)
        self.performance_metrics = defaultdict(dict)
        self.knowledge_graph = nx.DiGraph()

        # Initialize LangFuse for observability
        if langfuse_available:
            self.langfuse = Langfuse()
        else:
            self.langfuse = None

        # Manufacturing context
        self.current_context = ManufacturingContext(
            facility_id="main_facility",
            safety_level="standard"
        )

    async def initialize(self):
        """Initialize the orchestration system"""
        logger.info("🚀 Initializing XAgent Orchestration System")

        # Create manufacturing-specific agents
        await self._create_manufacturing_agents()

        # Build agent collaboration graph
        await self._build_agent_graph()

        # Initialize communication protocols
        await self._initialize_communication_protocols()

        # Start task processing
        asyncio.create_task(self._process_task_queue())

        logger.info("✅ XAgent Orchestration System initialized successfully")

    async def _create_manufacturing_agents(self):
        """Create specialized manufacturing agents"""

        # Safety Inspector
        safety_agent = ManufacturingSafetyInspector("safety_inspector_001", self)
        await safety_agent.initialize()
        self.agents[safety_agent.agent_id] = safety_agent

        # Quality Controller
        quality_agent = QualityController("quality_controller_001", self)
        await quality_agent.initialize()
        self.agents[quality_agent.agent_id] = quality_agent

        logger.info(f"🤖 Created {len(self.agents)} specialized manufacturing agents")

    async def _build_agent_graph(self):
        """Build agent collaboration graph"""
        for agent_id, agent in self.agents.items():
            self.agent_graph.add_node(agent_id, agent=agent)

            # Add edges based on role collaboration
            for other_id, other_agent in self.agents.items():
                if agent_id != other_id:
                    # Define collaboration relationships
                    if self._should_collaborate(agent.role, other_agent.role):
                        self.agent_graph.add_edge(agent_id, other_id, weight=1.0)

        logger.info(f"🔗 Built agent collaboration graph with {self.agent_graph.number_of_nodes()} nodes and {self.agent_graph.number_of_edges()} edges")

    def _should_collaborate(self, role1: AgentRole, role2: AgentRole) -> bool:
        """Determine if two agent roles should collaborate"""
        collaboration_matrix = {
            AgentRole.SAFETY_INSPECTOR: [AgentRole.QUALITY_CONTROLLER, AgentRole.PRODUCTION_MANAGER],
            AgentRole.QUALITY_CONTROLLER: [AgentRole.SAFETY_INSPECTOR, AgentRole.PRODUCTION_MANAGER],
            AgentRole.PRODUCTION_MANAGER: [AgentRole.SAFETY_INSPECTOR, AgentRole.QUALITY_CONTROLLER, AgentRole.MAINTENANCE_TECHNICIAN],
            AgentRole.MAINTENANCE_TECHNICIAN: [AgentRole.PRODUCTION_MANAGER, AgentRole.QUALITY_CONTROLLER],
        }

        return role2 in collaboration_matrix.get(role1, [])

    async def _initialize_communication_protocols(self):
        """Initialize inter-agent communication protocols"""
        # Setup message routing and channels
        for agent_id in self.agents:
            self.communication_hub[agent_id] = []

        logger.info("📡 Initialized communication protocols")

    async def submit_task(self, task: XAgentTask) -> str:
        """Submit a task to the orchestration system"""
        # Log task submission
        if self.langfuse:
            self.langfuse.trace(
                name="task_submission",
                inputs={
                    "task_id": task.task_id,
                    "title": task.title,
                    "priority": task.priority.name,
                    "required_capabilities": [cap.value for cap in task.required_capabilities]
                }
            )

        # Add to priority queue (lower number = higher priority)
        self.task_queue.put((task.priority.value, task))
        self.active_tasks[task.task_id] = task

        logger.info(f"📝 Task {task.task_id} submitted with priority {task.priority.name}")
        return task.task_id

    async def _process_task_queue(self):
        """Process tasks from the queue"""
        while True:
            try:
                # Get next task from queue
                priority, task = self.task_queue.get(timeout=1.0)

                # Find suitable agents
                suitable_agents = await self._find_suitable_agents(task)

                if suitable_agents:
                    # Assign task to best agent
                    best_agent = await self._select_best_agent(suitable_agents, task)
                    await best_agent.assign_task(task)

                    # Start task processing if not already running
                    if best_agent.status == "idle":
                        asyncio.create_task(best_agent.process_tasks())
                else:
                    # No suitable agents found, requeue with lower priority
                    if task.priority.value < TaskPriority.LOW.value:
                        task.priority = TaskPriority(task.priority.value + 1)
                        self.task_queue.put((task.priority.value, task))
                        logger.warning(f"⚠️ No suitable agents found for task {task.task_id}, requeuing with lower priority")

            except Empty:
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"❌ Error processing task queue: {e}")

    async def _find_suitable_agents(self, task: XAgentTask) -> List[XAgent]:
        """Find agents capable of handling the task"""
        suitable_agents = []

        for agent in self.agents.values():
            if await agent.can_handle_task(task):
                suitable_agents.append(agent)

        return suitable_agents

    async def _select_best_agent(self, agents: List[XAgent], task: XAgentTask) -> XAgent:
        """Select the best agent for the task"""
        # Score agents based on proficiency and availability
        best_agent = None
        best_score = -1

        for agent in agents:
            score = await self._calculate_agent_score(agent, task)
            if score > best_score:
                best_score = score
                best_agent = agent

        return best_agent

    async def _calculate_agent_score(self, agent: XAgent, task: XAgentTask) -> float:
        """Calculate agent suitability score for task"""
        score = 0.0

        # Proficiency in required capabilities
        for capability in task.required_capabilities:
            skill_name = f"{agent.role.value}_{capability.value}"
            if skill_name in agent.skills:
                score += agent.skills[skill_name].proficiency * 0.4

        # Success rate bonus
        score += agent.performance_metrics["success_rate"] * 0.3

        # Experience bonus
        score += min(agent.performance_metrics["tasks_completed"] / 100, 1.0) * 0.2

        # Availability bonus
        if agent.status == "idle":
            score += 0.1

        return score

    async def notify_task_completion(self, task: XAgentTask, agent_id: str, result: Dict[str, Any]):
        """Handle task completion notification"""
        task.status = TaskStatus.COMPLETED
        task.progress = 1.0
        task.result = result

        # Remove from active tasks
        if task.task_id in self.active_tasks:
            del self.active_tasks[task.task_id]

        # Add to completed tasks
        self.completed_tasks.append(task)

        # Log completion
        if self.langfuse:
            self.langfuse.trace(
                name="task_completion",
                inputs={"task_id": task.task_id, "agent_id": agent_id},
                output={"result": result}
            )

        logger.info(f"✅ Task {task.task_id} completed by agent {agent_id}")

    async def notify_task_failure(self, task: XAgentTask, agent_id: str, error: str):
        """Handle task failure notification"""
        task.error_message = error

        # Retry logic
        if task.retry_count < task.max_retries:
            task.retry_count += 1
            task.status = TaskStatus.QUEUED
            await self.submit_task(task)
            logger.info(f"🔄 Retrying task {task.task_id} (attempt {task.retry_count}/{task.max_retries})")
        else:
            task.status = TaskStatus.FAILED
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]

            logger.error(f"❌ Task {task.task_id} failed permanently: {error}")

    async def route_message(self, from_agent: str, to_agent: str, message: Dict[str, Any]):
        """Route message between agents"""
        self.communication_hub[to_agent].append({
            "from": from_agent,
            "to": to_agent,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })

        # Deliver message if target agent exists
        if to_agent in self.agents:
            await self.agents[to_agent].handle_message(message, from_agent)

    async def get_system_status(self) -> Dict[str, Any]:
        """Get orchestration system status"""
        return {
            "agents": {
                agent_id: {
                    "name": agent.name,
                    "role": agent.role.value,
                    "status": agent.status,
                    "capabilities": [cap.value for cap in agent.capabilities],
                    "performance": agent.performance_metrics
                }
                for agent_id, agent in self.agents.items()
            },
            "tasks": {
                "active": len(self.active_tasks),
                "queued": self.task_queue.qsize(),
                "completed": len(self.completed_tasks)
            },
            "communication": {
                "pending_messages": sum(len(msgs) for msgs in self.communication_hub.values())
            }
        }

# Global orchestrator instance
orchestrator = XAgentOrchestrator()

async def main():
    """Main function to demonstrate XAgent system"""
    logger.info("🎯 Starting XAgent Manufacturing Orchestration System")

    # Initialize orchestrator
    await orchestrator.initialize()

    # Create sample tasks
    safety_task = XAgentTask(
        task_id="safety_inspection_001",
        title="Daily Safety Inspection",
        description="Perform daily safety inspection of production line",
        priority=TaskPriority.HIGH,
        required_capabilities=[AgentCapability.SAFETY_CHECK],
        context=ManufacturingContext(
            facility_id="main_facility",
            production_line="line_1",
            equipment_type="cnc_milling",
            safety_level="standard"
        ),
        input_data={
            "safety_check": "equipment_safety"
        }
    )

    quality_task = XAgentTask(
        task_id="quality_check_001",
        title="Quality Control Check",
        description="Perform quality control inspection for current batch",
        priority=TaskPriority.NORMAL,
        required_capabilities=[AgentCapability.QUALITY_INSPECTION],
        context=ManufacturingContext(
            facility_id="main_facility",
            production_line="line_1",
            process_stage="final_inspection"
        ),
        input_data={
            "quality_inspection": {
                "measurements": [
                    {"parameter": "diameter", "specified": 10.0, "measured": 10.05, "tolerance": 0.1},
                    {"parameter": "length", "specified": 50.0, "measured": 50.02, "tolerance": 0.2}
                ]
            }
        }
    )

    # Submit tasks
    await orchestrator.submit_task(safety_task)
    await orchestrator.submit_task(quality_task)

    # Monitor system
    while True:
        await asyncio.sleep(10)
        status = await orchestrator.get_system_status()
        logger.info(f"📊 System Status: {status['tasks']['active']} active tasks, {status['tasks']['queued']} queued")

if __name__ == "__main__":
    asyncio.run(main())