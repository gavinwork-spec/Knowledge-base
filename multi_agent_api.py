#!/usr/bin/env python3
"""
Multi-Agent API Server
RESTful API for the multi-agent orchestration system with task management,
agent coordination, and result synthesis capabilities.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Body, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import redis
from contextlib import asynccontextmanager

# Import multi-agent systems
from multi_agent_orchestrator import (
    MultiAgentOrchestrator, AgentTask, AgentStatus, TaskStatus, TaskPriority,
    AgentCapability, BaseAgent, AgentMessage
)
from specialized_agents import (
    LearningAgent, DataProcessingAgent, DocumentAnalysisAgent, NotificationAgent
)
from agent_result_synthesizer import (
    ResultSynthesizer, SynthesisStrategy, AgentResult, SynthesisResult
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
orchestrator = None
result_synthesizer = None
redis_client = None
active_websockets = {}

# Pydantic models for API
class AgentRegistration(BaseModel):
    name: str = Field(..., description="Agent name")
    agent_type: str = Field(..., description="Agent type")
    description: Optional[str] = Field(None, description="Agent description")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Agent configuration")

class TaskSubmission(BaseModel):
    task_type: str = Field(..., description="Type of task")
    title: str = Field(..., description="Task title")
    description: str = Field(..., description="Task description")
    required_capabilities: List[str] = Field(default_factory=list, description="Required capabilities")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    priority: str = Field("normal", description="Task priority")
    deadline: Optional[str] = Field(None, description="Task deadline (ISO format)")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
    max_retry_attempts: int = Field(3, description="Maximum retry attempts")
    decompose: bool = Field(False, description="Whether to decompose complex task")

class TaskSynthesis(BaseModel):
    task_ids: List[str] = Field(..., description="List of task IDs to synthesize")
    synthesis_strategy: str = Field("weighted_average", description="Synthesis strategy")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for synthesis")

class AgentStatusResponse(BaseModel):
    agent_id: str
    name: str
    status: str
    current_tasks: int
    completed_tasks: int
    failed_tasks: int
    capabilities: List[str]
    last_heartbeat: str
    performance_metrics: Dict[str, Any]

class TaskStatusResponse(BaseModel):
    task_id: str
    parent_task_id: Optional[str]
    task_type: str
    title: str
    description: str
    status: str
    priority: str
    assigned_agent_id: Optional[str]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    deadline: Optional[str]
    retry_count: int
    subtasks: List[str]
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]

class SystemStatusResponse(BaseModel):
    orchestrator: Dict[str, Any]
    agents: Dict[str, Any]
    metrics: Dict[str, Any]

class SynthesisResponse(BaseModel):
    synthesis_id: str
    synthesized_data: Dict[str, Any]
    confidence: float
    contributing_agents: List[str]
    synthesis_strategy: str
    conflicts_detected: int
    conflicts_resolved: int
    quality_score: float
    created_at: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup multi-agent systems"""
    global orchestrator, result_synthesizer, redis_client

    logger.info("Initializing Multi-Agent API Server...")

    try:
        # Initialize Redis client
        redis_client = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)
        redis_client.ping()
        logger.info("Redis client initialized")

        # Initialize orchestrator
        orchestrator = MultiAgentOrchestrator(redis_client=redis_client)
        await orchestrator.start()
        logger.info("Multi-agent orchestrator initialized")

        # Initialize result synthesizer
        result_synthesizer = ResultSynthesizer()
        logger.info("Result synthesizer initialized")

        # Register default agents
        await register_default_agents()

        logger.info("Multi-Agent API Server ready!")

        yield

    except Exception as e:
        logger.error(f"Failed to initialize systems: {e}")
        raise

    finally:
        logger.info("Shutting down Multi-Agent API Server...")
        if orchestrator:
            await orchestrator.stop()
        if redis_client:
            redis_client.close()

# Create FastAPI app
app = FastAPI(
    title="Multi-Agent Orchestration API",
    description="Advanced multi-agent orchestration system with task management and result synthesis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def register_default_agents():
    """Register default specialized agents"""
    global orchestrator

    # Register Learning Agent
    learning_config = {
        'database': {'path': 'knowledge_base.db'},
        'scan_directories': [
            '/Users/gavin/Nutstore Files/.symlinks/坚果云/005-询盘询价和/',
            '/Users/gavin/Nutstore Files/.symlinks/坚果云/002-客户中/'
        ],
        'supported_formats': ['.pdf', '.xlsx', '.docx', '.txt', '.csv'],
        'confidence_threshold': 0.7
    }

    learning_agent = LearningAgent(
        agent_id="learning_agent_001",
        orchestrator=orchestrator,
        config=learning_config
    )
    await orchestrator.register_agent(learning_agent)

    # Register Data Processing Agent
    data_agent = DataProcessingAgent(
        agent_id="data_processing_agent_001",
        orchestrator=orchestrator
    )
    await orchestrator.register_agent(data_agent)

    # Register Document Analysis Agent
    doc_agent = DocumentAnalysisAgent(
        agent_id="document_analysis_agent_001",
        orchestrator=orchestrator
    )
    await orchestrator.register_agent(doc_agent)

    # Register Notification Agent
    notification_agent = NotificationAgent(
        agent_id="notification_agent_001",
        orchestrator=orchestrator
    )
    await orchestrator.register_agent(notification_agent)

    logger.info("Default agents registered successfully")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_agents: Dict[WebSocket, str] = {}

    async def connect(self, websocket: WebSocket, agent_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_agents[websocket] = agent_id
        logger.info(f"WebSocket connected for agent {agent_id}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.connection_agents:
            agent_id = self.connection_agents[websocket]
            del self.connection_agents[websocket]
            logger.info(f"WebSocket disconnected for agent {agent_id}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove disconnected connections
                self.disconnect(connection)

manager = ConnectionManager()

# API Endpoints

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint"""
    global orchestrator

    components = {
        "orchestrator": "running" if orchestrator and orchestrator.running else "stopped",
        "redis": "connected" if redis_client and redis_client.ping() else "disconnected",
        "synthesizer": "initialized" if result_synthesizer else "not_initialized"
    }

    overall_status = "healthy" if all(status == "running" or status == "connected" or status == "initialized"
                                        for status in components.values()) else "unhealthy"

    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "components": components,
        "version": "1.0.0"
    }

@app.get("/api/v1/system/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get overall system status"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        status = await orchestrator.get_system_status()
        return SystemStatusResponse(
            orchestrator=status['orchestrator'],
            agents=status['agents'],
            metrics=status['orchestrator']['metrics']
        )
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting system status: {str(e)}")

@app.post("/api/v1/agents/register", response_model=Dict[str, str])
async def register_agent(registration: AgentRegistration):
    """Register a new agent"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        # Create agent based on type
        agent_id = f"{registration.agent_type}_{str(uuid.uuid4())[:8]}"

        if registration.agent_type == "learning":
            agent = LearningAgent(agent_id, registration.name, orchestrator, registration.config)
        elif registration.agent_type == "data_processing":
            agent = DataProcessingAgent(agent_id, registration.name, orchestrator)
        elif registration.agent_type == "document_analysis":
            agent = DocumentAnalysisAgent(agent_id, registration.name, orchestrator)
        elif registration.agent_type == "notification":
            agent = NotificationAgent(agent_id, registration.name, orchestrator)
        else:
            # Create a generic agent
            class GenericAgent(BaseAgent):
                def __init__(self, agent_id, name, orchestrator):
                    super().__init__(agent_id, name, orchestrator)

                async def initialize(self):
                    pass

                async def execute_task(self, task):
                    return {"status": "completed", "message": "Generic agent execution"}

                def get_capabilities(self):
                    return [AgentCapability(cap) for cap in registration.capabilities]

            agent = GenericAgent(agent_id, registration.name, orchestrator)

        # Register agent
        success = await orchestrator.register_agent(agent)

        if success:
            logger.info(f"Agent {registration.name} registered successfully with ID {agent_id}")
            return {"agent_id": agent_id, "status": "registered", "message": "Agent registered successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to register agent")

    except Exception as e:
        logger.error(f"Error registering agent: {e}")
        raise HTTPException(status_code=500, detail=f"Error registering agent: {str(e)}")

@app.delete("/api/v1/agents/{agent_id}")
async def unregister_agent(agent_id: str):
    """Unregister an agent"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        success = await orchestrator.unregister_agent(agent_id)

        if success:
            return {"message": f"Agent {agent_id} unregistered successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    except Exception as e:
        logger.error(f"Error unregistering agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error unregistering agent: {str(e)}")

@app.get("/api/v1/agents", response_model=List[AgentStatusResponse])
async def list_agents():
    """List all registered agents"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        agents = []
        for agent_id, agent in orchestrator.agents.items():
            status = await agent.get_status()
            agents.append(AgentStatusResponse(**status))

        return agents

    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing agents: {str(e)}")

@app.get("/api/v1/agents/{agent_id}/status", response_model=AgentStatusResponse)
async def get_agent_status(agent_id: str):
    """Get status of a specific agent"""
    if not orchestrator or agent_id not in orchestrator.agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    try:
        agent = orchestrator.agents[agent_id]
        status = await agent.get_status()
        return AgentStatusResponse(**status)

    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting agent status: {str(e)}")

@app.post("/api/v1/tasks/submit", response_model=Dict[str, str])
async def submit_task(task: TaskSubmission):
    """Submit a new task for execution"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        # Convert task data
        required_capabilities = [
            AgentCapability(cap) for cap in task.required_capabilities
        ]

        priority_map = {
            "critical": TaskPriority.CRITICAL,
            "high": TaskPriority.HIGH,
            "normal": TaskPriority.NORMAL,
            "low": TaskPriority.LOW,
            "background": TaskPriority.BACKGROUND
        }

        agent_task = AgentTask(
            task_type=task.task_type,
            title=task.title,
            description=task.description,
            required_capabilities=required_capabilities,
            parameters=task.parameters,
            dependencies=task.dependencies,
            priority=priority_map.get(task.priority, TaskPriority.NORMAL),
            deadline=datetime.fromisoformat(task.deadline) if task.deadline else None,
            max_retry_attempts=task.max_retry_attempts
        )

        # Submit task
        task_id = await orchestrator.submit_task(agent_task)

        return {
            "task_id": task_id,
            "status": "submitted",
            "message": "Task submitted successfully"
        }

    except Exception as e:
        logger.error(f"Error submitting task: {e}")
        raise HTTPException(status_code=500, detail=f"Error submitting task: {str(e)}")

@app.get("/api/v1/tasks", response_model=List[TaskStatusResponse])
async def list_tasks(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, description="Maximum number of tasks to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """List tasks with optional filtering"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        tasks = []
        task_list = list(orchestrator.tasks.values())

        # Filter by status if specified
        if status:
            task_list = [t for t in task_list if t.status.value == status]

        # Apply pagination
        task_list = task_list[offset:offset + limit]

        for task in task_list:
            task_dict = task.to_dict()
            tasks.append(TaskStatusResponse(**task_dict))

        return tasks

    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing tasks: {str(e)}")

@app.get("/api/v1/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get status of a specific task"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        task_status = await orchestrator.get_task_status(task_id)

        if not task_status:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        return TaskStatusResponse(**task_status)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting task status: {str(e)}")

@app.delete("/api/v1/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a task"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        success = await orchestrator.cancel_task(task_id)

        if success:
            return {"message": f"Task {task_id} cancelled successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found or cannot be cancelled")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling task: {e}")
        raise HTTPException(status_code=500, detail=f"Error cancelling task: {str(e)}")

@app.post("/api/v1/tasks/synthesize", response_model=SynthesisResponse)
async def synthesize_task_results(synthesis_request: TaskSynthesis):
    """Synthesize results from multiple tasks"""
    if not orchestrator or not result_synthesizer:
        raise HTTPException(status_code=503, detail="Required systems not initialized")

    try:
        # Collect results from tasks
        agent_results = []
        for task_id in synthesis_request.task_ids:
            if task_id in orchestrator.tasks:
                task = orchestrator.tasks[task_id]
                if task.result:
                    # Create AgentResult from task
                    agent_result = AgentResult(
                        agent_id=task.assigned_agent_id or "unknown",
                        task_id=task_id,
                        result_data=task.result,
                        confidence=0.8,  # Default confidence
                        timestamp=task.completed_at or datetime.now()
                    )
                    agent_results.append(agent_result)

        if not agent_results:
            raise HTTPException(status_code=400, detail="No results found to synthesize")

        # Convert synthesis strategy
        strategy_map = {
            "majority_vote": SynthesisStrategy.MAJORITY_VOTE,
            "weighted_average": SynthesisStrategy.WEIGHTED_AVERAGE,
            "confidence_based": SynthesisStrategy.CONFIDENCE_BASED,
            "expert_consensus": SynthesisStrategy.EXPERT_CONSENSUS,
            "hierarchical_merge": SynthesisStrategy.HIERARCHICAL_MERGE,
            "conflict_resolution": SynthesisStrategy.CONFLICT_RESOLUTION,
            "temporal_sequence": SynthesisStrategy.TEMPORAL_SEQUENCE
        }

        strategy = strategy_map.get(synthesis_request.synthesis_strategy, SynthesisStrategy.WEIGHTED_AVERAGE)

        # Perform synthesis
        synthesis_result = await result_synthesizer.synthesize_results(
            agent_results,
            strategy=strategy,
            context=synthesis_request.context
        )

        return SynthesisResponse(
            synthesis_id=synthesis_result.synthesis_id,
            synthesized_data=synthesis_result.synthesized_data,
            confidence=synthesis_result.confidence,
            contributing_agents=synthesis_result.contributing_agents,
            synthesis_strategy=synthesis_result.synthesis_strategy.value,
            conflicts_detected=len(synthesis_result.conflicts_detected),
            conflicts_resolved=len(synthesis_result.conflicts_resolved),
            quality_score=synthesis_result.quality_score,
            created_at=synthesis_result.created_at.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error synthesizing task results: {e}")
        raise HTTPException(status_code=500, detail=f"Error synthesizing task results: {str(e)}")

@app.post("/api/v1/agents/{agent_id}/message")
async def send_agent_message(agent_id: str, message: Dict[str, Any]):
    """Send a message to a specific agent"""
    if not orchestrator or agent_id not in orchestrator.agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    try:
        # Create message
        agent_message = AgentMessage(
            sender_id="api_client",
            receiver_id=agent_id,
            message_type=message.get("message_type", "custom"),
            content=message.get("content", {}),
            priority=MessagePriority.NORMAL
        )

        # Send message
        success = await orchestrator.send_message(agent_message)

        if success:
            return {"message": "Message sent successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send message")

    except Exception as e:
        logger.error(f"Error sending message to agent: {e}")
        raise HTTPException(status_code=500, detail=f"Error sending message: {str(e)}")

@app.get("/api/v1/metrics", response_model=Dict[str, Any])
async def get_system_metrics():
    """Get detailed system metrics"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        system_status = await orchestrator.get_system_status()
        base_metrics = system_status['orchestrator']['metrics']

        # Additional detailed metrics
        detailed_metrics = {
            **base_metrics,
            "agent_performance": {},
            "task_performance": {},
            "system_efficiency": {}
        }

        # Agent performance metrics
        for agent_id, agent in orchestrator.agents.items():
            detailed_metrics["agent_performance"][agent_id] = {
                "tasks_completed": agent.performance_metrics['tasks_completed'],
                "tasks_failed": agent.performance_metrics['tasks_failed'],
                "success_rate": agent.performance_metrics['success_rate'],
                "average_response_time": agent.performance_metrics['average_response_time'],
                "current_workload": len(agent.current_tasks),
                "status": agent.status.value
            }

        # Task performance metrics
        all_tasks = list(orchestrator.tasks.values())
        if all_tasks:
            completed_tasks = [t for t in all_tasks if t.status == TaskStatus.COMPLETED]
            failed_tasks = [t for t in all_tasks if t.status == TaskStatus.FAILED]

            detailed_metrics["task_performance"] = {
                "total_tasks": len(all_tasks),
                "completed_tasks": len(completed_tasks),
                "failed_tasks": len(failed_tasks),
                "success_rate": len(completed_tasks) / len(all_tasks) if all_tasks else 0,
                "failure_rate": len(failed_tasks) / len(all_tasks) if all_tasks else 0,
                "average_execution_time": orchestrator.metrics['average_task_duration']
            }

        # System efficiency metrics
        total_agents = len(orchestrator.agents)
        active_agents = len([a for a in orchestrator.agents.values() if a.status == AgentStatus.BUSY])

        detailed_metrics["system_efficiency"] = {
            "agent_utilization": active_agents / total_agents if total_agents > 0 else 0,
            "task_throughput": base_metrics['total_tasks_completed'] / (base_metrics['system_uptime'] / 3600) if base_metrics['system_uptime'] > 0 else 0,
            "message_rate": base_metrics['message_count'] / (base_metrics['system_uptime'] / 60) if base_metrics['system_uptime'] > 0 else 0,
            "uptime_hours": base_metrics['system_uptime'] / 3600
        }

        return detailed_metrics

    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting system metrics: {str(e)}")

@app.get("/api/v1/capabilities")
async def get_available_capabilities():
    """Get list of available agent capabilities"""
    capabilities = {
        capability.value: {
            "name": capability.value.replace('_', ' ').title(),
            "description": f"Agents capable of {capability.value.replace('_', ' ')}"
        }
        for capability in AgentCapability
    }

    return {"capabilities": capabilities, "total": len(capabilities)}

@app.websocket("/ws/agents/{agent_id}")
async def websocket_agent_endpoint(websocket: WebSocket, agent_id: str):
    """WebSocket endpoint for agent communication"""
    await manager.connect(websocket, agent_id)

    try:
        while True:
            # Receive message from agent
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # Create message
            agent_message = AgentMessage(
                sender_id=agent_id,
                receiver_id=message_data.get("receiver_id", "orchestrator"),
                message_type=message_data.get("message_type", "websocket"),
                content=message_data.get("content", {}),
                correlation_id=message_data.get("correlation_id")
            )

            # Forward message to orchestrator
            if orchestrator:
                await orchestrator.send_message(agent_message)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error for agent {agent_id}: {e}")
        manager.disconnect(websocket)

@app.websocket("/ws/monitor")
async def websocket_monitor_endpoint(websocket: WebSocket):
    """WebSocket endpoint for system monitoring"""
    await websocket.accept()

    try:
        while True:
            # Send periodic status updates
            if orchestrator:
                status = await orchestrator.get_system_status()
                await websocket.send_text(json.dumps({
                    "type": "status_update",
                    "timestamp": datetime.now().isoformat(),
                    "data": status
                }))

            await asyncio.sleep(5)  # Update every 5 seconds

    except WebSocketDisconnect:
        logger.info("Monitor WebSocket disconnected")
    except Exception as e:
        logger.error(f"Monitor WebSocket error: {e}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.now().isoformat()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "timestamp": datetime.now().isoformat()
            }
        }
    )

# Background task examples
@app.post("/api/v1/examples/learning-task")
async def create_learning_task_example():
    """Create an example learning task"""
    task_data = TaskSubmission(
        task_type="file_scan_learning",
        title="Learn from recent documents",
        description="Scan and learn from recent document updates",
        required_capabilities=["data_processing", "knowledge_extraction", "document_processing"],
        parameters={
            "directories": [
                "/Users/gavin/Nutstore Files/.symlinks/坚果云/005-询盘询价和/",
                "/Users/gavin/Nutstore Files/.symlinks/坚果云/002-客户中/"
            ],
            "days_back": 7,
            "batch_size": 20
        },
        priority="high"
    )

    return await submit_task(task_data)

@app.post("/api/v1/examples/data-processing-task")
async def create_data_processing_task_example():
    """Create an example data processing task"""
    task_data = TaskSubmission(
        task_type="data_transformation",
        title="Process sales data",
        description="Transform and validate recent sales data",
        required_capabilities=["data_processing"],
        parameters={
            "source_data": [
                {"product": "A", "sales": 100, "region": "North"},
                {"product": "B", "sales": 150, "region": "South"}
            ],
            "transformations": [
                {"type": "filter", "criteria": {"region": "North"}},
                {"type": "map", "mapping": {"sales": "revenue"}}
            ]
        },
        priority="normal"
    )

    return await submit_task(task_data)

@app.post("/api/v1/examples/document-analysis-task")
async def create_document_analysis_task_example():
    """Create an example document analysis task"""
    task_data = TaskSubmission(
        task_type="document_parsing",
        title="Analyze contract document",
        description="Parse and analyze a contract document for key terms",
        required_capabilities=["document_processing", "text_analysis"],
        parameters={
            "file_path": "/path/to/contract.pdf",
            "options": {
                "extract_entities": True,
                "analyze_sentiment": False,
                "generate_summary": True
            }
        },
        priority="high"
    )

    return await submit_task(task_data)

# Main execution
if __name__ == "__main__":
    uvicorn.run(
        "multi_agent_api:app",
        host="0.0.0.0",
        port=8004,
        reload=True,
        log_level="info"
    )