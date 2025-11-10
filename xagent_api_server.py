#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAgent API Server
XAgent API服务器

This server provides RESTful API endpoints for the XAgent orchestration system,
enabling integration with external systems and real-time monitoring of agent activities.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import uuid
import threading
import time
from pathlib import Path

# Import XAgent components
from xagent_orchestrator import XAgentOrchestrator, XAgentTask, TaskPriority, TaskStatus, ManufacturingContext
from xagent_config_manager import XAgentConfigManager

# Import existing API components for integration
from multi_agent_orchestrator import AgentStatus, AgentCapability

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler('data/processed/xagent_api_server.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize XAgent components
config_manager = XAgentConfigManager()
orchestrator = None
api_server_running = False

# Configuration
API_PORT = 8003
API_VERSION = "2.0.0"

def initialize_xagent_system():
    """Initialize the XAgent system"""
    global orchestrator, api_server_running

    try:
        logger.info("🚀 Initializing XAgent System")

        # Load configurations
        asyncio.run(config_manager.load_configurations())

        # Initialize orchestrator
        orchestrator = XAgentOrchestrator()
        asyncio.run(orchestrator.initialize())

        # Load agent configurations into orchestrator
        await _load_agent_configurations()

        api_server_running = True
        logger.info("✅ XAgent System initialized successfully")

    except Exception as e:
        logger.error(f"❌ Failed to initialize XAgent system: {e}")
        api_server_running = False

async def _load_agent_configurations():
    """Load agent configurations into orchestrator"""
    if not orchestrator:
        return

    agent_configs = config_manager.get_all_agent_configs()

    for agent_id, config in agent_configs.items():
        try:
            # Create agent instance based on configuration
            agent_class = _get_agent_class(config.get("class", "GenericAgent"))

            # Convert config to XAgent format
            xagent_config = await _convert_config_to_xagent(config)

            # Initialize agent
            agent = agent_class(
                agent_id=config["id"],
                name=config["name"],
                role=AgentRole(config["role"]),
                capabilities=[AgentCapability(cap) for cap in config.get("capabilities", [])],
                orchestrator=orchestrator
            )

            # Initialize agent
            await agent.initialize()

            # Register with orchestrator
            orchestrator.agents[agent_id] = agent

            logger.info(f"✅ Loaded agent: {config['name']} ({config['role']})")

        except Exception as e:
            logger.error(f"❌ Failed to load agent {agent_id}: {e}")

def _get_agent_class(class_name: str):
    """Get agent class by name"""
    # Import agent classes (would be in separate files)
    from xagent_orchestrator import XAgent
    return XAgent  # Simplified for now

async def _convert_config_to_xagent(config: Dict) -> Dict:
    """Convert config manager format to XAgent format"""
    return {
        "task_id": config["id"],
        "title": config["name"],
        "description": config["description"],
        "priority": TaskPriority.NORMAL,
        "required_capabilities": [AgentCapability(cap) for cap in config.get("capabilities", [])],
        "context": ManufacturingContext(
            facility_id="main_facility",
            production_line=config.get("config", {}).get("production_line"),
            equipment_type=config.get("config", {}).get("equipment_type"),
            process_stage=config.get("config", {}).get("process_stage")
        ),
        "input_data": config.get("config", {}),
        "output_requirements": config.get("config", {}),
        "dependencies": [],
        "estimated_duration": timedelta(minutes=5),
        "max_duration": timedelta(hours=1)
    }

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            "status": "healthy" if api_server_running else "initializing",
            "timestamp": datetime.now().isoformat(),
            "version": API_VERSION,
            "xagent_system": {
                "initialized": api_server_running,
                "agents_loaded": len(orchestrator.agents) if orchestrator else 0
            }
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "unhealthy"}), 500

@app.route('/api/xagent/system/status', methods=['GET'])
def get_system_status():
    """Get detailed XAgent system status"""
    try:
        if not orchestrator:
            return jsonify({"error": "XAgent system not initialized"}), 503

        status = await orchestrator.get_system_status()
        return jsonify(status)

    except Exception as e:
        logger.error(f"❌ Failed to get system status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/xagent/agents', methods=['GET'])
def get_agents():
    """Get all registered agents"""
    try:
        if not orchestrator:
            return jsonify({"error": "XAgent system not initialized"}), 503

        agents_info = {}
        for agent_id, agent in orchestrator.agents.items():
            agents_info[agent_id] = {
                "id": agent_id,
                "name": agent.name,
                "role": agent.role.value,
                "status": agent.status,
                "capabilities": [cap.value for cap in agent.capabilities],
                "current_task": agent.current_task.task_id if agent.current_task else None,
                "performance_metrics": agent.performance_metrics,
                "skills_count": len(agent.skills)
            }

        return jsonify({
            "agents": agents_info,
            "total": len(agents_info)
        })

    except Exception as e:
        logger.error(f"❌ Failed to get agents: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/xagent/agents/<agent_id>', methods=['GET'])
def get_agent_details(agent_id: str):
    """Get detailed information about a specific agent"""
    try:
        if not orchestrator or agent_id not in orchestrator.agents:
            return jsonify({"error": f"Agent {agent_id} not found"}), 404

        agent = orchestrator.agents[agent_id]

        agent_details = {
            "id": agent.agent_id,
            "name": agent.name,
            "role": agent.role.value,
            "capabilities": [cap.value for cap in agent.capabilities],
            "status": agent.status,
            "current_task": None,
            "performance_metrics": agent.performance_metrics,
            "skills": {
                skill_name: {
                    "name": skill.name,
                    "capability": skill.capability.value,
                    "proficiency": skill.proficiency,
                    "experience": skill.experience,
                    "success_rate": skill.success_rate,
                    "last_used": skill.last_used.isoformat()
                }
                for skill_name, skill in agent.skills.items()
            },
            "knowledge_base_size": len(agent.knowledge_base),
            "communication_channels": len(agent.communication_channels)
        }

        if agent.current_task:
            agent_details["current_task"] = {
                "task_id": agent.current_task.task_id,
                "title": agent.current_task.title,
                "status": agent.current_task.status.value,
                "progress": agent.current_task.progress,
                "created_at": agent.current_task.created_at.isoformat()
            }

        return jsonify(agent_details)

    except Exception as e:
        logger.error(f"❌ Failed to get agent details: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/xagent/tasks', methods=['POST'])
def submit_task():
    """Submit a new task to the XAgent system"""
    try:
        if not orchestrator:
            return jsonify({"error": "XAgent system not initialized"}), 503

        data = request.get_json()

        # Validate required fields
        required_fields = ["title", "description", "required_capabilities"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Convert capabilities
        required_caps = [AgentCapability(cap) for cap in data.get("required_capabilities", [])]

        # Determine priority
        priority = TaskPriority.NORMAL
        if data.get("priority"):
            priority_map = {
                "critical": TaskPriority.CRITICAL,
                "high": TaskPriority.HIGH,
                "normal": TaskPriority.NORMAL,
                "low": TaskPriority.LOW
            }
            priority = priority_map.get(data["priority"], TaskPriority.NORMAL)

        # Create manufacturing context
        context = ManufacturingContext(
            facility_id=data.get("context", {}).get("facility_id", "main_facility"),
            production_line=data.get("context", {}).get("production_line"),
            equipment_type=data.get("context", {}).get("equipment_type"),
            process_stage=data.get("context", {}).get("process_stage"),
            safety_level=data.get("context", {}).get("safety_level", "standard")
        )

        # Create task
        task = XAgentTask(
            task_id=data.get("task_id", str(uuid.uuid4())),
            title=data["title"],
            description=data["description"],
            priority=priority,
            required_capabilities=required_caps,
            context=context,
            input_data=data.get("input_data", {}),
            output_requirements=data.get("output_requirements", {}),
            dependencies=data.get("dependencies", []),
            estimated_duration=timedelta(minutes=data.get("estimated_duration", 5)),
            max_duration=timedelta(hours=data.get("max_duration", 1))
        )

        # Submit task
        task_id = await orchestrator.submit_task(task)

        return jsonify({
            "task_id": task_id,
            "status": "submitted",
            "message": "Task submitted successfully",
            "estimated_duration": str(task.estimated_duration)
        })

    except Exception as e:
        logger.error(f"❌ Failed to submit task: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/xagent/tasks/<task_id>', methods=['GET'])
def get_task_status(task_id: str):
    """Get status of a specific task"""
    try:
        if not orchestrator:
            return jsonify({"error": "XAgent system not initialized"}), 503

        # Check active tasks
        if task_id in orchestrator.active_tasks:
            task = orchestrator.active_tasks[task_id]
            return jsonify({
                "task_id": task.task_id,
                "title": task.title,
                "status": task.status.value,
                "progress": task.progress,
                "assigned_agents": task.assigned_agents,
                "created_at": task.created_at.isoformat(),
                "result": task.result,
                "error_message": task.error_message
            })

        # Check completed tasks
        for task in orchestrator.completed_tasks:
            if task.task_id == task_id:
                return jsonify({
                    "task_id": task.task_id,
                    "title": task.title,
                    "status": task.status.value,
                    "progress": task.progress,
                    "assigned_agents": task.assigned_agents,
                    "created_at": task.created_at.isoformat(),
                    "completed_at": datetime.now().isoformat(),
                    "result": task.result,
                    "error_message": task.error_message
                })

        return jsonify({"error": f"Task {task_id} not found"}), 404

    except Exception as e:
        logger.error(f"❌ Failed to get task status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/xagent/tasks', methods=['GET'])
def get_tasks():
    """Get all tasks (active and completed)"""
    try:
        if not orchestrator:
            return jsonify({"error": "XAgent system not initialized"}), 503

        active_tasks = []
        for task_id, task in orchestrator.active_tasks.items():
            active_tasks.append({
                "task_id": task.task_id,
                "title": task.title,
                "status": task.status.value,
                "progress": task.progress,
                "priority": task.priority.value,
                "assigned_agents": task.assigned_agents,
                "created_at": task.created_at.isoformat()
            })

        completed_tasks = []
        for task in orchestrator.completed_tasks[-100:]:  # Last 100 completed tasks
            completed_tasks.append({
                "task_id": task.task_id,
                "title": task.title,
                "status": task.status.value,
                "progress": task.progress,
                "priority": task.priority.value,
                "assigned_agents": task.assigned_agents,
                "created_at": task.created_at.isoformat(),
                "completed_at": datetime.now().isoformat(),
                "duration_seconds": None  # Would calculate actual duration
            })

        return jsonify({
            "active_tasks": active_tasks,
            "completed_tasks": completed_tasks,
            "total_active": len(active_tasks),
            "total_completed": len(completed_tasks)
        })

    except Exception as e:
        logger.error(f"❌ Failed to get tasks: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/xagent/config/reload', methods=['POST'])
def reload_configurations():
    """Reload XAgent configurations"""
    try:
        await config_manager.reload_configurations()
        return jsonify({
            "message": "Configurations reloaded successfully",
            "timestamp": datetime.now().isoformat()
        })

    except Exception ase:
        logger.error(f"❌ Failed to reload configurations: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/xagent/config/overview', methods=['GET'])
def get_config_overview():
    """Get configuration overview"""
    try:
        overview = config_manager.get_system_overview()
        return jsonify(overview)

    except Exception as e:
        logger.error(f"❌ Failed to get config overview: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/xagent/migration/log', methods=['GET'])
def get_migration_log():
    """Get migration log"""
    try:
        migration_log = config_manager.get_migration_log()
        return jsonify({
            "migration_log": migration_log,
            "total_migrated": len(migration_log),
            "successful": len([log for log in migration_log if log.get("status") == "success"]),
            "failed": len([log for log in migration_log if log.get("status") == "failed"])
        })

    except Exception as e:
        logger.error(f"❌ Failed to get migration log: {e}")
        return jsonify({"error": str(e)}), 500

# WebSocket Events
@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    logger.info("🔗 Client connected to XAgent WebSocket")
    emit('status', {
        'message': 'Connected to XAgent WebSocket',
        'server_time': datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    logger.info("🔌 Client disconnected from XAgent WebSocket")

@socketio.on('subscribe_tasks')
def handle_subscribe_tasks():
    """Subscribe to task updates"""
    join_room('task_updates')
    emit('subscribed', {'channel': 'task_updates'})

@socketio.on('subscribe_agent/<agent_id>')
def handle_subscribe_agent(agent_id):
    """Subscribe to specific agent updates"""
    join_room(f'agent_{agent_id}')
    emit('subscribed', {'agent_id': agent_id, 'channel': f'agent_{agent_id}'})

# Background initialization
def start_background_initialization():
    """Start background initialization of XAgent system"""
    def init_system():
        try:
            logger.info("🚀 Starting background XAgent initialization")
            initialize_xagent_system()
        except Exception as e:
            logger.error(f"❌ Background initialization failed: {e}")

    init_thread = threading.Thread(target=init_system)
    init_thread.daemon = True
    init_thread.start()

# Start background initialization on server start
start_background_initialization()

if __name__ == '__main__':
    logger.info(f"🌐 Starting XAgent API Server v{API_VERSION}")
    logger.info(f"📡 Server will be available at http://localhost:{API_PORT}")

    try:
        socketio.run(
            app,
            host='0.0.0.0',
            port=API_PORT,
            debug=False,
            allow_unsafe_werkzeug=True
        )
    except KeyboardInterrupt:
        logger.info("👋 XAgent API Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start XAgent API Server: {e}")