"""
XAgent Integration Framework
Manufacturing Knowledge Base - Advanced Multi-Agent System Integration

This integration provides XAgent-powered multi-agent orchestration with manufacturing-specific
agents, task decomposition, and autonomous execution capabilities.
"""

from .integration import XAgentIntegration
from .orchestrator import (
    ManufacturingOrchestrator,
    TaskDecomposer,
    AgentCoordinator,
    ExecutionEngine
)
from .agents import (
    ManufacturingAgent,
    SafetyInspectorAgent,
    QualityControlAgent,
    TechnicalExpertAgent,
    MaintenanceAgent,
    ProcessOptimizerAgent
)
from .tasks import (
    ManufacturingTask,
    SafetyTask,
    QualityTask,
    TechnicalTask,
    MaintenanceTask
)
from .protocols import (
    AgentProtocol,
    MessageProtocol,
    TaskProtocol,
    CoordinationProtocol
)
from .workflows import (
    ManufacturingWorkflow,
    SafetyWorkflow,
    QualityWorkflow,
    MaintenanceWorkflow,
    ProcessWorkflow
)
from .tools import (
    ManufacturingTools,
    SafetyTools,
    QualityTools,
    TechnicalTools,
    MaintenanceTools
)

__version__ = "2.0.0"
__all__ = [
    # Core integration
    "XAgentIntegration",

    # Orchestrator components
    "ManufacturingOrchestrator",
    "TaskDecomposer",
    "AgentCoordinator",
    "ExecutionEngine",

    # Manufacturing agents
    "ManufacturingAgent",
    "SafetyInspectorAgent",
    "QualityControlAgent",
    "TechnicalExpertAgent",
    "MaintenanceAgent",
    "ProcessOptimizerAgent",

    # Task management
    "ManufacturingTask",
    "SafetyTask",
    "QualityTask",
    "TechnicalTask",
    "MaintenanceTask",

    # Communication protocols
    "AgentProtocol",
    "MessageProtocol",
    "TaskProtocol",
    "CoordinationProtocol",

    # Workflows
    "ManufacturingWorkflow",
    "SafetyWorkflow",
    "QualityWorkflow",
    "MaintenanceWorkflow",
    "ProcessWorkflow",

    # Tools and utilities
    "ManufacturingTools",
    "SafetyTools",
    "QualityTools",
    "TechnicalTools",
    "MaintenanceTools",
]