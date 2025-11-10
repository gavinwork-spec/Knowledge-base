"""
Multi-Agent Orchestration System
XAgent-inspired comprehensive multi-agent orchestration system for manufacturing knowledge base management.

This system provides:
- Advanced agent communication protocols
- Specialized agents for document processing, price analysis, trend prediction, and customer insights
- Coordinator agent for complex task decomposition
- Agent marketplace for registration and discovery
- Performance monitoring and optimization
- YAML configuration compatibility layer

Core Components:
- protocols/agent_communication.py: Advanced message routing and task delegation
- agents/specialized_agents.py: Enhanced base agents with communication capabilities
- agents/coordinator_agent.py: Task coordination and planning
- agents/trend_predictor_agent.py: Trend forecasting and market analysis
- agents/customer_insights_agent.py: Customer behavior analysis and segmentation
- marketplace/agent_marketplace.py: Agent registration and discovery system
- monitoring/performance_monitor.py: Performance monitoring and optimization
- core/yaml_compatibility.py: Legacy YAML configuration compatibility layer

Usage:
    from multi_agent_system import XAgentOrchestrator

    orchestrator = XAgentOrchestrator()
    await orchestrator.initialize()

    # Or use individual components:
    from multi_agent_system.agents.specialized_agents import DocumentProcessorAgent
    from multi_agent_system.marketplace.agent_marketplace import create_agent_marketplace
    from multi_agent_system.core.yaml_compatibility import create_yaml_compatibility_layer

    processor = DocumentProcessorAgent(orchestrator)
    marketplace = create_agent_marketplace()
    compatibility = create_yaml_compatibility_layer()
"""

__version__ = "2.0.0"
__author__ = "Manufacturing Knowledge Base Team"
__description__ = "XAgent-inspired multi-agent orchestration system"

# Core imports for easy access
from .agents.coordinator_agent import CoordinatorAgent, create_coordinator_agent
from .agents.specialized_agents import EnhancedBaseAgent, DocumentProcessorAgent, create_document_processor_agent
from .agents.trend_predictor_agent import TrendPredictorAgent, create_trend_predictor_agent
from .agents.customer_insights_agent import CustomerInsightsAgent, create_customer_insights_agent
from .marketplace.agent_marketplace import AgentMarketplace, create_agent_marketplace, AgentType
from .monitoring.performance_monitor import PerformanceMonitor, create_performance_monitor
from .core.yaml_compatibility import YAMLCompatibilityLayer, create_yaml_compatibility_layer

# Main orchestrator that ties everything together
from multi_agent_orchestrator import MultiAgentOrchestrator

__all__ = [
    # Core orchestrator
    'MultiAgentOrchestrator',

    # Agent classes
    'CoordinatorAgent',
    'create_coordinator_agent',
    'EnhancedBaseAgent',
    'DocumentProcessorAgent',
    'create_document_processor_agent',
    'TrendPredictorAgent',
    'create_trend_predictor_agent',
    'CustomerInsightsAgent',
    'create_customer_insights_agent',

    # System components
    'AgentMarketplace',
    'create_agent_marketplace',
    'AgentType',
    'PerformanceMonitor',
    'create_performance_monitor',
    'YAMLCompatibilityLayer',
    'create_yaml_compatibility_layer',

    # Version info
    '__version__',
    '__author__',
    '__description__'
]