"""
Manufacturing Knowledge Base - Open-Source Component Integrations

This package provides comprehensive integration frameworks for leading open-source
AI and observability components, specifically adapted for manufacturing and
industrial knowledge management use cases.

Supported Integrations:
- LangChain: Advanced AI/LLM processing with manufacturing expertise
- LobeChat: Modern chat interface with manufacturing themes
- XAgent: Multi-agent orchestration for complex manufacturing workflows
- LangFuse: Comprehensive observability and compliance monitoring
"""

from .shared.manager import IntegrationManager
from .shared.base import ManufacturingContext

# Import individual integrations
try:
    from .langchain import LangChainIntegration
except ImportError:
    LangChainIntegration = None

try:
    from .lobechat import LobeChatIntegration
except ImportError:
    LobeChatIntegration = None

try:
    from .xagent import XAgentIntegration
except ImportError:
    XAgentIntegration = None

try:
    from .langfuse import LangFuseIntegration
except ImportError:
    LangFuseIntegration = None

__version__ = "2.0.0"

# Available integrations registry
AVAILABLE_INTEGRATIONS = {
    "langchain": LangChainIntegration,
    "lobechat": LobeChatIntegration,
    "xagent": XAgentIntegration,
    "langfuse": LangFuseIntegration,
}

# Integration metadata
INTEGRATION_INFO = {
    "langchain": {
        "name": "LangChain Integration",
        "description": "Advanced AI/LLM processing with manufacturing expertise",
        "version": "2.0.0",
        "capabilities": [
            "Manufacturing-specific AI chains",
            "Safety procedure generation",
            "Quality control assistance",
            "Technical specification analysis"
        ]
    },
    "lobechat": {
        "name": "LobeChat Integration",
        "description": "Modern chat interface with manufacturing themes",
        "version": "2.0.0",
        "capabilities": [
            "Manufacturing-specific themes",
            "Quick action templates",
            "Real-time collaboration",
            "Multi-modal support"
        ]
    },
    "xagent": {
        "name": "XAgent Integration",
        "description": "Multi-agent orchestration for manufacturing workflows",
        "version": "2.0.0",
        "capabilities": [
            "Manufacturing-specific agents",
            "Task decomposition",
            "Workflow orchestration",
            "Autonomous execution"
        ]
    },
    "langfuse": {
        "name": "LangFuse Integration",
        "description": "Observability and compliance monitoring",
        "version": "2.0.0",
        "capabilities": [
            "AI performance tracking",
            "Manufacturing metrics",
            "Compliance monitoring",
            "Cost optimization"
        ]
    }
}

__all__ = [
    "IntegrationManager",
    "ManufacturingContext",
    "LangChainIntegration",
    "LobeChatIntegration",
    "XAgentIntegration",
    "LangFuseIntegration",
    "AVAILABLE_INTEGRATIONS",
    "INTEGRATION_INFO"
]