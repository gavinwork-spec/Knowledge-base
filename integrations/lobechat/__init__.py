"""
LobeChat Integration Framework
Manufacturing Knowledge Base - Modern Chat Interface Integration

This integration provides LobeChat-powered chat interface capabilities with manufacturing-specific
enhancements including custom themes, quick actions, and workflow templates.
"""

from .integration import LobeChatIntegration
from .components import (
    ManufacturingChatInterface,
    SafetyProcedureChat,
    QualityControlChat,
    TechnicalSupportChat,
    MaintenanceChat
)
from .themes import (
    ManufacturingTheme,
    SafetyTheme,
    QualityTheme,
    DefaultTheme
)
from .templates import (
    ManufacturingTemplates,
    SafetyTemplates,
    QualityTemplates,
    TechnicalTemplates
)
from .websocket import (
    WebSocketManager,
    ChatSessionManager,
    RealtimeCollaboration
)
from .ui import (
    ChatComponents,
    MessageComponents,
    InputComponents,
    ActionComponents
)

__version__ = "2.0.0"
__all__ = [
    # Core integration
    "LobeChatIntegration",

    # Chat interfaces
    "ManufacturingChatInterface",
    "SafetyProcedureChat",
    "QualityControlChat",
    "TechnicalSupportChat",
    "MaintenanceChat",

    # Themes
    "ManufacturingTheme",
    "SafetyTheme",
    "QualityTheme",
    "DefaultTheme",

    # Templates
    "ManufacturingTemplates",
    "SafetyTemplates",
    "QualityTemplates",
    "TechnicalTemplates",

    # WebSocket management
    "WebSocketManager",
    "ChatSessionManager",
    "RealtimeCollaboration",

    # UI components
    "ChatComponents",
    "MessageComponents",
    "InputComponents",
    "ActionComponents",
]