"""
LangChain Integration Framework
Manufacturing Knowledge Base - Advanced AI/LLM Capabilities

This integration provides LangChain-powered AI capabilities with manufacturing-specific
enhancements including specialized chains, prompts, and workflows.
"""

from .integration import LangChainIntegration
from .chains import (
    ManufacturingQueryChain,
    SafetyProcedureChain,
    QualityControlChain,
    TechnicalSpecificationChain,
    MaintenanceWorkflowChain
)
from .prompts import (
    ManufacturingPromptTemplate,
    SafetyPromptTemplate,
    QualityPromptTemplate,
    TechnicalPromptTemplate
)
from .memory import (
    ManufacturingMemory,
    ConversationMemory,
    ProcedureMemory,
    ComplianceMemory
)
from .agents import (
    ManufacturingAgent,
    SafetyInspectorAgent,
    QualityControlAgent,
    TechnicalExpertAgent
)
from .retrievers import (
    ManufacturingRetriever,
    SafetyDocumentRetriever,
    TechnicalSpecRetriever,
    ComplianceRetriever
)

__version__ = "2.0.0"
__all__ = [
    # Core integration
    "LangChainIntegration",

    # Manufacturing-specific chains
    "ManufacturingQueryChain",
    "SafetyProcedureChain",
    "QualityControlChain",
    "TechnicalSpecificationChain",
    "MaintenanceWorkflowChain",

    # Specialized prompts
    "ManufacturingPromptTemplate",
    "SafetyPromptTemplate",
    "QualityPromptTemplate",
    "TechnicalPromptTemplate",

    # Enhanced memory systems
    "ManufacturingMemory",
    "ConversationMemory",
    "ProcedureMemory",
    "ComplianceMemory",

    # Specialized agents
    "ManufacturingAgent",
    "SafetyInspectorAgent",
    "QualityControlAgent",
    "TechnicalExpertAgent",

    # Advanced retrievers
    "ManufacturingRetriever",
    "SafetyDocumentRetriever",
    "TechnicalSpecRetriever",
    "ComplianceRetriever",
]