"""
Knowledge Base API Python SDK

A comprehensive Python SDK for interacting with the Knowledge Base API Suite.
Provides type-safe interfaces for search, document management, user personalization,
and real-time WebSocket interactions.

Features:
- 🔍 Hybrid search (semantic + keyword + knowledge graph)
- 📚 Document indexing and management
- 👤 Personalized search with privacy controls
- 🔄 Real-time WebSocket support
- 🎯 Type hints and Pydantic models
- 🛡️ Built-in error handling and retries
- 📊 Analytics and monitoring
"""

__version__ = "1.0.0"
__author__ = "Knowledge Base Team"
__email__ = "support@knowledgebase.com"

from .client import KnowledgeBaseClient
from .types import (
    # Search types
    SearchRequest,
    UnifiedSearchRequest,
    SemanticSearchRequest,
    KeywordSearchRequest,
    KnowledgeGraphSearchRequest,
    PersonalizedSearchRequest,
    SearchResult,
    SearchAnalytics,
    QueryExpansion,

    # Document types
    DocumentInfo,
    IndexDocumentRequest,
    BatchIndexDocumentsRequest,

    # User management types
    UserFeedback,
    ConsentRequest,
    PrivacyConfig,
    UserExpertiseProfile,

    # Knowledge base types
    KnowledgeEntry,
    Customer,

    # Response types
    APIResponse,
    SearchResponse,
    PersonalizedSearchResponse,
    DocumentIndexResponse,

    # Exceptions
    KnowledgeBaseError,
    AuthenticationError,
    APIError,
    RateLimitError,
    ValidationError,
)

# WebSocket support
from .websocket import WebSocketClient

__all__ = [
    # Main client
    "KnowledgeBaseClient",
    "WebSocketClient",

    # Search types
    "SearchRequest",
    "UnifiedSearchRequest",
    "SemanticSearchRequest",
    "KeywordSearchRequest",
    "KnowledgeGraphSearchRequest",
    "PersonalizedSearchRequest",
    "SearchResult",
    "SearchAnalytics",
    "QueryExpansion",

    # Document types
    "DocumentInfo",
    "IndexDocumentRequest",
    "BatchIndexDocumentsRequest",

    # User management types
    "UserFeedback",
    "ConsentRequest",
    "PrivacyConfig",
    "UserExpertiseProfile",

    # Knowledge base types
    "KnowledgeEntry",
    "Customer",

    # Response types
    "APIResponse",
    "SearchResponse",
    "PersonalizedSearchResponse",
    "DocumentIndexResponse",

    # Exceptions
    "KnowledgeBaseError",
    "AuthenticationError",
    "APIError",
    "RateLimitError",
    "ValidationError",
]