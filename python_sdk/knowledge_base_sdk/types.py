"""
Type definitions for the Knowledge Base API SDK.

Uses Pydantic for data validation and serialization, providing strong type hints
and runtime validation for all API interactions.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Literal
from uuid import UUID

from pydantic import BaseModel, Field, validator, HttpUrl


class SearchStrategy(str, Enum):
    """Search strategy options"""
    UNIFIED = "unified"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    GRAPH = "graph"
    AUTO = "auto"


class FeedbackType(str, Enum):
    """User feedback types"""
    CLICK = "click"
    SATISFACTION = "satisfaction"
    RATING = "rating"


class CustomerStatus(str, Enum):
    """Customer status options"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PROSPECT = "prospect"


class SearchDirection(str, Enum):
    """Knowledge graph search direction"""
    FORWARD = "forward"
    BACKWARD = "backward"
    BOTH = "both"


class ExpansionType(str, Enum):
    """Query expansion types"""
    SYNONYM = "synonym"
    ABBREVIATION = "abbreviation"
    RELATED = "related"
    HYPERNYM = "hypernym"
    HYPONYM = "hyponym"


class SourceType(str, Enum):
    """Result source types"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    KNOWLEDGE_GRAPH = "knowledge_graph"


class QueryComplexity(str, Enum):
    """Query complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


# Base Models
class BaseAPIModel(BaseModel):
    """Base model for all API models"""
    class Config:
        extra = "forbid"
        use_enum_values = True


# Request Models
class SearchRequest(BaseAPIModel):
    """Base search request"""
    query: str = Field(..., description="Search query", min_length=1, max_length=1000)
    top_k: int = Field(10, description="Number of results to return", ge=1, le=100)
    similarity_threshold: float = Field(
        0.7,
        description="Similarity threshold",
        ge=0.0,
        le=1.0
    )
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")


class UnifiedSearchRequest(SearchRequest):
    """Unified search request"""
    search_strategy: SearchStrategy = Field(SearchStrategy.UNIFIED, description="Search strategy")
    rerank: bool = Field(True, description="Whether to apply reranking")
    include_metadata: bool = Field(True, description="Whether to include metadata")


class SemanticSearchRequest(SearchRequest):
    """Semantic search request"""
    pass


class KeywordSearchRequest(SearchRequest):
    """Keyword search request"""
    pass


class KnowledgeGraphSearchRequest(BaseAPIModel):
    """Knowledge graph search request"""
    entity_name: str = Field(..., description="Entity name to search for", min_length=1)
    relation_type: Optional[str] = Field(None, description="Type of relation to follow")
    direction: SearchDirection = Field(SearchDirection.BOTH, description="Search direction")
    max_depth: int = Field(3, description="Maximum depth for graph traversal", ge=1, le=10)
    top_k: int = Field(10, description="Number of results to return", ge=1, le=100)


class PersonalizedSearchRequest(UnifiedSearchRequest):
    """Personalized search request"""
    user_id: str = Field(..., description="User identifier", min_length=1)
    session_id: Optional[str] = Field(None, description="Session identifier")
    personalization_level: float = Field(
        0.7,
        description="Personalization strength",
        ge=0.0,
        le=1.0
    )
    boost_expertise: bool = Field(True, description="Boost based on user expertise")
    boost_history: bool = Field(True, description="Boost based on search history")
    boost_preferences: bool = Field(True, description="Boost based on content preferences")


class DocumentInfo(BaseAPIModel):
    """Document information"""
    id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title", min_length=1)
    content: str = Field(..., description="Document content", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Document metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Document timestamp")


class IndexDocumentRequest(BaseAPIModel):
    """Document indexing request"""
    document_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title", min_length=1)
    content: str = Field(..., description="Document content", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Document metadata")


class BatchIndexDocumentsRequest(BaseAPIModel):
    """Batch document indexing request"""
    documents: List[IndexDocumentRequest] = Field(
        ...,
        description="List of documents to index",
        min_items=1,
        max_items=1000
    )


class UserFeedback(BaseAPIModel):
    """User feedback for search results"""
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    result_id: str = Field(..., description="Clicked result ID")
    feedback_type: FeedbackType = Field(..., description="Type of feedback")
    dwell_time: float = Field(0.0, description="Time spent on result in seconds", ge=0.0)
    satisfaction_score: Optional[float] = Field(
        None,
        description="Satisfaction score",
        ge=0.0,
        le=1.0
    )


class ConsentRequest(BaseAPIModel):
    """User consent request"""
    user_id: str = Field(..., description="User identifier")
    consent_given: bool = Field(..., description="Whether consent is given")
    consent_text: str = Field(..., description="Consent text description", min_length=1)
    data_purposes: List[str] = Field(default_factory=list, description="Purposes of data processing")


class PrivacyConfig(BaseAPIModel):
    """User privacy configuration"""
    user_id: str = Field(..., description="User identifier")
    tracking_enabled: bool = Field(True, description="Enable behavior tracking")
    query_history_retention_days: int = Field(
        30,
        description="Query history retention in days",
        ge=1,
        le=365
    )
    click_tracking_enabled: bool = Field(True, description="Enable click tracking")
    expertise_learning_enabled: bool = Field(True, description="Enable expertise learning")
    personalization_enabled: bool = Field(True, description="Enable result personalization")
    data_anonymization_enabled: bool = Field(True, description="Enable data anonymization")
    auto_delete_after_days: Optional[int] = Field(
        365,
        description="Auto-delete data after days",
        ge=1,
        le=2550
    )
    gdpr_compliant: bool = Field(True, description="GDPR compliance")
    ccpa_compliant: bool = Field(True, description="CCPA compliance")


class QuerySuggestionRequest(BaseAPIModel):
    """Query suggestion request"""
    user_id: str = Field(..., description="User identifier")
    partial_query: str = Field(..., description="Partial query for suggestions", min_length=1)
    max_suggestions: int = Field(5, description="Maximum number of suggestions", ge=1, le=20)


# Response Models
class SearchResult(BaseAPIModel):
    """Single search result"""
    id: str = Field(..., description="Result ID")
    title: str = Field(..., description="Result title")
    content: str = Field(..., description="Result content snippet")
    score: float = Field(..., description="Relevance score", ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Result metadata")
    source_type: SourceType = Field(..., description="Source of the result")
    explanation: Optional[str] = Field(None, description="Explanation of result relevance")


class QueryExpansion(BaseAPIModel):
    """Query expansion information"""
    original_term: str = Field(..., description="Original query term")
    expanded_terms: List[str] = Field(..., description="Expanded terms")
    expansion_type: ExpansionType = Field(..., description="Type of expansion")
    confidence: float = Field(..., description="Expansion confidence", ge=0.0, le=1.0)


class SearchAnalytics(BaseAPIModel):
    """Search analytics data"""
    query_complexity: QueryComplexity = Field(..., description="Query complexity")
    total_results: int = Field(..., description="Total number of results")
    search_time_ms: float = Field(..., description="Search time in milliseconds")
    cache_hit: bool = Field(..., description="Whether result was from cache")
    semantic_score_avg: Optional[float] = Field(None, description="Average semantic score")
    keyword_score_avg: Optional[float] = Field(None, description="Average keyword score")
    graph_traversal_depth: Optional[int] = Field(None, description="Graph traversal depth")


class UserExpertiseDomain(BaseAPIModel):
    """User expertise in a specific domain"""
    domain: str = Field(..., description="Expertise domain")
    score: float = Field(..., description="Expertise score", ge=0.0, le=1.0)


class UserExpertiseProfile(BaseAPIModel):
    """User expertise profile"""
    user_id: str = Field(..., description="User identifier")
    expertise_domains: Dict[str, float] = Field(default_factory=dict, description="Expertise scores by domain")
    technical_level: float = Field(..., description="Overall technical level", ge=0.0, le=1.0)
    vocabulary_richness: float = Field(..., description="Vocabulary richness score", ge=0.0, le=1.0)
    confidence_score: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    last_updated: datetime = Field(..., description="Last update timestamp")


# Knowledge Base Models
class KnowledgeEntry(BaseAPIModel):
    """Knowledge base entry"""
    id: int = Field(..., description="Entry ID")
    title: str = Field(..., description="Entry title")
    content: str = Field(..., description="Entry content")
    category: Optional[str] = Field(None, description="Entry category")
    tags: List[str] = Field(default_factory=list, description="Entry tags")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class KnowledgeEntryCreate(BaseAPIModel):
    """Knowledge entry creation request"""
    title: str = Field(..., description="Entry title", min_length=1)
    content: str = Field(..., description="Entry content", min_length=1)
    category: Optional[str] = Field(None, description="Entry category")
    tags: List[str] = Field(default_factory=list, description="Entry tags")


class KnowledgeEntryUpdate(BaseAPIModel):
    """Knowledge entry update request"""
    title: Optional[str] = Field(None, description="Entry title")
    content: Optional[str] = Field(None, description="Entry content")
    category: Optional[str] = Field(None, description="Entry category")
    tags: Optional[List[str]] = Field(None, description="Entry tags")


class Customer(BaseAPIModel):
    """Customer information"""
    id: int = Field(..., description="Customer ID")
    name: str = Field(..., description="Customer name")
    email: str = Field(..., description="Customer email")
    phone: Optional[str] = Field(None, description="Customer phone")
    address: Optional[str] = Field(None, description="Customer address")
    status: CustomerStatus = Field(..., description="Customer status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


# Response Wrapper Models
class APIResponse(BaseAPIModel):
    """Base API response"""
    success: bool = Field(..., description="Whether the request was successful")
    message: Optional[str] = Field(None, description="Response message")
    timestamp: datetime = Field(..., description="Response timestamp")


class SearchResponse(BaseAPIModel):
    """Search response"""
    search_id: Optional[str] = Field(None, description="Search session ID")
    query: str = Field(..., description="Original query")
    strategy: Optional[str] = Field(None, description="Search strategy used")
    results: List[SearchResult] = Field(..., description="Search results")
    query_expansions: List[QueryExpansion] = Field(default_factory=list, description="Query expansions")
    aggregated_result: Optional[SearchResult] = Field(None, description="Aggregated best result")
    execution_time: float = Field(..., description="Execution time in seconds")
    analytics: Optional[SearchAnalytics] = Field(None, description="Search analytics")
    timestamp: datetime = Field(..., description="Response timestamp")


class PersonalizedSearchResponse(SearchResponse):
    """Personalized search response"""
    session_id: str = Field(..., description="User session ID")
    personalized_results: List[SearchResult] = Field(..., description="Personalized results")
    personalization_applied: bool = Field(..., description="Whether personalization was applied")
    personalization_level: float = Field(..., description="Personalization level used")
    user_expertise_domains: Dict[str, float] = Field(default_factory=dict, description="User expertise domains")
    privacy_anonymized: bool = Field(True, description="Whether data was anonymized")


class DocumentIndexResponse(BaseAPIModel):
    """Document indexing response"""
    success: bool = Field(..., description="Whether indexing was successful")
    document_id: str = Field(..., description="Document ID")
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(..., description="Response timestamp")


class BatchIndexResponse(BaseAPIModel):
    """Batch indexing response"""
    results: List[bool] = Field(..., description="Indexing results for each document")
    total_documents: int = Field(..., description="Total number of documents")
    successful_indexed: int = Field(..., description="Number of successfully indexed documents")
    timestamp: datetime = Field(..., description="Response timestamp")


class AnalyticsResponse(BaseAPIModel):
    """Analytics response"""
    engine_analytics: Dict[str, Any] = Field(..., description="Engine analytics data")
    recent_sessions: List[Dict[str, Any]] = Field(default_factory=list, description="Recent search sessions")
    active_connections: int = Field(..., description="Number of active WebSocket connections")
    total_sessions: int = Field(..., description="Total number of sessions")
    timestamp: datetime = Field(..., description="Response timestamp")


class HealthCheckResponse(BaseAPIModel):
    """Health check response"""
    status: Literal["healthy", "unhealthy"] = Field(..., description="Health status")
    response_time: Optional[float] = Field(None, description="Response time in seconds")
    document_count: Optional[int] = Field(None, description="Number of indexed documents")
    active_connections: Optional[int] = Field(None, description="Number of active connections")
    timestamp: datetime = Field(..., description="Health check timestamp")


# WebSocket Message Models
class WebSocketMessage(BaseAPIModel):
    """Base WebSocket message"""
    type: str = Field(..., description="Message type")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")


class WebSocketSearchMessage(WebSocketMessage):
    """WebSocket search request message"""
    type: Literal["search"] = "search"
    query: str = Field(..., description="Search query")
    strategy: str = Field("unified", description="Search strategy")
    top_k: int = Field(10, description="Number of results", ge=1, le=100)
    threshold: float = Field(0.7, description="Similarity threshold", ge=0.0, le=1.0)


class WebSocketSearchResultsMessage(WebSocketMessage):
    """WebSocket search results message"""
    type: Literal["results"] = "results"
    search_id: str = Field(..., description="Search session ID")
    results: List[SearchResult] = Field(..., description="Search results")
    execution_time: float = Field(..., description="Execution time in seconds")
    analytics: Optional[SearchAnalytics] = Field(None, description="Search analytics")


class WebSocketProgressMessage(WebSocketMessage):
    """WebSocket progress message"""
    type: Literal["progress"] = "progress"
    search_id: str = Field(..., description="Search session ID")
    status: str = Field(..., description="Current status")
    message: str = Field(..., description="Status message")


class WebSocketSuggestionsMessage(WebSocketMessage):
    """WebSocket suggestions message"""
    type: Literal["suggestions"] = "suggestions"
    query: str = Field(..., description="Original query")
    suggestions: List[str] = Field(..., description="Query suggestions")


class WebSocketErrorMessage(WebSocketMessage):
    """WebSocket error message"""
    type: Literal["error"] = "error"
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")