package kbsdk

import (
	"time"
)

// SearchStrategy represents different search strategies
type SearchStrategy string

const (
	SearchStrategyUnified   SearchStrategy = "unified"
	SearchStrategySemantic  SearchStrategy = "semantic"
	SearchStrategyKeyword   SearchStrategy = "keyword"
	SearchStrategyGraph     SearchStrategy = "graph"
	SearchStrategyAuto      SearchStrategy = "auto"
)

// FeedbackType represents different types of user feedback
type FeedbackType string

const (
	FeedbackTypeClick       FeedbackType = "click"
	FeedbackTypeSatisfaction FeedbackType = "satisfaction"
	FeedbackTypeRating      FeedbackType = "rating"
)

// CustomerStatus represents customer status
type CustomerStatus string

const (
	CustomerStatusActive    CustomerStatus = "active"
	CustomerStatusInactive  CustomerStatus = "inactive"
	CustomerStatusProspect  CustomerStatus = "prospect"
)

// SearchDirection represents knowledge graph search direction
type SearchDirection string

const (
	SearchDirectionForward  SearchDirection = "forward"
	SearchDirectionBackward SearchDirection = "backward"
	SearchDirectionBoth     SearchDirection = "both"
)

// ExpansionType represents query expansion types
type ExpansionType string

const (
	ExpansionTypeSynonym   ExpansionType = "synonym"
	ExpansionTypeAbbrev    ExpansionType = "abbreviation"
	ExpansionTypeRelated   ExpansionType = "related"
	ExpansionTypeHypernym  ExpansionType = "hypernym"
	ExpansionTypeHyponym   ExpansionType = "hyponym"
)

// SourceType represents result source types
type SourceType string

const (
	SourceTypeSemantic       SourceType = "semantic"
	SourceTypeKeyword        SourceType = "keyword"
	SourceTypeKnowledgeGraph SourceType = "knowledge_graph"
)

// QueryComplexity represents query complexity levels
type QueryComplexity string

const (
	QueryComplexitySimple    QueryComplexity = "simple"
	QueryComplexityModerate  QueryComplexity = "moderate"
	QueryComplexityComplex   QueryComplexity = "complex"
)

// ============================================================================
// Request Types
// ============================================================================

// SearchRequest represents a base search request
type SearchRequest struct {
	Query               string                 `json:"query"`
	TopK                int                    `json:"topK,omitempty"`
	SimilarityThreshold float64                `json:"similarityThreshold,omitempty"`
	Filters             map[string]interface{} `json:"filters,omitempty"`
}

// UnifiedSearchRequest represents a unified search request
type UnifiedSearchRequest struct {
	Query               string                 `json:"query"`
	SearchStrategy      SearchStrategy         `json:"searchStrategy,omitempty"`
	TopK                int                    `json:"topK,omitempty"`
	SimilarityThreshold float64                `json:"similarityThreshold,omitempty"`
	Rerank              bool                   `json:"rerank,omitempty"`
	IncludeMetadata     bool                   `json:"includeMetadata,omitempty"`
	Filters             map[string]interface{} `json:"filters,omitempty"`
}

// SemanticSearchRequest represents a semantic search request
type SemanticSearchRequest struct {
	Query               string                 `json:"query"`
	TopK                int                    `json:"topK,omitempty"`
	SimilarityThreshold float64                `json:"similarityThreshold,omitempty"`
	Filters             map[string]interface{} `json:"filters,omitempty"`
}

// KeywordSearchRequest represents a keyword search request
type KeywordSearchRequest struct {
	Query               string                 `json:"query"`
	TopK                int                    `json:"topK,omitempty"`
	SimilarityThreshold float64                `json:"similarityThreshold,omitempty"`
	Filters             map[string]interface{} `json:"filters,omitempty"`
}

// KnowledgeGraphSearchRequest represents a knowledge graph search request
type KnowledgeGraphSearchRequest struct {
	EntityName    string         `json:"entityName"`
	RelationType  *string        `json:"relationType,omitempty"`
	Direction     SearchDirection `json:"direction,omitempty"`
	MaxDepth      int            `json:"maxDepth,omitempty"`
	TopK          int            `json:"topK,omitempty"`
}

// PersonalizedSearchRequest represents a personalized search request
type PersonalizedSearchRequest struct {
	Query                  string                 `json:"query"`
	UserID                 string                 `json:"userId"`
	SessionID              *string                `json:"sessionId,omitempty"`
	SearchStrategy         SearchStrategy         `json:"searchStrategy,omitempty"`
	TopK                   int                    `json:"topK,omitempty"`
	SimilarityThreshold    float64                `json:"similarityThreshold,omitempty"`
	PersonalizationLevel   float64                `json:"personalizationLevel,omitempty"`
	BoostExpertise         bool                   `json:"boostExpertise,omitempty"`
	BoostHistory           bool                   `json:"boostHistory,omitempty"`
	BoostPreferences       bool                   `json:"boostPreferences,omitempty"`
	Filters                map[string]interface{} `json:"filters,omitempty"`
}

// QuerySuggestionRequest represents a query suggestion request
type QuerySuggestionRequest struct {
	UserID         string `json:"userId"`
	PartialQuery   string `json:"partialQuery"`
	MaxSuggestions int    `json:"maxSuggestions,omitempty"`
}

// ============================================================================
// Document Types
// ============================================================================

// DocumentInfo represents document information
type DocumentInfo struct {
	ID        string                 `json:"id"`
	Title     string                 `json:"title"`
	Content   string                 `json:"content"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	Timestamp *time.Time             `json:"timestamp,omitempty"`
}

// IndexDocumentRequest represents a document indexing request
type IndexDocumentRequest struct {
	DocumentID string                 `json:"documentId"`
	Title      string                 `json:"title"`
	Content    string                 `json:"content"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

// BatchIndexDocumentsRequest represents a batch document indexing request
type BatchIndexDocumentsRequest struct {
	Documents []IndexDocumentRequest `json:"documents"`
}

// ============================================================================
// User Management Types
// ============================================================================

// UserFeedback represents user feedback
type UserFeedback struct {
	UserID           string       `json:"userId"`
	SessionID        string       `json:"sessionId"`
	ResultID         string       `json:"resultId"`
	FeedbackType     FeedbackType `json:"feedbackType"`
	DwellTime        float64      `json:"dwellTime,omitempty"`
	SatisfactionScore *float64    `json:"satisfactionScore,omitempty"`
}

// ConsentRequest represents a user consent request
type ConsentRequest struct {
	UserID      string   `json:"userId"`
	ConsentGiven bool     `json:"consentGiven"`
	ConsentText string   `json:"consentText"`
	DataPurposes []string `json:"dataPurposes,omitempty"`
}

// PrivacyConfig represents user privacy configuration
type PrivacyConfig struct {
	UserID                    string  `json:"userId"`
	TrackingEnabled           *bool   `json:"trackingEnabled,omitempty"`
	QueryHistoryRetentionDays *int    `json:"queryHistoryRetentionDays,omitempty"`
	ClickTrackingEnabled      *bool   `json:"clickTrackingEnabled,omitempty"`
	ExpertiseLearningEnabled  *bool   `json:"expertiseLearningEnabled,omitempty"`
	PersonalizationEnabled    *bool   `json:"personalizationEnabled,omitempty"`
	DataAnonymizationEnabled  *bool   `json:"dataAnonymizationEnabled,omitempty"`
	AutoDeleteAfterDays       *int    `json:"autoDeleteAfterDays,omitempty"`
	GDPRCompliant             *bool   `json:"gdprCompliant,omitempty"`
	CCPACompliant             *bool   `json:"ccpaCompliant,omitempty"`
}

// ============================================================================
// Result Types
// ============================================================================

// SearchResult represents a single search result
type SearchResult struct {
	ID          string                 `json:"id"`
	Title       string                 `json:"title"`
	Content     string                 `json:"content"`
	Score       float64                `json:"score"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	SourceType  SourceType             `json:"sourceType"`
	Explanation *string                `json:"explanation,omitempty"`
}

// QueryExpansion represents query expansion information
type QueryExpansion struct {
	OriginalTerm   string        `json:"originalTerm"`
	ExpandedTerms  []string      `json:"expandedTerms"`
	ExpansionType  ExpansionType `json:"expansionType"`
	Confidence     float64       `json:"confidence"`
}

// SearchAnalytics represents search analytics data
type SearchAnalytics struct {
	QueryComplexity      QueryComplexity `json:"queryComplexity"`
	TotalResults         int             `json:"totalResults"`
	SearchTimeMs         float64         `json:"searchTimeMs"`
	CacheHit             bool            `json:"cacheHit"`
	SemanticScoreAvg     *float64        `json:"semanticScoreAvg,omitempty"`
	KeywordScoreAvg      *float64        `json:"keywordScoreAvg,omitempty"`
	GraphTraversalDepth  *int            `json:"graphTraversalDepth,omitempty"`
}

// UserExpertiseProfile represents user expertise profile
type UserExpertiseProfile struct {
	UserID             string            `json:"userId"`
	ExpertiseDomains   map[string]float64 `json:"expertiseDomains"`
	TechnicalLevel     float64           `json:"technicalLevel"`
	VocabularyRichness float64           `json:"vocabularyRichness"`
	ConfidenceScore    float64           `json:"confidenceScore"`
	LastUpdated        time.Time         `json:"lastUpdated"`
}

// ============================================================================
// Knowledge Base Types
// ============================================================================

// KnowledgeEntry represents a knowledge base entry
type KnowledgeEntry struct {
	ID        int       `json:"id"`
	Title     string    `json:"title"`
	Content   string    `json:"content"`
	Category  *string   `json:"category,omitempty"`
	Tags      []string  `json:"tags,omitempty"`
	CreatedAt time.Time `json:"createdAt"`
	UpdatedAt time.Time `json:"updatedAt"`
}

// KnowledgeEntryCreate represents a knowledge entry creation request
type KnowledgeEntryCreate struct {
	Title    string   `json:"title"`
	Content  string   `json:"content"`
	Category *string  `json:"category,omitempty"`
	Tags     []string `json:"tags,omitempty"`
}

// KnowledgeEntryUpdate represents a knowledge entry update request
type KnowledgeEntryUpdate struct {
	Title    *string  `json:"title,omitempty"`
	Content  *string  `json:"content,omitempty"`
	Category *string  `json:"category,omitempty"`
	Tags     []string `json:"tags,omitempty"`
}

// Customer represents customer information
type Customer struct {
	ID        int           `json:"id"`
	Name      string        `json:"name"`
	Email     string        `json:"email"`
	Phone     *string       `json:"phone,omitempty"`
	Address   *string       `json:"address,omitempty"`
	Status    CustomerStatus `json:"status"`
	CreatedAt time.Time     `json:"createdAt"`
	UpdatedAt time.Time     `json:"updatedAt"`
}

// ============================================================================
// Response Types
// ============================================================================

// APIResponse represents a base API response
type APIResponse struct {
	Success   bool   `json:"success"`
	Message   *string `json:"message,omitempty"`
	Timestamp string `json:"timestamp"`
}

// SearchResponse represents a search response
type SearchResponse struct {
	SearchID         *string           `json:"searchId,omitempty"`
	Query            string            `json:"query"`
	Strategy         *string           `json:"strategy,omitempty"`
	Results          []SearchResult    `json:"results"`
	QueryExpansions  []QueryExpansion  `json:"queryExpansions,omitempty"`
	AggregatedResult *SearchResult     `json:"aggregatedResult,omitempty"`
	ExecutionTime    float64           `json:"executionTime"`
	Analytics        *SearchAnalytics  `json:"analytics,omitempty"`
	Timestamp        time.Time         `json:"timestamp"`
}

// PersonalizedSearchResponse represents a personalized search response
type PersonalizedSearchResponse struct {
	SessionID            string                 `json:"sessionId"`
	Query                string                 `json:"query"`
	Strategy             *string                `json:"strategy,omitempty"`
	PersonalizedResults  []SearchResult         `json:"personalizedResults"`
	PersonalizationApplied bool                  `json:"personalizationApplied"`
	PersonalizationLevel float64                `json:"personalizationLevel"`
	UserExpertiseDomains  map[string]float64     `json:"userExpertiseDomains"`
	PrivacyAnonymized    bool                   `json:"privacyAnonymized"`
	ExecutionTime        float64                `json:"executionTime"`
	Timestamp            time.Time              `json:"timestamp"`
}

// DocumentIndexResponse represents a document indexing response
type DocumentIndexResponse struct {
	Success   bool   `json:"success"`
	DocumentID string `json:"documentId"`
	Message   string `json:"message"`
	Timestamp string `json:"timestamp"`
}

// BatchIndexResponse represents a batch indexing response
type BatchIndexResponse struct {
	Results           []bool  `json:"results"`
	TotalDocuments    int     `json:"totalDocuments"`
	SuccessfulIndexed int     `json:"successfulIndexed"`
	Timestamp         string  `json:"timestamp"`
}

// AnalyticsResponse represents an analytics response
type AnalyticsResponse struct {
	EngineAnalytics   map[string]interface{} `json:"engineAnalytics"`
	RecentSessions    []map[string]interface{} `json:"recentSessions"`
	ActiveConnections int                     `json:"activeConnections"`
	TotalSessions     int                     `json:"totalSessions"`
	Timestamp         string                  `json:"timestamp"`
}

// HealthCheckResponse represents a health check response
type HealthCheckResponse struct {
	Status           string    `json:"status"`
	ResponseTime     *float64  `json:"responseTime,omitempty"`
	DocumentCount    *int      `json:"documentCount,omitempty"`
	ActiveConnections *int     `json:"activeConnections,omitempty"`
	Timestamp        time.Time `json:"timestamp"`
}

// ============================================================================
// WebSocket Message Types
// ============================================================================

// WebSocketMessage represents a base WebSocket message
type WebSocketMessage struct {
	Type      string    `json:"type"`
	Timestamp time.Time `json:"timestamp,omitempty"`
}

// WebSocketSearchMessage represents a WebSocket search request
type WebSocketSearchMessage struct {
	Type      string  `json:"type"`
	Query     string  `json:"query"`
	Strategy  string  `json:"strategy,omitempty"`
	TopK      int     `json:"topK,omitempty"`
	Threshold float64 `json:"threshold,omitempty"`
	Timestamp time.Time `json:"timestamp,omitempty"`
}

// WebSocketSearchResultsMessage represents WebSocket search results
type WebSocketSearchResultsMessage struct {
	Type         string          `json:"type"`
	SearchID     string          `json:"searchId"`
	Results      []SearchResult  `json:"results"`
	ExecutionTime float64        `json:"executionTime"`
	Analytics    *SearchAnalytics `json:"analytics,omitempty"`
	Timestamp    time.Time       `json:"timestamp,omitempty"`
}

// WebSocketProgressMessage represents WebSocket progress update
type WebSocketProgressMessage struct {
	Type      string    `json:"type"`
	SearchID  string    `json:"searchId"`
	Status    string    `json:"status"`
	Message   string    `json:"message"`
	Timestamp time.Time `json:"timestamp,omitempty"`
}

// WebSocketSuggestionsMessage represents WebSocket suggestions
type WebSocketSuggestionsMessage struct {
	Type         string    `json:"type"`
	Query        string    `json:"query"`
	Suggestions  []string  `json:"suggestions"`
	Personalized *bool     `json:"personalized,omitempty"`
	Timestamp    time.Time `json:"timestamp,omitempty"`
}

// WebSocketErrorMessage represents WebSocket error
type WebSocketErrorMessage struct {
	Type      string    `json:"type"`
	Message   string    `json:"message"`
	Code      *string   `json:"code,omitempty"`
	Timestamp time.Time `json:"timestamp,omitempty"`
}

// ============================================================================
// Configuration Types
// ============================================================================

// Config represents client configuration
type Config struct {
	BaseURL     string            `json:"baseURL,omitempty"`
	APIKey      string            `json:"apiKey,omitempty"`
	Timeout     time.Duration     `json:"timeout,omitempty"`
	MaxRetries  int               `json:"maxRetries,omitempty"`
	RetryDelay  time.Duration     `json:"retryDelay,omitempty"`
	Headers     map[string]string `json:"headers,omitempty"`
}

// WebSocketConfig represents WebSocket client configuration
type WebSocketConfig struct {
	BaseURL              string        `json:"baseURL,omitempty"`
	PersonalizedURL      string        `json:"personalizedURL,omitempty"`
	APIKey               string        `json:"apiKey,omitempty"`
	PingInterval         time.Duration `json:"pingInterval,omitempty"`
	PingTimeout          time.Duration `json:"pingTimeout,omitempty"`
	CloseTimeout         time.Duration `json:"closeTimeout,omitempty"`
	MaxReconnectAttempts int           `json:"maxReconnectAttempts,omitempty"`
	ReconnectDelay       time.Duration `json:"reconnectDelay,omitempty"`
}

// ServiceEndpoints represents API service endpoints
type ServiceEndpoints struct {
	Chat        string `json:"chat"`
	Knowledge   string `json:"knowledge"`
	Search      string `json:"search"`
	Personalized string `json:"personalized"`
}

// ============================================================================
// Pagination Types
// ============================================================================

// PaginationParams represents pagination parameters
type PaginationParams struct {
	Limit  int `json:"limit,omitempty"`
	Offset int `json:"offset,omitempty"`
}

// PaginatedResponse represents a paginated response
type PaginatedResponse struct {
	Success      bool        `json:"success"`
	Items        interface{} `json:"items"`
	Total        int         `json:"total"`
	Limit        int         `json:"limit"`
	Offset       int         `json:"offset"`
	HasNext      bool        `json:"hasNext"`
	HasPrevious  bool        `json:"hasPrevious"`
	Timestamp    string      `json:"timestamp"`
}