/**
 * Knowledge Base API TypeScript/JavaScript SDK
 *
 * A comprehensive TypeScript/JavaScript SDK for interacting with the Knowledge Base API Suite.
 * Provides type-safe interfaces for search, document management, user personalization,
 * and real-time WebSocket interactions.
 *
 * Features:
 * - 🔍 Hybrid search (semantic + keyword + knowledge graph)
 * - 📚 Document indexing and management
 * - 👤 Personalized search with privacy controls
 * - 🔄 Real-time WebSocket support
 * - 🎯 Full TypeScript support with type definitions
 * - 🛡️ Built-in error handling and retries
 * - 📊 Analytics and monitoring
 */

// Main client
export { KnowledgeBaseClient, createClient, createDefaultClient } from './client';

// Service managers
export {
  SearchManager,
  DocumentManager,
  UserManager,
  KnowledgeManager,
  AnalyticsManager
} from './client';

// WebSocket client
export { WebSocketClient, createSearchWebSocket, createPersonalizedWebSocket } from './websocket';

// HTTP client
export { HTTPClient, getHTTPClient, resetHTTPClient } from './http';

// Error classes
export {
  KnowledgeBaseError,
  AuthenticationError,
  ConsentRequiredError,
  PersonalizationDisabledError,
  APIError,
  ValidationError,
  NotFoundError,
  RateLimitError,
  ServiceUnavailableError,
  ConnectionError,
  TimeoutError,
  WebSocketError,
  DocumentIndexError,
  SearchError,
  createExceptionFromResponse,
  isInstanceOf,
  isAPIError,
  isValidationError,
  isAuthenticationError,
  isRateLimitError,
  isWebSocketError
} from './errors';

// Type definitions
export type {
  // Enums
  SearchStrategy,
  FeedbackType,
  CustomerStatus,
  SearchDirection,
  ExpansionType,
  SourceType,
  QueryComplexity,
  WebSocketMessageType,

  // Base interfaces
  BaseAPIModel,
  APIResponse,

  // Search request types
  SearchRequest,
  UnifiedSearchRequest,
  SemanticSearchRequest,
  KeywordSearchRequest,
  KnowledgeGraphSearchRequest,
  PersonalizedSearchRequest,
  QuerySuggestionRequest,

  // Document types
  DocumentInfo,
  IndexDocumentRequest,
  BatchIndexDocumentsRequest,

  // User management types
  UserFeedback,
  ConsentRequest,
  PrivacyConfig,
  UserExpertiseProfile,

  // Search result types
  SearchResult,
  QueryExpansion,
  SearchAnalytics,

  // Knowledge base types
  KnowledgeEntry,
  KnowledgeEntryCreate,
  KnowledgeEntryUpdate,
  Customer,

  // Response types
  SearchResponse,
  PersonalizedSearchResponse,
  DocumentIndexResponse,
  BatchIndexResponse,
  AnalyticsResponse,
  HealthCheckResponse,

  // WebSocket message types
  BaseWebSocketMessage,
  WebSocketSearchMessage,
  WebSocketSearchResultsMessage,
  WebSocketProgressMessage,
  WebSocketSuggestionsMessage,
  WebSocketErrorMessage,
  WebSocketMessage,

  // Client configuration
  KnowledgeBaseClientConfig,
  WebSocketClientConfig,
  ServiceEndpoints,

  // Event handlers
  EventHandler,
  WebSocketEventHandlers,

  // Error types
  APIError as APIErrorType,
  WebSocketError as WebSocketErrorType,
  ValidationError as ValidationErrorType,
  AuthenticationError as AuthenticationErrorType,
  RateLimitError as RateLimitErrorType,
  NotFoundError as NotFoundErrorType,

  // Utility types
  DeepPartial,
  RequiredFields,
  OptionalFields,
  SearchRequestType,
  SearchResponseType,

  // HTTP types
  HTTPResponse,
  HTTPRequestConfig,

  // Pagination types
  PaginationParams,
  PaginatedResponse
} from './types';

// Version
export const VERSION = '1.0.0';

// Default configuration
export const DEFAULT_CONFIG = {
  baseURL: 'http://localhost:8000',
  timeout: 30000,
  maxRetries: 3,
  retryDelay: 1000,
} as const;

// Re-export main client as default
export { KnowledgeBaseClient as default } from './client';