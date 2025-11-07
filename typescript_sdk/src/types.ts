/**
 * Type definitions for the Knowledge Base API TypeScript/JavaScript SDK.
 *
 * Provides comprehensive type definitions for all API requests, responses,
 * and data structures with full TypeScript support.
 */

// ============================================================================
// Enums
// ============================================================================

export enum SearchStrategy {
  UNIFIED = 'unified',
  SEMANTIC = 'semantic',
  KEYWORD = 'keyword',
  GRAPH = 'graph',
  AUTO = 'auto'
}

export enum FeedbackType {
  CLICK = 'click',
  SATISFACTION = 'satisfaction',
  RATING = 'rating'
}

export enum CustomerStatus {
  ACTIVE = 'active',
  INACTIVE = 'inactive',
  PROSPECT = 'prospect'
}

export enum SearchDirection {
  FORWARD = 'forward',
  BACKWARD = 'backward',
  BOTH = 'both'
}

export enum ExpansionType {
  SYNONYM = 'synonym',
  ABBREVIATION = 'abbreviation',
  RELATED = 'related',
  HYPERNYM = 'hypernym',
  HYPONYM = 'hyponym'
}

export enum SourceType {
  SEMANTIC = 'semantic',
  KEYWORD = 'keyword',
  KNOWLEDGE_GRAPH = 'knowledge_graph'
}

export enum QueryComplexity {
  SIMPLE = 'simple',
  MODERATE = 'moderate',
  COMPLEX = 'complex'
}

export enum WebSocketMessageType {
  SEARCH = 'search',
  RESULTS = 'results',
  PROGRESS = 'progress',
  SUGGESTIONS = 'suggestions',
  ERROR = 'error'
}

// ============================================================================
// Base Interfaces
// ============================================================================

export interface BaseAPIModel {
  [key: string]: unknown;
}

export interface APIResponse {
  success: boolean;
  message?: string;
  timestamp: string;
}

// ============================================================================
// Search Request Types
// ============================================================================

export interface SearchRequest extends BaseAPIModel {
  query: string;
  topK?: number;
  similarityThreshold?: number;
  filters?: Record<string, unknown>;
}

export interface UnifiedSearchRequest extends SearchRequest {
  searchStrategy?: SearchStrategy;
  rerank?: boolean;
  includeMetadata?: boolean;
}

export interface SemanticSearchRequest extends SearchRequest {
  // Inherits all properties from SearchRequest
}

export interface KeywordSearchRequest extends SearchRequest {
  // Inherits all properties from SearchRequest
}

export interface KnowledgeGraphSearchRequest extends BaseAPIModel {
  entityName: string;
  relationType?: string;
  direction?: SearchDirection;
  maxDepth?: number;
  topK?: number;
}

export interface PersonalizedSearchRequest extends UnifiedSearchRequest {
  userId: string;
  sessionId?: string;
  personalizationLevel?: number;
  boostExpertise?: boolean;
  boostHistory?: boolean;
  boostPreferences?: boolean;
}

export interface QuerySuggestionRequest extends BaseAPIModel {
  userId: string;
  partialQuery: string;
  maxSuggestions?: number;
}

// ============================================================================
// Document Types
// ============================================================================

export interface DocumentInfo extends BaseAPIModel {
  id: string;
  title: string;
  content: string;
  metadata?: Record<string, unknown>;
  timestamp?: string;
}

export interface IndexDocumentRequest extends BaseAPIModel {
  documentId: string;
  title: string;
  content: string;
  metadata?: Record<string, unknown>;
}

export interface BatchIndexDocumentsRequest extends BaseAPIModel {
  documents: IndexDocumentRequest[];
}

// ============================================================================
// User Management Types
// ============================================================================

export interface UserFeedback extends BaseAPIModel {
  userId: string;
  sessionId: string;
  resultId: string;
  feedbackType: FeedbackType;
  dwellTime?: number;
  satisfactionScore?: number;
}

export interface ConsentRequest extends BaseAPIModel {
  userId: string;
  consentGiven: boolean;
  consentText: string;
  dataPurposes?: string[];
}

export interface PrivacyConfig extends BaseAPIModel {
  userId: string;
  trackingEnabled?: boolean;
  queryHistoryRetentionDays?: number;
  clickTrackingEnabled?: boolean;
  expertiseLearningEnabled?: boolean;
  personalizationEnabled?: boolean;
  dataAnonymizationEnabled?: boolean;
  autoDeleteAfterDays?: number;
  gdprCompliant?: boolean;
  ccpaCompliant?: boolean;
}

export interface UserExpertiseProfile extends BaseAPIModel {
  userId: string;
  expertiseDomains: Record<string, number>;
  technicalLevel: number;
  vocabularyRichness: number;
  confidenceScore: number;
  lastUpdated: string;
}

// ============================================================================
// Search Result Types
// ============================================================================

export interface SearchResult extends BaseAPIModel {
  id: string;
  title: string;
  content: string;
  score: number;
  metadata?: Record<string, unknown>;
  sourceType: SourceType;
  explanation?: string;
}

export interface QueryExpansion extends BaseAPIModel {
  originalTerm: string;
  expandedTerms: string[];
  expansionType: ExpansionType;
  confidence: number;
}

export interface SearchAnalytics extends BaseAPIModel {
  queryComplexity: QueryComplexity;
  totalResults: number;
  searchTimeMs: number;
  cacheHit: boolean;
  semanticScoreAvg?: number;
  keywordScoreAvg?: number;
  graphTraversalDepth?: number;
}

// ============================================================================
// Knowledge Base Types
// ============================================================================

export interface KnowledgeEntry extends BaseAPIModel {
  id: number;
  title: string;
  content: string;
  category?: string;
  tags?: string[];
  createdAt: string;
  updatedAt: string;
}

export interface KnowledgeEntryCreate extends BaseAPIModel {
  title: string;
  content: string;
  category?: string;
  tags?: string[];
}

export interface KnowledgeEntryUpdate extends BaseAPIModel {
  title?: string;
  content?: string;
  category?: string;
  tags?: string[];
}

export interface Customer extends BaseAPIModel {
  id: number;
  name: string;
  email: string;
  phone?: string;
  address?: string;
  status: CustomerStatus;
  createdAt: string;
  updatedAt: string;
}

// ============================================================================
// Response Types
// ============================================================================

export interface SearchResponse extends APIResponse {
  searchId?: string;
  query: string;
  strategy?: string;
  results: SearchResult[];
  queryExpansions?: QueryExpansion[];
  aggregatedResult?: SearchResult;
  executionTime: number;
  analytics?: SearchAnalytics;
}

export interface PersonalizedSearchResponse extends SearchResponse {
  sessionId: string;
  personalizedResults: SearchResult[];
  personalizationApplied: boolean;
  personalizationLevel: number;
  userExpertiseDomains: Record<string, number>;
  privacyAnonymized: boolean;
}

export interface DocumentIndexResponse extends APIResponse {
  documentId: string;
}

export interface BatchIndexResponse extends APIResponse {
  results: boolean[];
  totalDocuments: number;
  successfulIndexed: number;
}

export interface AnalyticsResponse extends APIResponse {
  engineAnalytics: Record<string, unknown>;
  recentSessions: Record<string, unknown>[];
  activeConnections: number;
  totalSessions: number;
}

export interface HealthCheckResponse extends BaseAPIModel {
  status: 'healthy' | 'unhealthy';
  responseTime?: number;
  documentCount?: number;
  activeConnections?: number;
  timestamp: string;
}

// ============================================================================
// WebSocket Message Types
// ============================================================================

export interface BaseWebSocketMessage extends BaseAPIModel {
  type: WebSocketMessageType;
  timestamp?: string;
}

export interface WebSocketSearchMessage extends BaseWebSocketMessage {
  type: WebSocketMessageType.SEARCH;
  query: string;
  strategy?: string;
  topK?: number;
  threshold?: number;
}

export interface WebSocketSearchResultsMessage extends BaseWebSocketMessage {
  type: WebSocketMessageType.RESULTS;
  searchId: string;
  results: SearchResult[];
  executionTime: number;
  analytics?: SearchAnalytics;
}

export interface WebSocketProgressMessage extends BaseWebSocketMessage {
  type: WebSocketMessageType.PROGRESS;
  searchId: string;
  status: string;
  message: string;
}

export interface WebSocketSuggestionsMessage extends BaseWebSocketMessage {
  type: WebSocketMessageType.SUGGESTIONS;
  query: string;
  suggestions: string[];
  personalized?: boolean;
}

export interface WebSocketErrorMessage extends BaseWebSocketMessage {
  type: WebSocketMessageType.ERROR;
  message: string;
  code?: string;
}

export type WebSocketMessage =
  | WebSocketSearchMessage
  | WebSocketSearchResultsMessage
  | WebSocketProgressMessage
  | WebSocketSuggestionsMessage
  | WebSocketErrorMessage;

// ============================================================================
// Client Configuration Types
// ============================================================================

export interface KnowledgeBaseClientConfig {
  baseURL?: string;
  apiKey?: string;
  timeout?: number;
  maxRetries?: number;
  retryDelay?: number;
  headers?: Record<string, string>;
}

export interface WebSocketClientConfig {
  baseURL?: string;
  personalizedURL?: string;
  apiKey?: string;
  pingInterval?: number;
  pingTimeout?: number;
  closeTimeout?: number;
  maxReconnectAttempts?: number;
  reconnectDelay?: number;
}

// ============================================================================
// Service Endpoints
// ============================================================================

export interface ServiceEndpoints {
  chat: string;
  knowledge: string;
  search: string;
  personalized: string;
}

// ============================================================================
// Event Handler Types
// ============================================================================

export type EventHandler<T = unknown> = (data: T) => void | Promise<void>;

export interface WebSocketEventHandlers {
  open?: (connectionType: 'search' | 'personalized') => void;
  close?: (connectionType: 'search' | 'personalized') => void;
  error?: (error: Error) => void;
  message?: (data: WebSocketMessage, connectionType: string) => void;
  results?: (data: {
    searchId: string;
    results: SearchResult[];
    executionTime: number;
    analytics?: SearchAnalytics;
    timestamp: Date;
  }) => void;
  progress?: (data: {
    searchId: string;
    status: string;
    message: string;
    timestamp: Date;
  }) => void;
  suggestions?: (data: {
    query: string;
    suggestions: string[];
    personalized?: boolean;
    timestamp: Date;
  }) => void;
  searchError?: (error: Error) => void;
}

// ============================================================================
// Error Types
// ============================================================================

export interface APIError extends Error {
  status?: number;
  code?: string;
  response?: unknown;
  requestId?: string;
}

export interface WebSocketError extends Error {
  code?: number;
  reason?: string;
}

export interface ValidationError extends APIError {
  validationErrors?: Record<string, unknown>;
}

export interface AuthenticationError extends APIError {
  userId?: string;
}

export interface RateLimitError extends APIError {
  retryAfter?: number;
}

export interface NotFoundError extends APIError {
  resourceType?: string;
  resourceId?: string;
}

// ============================================================================
// Utility Types
// ============================================================================

export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>;

export type OptionalFields<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

export type SearchRequestType =
  | UnifiedSearchRequest
  | SemanticSearchRequest
  | KeywordSearchRequest
  | KnowledgeGraphSearchRequest
  | PersonalizedSearchRequest;

export type SearchResponseType = SearchResponse | PersonalizedSearchResponse;

// ============================================================================
// HTTP Response Types
// ============================================================================

export interface HTTPResponse<T = unknown> {
  data: T;
  status: number;
  statusText: string;
  headers: Record<string, string>;
}

export interface HTTPRequestConfig {
  method: string;
  url: string;
  data?: unknown;
  params?: Record<string, unknown>;
  headers?: Record<string, string>;
  timeout?: number;
}

// ============================================================================
// Pagination Types
// ============================================================================

export interface PaginationParams {
  limit?: number;
  offset?: number;
}

export interface PaginatedResponse<T> extends APIResponse {
  items: T[];
  total: number;
  limit: number;
  offset: number;
  hasNext: boolean;
  hasPrevious: boolean;
}