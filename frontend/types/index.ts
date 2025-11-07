/**
 * Type definitions for the Knowledge Hub frontend
 */

// Theme and UI Types
export type Theme = 'light' | 'dark' | 'system'

export interface ThemeConfig {
  theme: Theme
  reducedMotion: boolean
  highContrast: boolean
  fontSize: 'sm' | 'md' | 'lg' | 'xl'
}

// Search Types
export interface SearchQuery {
  id: string
  query: string
  timestamp: Date
  userId?: string
  sessionId: string
  searchStrategy: SearchStrategy
  results: SearchResult[]
  filters?: SearchFilters
  personalization?: PersonalizationConfig
}

export type SearchStrategy = 'unified' | 'semantic' | 'keyword' | 'graph' | 'auto'

export interface SearchResult {
  documentId: string
  title: string
  content: string
  score: number
  searchType: string
  metadata: Record<string, any>
  explanation?: string
  highlightedContent?: string
  expertiseLevel?: ExpertiseLevel
  relevanceScore?: number
  timestamp?: Date
}

export interface SearchFilters {
  category?: string
  dateRange?: {
    start: Date
    end: Date
  }
  expertise?: ExpertiseLevel[]
  contentType?: string[]
  tags?: string[]
  author?: string
}

export interface PersonalizationConfig {
  enabled: boolean
  level: number // 0-1
  boostExpertise: boolean
  boostHistory: boolean
  boostPreferences: boolean
}

export interface SearchSuggestion {
  id: string
  text: string
  type: 'history' | 'popular' | 'semantic' | 'auto'
  score?: number
  category?: string
}

export interface SearchAnalytics {
  totalQueries: number
  averageResponseTime: number
  topQueries: Array<{ query: string; count: number }>
  searchStrategies: Record<SearchStrategy, number>
  userSatisfaction: number
  errorRate: number
}

// Expertise Types
export type ExpertiseLevel = 'beginner' | 'intermediate' | 'advanced' | 'expert'

export interface ExpertiseDomain {
  name: string
  score: number // 0-1
  keywords: string[]
  confidence: number // 0-1
  lastUpdated: Date
}

export interface UserExpertiseProfile {
  userId: string
  domains: Record<string, ExpertiseDomain>
  technicalLevel: number // 0-1
  vocabularyRichness: number // 0-1
  preferredLanguages: string[]
  contentPreferences: Record<string, number>
  confidenceScore: number // 0-1
  lastUpdated: Date
}

// User and Session Types
export interface User {
  id: string
  email?: string
  name?: string
  avatar?: string
  role: UserRole
  expertiseProfile?: UserExpertiseProfile
  preferences: UserPreferences
  privacySettings: PrivacySettings
  createdAt: Date
  lastActiveAt: Date
}

export type UserRole = 'guest' | 'user' | 'admin' | 'moderator'

export interface UserPreferences {
  theme: Theme
  language: string
  timezone: string
  notifications: NotificationSettings
  searchSettings: SearchUserSettings
  ui: UISettings
}

export interface NotificationSettings {
  email: boolean
  push: boolean
  search: boolean
  system: boolean
  frequency: 'immediate' | 'daily' | 'weekly' | 'never'
}

export interface SearchUserSettings {
  defaultStrategy: SearchStrategy
  resultsPerPage: number
  autoSuggest: boolean
  saveHistory: boolean
  personalizationEnabled: boolean
  showExplanations: boolean
  highlightResults: boolean
}

export interface UISettings {
  sidebarCollapsed: boolean
  compactMode: boolean
  showThumbnails: boolean
  animationsEnabled: boolean
  reducedMotion: boolean
  fontSize: 'sm' | 'md' | 'lg' | 'xl'
  density: 'comfortable' | 'normal' | 'compact'
}

export interface PrivacySettings {
  trackingEnabled: boolean
  personalizationEnabled: boolean
  dataRetentionDays: number
  analyticsEnabled: boolean
  cookiesEnabled: boolean
  gdprConsent: boolean
  ccpaOptOut: boolean
  consentVersion: string
  consentDate?: Date
}

export interface UserSession {
  id: string
  userId?: string
  createdAt: Date
  lastActivity: Date
  searchCount: number
  queries: string[]
  userAgent: string
  ipAddress: string
  deviceType: DeviceType
}

export type DeviceType = 'desktop' | 'tablet' | 'mobile' | 'unknown'

// Knowledge Base Types
export interface Document {
  id: string
  title: string
  content: string
  excerpt: string
  author?: string
  category: string
  tags: string[]
  metadata: DocumentMetadata
  createdAt: Date
  updatedAt: Date
  version: number
  status: DocumentStatus
  accessLevel: AccessLevel
}

export interface DocumentMetadata {
  wordCount: number
  readingTime: number
  difficulty: ExpertiseLevel
  language: string
  format: string
  size: number
  checksum: string
  lastIndexed: Date
  searchScore?: number
  clickCount: number
}

export type DocumentStatus = 'draft' | 'published' | 'archived' | 'deleted'

export type AccessLevel = 'public' | 'internal' | 'restricted' | 'private'

export interface KnowledgeGraph {
  nodes: GraphNode[]
  edges: GraphEdge[]
  metadata: GraphMetadata
}

export interface GraphNode {
  id: string
  label: string
  type: string
  properties: Record<string, any>
  importance: number
  createdAt: Date
}

export interface GraphEdge {
  id: string
  source: string
  target: string
  type: string
  weight: number
  properties: Record<string, any>
  createdAt: Date
}

export interface GraphMetadata {
  nodeCount: number
  edgeCount: number
  lastUpdated: Date
  version: string
}

// API Response Types
export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  error?: ApiError
  message?: string
  timestamp: Date
  requestId: string
}

export interface ApiError {
  code: string
  message: string
  details?: Record<string, any>
  stack?: string
}

export interface PaginatedResponse<T> {
  items: T[]
  pagination: PaginationInfo
  filters?: SearchFilters
  sorting?: SortingInfo
}

export interface PaginationInfo {
  page: number
  pageSize: number
  totalItems: number
  totalPages: number
  hasNext: boolean
  hasPrev: boolean
}

export interface SortingInfo {
  field: string
  direction: 'asc' | 'desc'
}

// UI Component Types
export interface ComponentSize {
  width?: number | string
  height?: number | string
  maxWidth?: number | string
  maxHeight?: number | string
}

export interface ComponentPosition {
  x: number
  y: number
  zIndex?: number
}

export interface AnimationConfig {
  duration: number
  easing: string
  delay?: number
}

export interface KeyboardShortcut {
  key: string
  ctrlKey?: boolean
  shiftKey?: boolean
  altKey?: boolean
  metaKey?: boolean
  action: () => void
  description: string
}

// Analytics and Monitoring Types
export interface UserAnalytics {
  userId: string
  sessionId: string
  events: AnalyticsEvent[]
  metrics: UserMetrics
  timestamp: Date
}

export interface AnalyticsEvent {
  type: string
  name: string
  properties: Record<string, any>
  timestamp: Date
}

export interface UserMetrics {
  sessionDuration: number
  queriesCount: number
  clicksCount: number
  satisfactionScore: number
  bounceRate: number
  conversionRate: number
}

export interface SystemMetrics {
  uptime: number
  responseTime: number
  errorRate: number
  throughput: number
  memoryUsage: number
  cpuUsage: number
  activeUsers: number
  totalQueries: number
}

// Form Types
export interface FormField {
  name: string
  type: string
  label: string
  placeholder?: string
  required?: boolean
  disabled?: boolean
  validation?: ValidationRule[]
  defaultValue?: any
  options?: FormOption[]
}

export interface FormOption {
  label: string
  value: any
  disabled?: boolean
}

export interface ValidationRule {
  type: string
  value?: any
  message: string
}

export interface FormState {
  values: Record<string, any>
  errors: Record<string, string>
  touched: Record<string, boolean>
  isValid: boolean
  isSubmitting: boolean
}

// Toast/Notification Types
export interface Toast {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message?: string
  duration?: number
  action?: ToastAction
  persistent?: boolean
}

export interface ToastAction {
  label: string
  action: () => void
}

// Modal Types
export interface Modal {
  id: string
  title: string
  content: React.ReactNode
  size: 'sm' | 'md' | 'lg' | 'xl' | 'full'
  closable?: boolean
  persistent?: boolean
}

// Layout Types
export interface LayoutConfig {
  sidebar: {
    collapsed: boolean
    width: number
    resizable: boolean
  }
  header: {
    height: number
    fixed: boolean
  }
  content: {
    padding: number
    maxWidth: number
  }
  footer: {
    height: number
    fixed: boolean
  }
}

// Export all types for easy importing
export type {
  // Search
  SearchQuery,
  SearchResult,
  SearchFilters,
  SearchSuggestion,
  SearchAnalytics,
  SearchUserSettings,

  // Expertise
  ExpertiseDomain,
  UserExpertiseProfile,

  // User
  User,
  UserPreferences,
  UserSession,
  PrivacySettings,

  // Knowledge
  Document,
  DocumentMetadata,
  KnowledgeGraph,
  GraphNode,
  GraphEdge,

  // API
  ApiResponse,
  ApiError,
  PaginatedResponse,

  // UI
  ComponentSize,
  ComponentPosition,
  AnimationConfig,
  KeyboardShortcut,

  // Analytics
  UserAnalytics,
  AnalyticsEvent,
  UserMetrics,
  SystemMetrics,

  // Forms
  FormField,
  FormOption,
  ValidationRule,
  FormState,

  // Components
  Toast,
  ToastAction,
  Modal,

  // Layout
  LayoutConfig,
}