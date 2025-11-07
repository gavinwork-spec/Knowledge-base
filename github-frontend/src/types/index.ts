// Manufacturing system types

export interface ManufacturingSystem {
  id: string;
  name: string;
  status: SystemStatus;
  productionRate: number;
  efficiency: number;
  lastMaintenance: string;
  nextMaintenance: string;
  location: string;
  operator: string;
  temperature: number;
  pressure: number;
  vibration: number;
  powerConsumption: number;
}

export enum SystemStatus {
  ONLINE = 'online',
  OFFLINE = 'offline',
  MAINTENANCE = 'maintenance',
  WARNING = 'warning',
  ERROR = 'error'
}

export interface ProductionMetrics {
  totalUnits: number;
  goodUnits: number;
  defectiveUnits: number;
  efficiency: number;
  uptime: number;
  downtime: number;
  cycleTime: number;
  oee: number; // Overall Equipment Effectiveness
}

export interface KnowledgeEntry {
  id: string;
  title: string;
  content: string;
  category: string;
  tags: string[];
  author: string;
  createdAt: string;
  updatedAt: string;
  attachments?: Attachment[];
  metadata?: Record<string, any>;
  viewCount: number;
  rating?: number;
}

export interface Attachment {
  id: string;
  name: string;
  type: string;
  size: number;
  url: string;
  thumbnailUrl?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: {
    model?: string;
    tokens?: number;
    responseTime?: number;
    relatedEntries?: string[];
    sources?: Source[];
  };
}

export interface Source {
  id: string;
  title: string;
  url: string;
  relevanceScore?: number;
  snippet?: string;
}

export interface SearchFilters {
  category?: string;
  tags?: string[];
  dateRange?: {
    start: string;
    end: string;
  };
  author?: string;
  rating?: number;
}

export interface ComparisonItem {
  id: string;
  name: string;
  category: string;
  specifications: Record<string, any>;
  performance: PerformanceMetrics;
  cost: CostMetrics;
  maintenance: MaintenanceMetrics;
}

export interface PerformanceMetrics {
  efficiency: number;
  throughput: number;
  quality: number;
  reliability: number;
  uptime: number;
}

export interface CostMetrics {
  purchasePrice: number;
  operatingCost: number;
  maintenanceCost: number;
  energyConsumption: number;
  totalCostOfOwnership: number;
}

export interface MaintenanceMetrics {
  mtbf: number; // Mean Time Between Failures
  mttr: number; // Mean Time To Repair
  maintenanceInterval: number;
  lastMaintenanceDate: string;
  nextMaintenanceDate: string;
  scheduledMaintenanceCount: number;
  unscheduledMaintenanceCount: number;
}

export interface User {
  id: string;
  name: string;
  email: string;
  role: UserRole;
  department: string;
  permissions: Permission[];
  preferences: UserPreferences;
}

export enum UserRole {
  ADMIN = 'admin',
  MANAGER = 'manager',
  OPERATOR = 'operator',
  VIEWER = 'viewer'
}

export enum Permission {
  VIEW_DASHBOARD = 'view_dashboard',
  MANAGE_SYSTEMS = 'manage_systems',
  VIEW_KNOWLEDGE_BASE = 'view_knowledge_base',
  EDIT_KNOWLEDGE_BASE = 'edit_knowledge_base',
  CHAT_WITH_AI = 'chat_with_ai',
  VIEW_ANALYTICS = 'view_analytics',
  MANAGE_USERS = 'manage_users'
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'system';
  language: string;
  notifications: NotificationSettings;
  dashboard: DashboardSettings;
}

export interface NotificationSettings {
  systemAlerts: boolean;
  maintenanceReminders: boolean;
  performanceReports: boolean;
  chatNotifications: boolean;
}

export interface DashboardSettings {
  layout: string;
  widgets: string[];
  refreshInterval: number;
  defaultView: string;
}

export interface Alert {
  id: string;
  type: AlertType;
  severity: AlertSeverity;
  title: string;
  message: string;
  systemId?: string;
  timestamp: string;
  acknowledged: boolean;
  acknowledgedBy?: string;
  acknowledgedAt?: string;
  resolved: boolean;
  resolvedBy?: string;
  resolvedAt?: string;
}

export enum AlertType {
  SYSTEM_DOWN = 'system_down',
  PERFORMANCE_DEGRADATION = 'performance_degradation',
  MAINTENANCE_REQUIRED = 'maintenance_required',
  QUALITY_ISSUE = 'quality_issue',
  SAFETY_CONCERN = 'safety_concern',
  COST_ANOMALY = 'cost_anomaly'
}

export enum AlertSeverity {
  INFO = 'info',
  WARNING = 'warning',
  ERROR = 'error',
  CRITICAL = 'critical'
}

export interface AnalyticsData {
  production: ProductionAnalytics;
  efficiency: EfficiencyAnalytics;
  cost: CostAnalytics;
  quality: QualityAnalytics;
  maintenance: MaintenanceAnalytics;
}

export interface ProductionAnalytics {
  daily: Array<{
    date: string;
    units: number;
    target: number;
    variance: number;
  }>;
  weekly: Array<{
    week: string;
    units: number;
    target: number;
    variance: number;
  }>;
  monthly: Array<{
    month: string;
    units: number;
    target: number;
    variance: number;
  }>;
}

export interface EfficiencyAnalytics {
  oee: Array<{
    timestamp: string;
    value: number;
    target: number;
  }>;
  availability: Array<{
    timestamp: string;
    value: number;
  }>;
  performance: Array<{
    timestamp: string;
    value: number;
  }>;
  quality: Array<{
    timestamp: string;
    value: number;
  }>;
}

export interface CostAnalytics {
  operatingCosts: Array<{
    date: string;
    labor: number;
    materials: number;
    energy: number;
    maintenance: number;
    total: number;
  }>;
  costPerUnit: Array<{
    date: string;
    value: number;
    target: number;
  }>;
}

export interface QualityAnalytics {
  defectRates: Array<{
    date: string;
    rate: number;
    target: number;
  }>;
  qualityScores: Array<{
    date: string;
    score: number;
  }>;
  defectTypes: Array<{
    type: string;
    count: number;
    percentage: number;
  }>;
}

export interface MaintenanceAnalytics {
  scheduledVsUnscheduled: Array<{
    date: string;
    scheduled: number;
    unscheduled: number;
  }>;
  mtbfTrends: Array<{
    date: string;
    value: number;
  }>;
  maintenanceCosts: Array<{
    date: string;
    cost: number;
  }>;
}

// API Response types
export interface ApiResponse<T> {
  data: T;
  success: boolean;
  message?: string;
  error?: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
  success: boolean;
}

// Search and Chat types
export interface SearchQuery {
  query: string;
  filters?: SearchFilters;
  limit?: number;
  offset?: number;
}

export interface SearchResult {
  entries: KnowledgeEntry[];
  total: number;
  suggestions?: string[];
  facets?: Record<string, Array<{ value: string; count: number }>>;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: string;
  updatedAt: string;
  systemId?: string;
}

export interface ChatRequest {
  message: string;
  sessionId?: string;
  systemId?: string;
  context?: string[];
  temperature?: number;
  maxTokens?: number;
}

export interface ChatResponse {
  message: ChatMessage;
  sessionId: string;
  suggestions?: string[];
  relatedEntries?: KnowledgeEntry[];
}