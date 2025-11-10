// Manufacturing Knowledge Base - Type Definitions

export interface ManufacturingContext {
  equipment_type?: string;
  user_role?: 'operator' | 'engineer' | 'quality_inspector' | 'safety_officer' | 'maintenance_tech';
  facility_id?: string;
  process_type?: 'machining' | 'inspection' | 'assembly' | 'maintenance' | 'quality_control';
  compliance_standards?: string[];
  is_safety_critical?: boolean;
  is_quality_critical?: boolean;
}

export interface SearchResult {
  id: string;
  title: string;
  content: string;
  excerpt: string;
  url?: string;
  score: number;
  relevance_score: number;
  expertise_level: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  document_type: 'technical_manual' | 'safety_procedure' | 'quality_specification' | 'equipment_manual' | 'general';
  metadata: {
    author?: string;
    created_at: string;
    updated_at?: string;
    equipment_type?: string;
    process_type?: string;
    compliance_standards?: string[];
    tags?: string[];
    file_size?: number;
    file_type?: string;
  };
  explanation?: string;
  manufacturing_context?: ManufacturingContext;
}

export interface SearchQuery {
  query: string;
  strategy: 'unified' | 'semantic' | 'keyword' | 'graph' | 'ai_enhanced';
  filters?: SearchFilters;
  limit?: number;
  offset?: number;
  manufacturing_context?: ManufacturingContext;
}

export interface SearchFilters {
  document_type?: string[];
  expertise_level?: string[];
  equipment_type?: string[];
  process_type?: string[];
  date_range?: {
    start: string;
    end: string;
  };
  compliance_standards?: string[];
}

export interface SearchResponse {
  results: SearchResult[];
  total_results: number;
  query: SearchQuery;
  search_time: number;
  suggestions?: string[];
  facets?: {
    document_types: { [key: string]: number };
    expertise_levels: { [key: string]: number };
    equipment_types: { [key: string]: number };
  };
  manufacturing_insights?: {
    safety_alerts?: string[];
    quality_notes?: string[];
    equipment_warnings?: string[];
  };
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  manufacturing_context?: ManufacturingContext;
  metadata?: {
    model?: string;
    tokens_used?: number;
    response_time?: number;
    sources?: SearchResult[];
    quick_actions?: string[];
  };
}

export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: ChatMessage[];
  manufacturing_context?: ManufacturingContext;
  settings: {
    model: string;
    temperature: number;
    max_tokens: number;
  };
}

export interface EquipmentStatus {
  id: string;
  name: string;
  type: string;
  status: 'operational' | 'maintenance' | 'error' | 'offline';
  location?: string;
  last_maintenance?: string;
  next_maintenance?: string;
  operating_hours?: number;
  performance_metrics?: {
    efficiency: number;
    uptime: number;
    error_rate: number;
  };
  safety_notes?: string[];
  quality_impact?: string;
}

export interface QualityMetric {
  id: string;
  measurement: string;
  value: number;
  unit: string;
  tolerance: {
    min: number;
    max: number;
    target?: number;
  };
  status: 'pass' | 'fail' | 'marginal';
  timestamp: string;
  equipment_id?: string;
  operator_id?: string;
  inspector_id?: string;
  compliance_standard?: string;
}

export interface SafetyEvent {
  id: string;
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  equipment_type?: string;
  location?: string;
  timestamp: string;
  reported_by: string;
  action_taken?: string;
  resolved_at?: string;
  compliance_impact?: string;
  preventive_measures?: string[];
}

export interface ComplianceReport {
  id: string;
  standard: string;
  status: 'compliant' | 'non_compliant' | 'pending';
  score: number;
  period: {
    start: string;
    end: string;
  };
  requirements: {
    id: string;
    name: string;
    status: 'compliant' | 'non_compliant' | 'pending';
    score: number;
    last_check: string;
    evidence?: string[];
    corrective_actions?: string[];
  }[];
  generated_at: string;
  next_review: string;
}

export interface APIError {
  code: string;
  message: string;
  details?: any;
  timestamp: string;
}

export interface APIResponse<T = any> {
  success: boolean;
  data?: T;
  error?: APIError;
  meta?: {
    timestamp: string;
    request_id: string;
    processing_time: number;
  };
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'system';
  language: string;
  default_search_strategy: string;
  expertise_level: string;
  equipment_types: string[];
  compliance_standards: string[];
  notifications: {
    safety_alerts: boolean;
    quality_issues: boolean;
    maintenance_reminders: boolean;
    compliance_updates: boolean;
  };
  ui: {
    sidebar_collapsed: boolean;
    compact_mode: boolean;
    show_explanations: boolean;
    auto_save_searches: boolean;
  };
}

export interface DashboardMetrics {
  search_performance: {
    total_searches: number;
    avg_response_time: number;
    success_rate: number;
    popular_queries: string[];
  };
  quality_metrics: {
    total_inspections: number;
    pass_rate: number;
    critical_issues: number;
    trend: 'improving' | 'stable' | 'declining';
  };
  safety_metrics: {
    total_events: number;
    critical_events: number;
    resolved_events: number;
    compliance_rate: number;
  };
  equipment_status: {
    operational: number;
    maintenance: number;
    error: number;
    offline: number;
  };
  ai_performance: {
    total_queries: number;
    avg_response_time: number;
    user_satisfaction: number;
    cost_usage: number;
  };
}

// LobeChat Integration Types
export interface LobeChatConfig {
  enabled: boolean;
  websocket_url: string;
  theme: 'default' | 'manufacturing' | 'safety' | 'quality';
  features: {
    file_upload: boolean;
    image_recognition: boolean;
    voice_input: boolean;
    suggested_responses: boolean;
    manufacturing_templates: boolean;
  };
  quick_actions: {
    id: string;
    title: string;
    description: string;
    template: string;
    equipment_type?: string;
  }[];
}

export interface QuickAction {
  id: string;
  title: string;
  description: string;
  icon: string;
  template: string;
  category: 'safety' | 'quality' | 'maintenance' | 'technical' | 'general';
  equipment_types?: string[];
  user_roles?: string[];
  process_types?: string[];
}

export interface KnowledgeGraphNode {
  id: string;
  label: string;
  type: 'equipment' | 'process' | 'material' | 'standard' | 'document' | 'person';
  properties: {
    description?: string;
    specification?: string;
    compliance_standards?: string[];
    safety_notes?: string[];
    quality_requirements?: string[];
  };
  relationships: {
    target_id: string;
    type: string;
    weight: number;
  }[];
}

export interface KnowledgeGraphPath {
  nodes: KnowledgeGraphNode[];
  edges: Array<{
    source: string;
    target: string;
    type: string;
    weight: number;
  }>;
  total_weight: number;
  explanation: string;
}

export interface ManufacturingWorkflow {
  id: string;
  name: string;
  description: string;
  steps: {
    id: string;
    name: string;
    type: 'inspection' | 'machining' | 'assembly' | 'quality_check' | 'safety_check';
    equipment_type?: string;
    estimated_duration: number;
    dependencies?: string[];
    safety_requirements?: string[];
    quality_standards?: string[];
  }[];
  current_step?: string;
  status: 'pending' | 'in_progress' | 'completed' | 'error' | 'paused';
  progress: number;
  started_at?: string;
  completed_at?: string;
}

// Integration Types
export interface LangChainConfig {
  enabled: boolean;
  llm: {
    provider: string;
    model: string;
    temperature: number;
    max_tokens: number;
  };
  manufacturing_prompts: {
    system_prompt: string;
    safety_procedure: string;
    quality_control: string;
    technical_specification: string;
  };
}

export interface XAgentConfig {
  enabled: boolean;
  agents: {
    [key: string]: {
      enabled: boolean;
      capabilities: string[];
      max_concurrent_tasks: number;
    };
  };
  task_decomposition: {
    max_subtasks: number;
    complexity_threshold: number;
  };
}

export interface LangFuseConfig {
  enabled: boolean;
  connection: {
    public_key?: string;
    secret_key?: string;
    host: string;
    environment: string;
  };
  tracing: {
    sample_rate: number;
    batch_size: number;
  };
  manufacturing_metrics: boolean;
}

export interface IntegrationConfig {
  langchain: LangChainConfig;
  lobechat: LobeChatConfig;
  xagent: XAgentConfig;
  langfuse: LangFuseConfig;
}

// React Query Types
export interface QueryOptions {
  staleTime?: number;
  cacheTime?: number;
  refetchOnWindowFocus?: boolean;
  refetchOnReconnect?: boolean;
  retry?: number;
  retryDelay?: number;
}

// Form Types
export interface SearchFormData {
  query: string;
  strategy: string;
  document_type: string[];
  expertise_level: string;
  equipment_type: string;
}

export interface ChatFormData {
  message: string;
  quick_action?: string;
  context?: ManufacturingContext;
}

export interface QualityFormData {
  measurement: string;
  value: number;
  unit: string;
  tolerance_min: number;
  tolerance_max: number;
  target?: number;
  equipment_id?: string;
  compliance_standard?: string;
}