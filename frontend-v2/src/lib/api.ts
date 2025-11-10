// API Service Layer - Manufacturing Knowledge Base
// Preserves compatibility with existing backend endpoints

import {
  SearchQuery,
  SearchResponse,
  ChatMessage,
  ChatSession,
  ManufacturingContext,
  APIResponse,
  EquipmentStatus,
  QualityMetric,
  SafetyEvent,
  ComplianceReport,
  DashboardMetrics,
  UserPreferences
} from '../types';

// API Base URLs - Preserve existing endpoint structure
const API_BASE_URLS = {
  knowledge: 'http://localhost:8001',
  chat: 'http://localhost:8002',
  advanced_rag: 'http://localhost:8003',
  multi_agent: 'http://localhost:8004',
  unified_search: 'http://localhost:8006',
  personalized_search: 'http://localhost:8007',
  reminders: 'http://localhost:8005',
};

class APIService {
  private baseURLs = API_BASE_URLS;

  // Generic request method with error handling
  private async request<T>(
    endpoint: string,
    options: RequestInit = {},
    baseURL: keyof typeof API_BASE_URLS = 'knowledge'
  ): Promise<APIResponse<T>> {
    const url = `${this.baseURLs[baseURL]}${endpoint}`;

    const defaultHeaders = {
      'Content-Type': 'application/json',
      'X-Manufacturing-Context': JSON.stringify(this.getCurrentManufacturingContext()),
    };

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          ...defaultHeaders,
          ...options.headers,
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      return {
        success: true,
        data,
        meta: {
          timestamp: new Date().toISOString(),
          request_id: this.generateRequestId(),
          processing_time: 0,
        },
      };
    } catch (error) {
      return {
        success: false,
        error: {
          code: 'REQUEST_FAILED',
          message: error instanceof Error ? error.message : 'Unknown error',
          timestamp: new Date().toISOString(),
        },
        meta: {
          timestamp: new Date().toISOString(),
          request_id: this.generateRequestId(),
          processing_time: 0,
        },
      };
    }
  }

  private generateRequestId(): string {
    return Math.random().toString(36).substring(2, 15);
  }

  private getCurrentManufacturingContext(): ManufacturingContext {
    // Get from localStorage or use defaults
    const stored = localStorage.getItem('manufacturing_context');
    if (stored) {
      return JSON.parse(stored);
    }
    return {
      equipment_type: 'general',
      user_role: 'operator',
      facility_id: 'default',
    };
  }

  // Search API Methods
  async search(query: SearchQuery): Promise<APIResponse<SearchResponse>> {
    const searchParams = new URLSearchParams({
      query: query.query,
      strategy: query.strategy,
      limit: (query.limit || 10).toString(),
      offset: (query.offset || 0).toString(),
    });

    if (query.manufacturing_context) {
      searchParams.append('manufacturing_context', JSON.stringify(query.manufacturing_context));
    }

    if (query.filters) {
      searchParams.append('filters', JSON.stringify(query.filters));
    }

    return this.request<SearchResponse>(
      `/api/search?${searchParams}`,
      {},
      'unified_search'
    );
  }

  async getSearchSuggestions(query: string): Promise<APIResponse<string[]>> {
    return this.request<string[]>(
      `/api/search/suggestions?q=${encodeURIComponent(query)}`,
      {},
      'unified_search'
    );
  }

  async getSimilarDocuments(documentId: string): Promise<APIResponse<SearchResponse['results']>> {
    return this.request<SearchResponse['results']>(
      `/api/similar/${documentId}`,
      {},
      'advanced_rag'
    );
  }

  // Chat API Methods
  async sendMessage(
    message: string,
    sessionId?: string,
    context?: ManufacturingContext
  ): Promise<APIResponse<ChatMessage>> {
    const payload = {
      message,
      session_id: sessionId,
      manufacturing_context: context || this.getCurrentManufacturingContext(),
    };

    return this.request<ChatMessage>(
      '/api/chat/message',
      {
        method: 'POST',
        body: JSON.stringify(payload),
      },
      'chat'
    );
  }

  async createSession(title: string, context?: ManufacturingContext): Promise<APIResponse<ChatSession>> {
    const payload = {
      title,
      manufacturing_context: context || this.getCurrentManufacturingContext(),
    };

    return this.request<ChatSession>(
      '/api/chat/sessions',
      {
        method: 'POST',
        body: JSON.stringify(payload),
      },
      'chat'
    );
  }

  async getSessions(): Promise<APIResponse<ChatSession[]>> {
    return this.request<ChatSession[]>(
      '/api/chat/sessions',
      {},
      'chat'
    );
  }

  async getSession(sessionId: string): Promise<APIResponse<ChatSession>> {
    return this.request<ChatSession>(
      `/api/chat/sessions/${sessionId}`,
      {},
      'chat'
    );
  }

  async deleteSession(sessionId: string): Promise<APIResponse<void>> {
    return this.request<void>(
      `/api/chat/sessions/${sessionId}`,
      {
        method: 'DELETE',
      },
      'chat'
    );
  }

  async getQuickActions(category?: string): Promise<APIResponse<any[]>> {
    const endpoint = category ? `/api/chat/quick-actions?category=${category}` : '/api/chat/quick-actions';
    return this.request<any[]>(endpoint, {}, 'chat');
  }

  // Knowledge Base API Methods
  async getDocument(documentId: string): Promise<APIResponse<any>> {
    return this.request<any>(
      `/api/documents/${documentId}`,
      {},
      'knowledge'
    );
  }

  async uploadDocument(file: File, metadata?: any): Promise<APIResponse<any>> {
    const formData = new FormData();
    formData.append('file', file);
    if (metadata) {
      formData.append('metadata', JSON.stringify(metadata));
    }

    return this.request<any>(
      '/api/documents/upload',
      {
        method: 'POST',
        body: formData,
        headers: {}, // Let browser set Content-Type for FormData
      },
      'knowledge'
    );
  }

  async getDocuments(filters?: any): Promise<APIResponse<any[]>> {
    const searchParams = filters ? new URLSearchParams(filters) : '';
    return this.request<any[]>(
      `/api/documents${searchParams ? '?' + searchParams : ''}`,
      {},
      'knowledge'
    );
  }

  // Equipment API Methods
  async getEquipmentStatus(): Promise<APIResponse<EquipmentStatus[]>> {
    return this.request<EquipmentStatus[]>(
      '/api/equipment/status',
      {},
      'knowledge'
    );
  }

  async updateEquipmentStatus(
    equipmentId: string,
    status: Partial<EquipmentStatus>
  ): Promise<APIResponse<EquipmentStatus>> {
    return this.request<EquipmentStatus>(
      `/api/equipment/${equipmentId}/status`,
      {
        method: 'PUT',
        body: JSON.stringify(status),
      },
      'knowledge'
    );
  }

  // Quality API Methods
  async getQualityMetrics(filters?: any): Promise<APIResponse<QualityMetric[]>> {
    const searchParams = filters ? new URLSearchParams(filters) : '';
    return this.request<QualityMetric[]>(
      `/api/quality/metrics${searchParams ? '?' + searchParams : ''}`,
      {},
      'knowledge'
    );
  }

  async recordQualityMetric(metric: Omit<QualityMetric, 'id' | 'timestamp'>): Promise<APIResponse<QualityMetric>> {
    return this.request<QualityMetric>(
      '/api/quality/metrics',
      {
        method: 'POST',
        body: JSON.stringify(metric),
      },
      'knowledge'
    );
  }

  async getQualityReports(timeRange?: string): Promise<APIResponse<any[]>> {
    const endpoint = timeRange ? `/api/quality/reports?range=${timeRange}` : '/api/quality/reports';
    return this.request<any[]>(endpoint, {}, 'knowledge');
  }

  // Safety API Methods
  async getSafetyEvents(filters?: any): Promise<APIResponse<SafetyEvent[]>> {
    const searchParams = filters ? new URLSearchParams(filters) : '';
    return this.request<SafetyEvent[]>(
      `/api/safety/events${searchParams ? '?' + searchParams : ''}`,
      {},
      'knowledge'
    );
  }

  async reportSafetyEvent(event: Omit<SafetyEvent, 'id' | 'timestamp'>): Promise<APIResponse<SafetyEvent>> {
    return this.request<SafetyEvent>(
      '/api/safety/events',
      {
        method: 'POST',
        body: JSON.stringify(event),
      },
      'knowledge'
    );
  }

  async resolveSafetyEvent(eventId: string, resolution: string): Promise<APIResponse<SafetyEvent>> {
    return this.request<SafetyEvent>(
      `/api/safety/events/${eventId}/resolve`,
      {
        method: 'POST',
        body: JSON.stringify({ action_taken: resolution }),
      },
      'knowledge'
    );
  }

  // Compliance API Methods
  async getComplianceReports(standard?: string): Promise<APIResponse<ComplianceReport[]>> {
    const endpoint = standard ? `/api/compliance/reports?standard=${standard}` : '/api/compliance/reports';
    return this.request<ComplianceReport[]>(endpoint, {}, 'knowledge');
  }

  async generateComplianceReport(
    standard: string,
    startDate: string,
    endDate: string
  ): Promise<APIResponse<ComplianceReport>> {
    const payload = {
      standard,
      start_date: startDate,
      end_date: endDate,
    };

    return this.request<ComplianceReport>(
      '/api/compliance/reports/generate',
      {
        method: 'POST',
        body: JSON.stringify(payload),
      },
      'knowledge'
    );
  }

  // Dashboard API Methods
  async getDashboardMetrics(): Promise<APIResponse<DashboardMetrics>> {
    return this.request<DashboardMetrics>(
      '/api/dashboard/metrics',
      {},
      'knowledge'
    );
  }

  async getSearchAnalytics(timeRange?: string): Promise<APIResponse<any>> {
    const endpoint = timeRange ? `/api/analytics/search?range=${timeRange}` : '/api/analytics/search';
    return this.request<any>(endpoint, {}, 'unified_search');
  }

  // User Preferences API Methods
  async getUserPreferences(): Promise<APIResponse<UserPreferences>> {
    return this.request<UserPreferences>(
      '/api/user/preferences',
      {},
      'knowledge'
    );
  }

  async updateUserPreferences(preferences: Partial<UserPreferences>): Promise<APIResponse<UserPreferences>> {
    return this.request<UserPreferences>(
      '/api/user/preferences',
      {
        method: 'PUT',
        body: JSON.stringify(preferences),
      },
      'knowledge'
    );
  }

  // Multi-Agent API Methods
  async executeAgentTask(
    agentType: string,
    task: any,
    context?: ManufacturingContext
  ): Promise<APIResponse<any>> {
    const payload = {
      agent_type: agentType,
      task,
      manufacturing_context: context || this.getCurrentManufacturingContext(),
    };

    return this.request<any>(
      '/api/agents/execute',
      {
        method: 'POST',
        body: JSON.stringify(payload),
      },
      'multi_agent'
    );
  }

  async getAgentStatus(): Promise<APIResponse<any>> {
    return this.request<any>(
      '/api/agents/status',
      {},
      'multi_agent'
    );
  }

  async decomposeTask(task: any): Promise<APIResponse<any>> {
    return this.request<any>(
      '/api/agents/decompose',
      {
        method: 'POST',
        body: JSON.stringify(task),
      },
      'multi_agent'
    );
  }

  // WebSocket connection for real-time updates
  connectWebSocket(endpoint: keyof typeof API_BASE_URLS = 'chat'): WebSocket {
    const wsUrl = this.baseURLs[endpoint].replace('http://', 'ws://').replace('https://', 'wss://');
    return new WebSocket(`${wsUrl}/ws`);
  }

  // Integration specific methods
  async getIntegrationStatus(): Promise<APIResponse<any>> {
    return this.request<any>(
      '/api/integrations/status',
      {},
      'knowledge'
    );
  }

  async testIntegration(integrationName: string): Promise<APIResponse<any>> {
    return this.request<any>(
      `/api/integrations/${integrationName}/test`,
      {},
      'knowledge'
    );
  }
}

export const apiService = new APIService();
export default apiService;