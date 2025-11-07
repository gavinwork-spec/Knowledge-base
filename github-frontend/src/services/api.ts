import axios from 'axios';
import type {
  ManufacturingSystem,
  KnowledgeEntry,
  ChatRequest,
  ChatResponse,
  SearchQuery,
  SearchResult,
  AnalyticsData,
  Alert,
  User
} from '@/types';

// Configure axios defaults
const apiClient = axios.create({
  baseURL: process.env.NODE_ENV === 'production' ? '/api' : 'http://localhost:8001/api',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

const chatClient = axios.create({
  baseURL: process.env.NODE_ENV === 'production' ? '/chat' : 'http://localhost:8002/api/v1',
  timeout: 30000, // Longer timeout for AI responses
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for authentication
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

chatClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

chatClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Manufacturing System API
export const manufacturingAPI = {
  // Get all manufacturing systems
  getSystems: async (): Promise<ManufacturingSystem[]> => {
    const response = await apiClient.get('/systems');
    return response.data.data;
  },

  // Get system by ID
  getSystem: async (id: string): Promise<ManufacturingSystem> => {
    const response = await apiClient.get(`/systems/${id}`);
    return response.data.data;
  },

  // Create new system
  createSystem: async (system: Omit<ManufacturingSystem, 'id'>): Promise<ManufacturingSystem> => {
    const response = await apiClient.post('/systems', system);
    return response.data.data;
  },

  // Update system
  updateSystem: async (id: string, system: Partial<ManufacturingSystem>): Promise<ManufacturingSystem> => {
    const response = await apiClient.put(`/systems/${id}`, system);
    return response.data.data;
  },

  // Delete system
  deleteSystem: async (id: string): Promise<void> => {
    await apiClient.delete(`/systems/${id}`);
  },

  // Get system metrics
  getSystemMetrics: async (id: string): Promise<any> => {
    const response = await apiClient.get(`/systems/${id}/metrics`);
    return response.data.data;
  },

  // Get system alerts
  getSystemAlerts: async (id: string): Promise<Alert[]> => {
    const response = await apiClient.get(`/systems/${id}/alerts`);
    return response.data.data;
  },
};

// Knowledge Base API
export const knowledgeAPI = {
  // Search knowledge base
  search: async (query: SearchQuery): Promise<SearchResult> => {
    const response = await apiClient.post('/knowledge/search', query);
    return response.data.data;
  },

  // Get all entries
  getEntries: async (page: number = 1, limit: number = 20): Promise<{ entries: KnowledgeEntry[], total: number }> => {
    const response = await apiClient.get('/knowledge/entries', {
      params: { page, limit }
    });
    return response.data.data;
  },

  // Get entry by ID
  getEntry: async (id: string): Promise<KnowledgeEntry> => {
    const response = await apiClient.get(`/knowledge/entries/${id}`);
    return response.data.data;
  },

  // Create new entry
  createEntry: async (entry: Omit<KnowledgeEntry, 'id' | 'createdAt' | 'updatedAt'>): Promise<KnowledgeEntry> => {
    const response = await apiClient.post('/knowledge/entries', entry);
    return response.data.data;
  },

  // Update entry
  updateEntry: async (id: string, entry: Partial<KnowledgeEntry>): Promise<KnowledgeEntry> => {
    const response = await apiClient.put(`/knowledge/entries/${id}`, entry);
    return response.data.data;
  },

  // Delete entry
  deleteEntry: async (id: string): Promise<void> => {
    await apiClient.delete(`/knowledge/entries/${id}`);
  },

  // Get categories
  getCategories: async (): Promise<string[]> => {
    const response = await apiClient.get('/knowledge/categories');
    return response.data.data;
  },

  // Get tags
  getTags: async (): Promise<string[]> => {
    const response = await apiClient.get('/knowledge/tags');
    return response.data.data;
  },

  // Upload attachment
  uploadAttachment: async (file: File): Promise<{ id: string; url: string }> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post('/knowledge/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data.data;
  },
};

// Chat API
export const chatAPI = {
  // Send chat message
  sendMessage: async (request: ChatRequest): Promise<ChatResponse> => {
    const response = await chatClient.post('/chat', request);
    return response.data.data;
  },

  // Get chat history
  getChatHistory: async (sessionId: string): Promise<any> => {
    const response = await chatClient.get(`/chat/history/${sessionId}`);
    return response.data.data;
  },

  // Delete chat session
  deleteChatSession: async (sessionId: string): Promise<void> => {
    await chatClient.delete(`/chat/session/${sessionId}`);
  },

  // Get chat suggestions
  getSuggestions: async (): Promise<string[]> => {
    const response = await chatClient.get('/chat/suggestions');
    return response.data.data;
  },
};

// Analytics API
export const analyticsAPI = {
  // Get dashboard analytics
  getDashboardAnalytics: async (): Promise<AnalyticsData> => {
    const response = await apiClient.get('/analytics/dashboard');
    return response.data.data;
  },

  // Get production analytics
  getProductionAnalytics: async (timeRange: string = '7d'): Promise<any> => {
    const response = await apiClient.get('/analytics/production', {
      params: { timeRange }
    });
    return response.data.data;
  },

  // Get efficiency analytics
  getEfficiencyAnalytics: async (systemId?: string): Promise<any> => {
    const response = await apiClient.get('/analytics/efficiency', {
      params: { systemId }
    });
    return response.data.data;
  },

  // Get cost analytics
  getCostAnalytics: async (timeRange: string = '30d'): Promise<any> => {
    const response = await apiClient.get('/analytics/cost', {
      params: { timeRange }
    });
    return response.data.data;
  },

  // Export analytics data
  exportAnalytics: async (type: string, format: string = 'csv'): Promise<Blob> => {
    const response = await apiClient.get(`/analytics/export/${type}`, {
      params: { format },
      responseType: 'blob'
    });
    return response.data;
  },
};

// Alert API
export const alertAPI = {
  // Get all alerts
  getAlerts: async (filters?: any): Promise<Alert[]> => {
    const response = await apiClient.get('/alerts', { params: filters });
    return response.data.data;
  },

  // Get alert by ID
  getAlert: async (id: string): Promise<Alert> => {
    const response = await apiClient.get(`/alerts/${id}`);
    return response.data.data;
  },

  // Acknowledge alert
  acknowledgeAlert: async (id: string): Promise<void> => {
    await apiClient.post(`/alerts/${id}/acknowledge`);
  },

  // Resolve alert
  resolveAlert: async (id: string): Promise<void> => {
    await apiClient.post(`/alerts/${id}/resolve`);
  },

  // Create custom alert
  createAlert: async (alert: Omit<Alert, 'id' | 'timestamp'>): Promise<Alert> => {
    const response = await apiClient.post('/alerts', alert);
    return response.data.data;
  },
};

// User API
export const userAPI = {
  // Get current user
  getCurrentUser: async (): Promise<User> => {
    const response = await apiClient.get('/user/me');
    return response.data.data;
  },

  // Update user profile
  updateProfile: async (profile: Partial<User>): Promise<User> => {
    const response = await apiClient.put('/user/profile', profile);
    return response.data.data;
  },

  // Update user preferences
  updatePreferences: async (preferences: any): Promise<void> => {
    await apiClient.put('/user/preferences', preferences);
  },

  // Get user activity
  getActivity: async (): Promise<any[]> => {
    const response = await apiClient.get('/user/activity');
    return response.data.data;
  },
};

// Health Check API
export const healthAPI = {
  // Check system health
  checkHealth: async (): Promise<any> => {
    const response = await apiClient.get('/health');
    return response.data.data;
  },

  // Get system status
  getStatus: async (): Promise<any> => {
    const response = await apiClient.get('/status');
    return response.data.data;
  },
};

// Export all APIs
export const api = {
  manufacturing: manufacturingAPI,
  knowledge: knowledgeAPI,
  chat: chatAPI,
  analytics: analyticsAPI,
  alerts: alertAPI,
  users: userAPI,
  health: healthAPI,
};

export default api;