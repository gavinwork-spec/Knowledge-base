// Zustand Store Configuration - Manufacturing Knowledge Base
// Maintains compatibility with existing state management patterns

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { devtools } from 'zustand/middleware';
import {
  SearchQuery,
  SearchResult,
  ChatSession,
  ChatMessage,
  ManufacturingContext,
  UserPreferences,
  EquipmentStatus,
  QuickAction,
  SearchFilters
} from '../types';

// Theme Store
interface ThemeState {
  theme: 'light' | 'dark' | 'system';
  toggleTheme: () => void;
  setTheme: (theme: 'light' | 'dark' | 'system') => void;
}

export const useThemeStore = create<ThemeState>()(
  devtools(
    persist(
      (set, get) => ({
        theme: 'system',
        toggleTheme: () => {
          const current = get().theme;
          const next = current === 'light' ? 'dark' : current === 'dark' ? 'system' : 'light';
          set({ theme: next });
        },
        setTheme: (theme) => set({ theme }),
      }),
      {
        name: 'theme-storage',
      }
    ),
    { name: 'theme-store' }
  )
);

// Manufacturing Context Store
interface ManufacturingState {
  context: ManufacturingContext;
  updateContext: (updates: Partial<ManufacturingContext>) => void;
  setContext: (context: ManufacturingContext) => void;
  resetContext: () => void;
}

const defaultManufacturingContext: ManufacturingContext = {
  equipment_type: 'general',
  user_role: 'operator',
  facility_id: 'default',
  process_type: 'machining',
  compliance_standards: ['ISO_9001'],
};

export const useManufacturingStore = create<ManufacturingState>()(
  devtools(
    persist(
      (set, get) => ({
        context: defaultManufacturingContext,
        updateContext: (updates) => {
          const current = get().context;
          set({ context: { ...current, ...updates } });
        },
        setContext: (context) => set({ context }),
        resetContext: () => set({ context: defaultManufacturingContext }),
      }),
      {
        name: 'manufacturing-context-storage',
      }
    ),
    { name: 'manufacturing-store' }
  )
);

// Search Store
interface SearchState {
  query: string;
  results: SearchResult[];
  isLoading: boolean;
  error: string | null;
  searchHistory: string[];
  filters: SearchFilters;
  searchStrategy: 'unified' | 'semantic' | 'keyword' | 'graph' | 'ai_enhanced';
  totalResults: number;
  searchTime: number;
  suggestions: string[];

  setQuery: (query: string) => void;
  setResults: (results: SearchResult[], totalResults?: number) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  addToHistory: (query: string) => void;
  clearHistory: () => void;
  setFilters: (filters: SearchFilters) => void;
  setSearchStrategy: (strategy: SearchState['searchStrategy']) => void;
  setSuggestions: (suggestions: string[]) => void;
  clearResults: () => void;
}

export const useSearchStore = create<SearchState>()(
  devtools(
    persist(
      (set, get) => ({
        query: '',
        results: [],
        isLoading: false,
        error: null,
        searchHistory: [],
        filters: {},
        searchStrategy: 'unified',
        totalResults: 0,
        searchTime: 0,
        suggestions: [],

        setQuery: (query) => set({ query }),
        setResults: (results, totalResults) => set({ results, totalResults: totalResults || results.length }),
        setLoading: (isLoading) => set({ isLoading }),
        setError: (error) => set({ error }),
        addToHistory: (query) => {
          const history = get().searchHistory;
          const filtered = history.filter(item => item !== query);
          set({ searchHistory: [query, ...filtered.slice(0, 19)] }); // Keep last 20
        },
        clearHistory: () => set({ searchHistory: [] }),
        setFilters: (filters) => set({ filters }),
        setSearchStrategy: (searchStrategy) => set({ searchStrategy }),
        setSuggestions: (suggestions) => set({ suggestions }),
        clearResults: () => set({ results: [], totalResults: 0, error: null }),
      }),
      {
        name: 'search-storage',
        partialize: (state) => ({
          searchHistory: state.searchHistory,
          filters: state.filters,
          searchStrategy: state.searchStrategy,
        }),
      }
    ),
    { name: 'search-store' }
  )
);

// Chat Store
interface ChatState {
  currentSession: ChatSession | null;
  sessions: ChatSession[];
  isLoading: boolean;
  error: string | null;
  quickActions: QuickAction[];

  setCurrentSession: (session: ChatSession | null) => void;
  addSession: (session: ChatSession) => void;
  updateSession: (sessionId: string, updates: Partial<ChatSession>) => void;
  removeSession: (sessionId: string) => void;
  addMessage: (sessionId: string, message: ChatMessage) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setQuickActions: (actions: QuickAction[]) => void;
}

export const useChatStore = create<ChatState>()(
  devtools(
    persist(
      (set, get) => ({
        currentSession: null,
        sessions: [],
        isLoading: false,
        error: null,
        quickActions: [],

        setCurrentSession: (currentSession) => set({ currentSession }),
        addSession: (session) => {
          const sessions = get().sessions;
          set({ sessions: [session, ...sessions.filter(s => s.id !== session.id)] });
        },
        updateSession: (sessionId, updates) => {
          const sessions = get().sessions;
          const updated = sessions.map(s => s.id === sessionId ? { ...s, ...updates } : s);
          set({ sessions: updated });

          // Update current session if it's the one being updated
          const current = get().currentSession;
          if (current && current.id === sessionId) {
            set({ currentSession: { ...current, ...updates } });
          }
        },
        removeSession: (sessionId) => {
          const sessions = get().sessions;
          const filtered = sessions.filter(s => s.id !== sessionId);
          set({ sessions: filtered });

          // Clear current session if it's the one being removed
          const current = get().currentSession;
          if (current && current.id === sessionId) {
            set({ currentSession: null });
          }
        },
        addMessage: (sessionId, message) => {
          const sessions = get().sessions;
          const updated = sessions.map(s =>
            s.id === sessionId
              ? {
                  ...s,
                  messages: [...s.messages, message],
                  updated_at: message.timestamp
                }
              : s
          );
          set({ sessions: updated });

          // Update current session if it's the one receiving the message
          const current = get().currentSession;
          if (current && current.id === sessionId) {
            set({
              currentSession: {
                ...current,
                messages: [...current.messages, message],
                updated_at: message.timestamp
              }
            });
          }
        },
        setLoading: (isLoading) => set({ isLoading }),
        setError: (error) => set({ error }),
        setQuickActions: (quickActions) => set({ quickActions }),
      }),
      {
        name: 'chat-storage',
        partialize: (state) => ({
          sessions: state.sessions.slice(0, 10), // Keep only last 10 sessions
          quickActions: state.quickActions,
        }),
      }
    ),
    { name: 'chat-store' }
  )
);

// Equipment Store
interface EquipmentState {
  equipment: EquipmentStatus[];
  selectedEquipment: EquipmentStatus | null;
  isLoading: boolean;
  error: string | null;

  setEquipment: (equipment: EquipmentStatus[]) => void;
  setSelectedEquipment: (equipment: EquipmentStatus | null) => void;
  updateEquipmentStatus: (equipmentId: string, status: Partial<EquipmentStatus>) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useEquipmentStore = create<EquipmentState>()(
  devtools(
    (set, get) => ({
      equipment: [],
      selectedEquipment: null,
      isLoading: false,
      error: null,

      setEquipment: (equipment) => set({ equipment }),
      setSelectedEquipment: (selectedEquipment) => set({ selectedEquipment }),
      updateEquipmentStatus: (equipmentId, status) => {
        const equipment = get().equipment;
        const updated = equipment.map(e =>
          e.id === equipmentId ? { ...e, ...status } : e
        );
        set({ equipment: updated });

        // Update selected equipment if it's the one being updated
        const selected = get().selectedEquipment;
        if (selected && selected.id === equipmentId) {
          set({ selectedEquipment: { ...selected, ...status } });
        }
      },
      setLoading: (isLoading) => set({ isLoading }),
      setError: (error) => set({ error }),
    }),
    { name: 'equipment-store' }
  )
);

// User Preferences Store
interface UserPreferencesState {
  preferences: UserPreferences;
  isLoading: boolean;
  error: string | null;

  setPreferences: (preferences: UserPreferences) => void;
  updatePreferences: (updates: Partial<UserPreferences>) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

const defaultUserPreferences: UserPreferences = {
  theme: 'system',
  language: 'en',
  default_search_strategy: 'unified',
  expertise_level: 'intermediate',
  equipment_types: [],
  compliance_standards: ['ISO_9001'],
  notifications: {
    safety_alerts: true,
    quality_issues: true,
    maintenance_reminders: true,
    compliance_updates: false,
  },
  ui: {
    sidebar_collapsed: false,
    compact_mode: false,
    show_explanations: true,
    auto_save_searches: true,
  },
};

export const useUserPreferencesStore = create<UserPreferencesState>()(
  devtools(
    persist(
      (set, get) => ({
        preferences: defaultUserPreferences,
        isLoading: false,
        error: null,

        setPreferences: (preferences) => set({ preferences }),
        updatePreferences: (updates) => {
          const current = get().preferences;
          set({ preferences: { ...current, ...updates } });
        },
        setLoading: (isLoading) => set({ isLoading }),
        setError: (error) => set({ error }),
      }),
      {
        name: 'user-preferences-storage',
      }
    ),
    { name: 'user-preferences-store' }
  )
);

// UI Store for component states
interface UIState {
  sidebarOpen: boolean;
  activeTab: string;
  notifications: Array<{
    id: string;
    type: 'info' | 'success' | 'warning' | 'error';
    title: string;
    message: string;
    timestamp: string;
    autoClose?: boolean;
  }>;
  modals: {
    searchFilters: boolean;
    equipmentDetails: boolean;
    qualityReport: boolean;
    safetyEvent: boolean;
    settings: boolean;
  };

  setSidebarOpen: (open: boolean) => void;
  setActiveTab: (tab: string) => void;
  addNotification: (notification: Omit<UIState['notifications'][0], 'id' | 'timestamp'>) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
  openModal: (modal: keyof UIState['modals']) => void;
  closeModal: (modal: keyof UIState['modals']) => void;
  closeAllModals: () => void;
}

export const useUIStore = create<UIState>()(
  devtools(
    (set, get) => ({
      sidebarOpen: true,
      activeTab: 'search',
      notifications: [],
      modals: {
        searchFilters: false,
        equipmentDetails: false,
        qualityReport: false,
        safetyEvent: false,
        settings: false,
      },

      setSidebarOpen: (sidebarOpen) => set({ sidebarOpen }),
      setActiveTab: (activeTab) => set({ activeTab }),
      addNotification: (notification) => {
        const id = Math.random().toString(36).substring(2);
        const timestamp = new Date().toISOString();
        const notifications = [...get().notifications, { ...notification, id, timestamp }];
        set({ notifications });

        // Auto close notification after 5 seconds if specified
        if (notification.autoClose !== false) {
          setTimeout(() => {
            set(state => ({
              notifications: state.notifications.filter(n => n.id !== id)
            }));
          }, 5000);
        }
      },
      removeNotification: (id) => {
        const notifications = get().notifications.filter(n => n.id !== id);
        set({ notifications });
      },
      clearNotifications: () => set({ notifications: [] }),
      openModal: (modal) => {
        const modals = { ...get().modals, [modal]: true };
        set({ modals });
      },
      closeModal: (modal) => {
        const modals = { ...get().modals, [modal]: false };
        set({ modals });
      },
      closeAllModals: () => {
        const modals = Object.keys(get().modals).reduce((acc, key) => ({
          ...acc,
          [key]: false
        }), {} as UIState['modals']);
        set({ modals });
      },
    }),
    { name: 'ui-store' }
  )
);

// Combined store selectors for commonly used combinations
export const useAppStores = () => ({
  theme: useThemeStore(),
  manufacturing: useManufacturingStore(),
  search: useSearchStore(),
  chat: useChatStore(),
  equipment: useEquipmentStore(),
  userPreferences: useUserPreferencesStore(),
  ui: useUIStore(),
});

export default {
  useThemeStore,
  useManufacturingStore,
  useSearchStore,
  useChatStore,
  useEquipmentStore,
  useUserPreferencesStore,
  useUIStore,
  useAppStores,
};