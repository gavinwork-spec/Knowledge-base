import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import { motion } from 'framer-motion';
import { Layout } from '@/components/layout/layout';
import { LayoutDashboard } from '@/components/dashboard/dashboard';
import { ChatInterface } from '@/components/chat/chat-interface';
import { SystemStatus } from '@/components/dashboard/system-status';
import { ThemeProvider } from '@/components/theme-provider';
import { initializeTheme } from '@/store/theme';
import '@/index.css';

// Create a client for React Query
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
    },
  },
});

// Page Components
const DashboardPage = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Manufacturing Dashboard</h1>
          <p className="text-muted-foreground">
            Real-time overview of your manufacturing systems and operations.
          </p>
        </div>
        <SystemStatus />
      </div>
    </motion.div>
  );
};

const ChatPage = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="h-[calc(100vh-8rem)]"
    >
      <ChatInterface />
    </motion.div>
  );
};

const KnowledgePage = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Knowledge Base</h1>
          <p className="text-muted-foreground">
            Access documentation, guides, and manufacturing best practices.
          </p>
        </div>
        {/* TODO: Implement Knowledge Base component */}
        <div className="bg-muted/50 rounded-lg p-8 text-center">
          <h3 className="text-lg font-semibold mb-2">Knowledge Base</h3>
          <p className="text-muted-foreground">
            Knowledge base interface coming soon...
          </p>
        </div>
      </div>
    </motion.div>
  );
};

const AnalyticsPage = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Analytics</h1>
          <p className="text-muted-foreground">
            Production analytics, performance metrics, and insights.
          </p>
        </div>
        {/* TODO: Implement Analytics component */}
        <div className="bg-muted/50 rounded-lg p-8 text-center">
          <h3 className="text-lg font-semibold mb-2">Analytics Dashboard</h3>
          <p className="text-muted-foreground">
            Analytics interface coming soon...
          </p>
        </div>
      </div>
    </motion.div>
  );
};

const SystemsPage = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Systems Management</h1>
          <p className="text-muted-foreground">
            Configure and manage your manufacturing systems.
          </p>
        </div>
        <SystemStatus />
      </div>
    </motion.div>
  );
};

const AlertsPage = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">System Alerts</h1>
          <p className="text-muted-foreground">
            Monitor and respond to system alerts and issues.
          </p>
        </div>
        {/* TODO: Implement Alerts component */}
        <div className="bg-muted/50 rounded-lg p-8 text-center">
          <h3 className="text-lg font-semibold mb-2">Alerts Center</h3>
          <p className="text-muted-foreground">
            Alerts interface coming soon...
          </p>
        </div>
      </div>
    </motion.div>
  );
};

const UsersPage = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">User Management</h1>
          <p className="text-muted-foreground">
            Manage user accounts and permissions.
          </p>
        </div>
        {/* TODO: Implement User Management component */}
        <div className="bg-muted/50 rounded-lg p-8 text-center">
          <h3 className="text-lg font-semibold mb-2">User Management</h3>
          <p className="text-muted-foreground">
            User management interface coming soon...
          </p>
        </div>
      </div>
    </motion.div>
  );
};

const SettingsPage = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
          <p className="text-muted-foreground">
            Configure system settings and preferences.
          </p>
        </div>
        {/* TODO: Implement Settings component */}
        <div className="bg-muted/50 rounded-lg p-8 text-center">
          <h3 className="text-lg font-semibold mb-2">System Settings</h3>
          <p className="text-muted-foreground">
            Settings interface coming soon...
          </p>
        </div>
      </div>
    </motion.div>
  );
};

function App() {
  const [currentPage, setCurrentPage] = useState('Dashboard');

  useEffect(() => {
    // Initialize theme
    const cleanup = initializeTheme();
    return cleanup;
  }, []);

  // Get current page name from path
  const getPageName = (path: string) => {
    const pageMap: Record<string, string> = {
      '/': 'Dashboard',
      '/chat': 'Chat',
      '/knowledge': 'Knowledge Base',
      '/analytics': 'Analytics',
      '/systems': 'Systems',
      '/alerts': 'Alerts',
      '/users': 'Users',
      '/settings': 'Settings',
    };
    return pageMap[path] || 'Dashboard';
  };

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <Router>
          <Layout currentPage={getPageName(window.location.pathname)}>
            <Routes>
              <Route path="/" element={<DashboardPage />} />
              <Route path="/chat" element={<ChatPage />} />
              <Route path="/knowledge" element={<KnowledgePage />} />
              <Route path="/analytics" element={<AnalyticsPage />} />
              <Route path="/systems" element={<SystemsPage />} />
              <Route path="/alerts" element={<AlertsPage />} />
              <Route path="/users" element={<UsersPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Layout>
        </Router>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;