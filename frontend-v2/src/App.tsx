import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter as Router } from 'react-router-dom';
import { Toaster } from 'sonner';

// Import components
import Layout from './components/layout/Layout';
import ManufacturingProvider from './components/manufacturing/ManufacturingProvider';
import NotificationSystem from './components/ui/NotificationSystem';
import ErrorBoundary from './components/ui/ErrorBoundary';

// Import styles
import './index.css';

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <Router>
          <ManufacturingProvider>
            <div className="min-h-screen bg-background text-foreground">
              <Layout />
              <NotificationSystem />
              <Toaster
                position="top-right"
                expand={false}
                richColors
                closeButton
              />
            </div>
          </ManufacturingProvider>
        </Router>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

export default App;