import React, { useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';

// Import store hooks
import { useThemeStore, useUIStore } from '../../stores';

// Import components
import Sidebar from './Sidebar';
import Header from './Header';
import SearchPage from '../search/SearchPage';
import ChatPage from '../chat/ChatPage';
import DashboardPage from '../dashboard/DashboardPage';
import EquipmentPage from '../manufacturing/EquipmentPage';
import QualityPage from '../manufacturing/QualityPage';
import SafetyPage from '../manufacturing/SafetyPage';
import SettingsPage from '../settings/SettingsPage';

const Layout: React.FC = () => {
  const { theme } = useThemeStore();
  const { sidebarOpen } = useUIStore();

  // Apply theme to document
  useEffect(() => {
    const root = document.documentElement;

    if (theme === 'system') {
      const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
      root.classList.toggle('dark', systemTheme === 'dark');
    } else {
      root.classList.toggle('dark', theme === 'dark');
    }
  }, [theme]);

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ x: -300 }}
            animate={{ x: 0 }}
            exit={{ x: -300 }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            className="fixed lg:relative z-30 h-full"
          >
            <Sidebar />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content Area */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Header */}
        <Header />

        {/* Page Content */}
        <main className="flex-1 overflow-auto bg-background">
          <AnimatePresence mode="wait">
            <Routes>
              <Route path="/" element={<SearchPage />} />
              <Route path="/search" element={<SearchPage />} />
              <Route path="/chat" element={<ChatPage />} />
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/equipment" element={<EquipmentPage />} />
              <Route path="/quality" element={<QualityPage />} />
              <Route path="/safety" element={<SafetyPage />} />
              <Route path="/settings" element={<SettingsPage />} />
            </Routes>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
};

export default Layout;