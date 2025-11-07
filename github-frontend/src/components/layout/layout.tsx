import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LayoutDashboard,
  MessageSquare,
  BookOpen,
  Settings,
  BarChart3,
  AlertTriangle,
  Users,
  Search,
  Menu,
  X,
  Factory,
  Cpu,
  Zap,
  Wrench,
  TrendingUp
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ThemeToggle } from '@/components/theme-toggle';
import { useThemeStore } from '@/store/theme';
import { cn } from '@/lib/utils';

interface LayoutProps {
  children: React.ReactNode;
  currentPage?: string;
}

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard, description: 'System overview and metrics' },
  { name: 'Chat', href: '/chat', icon: MessageSquare, description: 'AI manufacturing assistant' },
  { name: 'Knowledge Base', href: '/knowledge', icon: BookOpen, description: 'Documentation and guides' },
  { name: 'Analytics', href: '/analytics', icon: BarChart3, description: 'Production analytics' },
  { name: 'Systems', href: '/systems', icon: Cpu, description: 'Manufacturing systems' },
  { name: 'Alerts', href: '/alerts', icon: AlertTriangle, description: 'System alerts and issues' },
  { name: 'Users', href: '/users', icon: Users, description: 'User management' },
  { name: 'Settings', href: '/settings', icon: Settings, description: 'System configuration' },
];

const quickActions = [
  { name: 'Start Production', icon: Zap, color: 'text-green-600' },
  { name: 'Schedule Maintenance', icon: Wrench, color: 'text-orange-600' },
  { name: 'View Reports', icon: TrendingUp, color: 'text-blue-600' },
  { name: 'System Status', icon: Factory, color: 'text-purple-600' },
];

export function Layout({ children, currentPage }: LayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const { resolvedTheme } = useThemeStore();

  return (
    <div className="min-h-screen bg-background">
      {/* Mobile sidebar backdrop */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 bg-black/50 lg:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ x: -300 }}
            animate={{ x: 0 }}
            exit={{ x: -300 }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
            className="fixed left-0 top-0 z-50 h-full w-72 bg-card border-r lg:hidden"
          >
            <SidebarContent currentPage={currentPage} onClose={() => setSidebarOpen(false)} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Desktop Sidebar */}
      <div className="hidden lg:fixed lg:inset-y-0 lg:left-0 lg:z-40 lg:w-72 lg:block">
        <SidebarContent currentPage={currentPage} />
      </div>

      {/* Main Content */}
      <div className="lg:pl-72">
        {/* Top Navigation */}
        <header className="sticky top-0 z-30 flex h-16 shrink-0 items-center justify-between gap-x-4 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 px-4 sm:gap-x-6 sm:px-6 lg:px-8">
          <div className="flex items-center gap-x-4 lg:gap-x-6">
            {/* Mobile menu button */}
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSidebarOpen(true)}
              className="lg:hidden"
            >
              <Menu className="h-6 w-6" />
            </Button>

            {/* Search */}
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <input
                type="text"
                placeholder="Search systems, documentation, or ask AI..."
                className="w-full pl-10 pr-4 py-2 bg-muted/50 border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent text-sm placeholder:text-muted-foreground"
                // TODO: Implement search functionality
              />
            </div>
          </div>

          <div className="flex items-center gap-x-4">
            {/* Quick Actions */}
            <div className="hidden md:flex items-center gap-x-2">
              {quickActions.slice(0, 2).map((action) => (
                <Button
                  key={action.name}
                  variant="ghost"
                  size="sm"
                  className="h-8 text-xs"
                >
                  <action.icon className={cn("h-3 w-3 mr-2", action.color)} />
                  {action.name}
                </Button>
              ))}
            </div>

            {/* Theme Toggle */}
            <ThemeToggle />

            {/* User Menu */}
            <div className="flex items-center">
              <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
                <span className="text-sm font-medium text-primary">A</span>
              </div>
            </div>
          </div>
        </header>

        {/* Page Content */}
        <main className="py-6">
          <div className="px-4 sm:px-6 lg:px-8">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}

interface SidebarContentProps {
  currentPage?: string;
  onClose?: () => void;
}

function SidebarContent({ currentPage, onClose }: SidebarContentProps) {
  return (
    <div className="flex h-full flex-col">
      {/* Logo */}
      <div className="flex h-16 shrink-0 items-center justify-between px-6 border-b">
        <div className="flex items-center space-x-3">
          <Factory className="h-8 w-8 text-primary" />
          <div>
            <h1 className="text-lg font-semibold">Manufacturing KB</h1>
            <p className="text-xs text-muted-foreground">Knowledge Base System</p>
          </div>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={onClose}
          className="lg:hidden"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Navigation */}
      <nav className="flex flex-1 flex-col px-4 pb-4">
        <ul role="list" className="space-y-1">
          {navigation.map((item) => (
            <li key={item.name}>
              <Button
                variant={currentPage === item.name ? 'secondary' : 'ghost'}
                className={cn(
                  "w-full justify-start h-12 text-sm",
                  currentPage === item.name && "bg-primary/10 text-primary"
                )}
                onClick={() => {
                  // TODO: Navigate to page
                  console.log(`Navigate to ${item.href}`);
                  onClose?.();
                }}
              >
                <item.icon className="mr-3 h-4 w-4" />
                <span className="block">{item.name}</span>
              </Button>
              {currentPage === item.name && (
                <div className="text-xs text-muted-foreground ml-12 mb-2">
                  {item.description}
                </div>
              )}
            </li>
          ))}
        </ul>

        {/* Quick Actions Section */}
        <div className="mt-auto">
          <div className="px-3 py-2">
            <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
              Quick Actions
            </h3>
            <div className="space-y-1">
              {quickActions.map((action) => (
                <Button
                  key={action.name}
                  variant="ghost"
                  className="w-full justify-start h-10 text-sm"
                  onClick={() => {
                    // TODO: Implement quick action
                    console.log(`Quick action: ${action.name}`);
                  }}
                >
                  <action.icon className={cn("mr-3 h-4 w-4", action.color)} />
                  {action.name}
                </Button>
              ))}
            </div>
          </div>

          {/* System Status Summary */}
          <div className="px-3 py-2 mt-4">
            <div className="bg-muted/50 rounded-lg p-3">
              <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                System Status
              </h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Online Systems</span>
                  <span className="font-medium text-green-600">8/10</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Average Efficiency</span>
                  <span className="font-medium">87.3%</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Active Alerts</span>
                  <span className="font-medium text-orange-600">3</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </nav>
    </div>
  );
}