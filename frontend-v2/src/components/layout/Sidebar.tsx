import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Search,
  MessageSquare,
  LayoutDashboard,
  Settings,
  Wrench,
  Shield,
  Target,
  Menu,
  X
} from 'lucide-react';
import { useUIStore, useManufacturingStore } from '../../stores';
import Button from '../ui/button';

const Sidebar: React.FC = () => {
  const location = useLocation();
  const { sidebarOpen, setSidebarOpen } = useUIStore();
  const { context } = useManufacturingStore();

  const navigationItems = [
    { name: 'Search', href: '/', icon: Search },
    { name: 'Chat', href: '/chat', icon: MessageSquare },
    { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
    { name: 'Equipment', href: '/equipment', icon: Wrench },
    { name: 'Quality', href: '/quality', icon: Target },
    { name: 'Safety', href: '/safety', icon: Shield },
    { name: 'Settings', href: '/settings', icon: Settings },
  ];

  const isActive = (href: string) => {
    if (href === '/') {
      return location.pathname === '/' || location.pathname === '/search';
    }
    return location.pathname === href;
  };

  return (
    <div className="manufacturing-sidebar">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
            <span className="text-primary-foreground font-bold text-sm">M</span>
          </div>
          <span className="font-semibold">Manufacturing KB</span>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setSidebarOpen(false)}
          className="lg:hidden"
        >
          <X className="w-4 h-4" />
        </Button>
      </div>

      {/* Manufacturing Context */}
      <div className="p-4 border-b border-border">
        <div className="text-sm font-medium text-muted-foreground mb-2">Context</div>
        <div className="space-y-1 text-xs">
          <div className="text-foreground">{context.equipment_type || 'General'}</div>
          <div className="text-muted-foreground">{context.user_role || 'Operator'}</div>
          <div className="text-muted-foreground">{context.facility_id || 'Default Facility'}</div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4">
        <div className="space-y-2">
          {navigationItems.map((item) => {
            const Icon = item.icon;
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`
                  flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors
                  ${isActive(item.href)
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:text-foreground hover:bg-accent'
                  }
                `}
              >
                <Icon className="w-4 h-4" />
                {item.name}
              </Link>
            );
          })}
        </div>
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-border">
        <div className="text-xs text-muted-foreground">
          Manufacturing Knowledge Base
        </div>
        <div className="text-xs text-muted-foreground mt-1">
          v2.0.0
        </div>
      </div>
    </div>
  );
};

export default Sidebar;