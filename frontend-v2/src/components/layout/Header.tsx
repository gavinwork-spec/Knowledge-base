import React from 'react';
import { Menu, Search, Bell, User, Moon, Sun } from 'lucide-react';
import { useUIStore, useThemeStore } from '../../stores';
import Button from '../ui/button';

const Header: React.FC = () => {
  const { setSidebarOpen } = useUIStore();
  const { theme, toggleTheme } = useThemeStore();

  return (
    <header className="manufacturing-header">
      <div className="flex items-center justify-between px-6 py-4">
        {/* Left side */}
        <div className="flex items-center gap-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setSidebarOpen(true)}
            className="lg:hidden text-white hover:bg-white/20"
          >
            <Menu className="w-5 h-5" />
          </Button>

          <div className="hidden lg:block">
            <h1 className="text-xl font-semibold">Manufacturing Knowledge Base</h1>
          </div>
        </div>

        {/* Right side */}
        <div className="flex items-center gap-3">
          {/* Search button for mobile */}
          <Button
            variant="ghost"
            size="icon"
            className="lg:hidden text-white hover:bg-white/20"
          >
            <Search className="w-5 h-5" />
          </Button>

          {/* Theme toggle */}
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleTheme}
            className="text-white hover:bg-white/20"
          >
            {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </Button>

          {/* Notifications */}
          <Button
            variant="ghost"
            size="icon"
            className="text-white hover:bg-white/20 relative"
          >
            <Bell className="w-5 h-5" />
            <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></span>
          </Button>

          {/* User menu */}
          <Button
            variant="ghost"
            size="icon"
            className="text-white hover:bg-white/20"
          >
            <User className="w-5 h-5" />
          </Button>
        </div>
      </div>
    </header>
  );
};

export default Header;