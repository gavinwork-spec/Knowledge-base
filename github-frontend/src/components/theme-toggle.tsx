import * as React from 'react';
import { Moon, Sun, Monitor } from 'lucide-react';
import { useThemeStore } from '@/store/theme';
import { Button } from '@/components/ui/button';

export function ThemeToggle() {
  const { theme, setTheme } = useThemeStore();

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={() => {
        const currentThemeIndex = ['light', 'dark', 'system'].indexOf(theme);
        const nextTheme = ['light', 'dark', 'system'][(currentThemeIndex + 1) % 3];
        setTheme(nextTheme as 'light' | 'dark' | 'system');
      }}
      className="h-9 w-9"
    >
      <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
      <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
      <span className="sr-only">Toggle theme</span>
    </Button>
  );
}

export function ThemeToggleExpanded() {
  const { theme, setTheme } = useThemeStore();

  return (
    <div className="flex items-center space-x-2 rounded-lg border p-1">
      <Button
        variant={theme === 'light' ? 'default' : 'ghost'}
        size="sm"
        onClick={() => setTheme('light')}
        className="h-8 px-3"
      >
        <Sun className="h-4 w-4" />
        <span className="ml-2 hidden sm:inline">Light</span>
      </Button>
      <Button
        variant={theme === 'dark' ? 'default' : 'ghost'}
        size="sm"
        onClick={() => setTheme('dark')}
        className="h-8 px-3"
      >
        <Moon className="h-4 w-4" />
        <span className="ml-2 hidden sm:inline">Dark</span>
      </Button>
      <Button
        variant={theme === 'system' ? 'default' : 'ghost'}
        size="sm"
        onClick={() => setTheme('system')}
        className="h-8 px-3"
      >
        <Monitor className="h-4 w-4" />
        <span className="ml-2 hidden sm:inline">System</span>
      </Button>
    </div>
  );
}