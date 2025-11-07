import React, { createContext, useContext, useEffect, ReactNode } from 'react';
import { useThemeStore } from '@/store/theme';

interface ThemeContextType {
  theme: string;
  resolvedTheme: 'light' | 'dark';
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

interface ThemeProviderProps {
  children: ReactNode;
}

export function ThemeProvider({ children }: ThemeProviderProps) {
  const { theme, resolvedTheme, resolveTheme } = useThemeStore();

  useEffect(() => {
    resolveTheme();
  }, [resolveTheme]);

  return (
    <ThemeContext.Provider value={{ theme, resolvedTheme }}>
      <div className={resolvedTheme}>
        {children}
      </div>
    </ThemeContext.Provider>
  );
}

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};