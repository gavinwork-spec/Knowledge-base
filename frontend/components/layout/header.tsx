'use client'

import * as React from 'react'
import { motion } from 'framer-motion'
import { Sun, Moon, Monitor, Search, Settings, User, Bell, Menu, X, Github, ExternalLink } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useTheme } from '@/store/theme-store'
import { cn } from '@/lib/utils'

interface HeaderProps {
  className?: string
  onSearchClick?: () => void
  onMenuClick?: () => void
  isMenuOpen?: boolean
  showSearch?: boolean
}

const Header = React.forwardRef<HTMLDivElement, HeaderProps>(
  ({ className, onSearchClick, onMenuClick, isMenuOpen, showSearch = true }, ref) => {
    const { theme, setTheme, toggleTheme } = useTheme()
    const [isUserMenuOpen, setIsUserMenuOpen] = React.useState(false)
    const [isNotificationMenuOpen, setIsNotificationMenuOpen] = React.useState(false)

    const handleThemeChange = (newTheme: 'light' | 'dark' | 'system') => {
      setTheme(newTheme)
    }

    const getThemeIcon = () => {
      switch (theme) {
        case 'light':
          return <Sun className="w-4 h-4" />
        case 'dark':
          return <Moon className="w-4 h-4" />
        case 'system':
          return <Monitor className="w-4 h-4" />
        default:
          return <Sun className="w-4 h-4" />
      }
    }

    const handleKeyDown = (e: React.KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        onSearchClick?.()
      }
    }

    React.useEffect(() => {
      document.addEventListener('keydown', handleKeyDown)
      return () => document.removeEventListener('keydown', handleKeyDown)
    }, [onSearchClick])

    return (
      <header
        ref={ref}
        className={cn(
          'sticky top-0 z-40 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60',
          className
        )}
      >
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          {/* Left side - Logo and navigation */}
          <div className="flex items-center space-x-4">
            <Button
              variant="ghost"
              size="icon"
              onClick={onMenuClick}
              className="lg:hidden"
            >
              {isMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </Button>

            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center space-x-3"
            >
              <div className="w-8 h-8 bg-gradient-to-br from-primary to-primary/60 rounded-lg flex items-center justify-center">
                <span className="text-lg font-bold text-primary-foreground">K</span>
              </div>
              <div className="hidden sm:block">
                <h1 className="text-lg font-semibold">Knowledge Hub</h1>
                <p className="text-xs text-muted-foreground">AI-Powered Search</p>
              </div>
            </motion.div>
          </div>

          {/* Center - Search bar (desktop only) */}
          {showSearch && (
            <div className="hidden md:flex flex-1 max-w-lg mx-8">
              <Button
                variant="outline"
                className="w-full justify-between text-muted-foreground h-10"
                onClick={onSearchClick}
              >
                <span className="flex items-center space-x-2">
                  <Search className="w-4 h-4" />
                  <span>Search knowledge base...</span>
                </span>
                <kbd className="px-2 py-1 text-xs bg-muted rounded">
                  ⌘K
                </kbd>
              </Button>
            </div>
          )}

          {/* Right side - Actions */}
          <div className="flex items-center space-x-2">
            {/* Theme selector */}
            <div className="relative group">
              <Button
                variant="ghost"
                size="icon"
                onClick={toggleTheme}
                className="h-9 w-9"
                title={`Current theme: ${theme}`}
              >
                {getThemeIcon()}
              </Button>

              {/* Theme dropdown */}
              <div className="absolute right-0 top-full mt-2 opacity-0 group-hover:opacity-100 pointer-events-none group-hover:pointer-events-auto transition-opacity duration-200">
                <div className="glass border border-border rounded-lg shadow-medium p-1 min-w-[140px]">
                  <Button
                    variant="ghost"
                    size="sm"
                    className={cn(
                      'w-full justify-start h-8 px-2',
                      theme === 'light' && 'bg-accent'
                    )}
                    onClick={() => handleThemeChange('light')}
                  >
                    <Sun className="w-4 h-4 mr-2" />
                    Light
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className={cn(
                      'w-full justify-start h-8 px-2',
                      theme === 'dark' && 'bg-accent'
                    )}
                    onClick={() => handleThemeChange('dark')}
                  >
                    <Moon className="w-4 h-4 mr-2" />
                    Dark
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className={cn(
                      'w-full justify-start h-8 px-2',
                      theme === 'system' && 'bg-accent'
                    )}
                    onClick={() => handleThemeChange('system')}
                  >
                    <Monitor className="w-4 h-4 mr-2" />
                    System
                  </Button>
                </div>
              </div>
            </div>

            {/* Notifications */}
            <Button
              variant="ghost"
              size="icon"
              className="h-9 w-9 relative"
              onClick={() => setIsNotificationMenuOpen(!isNotificationMenuOpen)}
            >
              <Bell className="w-4 h-4" />
              <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full" />
            </Button>

            {/* User menu */}
            <div className="relative">
              <Button
                variant="ghost"
                size="icon"
                className="h-9 w-9"
                onClick={() => setIsUserMenuOpen(!isUserMenuOpen)}
              >
                <User className="w-4 h-4" />
              </Button>

              {/* User dropdown */}
              <div className={cn(
                'absolute right-0 top-full mt-2 glass border border-border rounded-lg shadow-medium p-1 min-w-[180px]',
                'opacity-0 pointer-events-none transition-all duration-200',
                isUserMenuOpen && 'opacity-100 pointer-events-auto'
              )}>
                <Button variant="ghost" size="sm" className="w-full justify-start h-8 px-2">
                  <User className="w-4 h-4 mr-2" />
                  Profile
                </Button>
                <Button variant="ghost" size="sm" className="w-full justify-start h-8 px-2">
                  <Settings className="w-4 h-4 mr-2" />
                  Settings
                </Button>
                <hr className="my-1 border-border" />
                <Button variant="ghost" size="sm" className="w-full justify-start h-8 px-2">
                  <Github className="w-4 h-4 mr-2" />
                  GitHub
                </Button>
              </div>
            </div>

            {/* Mobile search button */}
            {showSearch && (
              <Button
                variant="ghost"
                size="icon"
                onClick={onSearchClick}
                className="md:hidden h-9 w-9"
              >
                <Search className="w-4 h-4" />
              </Button>
            )}
          </div>
        </div>
      </header>
    )
  }
)
Header.displayName = 'Header'

export { Header }