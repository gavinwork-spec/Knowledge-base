'use client'

import * as React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Menu, X, Home, Search, Brain, BarChart3, Settings, HelpCircle, BookOpen, Layers, Zap } from 'lucide-react'
import { Header } from './header'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { useTheme } from '@/store/theme-store'

interface LayoutProps {
  children: React.ReactNode
  className?: string
}

interface SidebarItem {
  id: string
  label: string
  icon: React.ReactNode
  href?: string
  badge?: string | number
  active?: boolean
  onClick?: () => void
}

const Layout = React.forwardRef<HTMLDivElement, LayoutProps>(({ children, className }, ref) => {
  const [isSidebarOpen, setIsSidebarOpen] = React.useState(false)
  const [isMobile, setIsMobile] = React.useState(false)
  const { isDark } = useTheme()

  // Detect mobile screen size
  React.useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 1024)
      if (window.innerWidth >= 1024) {
        setIsSidebarOpen(false)
      }
    }

    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

  const sidebarItems: SidebarItem[] = [
    {
      id: 'home',
      label: 'Home',
      icon: <Home className="w-5 h-5" />,
      href: '/',
      active: true,
    },
    {
      id: 'search',
      label: 'Search',
      icon: <Search className="w-5 h-5" />,
      href: '/search',
      badge: 'New',
    },
    {
      id: 'ai-search',
      label: 'AI Search',
      icon: <Brain className="w-5 h-5" />,
      href: '/ai-search',
    },
    {
      id: 'hybrid',
      label: 'Hybrid Search',
      icon: <Layers className="w-5 h-5" />,
      href: '/hybrid',
    },
    {
      id: 'personalized',
      label: 'Personalized',
      icon: <Zap className="w-5 h-5" />,
      href: '/personalized',
    },
    {
      id: 'analytics',
      label: 'Analytics',
      icon: <BarChart3 className="w-5 h-5" />,
      href: '/analytics',
    },
    {
      id: 'knowledge',
      label: 'Knowledge Base',
      icon: <BookOpen className="w-5 h-5" />,
      href: '/knowledge',
    },
  ]

  const sidebarBottomItems: SidebarItem[] = [
    {
      id: 'help',
      label: 'Help & Support',
      icon: <HelpCircle className="w-5 h-5" />,
      href: '/help',
    },
    {
      id: 'settings',
      label: 'Settings',
      icon: <Settings className="w-5 h-5" />,
      href: '/settings',
    },
  ]

  const handleSearchClick = () => {
    // Open search modal or navigate to search page
    setIsSidebarOpen(false)
  }

  const handleSidebarItemClick = (item: SidebarItem) => {
    item.onClick?.()
    if (isMobile) {
      setIsSidebarOpen(false)
    }
  }

  return (
    <div ref={ref} className={cn('min-h-screen bg-background', className)}>
      {/* Overlay for mobile sidebar */}
      <AnimatePresence>
        {isSidebarOpen && isMobile && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-40 bg-black/50 lg:hidden"
            onClick={() => setIsSidebarOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* Header */}
      <Header
        onSearchClick={handleSearchClick}
        onMenuClick={() => setIsSidebarOpen(!isSidebarOpen)}
        isMenuOpen={isSidebarOpen}
      />

      <div className="flex">
        {/* Sidebar */}
        <AnimatePresence>
          {(isSidebarOpen || !isMobile) && (
            <motion.aside
              initial={isMobile ? { x: -300 } : { x: 0 }}
              animate={{ x: 0 }}
              exit={isMobile ? { x: -300 } : { x: 0 }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className={cn(
                'fixed lg:static inset-y-0 left-0 z-30',
                'w-64 bg-card border-r border-border',
                'flex flex-col',
                isMobile && 'shadow-medium'
              )}
            >
              {/* Sidebar header */}
              <div className="flex items-center justify-between p-4 border-b border-border">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-gradient-to-br from-primary to-primary/60 rounded-lg flex items-center justify-center">
                    <span className="text-lg font-bold text-primary-foreground">K</span>
                  </div>
                  <div>
                    <h2 className="font-semibold">Knowledge Hub</h2>
                    <p className="text-xs text-muted-foreground">Smart Search Platform</p>
                  </div>
                </div>
                {isMobile && (
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setIsSidebarOpen(false)}
                  >
                    <X className="w-4 h-4" />
                  </Button>
                )}
              </div>

              {/* Navigation */}
              <nav className="flex-1 p-4 space-y-6">
                {/* Main navigation */}
                <div className="space-y-1">
                  <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider px-2">
                    Main
                  </h3>
                  {sidebarItems.map((item) => (
                    <Button
                      key={item.id}
                      variant={item.active ? 'secondary' : 'ghost'}
                      className={cn(
                        'w-full justify-start h-10',
                        item.active && 'bg-primary text-primary-foreground hover:bg-primary/90'
                      )}
                      onClick={() => handleSidebarItemClick(item)}
                    >
                      {item.icon}
                      <span className="ml-3">{item.label}</span>
                      {item.badge && (
                        <span className={cn(
                          'ml-auto text-xs px-2 py-1 rounded-full',
                          item.active
                            ? 'bg-primary-foreground/20 text-primary-foreground'
                            : 'bg-primary/10 text-primary'
                        )}>
                          {item.badge}
                        </span>
                      )}
                    </Button>
                  ))}
                </div>

                {/* Secondary navigation */}
                <div className="space-y-1">
                  <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider px-2">
                    Support
                  </h3>
                  {sidebarBottomItems.map((item) => (
                    <Button
                      key={item.id}
                      variant="ghost"
                      className="w-full justify-start h-10"
                      onClick={() => handleSidebarItemClick(item)}
                    >
                      {item.icon}
                      <span className="ml-3">{item.label}</span>
                    </Button>
                  ))}
                </div>
              </nav>

              {/* Sidebar footer */}
              <div className="p-4 border-t border-border">
                <div className="bg-gradient-to-r from-primary/10 to-primary/5 rounded-lg p-3">
                  <div className="flex items-center space-x-2 mb-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full" />
                    <span className="text-xs font-medium">System Status</span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    All systems operational
                  </p>
                </div>
              </div>
            </motion.aside>
          )}
        </AnimatePresence>

        {/* Main content */}
        <main className="flex-1 min-h-[calc(100vh-4rem)]">
          <div className="container mx-auto px-4 py-6">
            {children}
          </div>
        </main>
      </div>

      {/* Mobile search trigger (floating) */}
      {isMobile && (
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="fixed bottom-6 right-6 z-30 lg:hidden"
        >
          <Button
            size="lg"
            className="h-14 w-14 rounded-full shadow-medium"
            onClick={handleSearchClick}
          >
            <Search className="w-6 h-6" />
          </Button>
        </motion.div>
      )}
    </div>
  )
})
Layout.displayName = 'Layout'

export { Layout }