'use client'

import * as React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, X, Command, Sparkles, History, TrendingUp } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { LoadingSpinner } from '@/components/ui/loading'
import { cn } from '@/lib/utils'
import { SearchSuggestion, SearchStrategy } from '@/types'

interface SearchInputProps {
  value: string
  onChange: (value: string) => void
  onSubmit: (query: string) => void
  suggestions?: SearchSuggestion[]
  isLoading?: boolean
  placeholder?: string
  className?: string
  showSuggestions?: boolean
  onSuggestionSelect?: (suggestion: SearchSuggestion) => void
  recentQueries?: string[]
  strategy?: SearchStrategy
  onStrategyChange?: (strategy: SearchStrategy) => void
  disabled?: boolean
}

const SearchInput = React.forwardRef<HTMLInputElement, SearchInputProps>(
  ({
    value,
    onChange,
    onSubmit,
    suggestions = [],
    isLoading = false,
    placeholder = 'Search knowledge base...',
    className,
    showSuggestions = true,
    onSuggestionSelect,
    recentQueries = [],
    strategy = 'unified',
    onStrategyChange,
    disabled = false,
  }, ref) => {
    const [isFocused, setIsFocused] = React.useState(false)
    const [selectedIndex, setSelectedIndex] = React.useState(-1)
    const [showAllSuggestions, setShowAllSuggestions] = React.useState(false)
    const inputRef = React.useRef<HTMLInputElement>(null)
    const suggestionsRef = React.useRef<HTMLDivElement>(null)

    const allSuggestions = React.useMemo(() => {
      const filteredSuggestions = suggestions.filter(s =>
        s.text.toLowerCase().includes(value.toLowerCase())
      )

      const uniqueRecentQueries = recentQueries
        .filter(q => q.toLowerCase().includes(value.toLowerCase()))
        .slice(0, 3)
        .map(query => ({
          id: `recent-${query}`,
          text: query,
          type: 'history' as const,
        }))

      return [...uniqueRecentQueries, ...filteredSuggestions].slice(0, 8)
    }, [suggestions, value, recentQueries])

    React.useEffect(() => {
      setSelectedIndex(-1)
    }, [value])

    React.useEffect(() => {
      const handleClickOutside = (event: MouseEvent) => {
        if (suggestionsRef.current && !suggestionsRef.current.contains(event.target as Node)) {
          setIsFocused(false)
        }
      }

      document.addEventListener('mousedown', handleClickOutside)
      return () => document.removeEventListener('mousedown', handleClickOutside)
    }, [])

    const handleSubmit = (e: React.FormEvent) => {
      e.preventDefault()
      if (value.trim() && !disabled) {
        onSubmit(value.trim())
        setIsFocused(false)
      }
    }

    const handleKeyDown = (e: React.KeyboardEvent) => {
      if (!showSuggestions || allSuggestions.length === 0) return

      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault()
          setSelectedIndex(prev => (prev + 1) % allSuggestions.length)
          break
        case 'ArrowUp':
          e.preventDefault()
          setSelectedIndex(prev => prev <= 0 ? allSuggestions.length - 1 : prev - 1)
          break
        case 'Enter':
          e.preventDefault()
          if (selectedIndex >= 0) {
            const suggestion = allSuggestions[selectedIndex]
            onSuggestionSelect?.(suggestion)
            onChange(suggestion.text)
            setIsFocused(false)
          } else {
            handleSubmit(e)
          }
          break
        case 'Escape':
          setIsFocused(false)
          setSelectedIndex(-1)
          break
      }
    }

    const handleSuggestionClick = (suggestion: SearchSuggestion) => {
      onSuggestionSelect?.(suggestion)
      onChange(suggestion.text)
      setIsFocused(false)
      setSelectedIndex(-1)
    }

    const handleClear = () => {
      onChange('')
      inputRef.current?.focus()
    }

    const getSuggestionIcon = (type: SearchSuggestion['type']) => {
      switch (type) {
        case 'history':
          return <History className="w-4 h-4" />
        case 'popular':
          return <TrendingUp className="w-4 h-4" />
        case 'semantic':
          return <Sparkles className="w-4 h-4" />
        default:
          return <Search className="w-4 h-4" />
      }
    }

    const getSuggestionColor = (type: SearchSuggestion['type']) => {
      switch (type) {
        case 'history':
          return 'text-muted-foreground'
        case 'popular':
          return 'text-blue-500'
        case 'semantic':
          return 'text-purple-500'
        default:
          return 'text-muted-foreground'
      }
    }

    return (
      <div className={cn('relative w-full max-w-2xl', className)}>
        <form onSubmit={handleSubmit} className="relative">
          <div className="relative">
            <div className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">
              {isLoading ? (
                <LoadingSpinner size="sm" />
              ) : (
                <Search className="w-5 h-5" />
              )}
            </div>

            <Input
              ref={inputRef}
              type="text"
              value={value}
              onChange={(e) => onChange(e.target.value)}
              onFocus={() => setIsFocused(true)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              disabled={disabled}
              className={cn(
                'pl-10 pr-20 h-12 text-base',
                'border-2 focus:border-primary transition-all duration-200',
                'placeholder:text-muted-foreground/70',
                disabled:opacity-50
              )}
              autoComplete="off"
              spellCheck={false}
            />

            <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center space-x-1">
              {value && (
                <Button
                  type="button"
                  variant="ghost"
                  size="icon-sm"
                  onClick={handleClear}
                  className="h-8 w-8 text-muted-foreground hover:text-foreground"
                  disabled={disabled}
                >
                  <X className="w-4 h-4" />
                </Button>
              )}

              <Button
                type="submit"
                size="sm"
                disabled={!value.trim() || disabled}
                className="h-8 px-3"
              >
                Search
              </Button>
            </div>
          </div>

          {/* Keyboard shortcut hint */}
          <div className="absolute right-2 -bottom-6 text-xs text-muted-foreground">
            <kbd className="px-1.5 py-0.5 bg-muted rounded text-xs">⌘K</kbd>
          </div>
        </form>

        {/* Suggestions dropdown */}
        <AnimatePresence>
          {showSuggestions && isFocused && allSuggestions.length > 0 && (
            <motion.div
              ref={suggestionsRef}
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
              className={cn(
                'absolute top-full left-0 right-0 mt-2',
                'bg-popover border border-border rounded-lg shadow-medium',
                'max-h-80 overflow-y-auto',
                'z-50'
              )}
            >
              <div className="p-1">
                {allSuggestions.map((suggestion, index) => (
                  <motion.button
                    key={suggestion.id}
                    type="button"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className={cn(
                      'w-full flex items-center space-x-3 px-3 py-2.5 rounded-md',
                      'text-left hover:bg-accent hover:text-accent-foreground',
                      'transition-colors duration-150',
                      selectedIndex === index && 'bg-accent text-accent-foreground'
                    )}
                    onClick={() => handleSuggestionClick(suggestion)}
                  >
                    <div className={cn('flex-shrink-0', getSuggestionColor(suggestion.type))}>
                      {getSuggestionIcon(suggestion.type)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="font-medium truncate">{suggestion.text}</div>
                      {suggestion.category && (
                        <div className="text-xs text-muted-foreground capitalize">
                          {suggestion.category}
                        </div>
                      )}
                    </div>
                    {suggestion.score && (
                      <div className="flex-shrink-0 text-xs text-muted-foreground">
                        {Math.round(suggestion.score * 100)}%
                      </div>
                    )}
                  </motion.button>
                ))}
              </div>

              {allSuggestions.length === 0 && value && (
                <div className="p-4 text-center text-muted-foreground text-sm">
                  No suggestions found for "{value}"
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    )
  }
)
SearchInput.displayName = 'SearchInput'

export { SearchInput }