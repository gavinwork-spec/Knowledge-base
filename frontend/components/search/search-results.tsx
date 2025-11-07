'use client'

import * as React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, ExternalLink, Clock, TrendingUp, Star, ChevronRight, BookOpen, FileText, Hash } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { LoadingSkeleton } from '@/components/ui/loading'
import { cn, formatRelativeTime, highlightText } from '@/lib/utils'
import { SearchResult, ExpertiseLevel } from '@/types'

interface SearchResultsProps {
  results: SearchResult[]
  isLoading?: boolean
  query?: string
  onResultClick?: (result: SearchResult) => void
  className?: string
  showExplanations?: boolean
  showMetadata?: boolean
  maxResults?: number
}

const SearchResults = React.forwardRef<HTMLDivElement, SearchResultsProps>(
  ({
    results,
    isLoading = false,
    query,
    onResultClick,
    className,
    showExplanations = true,
    showMetadata = true,
    maxResults,
  }, ref) => {
    const displayResults = maxResults ? results.slice(0, maxResults) : results

    const getExpertiseIcon = (level?: ExpertiseLevel) => {
      switch (level) {
        case 'beginner':
          return '🟢'
        case 'intermediate':
          return '🟡'
        case 'advanced':
          return '🟠'
        case 'expert':
          return '🔴'
        default:
          return null
      }
    }

    const getExpertiseColor = (level?: ExpertiseLevel) => {
      switch (level) {
        case 'beginner':
          return 'expertise-beginner'
        case 'intermediate':
          return 'expertise-intermediate'
        case 'advanced':
          return 'expertise-advanced'
        case 'expert':
          return 'bg-red-500 text-white'
        default:
          return 'bg-gray-500 text-white'
      }
    }

    const getTypeIcon = (type: string) => {
      switch (type.toLowerCase()) {
        case 'document':
        case 'pdf':
          return <FileText className="w-4 h-4" />
        case 'book':
        case 'article':
          return <BookOpen className="w-4 h-4" />
        case 'tag':
        case 'category':
          return <Hash className="w-4 h-4" />
        default:
          return <Search className="w-4 h-4" />
      }
    }

    const getResultTypeColor = (type: string) => {
      switch (type.toLowerCase()) {
        case 'semantic':
          return 'text-blue-500'
        case 'keyword':
          return 'text-green-500'
        case 'graph':
          return 'text-purple-500'
        default:
          return 'text-muted-foreground'
      }
    }

    if (isLoading) {
      return (
        <div ref={ref} className={cn('space-y-4', className)}>
          {Array.from({ length: 5 }).map((_, index) => (
            <Card key={index} className="p-6">
              <div className="space-y-3">
                <LoadingSkeleton width="60%" height="20px" />
                <LoadingSkeleton width="100%" height="16px" />
                <LoadingSkeleton width="100%" height="16px" />
                <LoadingSkeleton width="80%" height="16px" />
                <div className="flex items-center space-x-2 pt-2">
                  <LoadingSkeleton width="80px" height="24px" />
                  <LoadingSkeleton width="60px" height="24px" />
                </div>
              </div>
            </Card>
          ))}
        </div>
      )
    }

    if (results.length === 0 && query) {
      return (
        <div ref={ref} className={cn('text-center py-12', className)}>
          <div className="max-w-md mx-auto">
            <Search className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-xl font-semibold mb-2">No results found</h3>
            <p className="text-muted-foreground mb-6">
              We couldn't find any results for "{query}". Try different keywords or check your spelling.
            </p>
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>Suggestions:</p>
              <ul className="space-y-1">
                <li>• Try more general terms</li>
                <li>• Check for typos</li>
                <li>• Use different keywords</li>
                <li>• Try advanced search operators</li>
              </ul>
            </div>
          </div>
        </div>
      )
    }

    return (
      <div ref={ref} className={cn('space-y-4', className)}>
        <AnimatePresence mode="wait">
          {displayResults.map((result, index) => (
            <motion.div
              key={result.documentId}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{
                duration: 0.3,
                delay: index * 0.1,
                ease: "easeOut"
              }}
            >
              <Card
                className={cn(
                  'group cursor-pointer transition-all duration-200',
                  'hover:shadow-medium hover:-translate-y-1',
                  'border-2 border-transparent hover:border-primary/20'
                )}
                onClick={() => onResultClick?.(result)}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-2">
                        {getTypeIcon(result.searchType)}
                        <span className={cn('text-xs font-medium', getResultTypeColor(result.searchType))}>
                          {result.searchType}
                        </span>
                        {result.relevanceScore && (
                          <Badge variant="secondary" className="text-xs">
                            {Math.round(result.relevanceScore * 100)}% match
                          </Badge>
                        )}
                      </div>
                      <CardTitle
                        className={cn(
                          'text-lg line-clamp-2 group-hover:text-primary transition-colors',
                          query && 'font-semibold'
                        )}
                      >
                        {query ? (
                          <span
                            dangerouslySetInnerHTML={{
                              __html: highlightText(result.title, [query], 'search-highlight')
                            }}
                          />
                        ) : (
                          result.title
                        )}
                      </CardTitle>
                    </div>
                    <div className="flex items-center space-x-2 flex-shrink-0">
                      {result.expertiseLevel && (
                        <Badge className={cn('text-xs', getExpertiseColor(result.expertiseLevel))}>
                          {getExpertiseIcon(result.expertiseLevel)} {result.expertiseLevel}
                        </Badge>
                      )}
                      <div className="flex items-center space-x-1 text-xs text-muted-foreground">
                        <Star className="w-3 h-3 fill-current" />
                        <span>{(result.score * 100).toFixed(0)}</span>
                      </div>
                    </div>
                  </div>
                </CardHeader>

                <CardContent className="pt-0">
                  <CardDescription
                    className={cn(
                      'text-sm line-clamp-3 mb-4',
                      query && 'font-medium'
                    )}
                  >
                    {query && result.highlightedContent ? (
                      <span
                        dangerouslySetInnerHTML={{
                          __html: highlightText(result.highlightedContent, [query], 'search-highlight')
                        }}
                      />
                    ) : (
                      result.content
                    )}
                  </CardDescription>

                  {showMetadata && result.metadata && (
                    <div className="flex flex-wrap items-center gap-4 text-xs text-muted-foreground mb-4">
                      {result.metadata.author && (
                        <div className="flex items-center space-x-1">
                          <span>Author:</span>
                          <span className="font-medium">{result.metadata.author}</span>
                        </div>
                      )}
                      {result.metadata.category && (
                        <Badge variant="outline" className="text-xs">
                          {result.metadata.category}
                        </Badge>
                      )}
                      {result.metadata.wordCount && (
                        <div className="flex items-center space-x-1">
                          <span>{result.metadata.wordCount} words</span>
                        </div>
                      )}
                      {result.timestamp && (
                        <div className="flex items-center space-x-1">
                          <Clock className="w-3 h-3" />
                          <span>{formatRelativeTime(result.timestamp)}</span>
                        </div>
                      )}
                    </div>
                  )}

                  {showExplanations && result.explanation && (
                    <div className="bg-muted/50 rounded-md p-3 mb-4">
                      <div className="flex items-center space-x-2 mb-1">
                        <TrendingUp className="w-4 h-4 text-primary" />
                        <span className="text-sm font-medium text-primary">Why this result?</span>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        {result.explanation}
                      </p>
                    </div>
                  )}

                  <div className="flex items-center justify-between pt-2 border-t">
                    <div className="flex items-center space-x-4 text-xs text-muted-foreground">
                      <span>ID: {result.documentId}</span>
                      {result.metadata.clickCount !== undefined && (
                        <div className="flex items-center space-x-1">
                          <TrendingUp className="w-3 h-3" />
                          <span>{result.metadata.clickCount} views</span>
                        </div>
                      )}
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-8 px-3 text-xs"
                      onClick={(e) => {
                        e.stopPropagation()
                        onResultClick?.(result)
                      }}
                    >
                      View
                      <ChevronRight className="w-3 h-3 ml-1" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </AnimatePresence>

        {maxResults && results.length > maxResults && (
          <div className="text-center pt-6">
            <Button
              variant="outline"
              onClick={() => {
                // Handle showing all results
              }}
            >
              Show all {results.length} results
              <ChevronRight className="w-4 h-4 ml-2" />
            </Button>
          </div>
        )}
      </div>
    )
  }
)
SearchResults.displayName = 'SearchResults'

export { SearchResults }