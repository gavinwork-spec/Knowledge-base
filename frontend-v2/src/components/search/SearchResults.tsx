import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  FileText,
  Clock,
  Eye,
  Download,
  Share,
  ExternalLink,
  Filter,
  ChevronDown,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Target,
  Shield,
  Wrench
} from 'lucide-react';
import { Button } from '../ui/button';
import { SearchResult, SearchFilters } from '../../types';
import {
  getExpertiseColorClass,
  getExpertiseColorClass as getExpertiseBadgeClass,
  formatDate,
  calculateRelevancePercentage,
  getComplianceStandardName
} from '@lib/utils';
import { cn } from '@lib/utils';

interface SearchResultsProps {
  results: SearchResult[];
  totalResults: number;
  searchTime: number;
  isLoading?: boolean;
  query?: string;
  onResultClick?: (result: SearchResult) => void;
  filters?: SearchFilters;
  onFilterChange?: (filters: SearchFilters) => void;
  className?: string;
}

const SearchResults: React.FC<SearchResultsProps> = ({
  results,
  totalResults,
  searchTime,
  isLoading = false,
  query,
  onResultClick,
  filters,
  onFilterChange,
  className
}) => {
  const [sortBy, setSortBy] = useState<'relevance' | 'date' | 'title'>('relevance');
  const [expandedResults, setExpandedResults] = useState<Set<string>>(new Set());

  // Document type icons
  const getDocumentIcon = (type: string) => {
    const icons = {
      technical_manual: <FileText className="w-4 h-4" />,
      safety_procedure: <Shield className="w-4 h-4" />,
      quality_specification: <Target className="w-4 h-4" />,
      equipment_manual: <Wrench className="w-4 h-4" />,
      general: <FileText className="w-4 h-4" />
    };
    return icons[type as keyof typeof icons] || icons.general;
  };

  // Sort results
  const sortedResults = React.useMemo(() => {
    const sorted = [...results];
    switch (sortBy) {
      case 'relevance':
        return sorted.sort((a, b) => b.relevance_score - a.relevance_score);
      case 'date':
        return sorted.sort((a, b) => new Date(b.metadata.created_at).getTime() - new Date(a.metadata.created_at).getTime());
      case 'title':
        return sorted.sort((a, b) => a.title.localeCompare(b.title));
      default:
        return sorted;
    }
  }, [results, sortBy]);

  // Toggle result expansion
  const toggleExpanded = (resultId: string) => {
    const newExpanded = new Set(expandedResults);
    if (newExpanded.has(resultId)) {
      newExpanded.delete(resultId);
    } else {
      newExpanded.add(resultId);
    }
    setExpandedResults(newExpanded);
  };

  // Render expertise badge
  const ExpertiseBadge: React.FC<{ level: string }> = ({ level }) => {
    return (
      <span className={cn(
        "inline-flex items-center px-2 py-1 rounded text-xs font-semibold",
        getExpertiseBadgeClass(level)
      )}>
        {level.charAt(0).toUpperCase() + level.slice(1)}
      </span>
    );
  };

  // Render compliance badges
  const ComplianceBadges: React.FC<{ standards?: string[] }> = ({ standards }) => {
    if (!standards || standards.length === 0) return null;

    return (
      <div className="flex flex-wrap gap-1">
        {standards.slice(0, 3).map((standard) => (
          <span
            key={standard}
            className="inline-flex items-center px-2 py-1 rounded text-xs bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400"
          >
            {getComplianceStandardName(standard)}
          </span>
        ))}
        {standards.length > 3 && (
          <span className="inline-flex items-center px-2 py-1 rounded text-xs bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400">
            +{standards.length - 3} more
          </span>
        )}
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        {[...Array(3)].map((_, index) => (
          <div key={index} className="manufacturing-card animate-pulse">
            <div className="space-y-3">
              <div className="h-4 bg-muted rounded w-3/4"></div>
              <div className="h-3 bg-muted rounded w-1/2"></div>
              <div className="h-20 bg-muted rounded"></div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className={cn("space-y-4", className)}>
      {/* Results Header */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-muted-foreground">
          {totalResults > 0 ? (
            <>
              Found <span className="font-semibold text-foreground">{totalResults.toLocaleString()}</span> results
              {query && <span> for "<span className="font-semibold">{query}</span>"</span>}
              <span className="ml-2">in {searchTime.toFixed(2)}s</span>
            </>
          ) : (
            'No results found'
          )}
        </div>

        {results.length > 1 && (
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Sort by:</span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSortBy(sortBy === 'relevance' ? 'date' : sortBy === 'date' ? 'title' : 'relevance')}
            >
              {sortBy === 'relevance' && <TrendingUp className="w-3 h-3 mr-1" />}
              {sortBy === 'date' && <Clock className="w-3 h-3 mr-1" />}
              {sortBy === 'title' && <FileText className="w-3 h-3 mr-1" />}
              {sortBy.charAt(0).toUpperCase() + sortBy.slice(1)}
              <ChevronDown className="w-3 h-3 ml-1" />
            </Button>
          </div>
        )}
      </div>

      {/* Results List */}
      <AnimatePresence mode="wait">
        {sortedResults.length === 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="text-center py-12"
          >
            <div className="flex flex-col items-center gap-4">
              <AlertCircle className="w-12 h-12 text-muted-foreground" />
              <div>
                <h3 className="text-lg font-semibold mb-2">No results found</h3>
                <p className="text-muted-foreground mb-4">
                  {query
                    ? `No results found for "${query}". Try different keywords or filters.`
                    : 'Enter a search query to find manufacturing knowledge and procedures.'
                  }
                </p>
                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">Try searching for:</p>
                  <div className="flex flex-wrap gap-2 justify-center">
                    {[
                      'CNC milling safety',
                      'Quality inspection',
                      'Equipment maintenance',
                      'ISO 9001 procedures'
                    ].map((suggestion) => (
                      <Button
                        key={suggestion}
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          if (onResultClick) {
                            // This would typically trigger a new search
                            console.log('Search for:', suggestion);
                          }
                        }}
                      >
                        {suggestion}
                      </Button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="space-y-4"
          >
            {sortedResults.map((result, index) => (
              <motion.div
                key={result.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                className={cn(
                  "search-result-card cursor-pointer group",
                  expandedResults.has(result.id) && "ring-2 ring-primary ring-offset-2"
                )}
                onClick={() => {
                  if (onResultClick) {
                    onResultClick(result);
                  }
                  toggleExpanded(result.id);
                }}
              >
                <div className="flex items-start gap-4">
                  {/* Document Icon */}
                  <div className="flex-shrink-0 mt-1">
                    <div className={cn(
                      "p-2 rounded-lg",
                      result.document_type === 'safety_procedure' && "bg-red-100 text-red-600 dark:bg-red-900/20 dark:text-red-400",
                      result.document_type === 'quality_specification' && "bg-green-100 text-green-600 dark:bg-green-900/20 dark:text-green-400",
                      result.document_type === 'technical_manual' && "bg-blue-100 text-blue-600 dark:bg-blue-900/20 dark:text-blue-400",
                      result.document_type === 'equipment_manual' && "bg-orange-100 text-orange-600 dark:bg-orange-900/20 dark:text-orange-400",
                      result.document_type === 'general' && "bg-gray-100 text-gray-600 dark:bg-gray-900/20 dark:text-gray-400"
                    )}>
                      {getDocumentIcon(result.document_type)}
                    </div>
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    {/* Title */}
                    <h3 className="text-lg font-semibold text-foreground mb-1 group-hover:text-primary transition-colors">
                      {result.title}
                    </h3>

                    {/* Excerpt */}
                    <p className="text-sm text-muted-foreground mb-3 line-clamp-2">
                      {result.excerpt}
                    </p>

                    {/* Metadata */}
                    <div className="flex flex-wrap items-center gap-4 text-xs text-muted-foreground mb-3">
                      {/* Expertise Level */}
                      <ExpertiseBadge level={result.expertise_level} />

                      {/* Relevance Score */}
                      <div className="flex items-center gap-1">
                        <span>Relevance:</span>
                        <span className={cn(
                          "font-medium",
                          result.relevance_score > 0.8 ? "text-green-600" :
                          result.relevance_score > 0.6 ? "text-yellow-600" :
                          "text-red-600"
                        )}>
                          {calculateRelevancePercentage(result.relevance_score)}%
                        </span>
                      </div>

                      {/* Date */}
                      <div className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {formatDate(result.metadata.created_at)}
                      </div>

                      {/* Author */}
                      {result.metadata.author && (
                        <span>by {result.metadata.author}</span>
                      )}
                    </div>

                    {/* Compliance Standards */}
                    <ComplianceBadges standards={result.metadata.compliance_standards} />

                    {/* Manufacturing Context */}
                    {(result.metadata.equipment_type || result.metadata.process_type) && (
                      <div className="flex flex-wrap gap-2 mt-2">
                        {result.metadata.equipment_type && (
                          <span className="inline-flex items-center px-2 py-1 rounded text-xs bg-purple-100 text-purple-800 dark:bg-purple-900/20 dark:text-purple-400">
                            <Wrench className="w-3 h-3 mr-1" />
                            {result.metadata.equipment_type.replace('_', ' ')}
                          </span>
                        )}
                        {result.metadata.process_type && (
                          <span className="inline-flex items-center px-2 py-1 rounded text-xs bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400">
                            <Target className="w-3 h-3 mr-1" />
                            {result.metadata.process_type.replace('_', ' ')}
                          </span>
                        )}
                      </div>
                    )}

                    {/* Expanded Content */}
                    <AnimatePresence>
                      {expandedResults.has(result.id) && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          className="overflow-hidden mt-3 pt-3 border-t border-border"
                        >
                          {/* Full Content Preview */}
                          <div className="text-sm text-muted-foreground mb-3">
                            {result.content.length > 500
                              ? result.content.substring(0, 500) + '...'
                              : result.content}
                          </div>

                          {/* Tags */}
                          {result.metadata.tags && result.metadata.tags.length > 0 && (
                            <div className="flex flex-wrap gap-1 mb-3">
                              {result.metadata.tags.map((tag, tagIndex) => (
                                <span
                                  key={tagIndex}
                                  className="inline-flex items-center px-2 py-1 rounded text-xs bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400"
                                >
                                  #{tag}
                                </span>
                              ))}
                            </div>
                          )}

                          {/* Actions */}
                          <div className="flex items-center gap-2 pt-2">
                            <Button variant="outline" size="sm">
                              <Eye className="w-3 h-3 mr-1" />
                              View
                            </Button>
                            {result.url && (
                              <Button variant="outline" size="sm">
                                <ExternalLink className="w-3 h-3 mr-1" />
                                Open
                              </Button>
                            )}
                            <Button variant="outline" size="sm">
                              <Download className="w-3 h-3 mr-1" />
                              Download
                            </Button>
                            <Button variant="outline" size="sm">
                              <Share className="w-3 h-3 mr-1" />
                              Share
                            </Button>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>

                    {/* Expansion Indicator */}
                    <div className="flex items-center gap-1 text-xs text-muted-foreground mt-2">
                      <span>
                        {expandedResults.has(result.id) ? 'Click to collapse' : 'Click to expand'}
                      </span>
                      <ChevronDown
                        className={cn(
                          "w-3 h-3 transition-transform",
                          expandedResults.has(result.id) && "rotate-180"
                        )}
                      />
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default SearchResults;