import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Search,
  Filter,
  X,
  ChevronDown,
  Sparkles,
  Brain,
  FileText,
  GitBranch,
  Target,
  Settings,
  Clock,
  TrendingUp
} from 'lucide-react';
import { Button } from '../ui/button';
import { cn } from '@lib/utils';
import { useSearchStore } from '../../stores';

interface AdvancedSearchProps {
  onSearch: (query: string, strategy: string, filters: any) => void;
  isLoading?: boolean;
  className?: string;
}

const AdvancedSearch: React.FC<AdvancedSearchProps> = ({
  onSearch,
  isLoading = false,
  className
}) => {
  const {
    query,
    setQuery,
    filters,
    setFilters,
    searchStrategy,
    setSearchStrategy,
    searchHistory,
    addToHistory,
    suggestions,
    setSuggestions
  } = useSearchStore();

  const [showFilters, setShowFilters] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [focusedIndex, setFocusedIndex] = useState(-1);

  // Search strategies with manufacturing context
  const searchStrategies = [
    {
      id: 'unified',
      name: 'Unified Search',
      description: 'Best overall results combining all methods',
      icon: Sparkles,
      color: 'from-blue-500 to-purple-600'
    },
    {
      id: 'semantic',
      name: 'Semantic Search',
      description: 'Understands meaning and context',
      icon: Brain,
      color: 'from-green-500 to-teal-600'
    },
    {
      id: 'keyword',
      name: 'Keyword Search',
      description: 'Exact text matching',
      icon: FileText,
      color: 'from-orange-500 to-red-600'
    },
    {
      id: 'graph',
      name: 'Knowledge Graph',
      description: 'Relationships and connections',
      icon: GitBranch,
      color: 'from-purple-500 to-pink-600'
    },
    {
      id: 'ai_enhanced',
      name: 'AI Enhanced',
      description: 'Advanced AI-powered search',
      icon: TrendingUp,
      color: 'from-indigo-500 to-blue-600'
    }
  ];

  // Manufacturing filter options
  const filterOptions = {
    document_type: [
      { id: 'technical_manual', label: 'Technical Manual', icon: '📚' },
      { id: 'safety_procedure', label: 'Safety Procedure', icon: '🛡️' },
      { id: 'quality_specification', label: 'Quality Specification', icon: '✅' },
      { id: 'equipment_manual', label: 'Equipment Manual', icon: '🔧' },
      { id: 'general', label: 'General', icon: '📄' }
    ],
    expertise_level: [
      { id: 'beginner', label: 'Beginner', color: 'text-green-600' },
      { id: 'intermediate', label: 'Intermediate', color: 'text-blue-600' },
      { id: 'advanced', label: 'Advanced', color: 'text-purple-600' },
      { id: 'expert', label: 'Expert', color: 'text-red-600' }
    ],
    equipment_type: [
      { id: 'cnc_milling', label: 'CNC Milling' },
      { id: 'cnc_turning', label: 'CNC Turning' },
      { id: 'grinding', label: 'Grinding' },
      { id: 'measurement', label: 'Measurement' },
      { id: 'assembly', label: 'Assembly' }
    ]
  };

  // Handle search submission
  const handleSearch = (searchQuery?: string) => {
    const finalQuery = searchQuery || query;
    if (finalQuery.trim()) {
      addToHistory(finalQuery.trim());
      onSearch(finalQuery.trim(), searchStrategy, filters);
      setShowSuggestions(false);
      setFocusedIndex(-1);
    }
  };

  // Handle input changes
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);
    setShowSuggestions(value.length > 0);
    setFocusedIndex(-1);

    // In a real implementation, this would fetch suggestions from the API
    if (value.length > 2) {
      // Mock suggestions for demo
      const mockSuggestions = [
        'CNC milling safety procedures',
        'Quality inspection guidelines',
        'Equipment maintenance schedule',
        'ISO 9001 requirements'
      ].filter(s => s.toLowerCase().includes(value.toLowerCase()));
      setSuggestions(mockSuggestions.slice(0, 5));
    } else {
      setSuggestions([]);
    }
  };

  // Handle keyboard navigation
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (showSuggestions) {
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setFocusedIndex(prev => Math.min(prev + 1, suggestions.length - 1));
          break;
        case 'ArrowUp':
          e.preventDefault();
          setFocusedIndex(prev => Math.max(prev - 1, -1));
          break;
        case 'Enter':
          e.preventDefault();
          if (focusedIndex >= 0 && focusedIndex < suggestions.length) {
            handleSearch(suggestions[focusedIndex]);
          } else {
            handleSearch();
          }
          break;
        case 'Escape':
          setShowSuggestions(false);
          setFocusedIndex(-1);
          break;
      }
    }
  };

  // Toggle filter
  const toggleFilter = (category: string, value: string) => {
    const currentFilters = { ...filters };
    if (!currentFilters[category]) {
      currentFilters[category] = [];
    }

    const index = currentFilters[category].indexOf(value);
    if (index > -1) {
      currentFilters[category].splice(index, 1);
    } else {
      currentFilters[category].push(value);
    }

    setFilters(currentFilters);
  };

  // Clear all filters
  const clearFilters = () => {
    setFilters({});
  };

  // Get active filter count
  const getActiveFilterCount = () => {
    return Object.values(filters).reduce((total, filterArray) => total + (Array.isArray(filterArray) ? filterArray.length : 0), 0);
  };

  return (
    <div className={cn("space-y-4", className)}>
      {/* Main Search Input */}
      <div className="relative">
        <div className="flex gap-2">
          {/* Search Icon */}
          <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground">
            <Search className="w-5 h-5" />
          </div>

          {/* Input Field */}
          <input
            type="text"
            value={query}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            onFocus={() => setShowSuggestions(query.length > 0)}
            placeholder="Search for manufacturing procedures, safety guidelines, technical specifications..."
            className={cn(
              "w-full pl-10 pr-32 py-3 border border-border rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all",
              isLoading && "opacity-50"
            )}
            disabled={isLoading}
          />

          {/* Right Side Actions */}
          <div className="absolute right-2 top-1/2 transform -translate-y-1/2 flex gap-1">
            {/* Clear Button */}
            {query && (
              <Button
                variant="ghost"
                size="sm"
                className="h-8 w-8 p-0"
                onClick={() => {
                  setQuery('');
                  setShowSuggestions(false);
                }}
              >
                <X className="w-4 h-4" />
              </Button>
            )}

            {/* Filters Button */}
            <Button
              variant="outline"
              size="sm"
              className={cn(
                "h-8 relative",
                getActiveFilterCount() > 0 && "border-primary text-primary"
              )}
              onClick={() => setShowFilters(!showFilters)}
            >
              <Filter className="w-4 h-4 mr-1" />
              Filters
              {getActiveFilterCount() > 0 && (
                <span className="absolute -top-1 -right-1 w-4 h-4 bg-primary text-primary-foreground rounded-full text-xs flex items-center justify-center">
                  {getActiveFilterCount()}
                </span>
              )}
            </Button>

            {/* Search Button */}
            <Button
              variant="quickAction"
              size="sm"
              onClick={() => handleSearch()}
              disabled={!query.trim() || isLoading}
            >
              {isLoading ? (
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}
                >
                  <Search className="w-4 h-4" />
                </motion.div>
              ) : (
                <Search className="w-4 h-4" />
              )}
            </Button>
          </div>
        </div>

        {/* Search Suggestions */}
        <AnimatePresence>
          {showSuggestions && suggestions.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="absolute top-full left-0 right-0 mt-1 bg-background border border-border rounded-lg shadow-lg z-50"
            >
              <div className="p-2">
                {suggestions.map((suggestion, index) => (
                  <button
                    key={index}
                    className={cn(
                      "w-full text-left px-3 py-2 rounded-md text-sm hover:bg-accent transition-colors",
                      focusedIndex === index && "bg-accent"
                    )}
                    onClick={() => handleSearch(suggestion)}
                  >
                    <div className="flex items-center gap-2">
                      <Clock className="w-3 h-3 text-muted-foreground" />
                      <span>{suggestion}</span>
                    </div>
                  </button>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Search Strategy Selection */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
        {searchStrategies.map((strategy) => {
          const Icon = strategy.icon;
          const isSelected = searchStrategy === strategy.id;
          return (
            <Button
              key={strategy.id}
              variant={isSelected ? "default" : "outline"}
              className={cn(
                "h-auto p-3 flex flex-col items-start gap-1 relative overflow-hidden",
                !isSelected && "hover:scale-105 transition-transform"
              )}
              onClick={() => setSearchStrategy(strategy.id)}
            >
              {/* Background Gradient */}
              <div className={cn(
                "absolute inset-0 opacity-10 bg-gradient-to-r",
                strategy.color
              )} />

              {/* Content */}
              <div className="relative z-10 flex items-center gap-2 w-full">
                <Icon className="w-4 h-4" />
                <span className="font-medium text-sm">{strategy.name}</span>
              </div>
              <span className="relative z-10 text-xs text-muted-foreground">
                {strategy.description}
              </span>
            </Button>
          );
        })}
      </div>

      {/* Filter Panel */}
      <AnimatePresence>
        {showFilters && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="manufacturing-card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold">Search Filters</h3>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={clearFilters}
                  >
                    Clear All
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowFilters(false)}
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Document Type Filter */}
                <div>
                  <h4 className="text-sm font-medium mb-3">Document Type</h4>
                  <div className="space-y-2">
                    {filterOptions.document_type.map((doc) => (
                      <label
                        key={doc.id}
                        className="flex items-center gap-2 cursor-pointer hover:bg-accent p-2 rounded"
                      >
                        <input
                          type="checkbox"
                          checked={filters.document_type?.includes(doc.id) || false}
                          onChange={() => toggleFilter('document_type', doc.id)}
                          className="rounded border-border"
                        />
                        <span className="text-sm">{doc.icon} {doc.label}</span>
                      </label>
                    ))}
                  </div>
                </div>

                {/* Expertise Level Filter */}
                <div>
                  <h4 className="text-sm font-medium mb-3">Expertise Level</h4>
                  <div className="space-y-2">
                    {filterOptions.expertise_level.map((level) => (
                      <label
                        key={level.id}
                        className="flex items-center gap-2 cursor-pointer hover:bg-accent p-2 rounded"
                      >
                        <input
                          type="checkbox"
                          checked={filters.expertise_level?.includes(level.id) || false}
                          onChange={() => toggleFilter('expertise_level', level.id)}
                          className="rounded border-border"
                        />
                        <span className={cn("text-sm font-medium", level.color)}>
                          {level.label}
                        </span>
                      </label>
                    ))}
                  </div>
                </div>

                {/* Equipment Type Filter */}
                <div>
                  <h4 className="text-sm font-medium mb-3">Equipment Type</h4>
                  <div className="space-y-2">
                    {filterOptions.equipment_type.map((equipment) => (
                      <label
                        key={equipment.id}
                        className="flex items-center gap-2 cursor-pointer hover:bg-accent p-2 rounded"
                      >
                        <input
                          type="checkbox"
                          checked={filters.equipment_type?.includes(equipment.id) || false}
                          onChange={() => toggleFilter('equipment_type', equipment.id)}
                          className="rounded border-border"
                        />
                        <span className="text-sm">{equipment.label}</span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Search History */}
      {searchHistory.length > 0 && !query && (
        <div className="manufacturing-card">
          <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
            <Clock className="w-4 h-4" />
            Recent Searches
          </h3>
          <div className="flex flex-wrap gap-2">
            {searchHistory.slice(0, 10).map((item, index) => (
              <Button
                key={index}
                variant="outline"
                size="sm"
                onClick={() => handleSearch(item)}
              >
                {item}
              </Button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default AdvancedSearch;