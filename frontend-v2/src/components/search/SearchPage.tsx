import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Search } from 'lucide-react';

// Import components
import AdvancedSearch from './AdvancedSearch';
import SearchResults from './SearchResults';

// Import services and stores
import apiService from '@lib/api';
import { useManufacturingStore } from '@stores';
import { SearchQuery, SearchResult, SearchResponse } from '@types';

// Import UI components
import Button from '../ui/button';

const SearchPage: React.FC = () => {
  const { context } = useManufacturingStore();
  const [results, setResults] = useState<SearchResult[]>([]);
  const [totalResults, setTotalResults] = useState(0);
  const [searchTime, setSearchTime] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  // Handle search request
  const handleSearch = useCallback(async (query: string, strategy: string, filters: any) => {
    if (!query.trim()) return;

    setIsLoading(true);
    setHasSearched(true);

    try {
      const searchQuery: SearchQuery = {
        query: query.trim(),
        strategy: strategy as any,
        filters,
        manufacturing_context: context,
        limit: 20
      };

      const response = await apiService.search(searchQuery);

      if (response.success && response.data) {
        setResults(response.data.results);
        setTotalResults(response.data.total_results);
        setSearchTime(response.data.search_time);
      } else {
        console.error('Search failed:', response.error);
        setResults([]);
        setTotalResults(0);
      }
    } catch (error) {
      console.error('Search error:', error);
      setResults([]);
      setTotalResults(0);
    } finally {
      setIsLoading(false);
    }
  }, [context]);

  // Handle result click
  const handleResultClick = useCallback((result: SearchResult) => {
    // In a real implementation, this would navigate to the document detail page
    console.log('Result clicked:', result);
  }, []);

  // Handle quick action searches
  const handleQuickAction = useCallback((prompt: string) => {
    handleSearch(prompt, 'unified', {});
  }, [handleSearch]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="manufacturing-content"
    >
      <div className="max-w-6xl mx-auto">
        {/* Search Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-primary to-primary-80 bg-clip-text text-transparent">
            Manufacturing Knowledge Search
          </h1>
          <p className="text-lg text-muted-foreground">
            Search through technical manuals, safety procedures, quality specifications, and equipment documentation.
          </p>
        </div>

        {/* Manufacturing Context Display */}
        {context && (
          <div className="manufacturing-card mb-6">
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
              <span className="w-2 h-2 bg-green-500 rounded-full"></span>
              Current Manufacturing Context
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Equipment Type:</span>
                <div className="font-medium">{context.equipment_type || 'General'}</div>
              </div>
              <div>
                <span className="text-muted-foreground">User Role:</span>
                <div className="font-medium">{context.user_role || 'Operator'}</div>
              </div>
              <div>
                <span className="text-muted-foreground">Facility:</span>
                <div className="font-medium">{context.facility_id || 'Default'}</div>
              </div>
              <div>
                <span className="text-muted-foreground">Process:</span>
                <div className="font-medium">{context.process_type || 'General'}</div>
              </div>
            </div>
          </div>
        )}

        {/* Advanced Search Component */}
        <div className="mb-8">
          <AdvancedSearch
            onSearch={handleSearch}
            isLoading={isLoading}
          />
        </div>

        {/* Quick Actions */}
        {!hasSearched && (
          <div className="manufacturing-card mb-8">
            <h3 className="font-semibold mb-4">Quick Actions</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <Button
                variant="outline"
                className="h-auto p-4 flex flex-col items-start gap-2 hover-lift"
                onClick={() => handleQuickAction('Generate comprehensive safety procedures for CNC milling machine setup')}
              >
                <div className="text-2xl">🛡️</div>
                <div className="text-left">
                  <div className="font-medium">Safety Procedures</div>
                  <div className="text-xs text-muted-foreground">CNC milling setup and operation</div>
                </div>
              </Button>

              <Button
                variant="outline"
                className="h-auto p-4 flex flex-col items-start gap-2 hover-lift"
                onClick={() => handleQuickAction('First article inspection procedures for aerospace components')}
              >
                <div className="text-2xl">✅</div>
                <div className="text-left">
                  <div className="font-medium">Quality Guidelines</div>
                  <div className="text-xs text-muted-foreground">Aerospace component inspection</div>
                </div>
              </Button>

              <Button
                variant="outline"
                className="h-auto p-4 flex flex-col items-start gap-2 hover-lift"
                onClick={() => handleQuickAction('Preventive maintenance schedule for CNC equipment')}
              >
                <div className="text-2xl">🔧</div>
                <div className="text-left">
                  <div className="font-medium">Maintenance Help</div>
                  <div className="text-xs text-muted-foreground">CNC equipment maintenance</div>
                </div>
              </Button>

              <Button
                variant="outline"
                className="h-auto p-4 flex flex-col items-start gap-2 hover-lift"
                onClick={() => handleQuickAction('ISO 9001 quality management system requirements')}
              >
                <div className="text-2xl">📋</div>
                <div className="text-left">
                  <div className="font-medium">Compliance Standards</div>
                  <div className="text-xs text-muted-foreground">ISO 9001 requirements</div>
                </div>
              </Button>
            </div>
          </div>
        )}

        {/* Search Results */}
        {hasSearched && (
          <div>
            <SearchResults
              results={results}
              totalResults={totalResults}
              searchTime={searchTime}
              isLoading={isLoading}
              query={results.length > 0 ? undefined : 'No results found'}
              onResultClick={handleResultClick}
            />
          </div>
        )}

        {/* Manufacturing Tips */}
        {!hasSearched && (
          <div className="manufacturing-card">
            <h3 className="font-semibold mb-4">Search Tips</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
              <div>
                <h4 className="font-medium mb-2 text-green-600">🔍 Effective Searches</h4>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• Use specific equipment names (e.g., "HAAS VF-2")</li>
                  <li>• Include process types (e.g., "CNC milling", "grinding")</li>
                  <li>• Try standard numbers (e.g., "ISO 9001", "OSHA 1910")</li>
                  <li>• Combine keywords for better results</li>
                </ul>
              </div>

              <div>
                <h4 className="font-medium mb-2 text-blue-600">⚡ Search Strategies</h4>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• <strong>Unified</strong>: Best overall results</li>
                  <li>• <strong>Semantic</strong>: Understands meaning</li>
                  <li>• <strong>Keyword</strong>: Exact text matching</li>
                  <li>• <strong>AI Enhanced</strong>: Advanced AI search</li>
                </ul>
              </div>

              <div>
                <h4 className="font-medium mb-2 text-purple-600">📚 Document Types</h4>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• <strong>Safety Procedures</strong>: OSHA, ANSI standards</li>
                  <li>• <strong>Quality Specs</strong>: ISO, AS9100 requirements</li>
                  <li>• <strong>Technical Manuals</strong>: Equipment operation</li>
                  <li>• <strong>General Docs</strong>: Policies and guidelines</li>
                </ul>
              </div>

              <div>
                <h4 className="font-medium mb-2 text-orange-600">🛡️ Quick Access</h4>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• Emergency stop procedures</li>
                  <li>• Lockout/tagout (LOTO) steps</li>
                  <li>• First article inspection</li>
                  <li>• Maintenance troubleshooting</li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default SearchPage;