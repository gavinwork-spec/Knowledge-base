'use client'

import * as React from 'react'
import { motion } from 'framer-motion'
import { Search, Brain, Layers, Zap, BarChart3, ArrowRight, Star, Users, Clock, TrendingUp } from 'lucide-react'
import { Layout } from '@/components/layout/layout'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { SearchInput } from '@/components/search/search-input'
import { SearchResults } from '@/components/search/search-results'
import { LoadingSkeleton } from '@/components/ui/loading'
import { cn } from '@/lib/utils'

export default function HomePage() {
  const [isSearchOpen, setIsSearchOpen] = React.useState(false)
  const [searchQuery, setSearchQuery] = React.useState('')
  const [searchResults, setSearchResults] = React.useState([])
  const [isSearching, setIsSearching] = React.useState(false)
  const [showResults, setShowResults] = React.useState(false)

  const features = [
    {
      icon: <Search className="w-6 h-6" />,
      title: 'Semantic Search',
      description: 'AI-powered search that understands context and meaning, not just keywords.',
      badge: 'Advanced',
      color: 'text-blue-500',
      bgColor: 'bg-blue-500/10'
    },
    {
      icon: <Brain className="w-6 h-6" />,
      title: 'Personalized Results',
      description: 'Tailored search results based on your expertise and preferences.',
      badge: 'AI-Driven',
      color: 'text-purple-500',
      bgColor: 'bg-purple-500/10'
    },
    {
      icon: <Layers className="w-6 h-6" />,
      title: 'Hybrid Technology',
      description: 'Combines semantic, keyword, and knowledge graph search for comprehensive results.',
      badge: 'Multi-Modal',
      color: 'text-green-500',
      bgColor: 'bg-green-500/10'
    },
    {
      icon: <Zap className="w-6 h-6" />,
      title: 'Real-time Processing',
      description: 'Lightning-fast search with intelligent caching and optimization.',
      badge: 'Fast',
      color: 'text-yellow-500',
      bgColor: 'bg-yellow-500/10'
    }
  ]

  const stats = [
    { label: 'Search Queries', value: '2.5M+', icon: <Search className="w-4 h-4" /> },
    { label: 'Documents Indexed', value: '50K+', icon: <Layers className="w-4 h-4" /> },
    { label: 'Active Users', value: '10K+', icon: <Users className="w-4 h-4" /> },
    { label: 'Avg Response Time', value: '<200ms', icon: <Clock className="w-4 h-4" /> }
  ]

  const handleSearch = async (query: string) => {
    if (!query.trim()) return

    setIsSearching(true)
    setShowResults(true)

    // Simulate search API call
    try {
      // This would be your actual API call
      await new Promise(resolve => setTimeout(resolve, 1000))

      // Mock results for demonstration
      const mockResults = [
        {
          documentId: '1',
          title: `Results for "${query}" - Document 1`,
          content: `This is a sample result for your search query "${query}". Our advanced search algorithms analyze the semantic meaning and context to provide you with the most relevant information.`,
          score: 0.95,
          searchType: 'semantic',
          metadata: {
            author: 'AI Assistant',
            category: 'Knowledge Base',
            wordCount: 250,
            clickCount: 42
          },
          timestamp: new Date(Date.now() - 1000 * 60 * 60), // 1 hour ago
          expertiseLevel: 'intermediate'
        },
        {
          documentId: '2',
          title: `Understanding ${query} - Advanced Guide`,
          content: `Comprehensive guide covering all aspects of ${query}. This document provides in-depth information with practical examples and best practices for implementation.`,
          score: 0.88,
          searchType: 'hybrid',
          metadata: {
            author: 'Expert Team',
            category: 'Tutorial',
            wordCount: 500,
            clickCount: 128
          },
          timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24), // 1 day ago
          expertiseLevel: 'advanced'
        }
      ]

      setSearchResults(mockResults)
    } catch (error) {
      console.error('Search error:', error)
    } finally {
      setIsSearching(false)
    }
  }

  const handleResultClick = (result: any) => {
    console.log('Result clicked:', result)
    // Navigate to result detail page
  }

  return (
    <Layout>
      <div className="max-w-7xl mx-auto">
        {/* Hero Section */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center py-12 lg:py-20"
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="mb-8"
          >
            <Badge className="mb-4 text-sm px-4 py-2 bg-gradient-to-r from-primary/20 to-primary/10 border-primary/20">
              ✨ AI-Powered Knowledge Discovery
            </Badge>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-4xl lg:text-6xl font-bold mb-6 bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent"
          >
            Discover Knowledge with
            <br />
            Intelligent Search
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="text-xl text-muted-foreground mb-8 max-w-3xl mx-auto leading-relaxed"
          >
            Experience the future of knowledge discovery with our AI-powered search platform.
            Combining semantic understanding, personalization, and hybrid search technologies
            to deliver exactly what you need, when you need it.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="max-w-2xl mx-auto mb-12"
          >
            <SearchInput
              value={searchQuery}
              onChange={setSearchQuery}
              onSubmit={handleSearch}
              isLoading={isSearching}
              placeholder="What would you like to discover today?"
              className="text-lg"
              suggestions={[
                { id: '1', text: 'machine learning algorithms', type: 'popular' },
                { id: '2', text: 'React best practices', type: 'semantic' },
                { id: '3', text: 'data science tutorials', type: 'popular' }
              ]}
            />
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.5 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <Button size="lg" className="text-base px-8 py-3">
              <Search className="w-5 h-5 mr-2" />
              Start Searching
            </Button>
            <Button variant="outline" size="lg" className="text-base px-8 py-3">
              <BarChart3 className="w-5 h-5 mr-2" />
              View Analytics
            </Button>
          </motion.div>
        </motion.section>

        {/* Stats Section */}
        <motion.section
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="py-12"
        >
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.7 + index * 0.1 }}
              >
                <Card className="text-center p-6 border-2 border-primary/10">
                  <div className="flex items-center justify-center w-12 h-12 mx-auto mb-4 rounded-lg bg-primary/10 text-primary">
                    {stat.icon}
                  </div>
                  <div className="text-2xl font-bold mb-1">{stat.value}</div>
                  <div className="text-sm text-muted-foreground">{stat.label}</div>
                </Card>
              </motion.div>
            ))}
          </div>
        </motion.section>

        {/* Features Section */}
        <motion.section
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.8 }}
          className="py-16"
        >
          <div className="text-center mb-12">
            <h2 className="text-3xl lg:text-4xl font-bold mb-4">
              Powerful Search Capabilities
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Our platform combines cutting-edge AI technologies to deliver unparalleled search experiences
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.9 + index * 0.1 }}
              >
                <Card className="group cursor-pointer transition-all duration-300 hover:shadow-medium hover:-translate-y-1 border-2 border-transparent hover:border-primary/20">
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className={cn(
                        'w-12 h-12 rounded-lg flex items-center justify-center',
                        feature.bgColor
                      )}>
                        <div className={feature.color}>
                          {feature.icon}
                        </div>
                      </div>
                      <Badge variant="secondary" className="text-xs">
                        {feature.badge}
                      </Badge>
                    </div>
                    <CardTitle className="text-xl mb-2 group-hover:text-primary transition-colors">
                      {feature.title}
                    </CardTitle>
                    <CardDescription className="text-base leading-relaxed">
                      {feature.description}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Button variant="ghost" className="group-hover:bg-primary/10">
                      Learn more
                      <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
                    </Button>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </motion.section>

        {/* Search Results Section */}
        {showResults && (
          <motion.section
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="py-12"
          >
            <div className="mb-8">
              <h2 className="text-2xl font-bold mb-2">Search Results</h2>
              <p className="text-muted-foreground">
                Found {searchResults.length} results for "{searchQuery}"
              </p>
            </div>

            <SearchResults
              results={searchResults}
              isLoading={isSearching}
              query={searchQuery}
              onResultClick={handleResultClick}
              showExplanations={true}
              showMetadata={true}
            />
          </motion.section>
        )}

        {/* CTA Section */}
        <motion.section
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1.0 }}
          className="py-16 text-center"
        >
          <Card className="max-w-4xl mx-auto border-2 border-primary/20 bg-gradient-to-br from-primary/5 to-primary/10">
            <CardHeader className="pb-8">
              <CardTitle className="text-3xl font-bold mb-4">
                Ready to Transform Your Search Experience?
              </CardTitle>
              <CardDescription className="text-xl max-w-2xl mx-auto">
                Join thousands of users who have already discovered the power of AI-driven search.
                Start exploring our knowledge base today.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Button size="lg" className="text-base px-8 py-3">
                  Get Started Free
                  <ArrowRight className="w-5 h-5 ml-2" />
                </Button>
                <Button variant="outline" size="lg" className="text-base px-8 py-3">
                  <TrendingUp className="w-5 h-5 mr-2" />
                  View Demo
                </Button>
              </div>

              <div className="flex items-center justify-center mt-8 space-x-6 text-sm text-muted-foreground">
                <div className="flex items-center space-x-1">
                  <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />
                  <span>4.9/5 Rating</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Users className="w-4 h-4" />
                  <span>10K+ Users</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Zap className="w-4 h-4" />
                  <span>Sub-second Search</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.section>
      </div>
    </Layout>
  )
}