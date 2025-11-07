'use client'

import * as React from 'react'
import { motion } from 'framer-motion'
import { Network, Brain, Activity, Clock, TrendingUp, Filter, Download, Share2, Zap } from 'lucide-react'
import { Layout } from '@/components/layout/layout'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { KnowledgeGraphVisualization } from '@/components/graph/graph-visualization'
import { cn } from '@/lib/utils'

export default function KnowledgeGraphPage() {
  const [selectedView, setSelectedView] = React.useState<'graph' | 'timeline' | 'analytics'>('graph')
  const [stats, setStats] = React.useState({
    totalNodes: 156,
    totalEdges: 342,
    avgConnections: 2.2,
    growthRate: 15.3,
    activeUsers: 48
  })

  return (
    <Layout>
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-8"
        >
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold mb-2 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Knowledge Graph
              </h1>
              <p className="text-lg text-muted-foreground">
                Explore relationships and connections in your knowledge base
              </p>
            </div>

            <div className="flex items-center space-x-2">
              <Button variant="outline" size="sm">
                <Download className="w-4 h-4 mr-2" />
                Export
              </Button>
              <Button variant="outline" size="sm">
                <Share2 className="w-4 h-4 mr-2" />
                Share
              </Button>
            </div>
          </div>
        </motion.div>

        {/* Stats Cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-8"
        >
          <Card className="p-4 border-2 border-blue-200 bg-blue-50/50">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-blue-600">Total Nodes</p>
                <p className="text-2xl font-bold text-blue-900">{stats.totalNodes}</p>
              </div>
              <div className="p-2 bg-blue-100 rounded-lg">
                <Network className="w-5 h-5 text-blue-600" />
              </div>
            </div>
          </Card>

          <Card className="p-4 border-2 border-green-200 bg-green-50/50">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-green-600">Connections</p>
                <p className="text-2xl font-bold text-green-900">{stats.totalEdges}</p>
              </div>
              <div className="p-2 bg-green-100 rounded-lg">
                <Activity className="w-5 h-5 text-green-600" />
              </div>
            </div>
          </Card>

          <Card className="p-4 border-2 border-purple-200 bg-purple-50/50">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-purple-600">Avg Connections</p>
                <p className="text-2xl font-bold text-purple-900">{stats.avgConnections}</p>
              </div>
              <div className="p-2 bg-purple-100 rounded-lg">
                <Brain className="w-5 h-5 text-purple-600" />
              </div>
            </div>
          </Card>

          <Card className="p-4 border-2 border-orange-200 bg-orange-50/50">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-orange-600">Growth Rate</p>
                <p className="text-2xl font-bold text-orange-900">+{stats.growthRate}%</p>
              </div>
              <div className="p-2 bg-orange-100 rounded-lg">
                <TrendingUp className="w-5 h-5 text-orange-600" />
              </div>
            </div>
          </Card>

          <Card className="p-4 border-2 border-pink-200 bg-pink-50/50">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-pink-600">Active Users</p>
                <p className="text-2xl font-bold text-pink-900">{stats.activeUsers}</p>
              </div>
              <div className="p-2 bg-pink-100 rounded-lg">
                <Zap className="w-5 h-5 text-pink-600" />
              </div>
            </div>
          </Card>
        </motion.div>

        {/* View Toggle */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="flex items-center space-x-2 mb-6"
        >
          <Button
            variant={selectedView === 'graph' ? 'default' : 'outline'}
            onClick={() => setSelectedView('graph')}
            className="flex items-center space-x-2"
          >
            <Network className="w-4 h-4" />
            Graph View
          </Button>
          <Button
            variant={selectedView === 'timeline' ? 'default' : 'outline'}
            onClick={() => setSelectedView('timeline')}
            className="flex items-center space-x-2"
          >
            <Clock className="w-4 h-4" />
            Timeline
          </Button>
          <Button
            variant={selectedView === 'analytics' ? 'default' : 'outline'}
            onClick={() => setSelectedView('analytics')}
            className="flex items-center space-x-2"
          >
            <Activity className="w-4 h-4" />
            Analytics
          </Button>
        </motion.div>

        {/* Main Content */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
        >
          {selectedView === 'graph' && (
            <div className="space-y-6">
              {/* Quick Actions */}
              <div className="flex items-center justify-between p-4 bg-card rounded-lg border">
                <div className="flex items-center space-x-4">
                  <h3 className="text-lg font-semibold">Interactive Graph</h3>
                  <Badge variant="secondary">Live</Badge>
                  <Badge variant="outline">156 nodes</Badge>
                </div>
                <div className="flex items-center space-x-2">
                  <Button variant="outline" size="sm">
                    <Filter className="w-4 h-4 mr-2" />
                    Advanced Filters
                  </Button>
                  <Button variant="outline" size="sm">
                    <Brain className="w-4 h-4 mr-2" />
                    AI Insights
                  </Button>
                </div>
              </div>

              {/* Graph Visualization */}
              <Card className="p-6">
                <KnowledgeGraphVisualization />
              </Card>
            </div>
          )}

          {selectedView === 'timeline' && (
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Clock className="w-5 h-5" />
                    Knowledge Evolution Timeline
                  </CardTitle>
                  <CardDescription>
                    Track how your knowledge base has grown and evolved over time
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {/* Timeline implementation would go here */}
                    <div className="text-center py-12 text-muted-600">
                      <Clock className="w-12 h-12 mx-auto mb-4 opacity-50" />
                      <p>Timeline view coming soon...</p>
                      <p className="text-sm">
                        This feature will show the evolution of your knowledge graph over time,
                        highlighting when new nodes were added and connections were made.
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}

          {selectedView === 'analytics' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Activity className="w-5 h-5" />
                      Graph Analytics
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <span>Node Density</span>
                        <span className="font-medium">High</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Connectivity</span>
                        <span className="font-medium">Well Connected</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Clustering</span>
                        <span className="font-medium">3 Clusters</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Popular Connections</CardTitle>
                    <CardDescription>
                      Most frequently traversed paths in your knowledge graph
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {[
                        { from: 'Machine Learning', to: 'Neural Networks', count: 45 },
                        { from: 'Python', to: 'Data Science', count: 38 },
                        { from: 'Deep Learning', to: 'TensorFlow', count: 32 },
                        { from: 'Algorithms', to: 'Data Structures', count: 28 }
                      ].map((connection, index) => (
                        <div key={index} className="flex items-center justify-between p-2 bg-muted/50 rounded">
                          <div className="flex-1">
                            <div className="text-sm font-medium">{connection.from}</div>
                            <div className="text-xs text-muted-foreground">→ {connection.to}</div>
                          </div>
                          <Badge variant="secondary">{connection.count}</Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>Growth Metrics</CardTitle>
                  <CardDescription>
                    Track the growth of your knowledge graph over time
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {/* Growth metrics implementation would go here */}
                    <div className="text-center py-12 text-muted-600">
                      <TrendingUp className="w-12 h-12 mx-auto mb-4 opacity-50" />
                      <p>Detailed analytics coming soon...</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </motion.div>
      </div>
    </Layout>
  )
}