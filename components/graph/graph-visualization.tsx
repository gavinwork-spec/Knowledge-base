'use client'

import * as React from 'react'
import { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import * as d3 from 'd3'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Network,
  Search,
  Filter,
  Clock,
  Zap,
  Layers,
  Settings,
  Maximize2,
  Minimize2,
  RotateCcw,
  Play,
  Pause,
  Download,
  Share2,
  HelpCircle
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { LoadingSkeleton } from '@/components/ui/loading'
import { cn, formatRelativeTime, generateId } from '@/lib/utils'

// Graph data types
interface GraphNode {
  id: string
  label: string
  type: 'document' | 'concept' | 'entity' | 'tag' | 'user'
  category?: string
  weight: number
  importance: number
  createdAt: Date
  lastModified: Date
  metadata: Record<string, any>
  x?: number
  y?: number
  fx?: number
  fy?: number
  vx?: number
  vy?: number
}

interface GraphEdge {
  id: string
  source: string | GraphNode
  target: string | GraphNode
  type: 'reference' | 'similarity' | 'containment' | 'relation' | 'temporal'
  weight: number
  strength: number
  createdAt: Date
  metadata: Record<string, any>
}

interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

interface GraphConfig {
  width: number
  height: number
  centerForce: number
  linkDistance: number
  chargeStrength: number
  collisionRadius: number
  showLabels: boolean
  showEdges: boolean
  nodeSize: number
  edgeWidth: number
  colorScheme: 'type' | 'category' | 'importance' | 'temporal'
  clustering: boolean
  animationDuration: number
}

interface GraphFilter {
  nodeTypes: string[]
  categories: string[]
  dateRange?: {
    start: Date
    end: Date
  }
  weightRange: {
    min: number
    max: number
  }
  searchText: string
}

interface PathFindingResult {
  path: GraphNode[]
  edges: GraphEdge[]
  distance: number
  hops: number
}

interface TimeSlice {
  timestamp: Date
  nodes: GraphNode[]
  edges: GraphEdge[]
  events: string[]
}

const KnowledgeGraphVisualization: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null)
  const simulationRef = useRef<d3.Simulation<GraphNode, GraphEdge>>()
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] })
  const [selectedNodes, setSelectedNodes] = useState<Set<string>>(new Set())
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(new Date())
  const [filter, setFilter] = useState<GraphFilter>({
    nodeTypes: [],
    categories: [],
    weightRange: { min: 0, max: 1 },
    searchText: ''
  })
  const [config, setConfig] = useState<GraphConfig>({
    width: 800,
    height: 600,
    centerForce: 0.1,
    linkDistance: 100,
    chargeStrength: -300,
    collisionRadius: 30,
    showLabels: true,
    showEdges: true,
    nodeSize: 8,
    edgeWidth: 2,
    colorScheme: 'type',
    clustering: false,
    animationDuration: 300
  })
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [showControls, setShowControls] = useState(true)
  const [pathResult, setPathResult] = useState<PathFindingResult | null>(null)
  const [timeSlices, setTimeSlices] = useState<TimeSlice[]>([])

  // Generate mock data for demonstration
  useEffect(() => {
    const mockNodes: GraphNode[] = [
      {
        id: 'ml-basics',
        label: 'Machine Learning Basics',
        type: 'document',
        category: 'technology',
        weight: 0.9,
        importance: 0.8,
        createdAt: new Date('2024-01-15'),
        lastModified: new Date('2024-01-20'),
        metadata: { author: 'AI Team', views: 1250 }
      },
      {
        id: 'neural-networks',
        label: 'Neural Networks',
        type: 'concept',
        category: 'technology',
        weight: 0.8,
        importance: 0.9,
        createdAt: new Date('2024-01-10'),
        lastModified: new Date('2024-01-18'),
        metadata: { complexity: 'advanced' }
      },
      {
        id: 'deep-learning',
        label: 'Deep Learning',
        type: 'concept',
        category: 'technology',
        weight: 0.85,
        importance: 0.95,
        createdAt: new Date('2024-01-12'),
        lastModified: new Date('2024-01-22'),
        metadata: { frameworks: ['tensorflow', 'pytorch'] }
      },
      {
        id: 'python',
        label: 'Python',
        type: 'entity',
        category: 'language',
        weight: 0.7,
        importance: 0.85,
        createdAt: new Date('2024-01-05'),
        lastModified: new Date('2024-01-25'),
        metadata: { popularity: 'high' }
      },
      {
        id: 'algorithms',
        label: 'Algorithms',
        type: 'concept',
        category: 'computer-science',
        weight: 0.75,
        importance: 0.8,
        createdAt: new Date('2024-01-08'),
        lastModified: new Date('2024-01-15'),
        metadata: { topics: ['sorting', 'searching', 'graph'] }
      },
      {
        id: 'data-science',
        label: 'Data Science',
        type: 'document',
        category: 'field',
        weight: 0.8,
        importance: 0.85,
        createdAt: new Date('2024-01-18'),
        lastModified: new Date('2024-01-26'),
        metadata: { tools: ['pandas', 'numpy', 'scikit-learn'] }
      }
    ]

    const mockEdges: GraphEdge[] = [
      {
        id: 'ml-basics-neural-networks',
        source: 'ml-basics',
        target: 'neural-networks',
        type: 'reference',
        weight: 0.7,
        strength: 0.8,
        createdAt: new Date('2024-01-16'),
        metadata: { context: 'introduction' }
      },
      {
        id: 'neural-networks-deep-learning',
        source: 'neural-networks',
        target: 'deep-learning',
        type: 'containment',
        weight: 0.9,
        strength: 0.9,
        createdAt: new Date('2024-01-14'),
        metadata: { relationship: 'specialization' }
      },
      {
        id: 'ml-basics-python',
        source: 'ml-basics',
        target: 'python',
        type: 'relation',
        weight: 0.6,
        strength: 0.7,
        createdAt: new Date('2024-01-17'),
        metadata: { tool: 'programming' }
      },
      {
        id: 'deep-learning-python',
        source: 'deep-learning',
        target: 'python',
        type: 'relation',
        weight: 0.8,
        strength: 0.8,
        createdAt: new Date('2024-01-19'),
        metadata: { tool: 'implementation' }
      },
      {
        id: 'algorithms-neural-networks',
        source: 'algorithms',
        target: 'neural-networks',
        type: 'reference',
        weight: 0.5,
        strength: 0.6,
        createdAt: new Date('2024-01-13'),
        metadata: { context: 'optimization' }
      },
      {
        id: 'data-science-ml-basics',
        source: 'data-science',
        target: 'ml-basics',
        type: 'reference',
        weight: 0.8,
        strength: 0.85,
        createdAt: new Date('2024-01-20'),
        metadata: { context: 'foundation' }
      }
    ]

    setGraphData({ nodes: mockNodes, edges: mockEdges })

    // Generate time slices for temporal visualization
    const slices: TimeSlice[] = []
    const startDate = new Date('2024-01-01')
    for (let i = 0; i < 30; i++) {
      const sliceDate = new Date(startDate.getTime() + i * 24 * 60 * 60 * 1000)
      const sliceNodes = mockNodes.filter(node => node.createdAt <= sliceDate)
      const sliceEdges = mockEdges.filter(edge => edge.createdAt <= sliceDate)

      slices.push({
        timestamp: sliceDate,
        nodes: sliceNodes,
        edges: sliceEdges,
        events: i % 7 === 0 ? ['New connection added'] : []
      })
    }
    setTimeSlices(slices)
  }, [])

  // Color schemes
  const getColorScale = useCallback(() => {
    const colorSchemes = {
      type: d3.scaleOrdinal(['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']),
      category: d3.scaleOrdinal(['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']),
      importance: d3.scaleSequential(d3.interpolateViridis).domain([0, 1]),
      temporal: d3.scaleSequential(d3.interpolatePlasma).domain([0, 30])
    }
    return colorSchemes[config.colorScheme]
  }, [config.colorScheme])

  // Filter data based on current filter
  const filteredData = useMemo(() => {
    const filteredNodes = graphData.nodes.filter(node => {
      if (filter.nodeTypes.length > 0 && !filter.nodeTypes.includes(node.type)) return false
      if (filter.categories.length > 0 && !filter.categories.includes(node.category || '')) return false
      if (filter.searchText && !node.label.toLowerCase().includes(filter.searchText.toLowerCase())) return false
      if (node.weight < filter.weightRange.min || node.weight > filter.weightRange.max) return false
      if (filter.dateRange) {
        const nodeDate = node.lastModified || node.createdAt
        if (nodeDate < filter.dateRange.start || nodeDate > filter.dateRange.end) return false
      }
      return true
    })

    const nodeIds = new Set(filteredNodes.map(n => n.id))
    const filteredEdges = graphData.edges.filter(edge => {
      const sourceId = typeof edge.source === 'string' ? edge.source : edge.source.id
      const targetId = typeof edge.target === 'string' ? edge.target : edge.target.id
      return nodeIds.has(sourceId) && nodeIds.has(targetId)
    })

    return { nodes: filteredNodes, edges: filteredEdges }
  }, [graphData, filter])

  // Path finding using Dijkstra's algorithm
  const findPath = useCallback((sourceId: string, targetId: string): PathFindingResult | null => {
    const nodeMap = new Map(filteredData.nodes.map(n => [n.id, n]))
    const distances = new Map<string, number>()
    const previous = new Map<string, string | null>()
    const unvisited = new Set(filteredData.nodes.map(n => n.id))

    // Initialize distances
    filteredData.nodes.forEach(node => {
      distances.set(node.id, node.id === sourceId ? 0 : Infinity)
      previous.set(node.id, null)
    })

    while (unvisited.size > 0) {
      // Find node with minimum distance
      let current: string | null = null
      let minDistance = Infinity
      unvisited.forEach(nodeId => {
        const distance = distances.get(nodeId) || Infinity
        if (distance < minDistance) {
          minDistance = distance
          current = nodeId
        }
      })

      if (current === null || current === targetId) break

      unvisited.delete(current)

      // Update distances to neighbors
      filteredData.edges.forEach(edge => {
        const sourceId = typeof edge.source === 'string' ? edge.source : edge.source.id
        const targetId = typeof edge.target === 'string' ? edge.target : edge.target.id

        let neighbor: string | null = null
        if (sourceId === current) neighbor = targetId
        else if (targetId === current) neighbor = sourceId

        if (neighbor && unvisited.has(neighbor)) {
          const alt = (distances.get(current) || 0) + (1 - edge.strength)
          if (alt < (distances.get(neighbor) || Infinity)) {
            distances.set(neighbor, alt)
            previous.set(neighbor, current)
          }
        }
      })
    }

    // Reconstruct path
    if (distances.get(targetId) === Infinity) return null

    const path: GraphNode[] = []
    const pathEdges: GraphEdge[] = []
    let current: string | null = targetId

    while (current !== null) {
      const node = nodeMap.get(current)
      if (node) path.unshift(node)

      const prev = previous.get(current)
      if (prev) {
        const edge = filteredData.edges.find(e => {
          const sourceId = typeof e.source === 'string' ? e.source : e.source.id
          const targetId = typeof e.target === 'string' ? e.target : e.target.id
          return (sourceId === prev && targetId === current) || (sourceId === current && targetId === prev)
        })
        if (edge) pathEdges.push(edge)
      }

      current = prev
    }

    return {
      path,
      edges: pathEdges,
      distance: distances.get(targetId) || 0,
      hops: path.length - 1
    }
  }, [filteredData])

  // D3 force simulation
  useEffect(() => {
    if (!svgRef.current || filteredData.nodes.length === 0) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const width = config.width
    const height = config.height

    // Create container groups
    const container = svg.append('g')
      .attr('class', 'graph-container')

    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        container.attr('transform', event.transform)
      })

    svg.call(zoom)

    // Create force simulation
    const simulation = d3.forceSimulation<GraphNode>(filteredData.nodes)
      .force('link', d3.forceLink<GraphNode, GraphEdge>(filteredData.edges)
        .id(d => d.id)
        .distance(config.linkDistance)
        .strength(d => d.strength))
      .force('charge', d3.forceManyBody().strength(config.chargeStrength))
      .force('center', d3.forceCenter(width / 2, height / 2).strength(config.centerForce))
      .force('collision', d3.forceCollide().radius(config.collisionRadius))
      .force('x', d3.forceX(width / 2).strength(0.1))
      .force('y', d3.forceY(height / 2).strength(0.1))

    simulationRef.current = simulation

    // Create edges
    const link = container.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(filteredData.edges)
      .enter().append('line')
      .attr('class', 'link')
      .attr('stroke', d => {
        const colors = {
          reference: '#94a3b8',
          similarity: '#10b981',
          containment: '#3b82f6',
          relation: '#f59e0b',
          temporal: '#ef4444'
        }
        return colors[d.type as keyof typeof colors] || '#94a3b8'
      })
      .attr('stroke-width', d => Math.max(1, d.weight * config.edgeWidth))
      .attr('stroke-opacity', 0.6)

    // Create nodes
    const node = container.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(filteredData.nodes)
      .enter().append('g')
      .attr('class', 'node')
      .call(d3.drag<SVGGElement, GraphNode>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart()
          d.fx = d.x
          d.fy = d.y
        })
        .on('drag', (event, d) => {
          d.fx = event.x
          d.fy = event.y
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0)
          d.fx = null
          d.fy = null
        }) as any
      )

    // Add node circles
    node.append('circle')
      .attr('class', 'node-circle')
      .attr('r', d => Math.max(4, d.weight * config.nodeSize))
      .attr('fill', d => {
        const colorScale = getColorScale()
        if (config.colorScheme === 'importance') {
          return colorScale(d.importance) as string
        } else if (config.colorScheme === 'temporal') {
          const daysSinceCreation = (Date.now() - d.createdAt.getTime()) / (1000 * 60 * 60 * 24)
          return colorScale(Math.min(daysSinceCreation, 30)) as string
        }
        return colorScale(d.type) as string
      })
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        event.stopPropagation()
        const newSelected = new Set(selectedNodes)
        if (newSelected.has(d.id)) {
          newSelected.delete(d.id)
        } else {
          newSelected.add(d.id)
        }
        setSelectedNodes(newSelected)
      })
      .on('mouseenter', (event, d) => {
        setHoveredNode(d)
      })
      .on('mouseleave', () => {
        setHoveredNode(null)
      })

    // Add node labels
    if (config.showLabels) {
      node.append('text')
        .attr('class', 'node-label')
        .attr('text-anchor', 'middle')
        .attr('dy', '.35em')
        .style('font-size', '12px')
        .style('font-weight', '500')
        .style('pointer-events', 'none')
        .style('user-select', 'none')
        .text(d => d.label.length > 15 ? d.label.substring(0, 12) + '...' : d.label)
    }

    // Add selection indicators
    const updateSelection = () => {
      node.selectAll('.node-circle')
        .attr('stroke', d => selectedNodes.has(d.id) ? '#3b82f6' : '#fff')
        .attr('stroke-width', d => selectedNodes.has(d.id) ? 3 : 2)

      if (pathResult) {
        const pathNodeIds = new Set(pathResult.path.map(n => n.id))
        const pathEdgeIds = new Set(pathResult.edges.map(e => e.id))

        link
          .attr('stroke', d => pathEdgeIds.has(d.id) ? '#3b82f6' : '#94a3b8')
          .attr('stroke-width', d => pathEdgeIds.has(d.id) ? 3 : d.weight * config.edgeWidth)

        node.selectAll('.node-circle')
          .attr('stroke', d => pathNodeIds.has(d.id) ? '#3b82f6' : (selectedNodes.has(d.id) ? '#3b82f6' : '#fff'))
          .attr('stroke-width', d => pathNodeIds.has(d.id) ? 3 : (selectedNodes.has(d.id) ? 3 : 2))
      }
    }

    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => {
          const source = d.source as GraphNode
          return source.x || 0
        })
        .attr('y1', d => {
          const source = d.source as GraphNode
          return source.y || 0
        })
        .attr('x2', d => {
          const target = d.target as GraphNode
          return target.x || 0
        })
        .attr('y2', d => {
          const target = d.target as GraphNode
          return target.y || 0
        })

      node.attr('transform', d => `translate(${d.x || 0},${d.y || 0})`)
      updateSelection()
    })

    return () => {
      simulation.stop()
    }
  }, [filteredData, config, selectedNodes, pathResult, getColorScale])

  // Handle node selection for path finding
  useEffect(() => {
    if (selectedNodes.size === 2) {
      const [sourceId, targetId] = Array.from(selectedNodes)
      const result = findPath(sourceId, targetId)
      setPathResult(result)
    } else {
      setPathResult(null)
    }
  }, [selectedNodes, findPath])

  // Handle time evolution
  useEffect(() => {
    if (!isPlaying || timeSlices.length === 0) return

    const interval = setInterval(() => {
      setCurrentTime(prev => {
        const nextTime = new Date(prev.getTime() + 24 * 60 * 60 * 1000) // Add 1 day
        if (nextTime > timeSlices[timeSlices.length - 1].timestamp) {
          setIsPlaying(false)
          return prev
        }
        return nextTime
      })
    }, 2000) // Change every 2 seconds

    return () => clearInterval(interval)
  }, [isPlaying, timeSlices])

  // Export graph as image
  const exportGraph = useCallback(() => {
    if (!svgRef.current) return

    const svgElement = svgRef.current
    const serializer = new XMLSerializer()
    const svgString = serializer.serializeToString(svgElement)
    const blob = new Blob([svgString], { type: 'image/svg+xml' })
    const url = URL.createObjectURL(blob)

    const link = document.createElement('a')
    link.href = url
    link.download = `knowledge-graph-${new Date().toISOString().split('T')[0]}.svg`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }, [])

  return (
    <div className={cn('w-full h-full flex flex-col', isFullscreen && 'fixed inset-0 z-50 bg-background')}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center space-x-4">
          <Network className="w-5 h-5" />
          <h2 className="text-lg font-semibold">Knowledge Graph</h2>
          <Badge variant="secondary">
            {filteredData.nodes.length} nodes, {filteredData.edges.length} edges
          </Badge>
        </div>

        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsPlaying(!isPlaying)}
            disabled={timeSlices.length === 0}
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isPlaying ? 'Pause' : 'Play'}
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={() => setConfig({ ...config, showLabels: !config.showLabels })}
          >
            <span className="text-xs">Labels</span>
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={() => setConfig({ ...config, showEdges: !config.showEdges })}
          >
            <span className="text-xs">Edges</span>
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowControls(!showControls)}
          >
            <Settings className="w-4 h-4" />
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsFullscreen(!isFullscreen)}
          >
            {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={exportGraph}
          >
            <Download className="w-4 h-4" />
          </Button>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Main graph area */}
        <div className="flex-1 relative bg-muted/10">
          <svg
            ref={svgRef}
            width={config.width}
            height={config.height}
            className="w-full h-full"
            style={{ minHeight: '600px' }}
          />

          {/* Time slider */}
          {timeSlices.length > 0 && (
            <div className="absolute bottom-4 left-4 right-4 glass border border-border rounded-lg p-4">
              <div className="flex items-center space-x-4">
                <Clock className="w-4 h-4" />
                <div className="flex-1">
                  <input
                    type="range"
                    min="0"
                    max={timeSlices.length - 1}
                    value={timeSlices.findIndex(slice => slice.timestamp <= currentTime)}
                    onChange={(e) => {
                      const index = parseInt(e.target.value)
                      setCurrentTime(timeSlices[index].timestamp)
                    }}
                    className="w-full"
                  />
                </div>
                <span className="text-sm text-muted-foreground min-w-[100px]">
                  {formatRelativeTime(currentTime)}
                </span>
              </div>
            </div>
          )}

          {/* Tooltip */}
          <AnimatePresence>
            {hoveredNode && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="absolute top-4 left-4 glass border border-border rounded-lg p-4 max-w-xs"
              >
                <div className="font-semibold mb-2">{hoveredNode.label}</div>
                <div className="text-sm text-muted-foreground space-y-1">
                  <div>Type: {hoveredNode.type}</div>
                  <div>Category: {hoveredNode.category}</div>
                  <div>Weight: {(hoveredNode.weight * 100).toFixed(0)}%</div>
                  <div>Importance: {(hoveredNode.importance * 100).toFixed(0)}%</div>
                  <div>Modified: {formatRelativeTime(hoveredNode.lastModified)}</div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Path result */}
          <AnimatePresence>
            {pathResult && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 20 }}
                className="absolute top-4 right-4 glass border border-border rounded-lg p-4 max-w-xs"
              >
                <div className="font-semibold mb-2">Path Found</div>
                <div className="text-sm text-muted-foreground space-y-1">
                  <div>Distance: {pathResult.distance.toFixed(2)}</div>
                  <div>Hops: {pathResult.hops}</div>
                  <div>Path: {pathResult.path.map(n => n.label).join(' → ')}</div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Control panel */}
        <AnimatePresence>
          {showControls && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className="w-80 border-l border-border bg-background overflow-y-auto"
            >
              <div className="p-4 space-y-6">
                {/* Filters */}
                <div>
                  <h3 className="font-semibold mb-3 flex items-center">
                    <Filter className="w-4 h-4 mr-2" />
                    Filters
                  </h3>

                  <div className="space-y-4">
                    <div>
                      <label className="text-sm font-medium mb-2">Search</label>
                      <Input
                        placeholder="Search nodes..."
                        value={filter.searchText}
                        onChange={(e) => setFilter({ ...filter, searchText: e.target.value })}
                      />
                    </div>

                    <div>
                      <label className="text-sm font-medium mb-2">Node Types</label>
                      <div className="space-y-2">
                        {['document', 'concept', 'entity', 'tag', 'user'].map(type => (
                          <label key={type} className="flex items-center space-x-2">
                            <input
                              type="checkbox"
                              checked={filter.nodeTypes.includes(type)}
                              onChange={(e) => {
                                if (e.target.checked) {
                                  setFilter({ ...filter, nodeTypes: [...filter.nodeTypes, type] })
                                } else {
                                  setFilter({ ...filter, nodeTypes: filter.nodeTypes.filter(t => t !== type) })
                                }
                              }}
                            />
                            <span className="text-sm capitalize">{type}</span>
                          </label>
                        ))}
                      </div>
                    </div>

                    <div>
                      <label className="text-sm font-medium mb-2">Weight Range</label>
                      <div className="space-y-2">
                        <div className="flex items-center space-x-2">
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={filter.weightRange.min}
                            onChange={(e) => setFilter({ ...filter, weightRange: { ...filter.weightRange, min: parseFloat(e.target.value) } })}
                            className="flex-1"
                          />
                          <span className="text-sm w-8">{filter.weightRange.min.toFixed(1)}</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={filter.weightRange.max}
                            onChange={(e) => setFilter({ ...filter, weightRange: { ...filter.weightRange, max: parseFloat(e.target.value) } })}
                            className="flex-1"
                          />
                          <span className="text-sm w-8">{filter.weightRange.max.toFixed(1)}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Configuration */}
                <div>
                  <h3 className="font-semibold mb-3 flex items-center">
                    <Settings className="w-4 h-4 mr-2" />
                    Configuration
                  </h3>

                  <div className="space-y-4">
                    <div>
                      <label className="text-sm font-medium mb-2">Color Scheme</label>
                      <select
                        value={config.colorScheme}
                        onChange={(e) => setConfig({ ...config, colorScheme: e.target.value as any })}
                        className="w-full p-2 border border-border rounded-md"
                      >
                        <option value="type">By Type</option>
                        <option value="category">By Category</option>
                        <option value="importance">By Importance</option>
                        <option value="temporal">By Time</option>
                      </select>
                    </div>

                    <div>
                      <label className="text-sm font-medium mb-2">Layout</label>
                      <div className="space-y-2">
                        <label className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            checked={config.clustering}
                            onChange={(e) => setConfig({ ...config, clustering: e.target.checked })}
                          />
                          <span className="text-sm">Enable Clustering</span>
                        </label>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Statistics */}
                <div>
                  <h3 className="font-semibold mb-3 flex items-center">
                    <BarChart3 className="w-4 h-4 mr-2" />
                    Statistics
                  </h3>

                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Nodes:</span>
                      <span className="font-medium">{filteredData.nodes.length}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Edges:</span>
                      <span className="font-medium">{filteredData.edges.length}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Density:</span>
                      <span className="font-medium">
                        {filteredData.nodes.length > 0
                          ? (filteredData.edges.length / filteredData.nodes.length).toFixed(2)
                          : '0.00'
                        }
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

export default KnowledgeGraphVisualization