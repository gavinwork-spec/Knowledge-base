/**
 * Performance-optimized graph engine for knowledge graph visualization
 * Provides efficient algorithms for clustering, path finding, and graph analysis
 */

export interface GraphNode {
  id: string
  label: string
  type: 'document' | 'concept' | 'entity' | 'topic'
  weight: number
  cluster?: number
  metadata: Record<string, any>
  x?: number
  y?: number
  vx?: number
  vy?: number
  fx?: number | null
  fy?: number | null
}

export interface GraphEdge {
  source: string | GraphNode
  target: string | GraphNode
  weight: number
  type: 'semantic' | 'reference' | 'hierarchy' | 'temporal'
  metadata: Record<string, any>
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface ClusterInfo {
  id: number
  nodes: string[]
  centroid: { x: number; y: number }
  keywords: string[]
  weight: number
}

export interface PathResult {
  path: string[]
  distance: number
  hops: number
  nodes: GraphNode[]
  edges: GraphEdge[]
}

/**
 * High-performance graph engine with optimized algorithms
 */
export class GraphEngine {
  private nodeMap: Map<string, GraphNode> = new Map()
  private adjacencyList: Map<string, Map<string, GraphEdge>> = new Map()
  private clusters: Map<number, ClusterInfo> = new Map()
  private nodeIndex: Map<string, number> = new Map()
  private distanceMatrix: number[][] = []
  private initialized = false

  constructor(private data: GraphData) {
    this.initialize()
  }

  private initialize(): void {
    // Build efficient data structures
    this.buildNodeMap()
    this.buildAdjacencyList()
    this.buildNodeIndex()
    this.initialized = true
  }

  private buildNodeMap(): void {
    this.nodeMap.clear()
    this.data.nodes.forEach(node => {
      this.nodeMap.set(node.id, node)
    })
  }

  private buildAdjacencyList(): void {
    this.adjacencyList.clear()

    // Initialize adjacency list
    this.data.nodes.forEach(node => {
      this.adjacencyList.set(node.id, new Map())
    })

    // Add edges
    this.data.edges.forEach(edge => {
      const sourceId = typeof edge.source === 'string' ? edge.source : edge.source.id
      const targetId = typeof edge.target === 'string' ? edge.target : edge.target.id

      if (this.adjacencyList.has(sourceId)) {
        this.adjacencyList.get(sourceId)!.set(targetId, edge)
      }

      // For undirected graphs, add reverse edge
      if (this.adjacencyList.has(targetId)) {
        this.adjacencyList.get(targetId)!.set(sourceId, edge)
      }
    })
  }

  private buildNodeIndex(): void {
    this.nodeIndex.clear()
    this.data.nodes.forEach((node, index) => {
      this.nodeIndex.set(node.id, index)
    })
  }

  /**
   * Find shortest path using Dijkstra's algorithm with priority queue
   */
  findShortestPath(sourceId: string, targetId: string): PathResult | null {
    if (!this.initialized || !this.nodeMap.has(sourceId) || !this.nodeMap.has(targetId)) {
      return null
    }

    const distances = new Map<string, number>()
    const previous = new Map<string, string | null>()
    const visited = new Set<string>()
    const queue: Array<{ node: string; distance: number }> = []

    // Initialize distances
    this.data.nodes.forEach(node => {
      distances.set(node.id, node.id === sourceId ? 0 : Infinity)
      previous.set(node.id, null)
      queue.push({ node: node.id, distance: node.id === sourceId ? 0 : Infinity })
    })

    // Priority queue implementation
    while (queue.length > 0) {
      // Sort by distance (simple implementation, for production use a proper priority queue)
      queue.sort((a, b) => a.distance - b.distance)
      const { node: currentId } = queue.shift()!

      if (currentId === targetId) {
        // Reconstruct path
        const path: string[] = []
        let current: string | null = targetId
        let totalDistance = 0

        while (current !== null) {
          path.unshift(current)
          const prev = previous.get(current)
          if (prev !== null) {
            const edge = this.getEdge(prev, current)
            if (edge) {
              totalDistance += edge.weight
            }
          }
          current = prev
        }

        const pathNodes = path.map(id => this.nodeMap.get(id)!).filter(Boolean)
        const pathEdges: GraphEdge[] = []
        for (let i = 0; i < path.length - 1; i++) {
          const edge = this.getEdge(path[i], path[i + 1])
          if (edge) pathEdges.push(edge)
        }

        return {
          path,
          distance: totalDistance,
          hops: path.length - 1,
          nodes: pathNodes,
          edges: pathEdges
        }
      }

      if (visited.has(currentId)) continue
      visited.add(currentId)

      // Update neighbors
      const neighbors = this.adjacencyList.get(currentId) || new Map()
      neighbors.forEach((edge, neighborId) => {
        if (visited.has(neighborId)) return

        const alt = distances.get(currentId)! + edge.weight
        if (alt < distances.get(neighborId)!) {
          distances.set(neighborId, alt)
          previous.set(neighborId, currentId)

          // Update queue
          const queueItem = queue.find(item => item.node === neighborId)
          if (queueItem) {
            queueItem.distance = alt
          }
        }
      })
    }

    return null
  }

  private getEdge(sourceId: string, targetId: string): GraphEdge | null {
    const neighbors = this.adjacencyList.get(sourceId)
    return neighbors?.get(targetId) || null
  }

  /**
   * Find all paths between two nodes up to a maximum depth
   */
  findAllPaths(sourceId: string, targetId: string, maxDepth: number = 4): PathResult[] {
    if (!this.initialized || !this.nodeMap.has(sourceId) || !this.nodeMap.has(targetId)) {
      return []
    }

    const paths: PathResult[] = []
    const visited = new Set<string>()

    const dfs = (currentId: string, currentPath: string[], currentDistance: number, depth: number): void => {
      if (depth > maxDepth) return

      if (currentId === targetId && currentPath.length > 1) {
        const pathNodes = currentPath.map(id => this.nodeMap.get(id)!).filter(Boolean)
        const pathEdges: GraphEdge[] = []
        let totalDistance = 0

        for (let i = 0; i < currentPath.length - 1; i++) {
          const edge = this.getEdge(currentPath[i], currentPath[i + 1])
          if (edge) {
            pathEdges.push(edge)
            totalDistance += edge.weight
          }
        }

        paths.push({
          path: [...currentPath],
          distance: totalDistance,
          hops: currentPath.length - 1,
          nodes: pathNodes,
          edges: pathEdges
        })
        return
      }

      visited.add(currentId)

      const neighbors = this.adjacencyList.get(currentId) || new Map()
      neighbors.forEach((edge, neighborId) => {
        if (!visited.has(neighborId)) {
          dfs(neighborId, [...currentPath, neighborId], currentDistance + edge.weight, depth + 1)
        }
      })

      visited.delete(currentId)
    }

    dfs(sourceId, [sourceId], 0, 0)
    return paths.sort((a, b) => a.distance - b.distance)
  }

  /**
   * Perform graph clustering using Louvain method (simplified)
   */
  performClustering(resolution: number = 1.0): ClusterInfo[] {
    if (!this.initialized) return []

    // Initialize each node as its own cluster
    const nodeClusters = new Map<string, number>()
    this.data.nodes.forEach((node, index) => {
      nodeClusters.set(node.id, index)
    })

    let improved = true
    let iteration = 0
    const maxIterations = 50

    while (improved && iteration < maxIterations) {
      improved = false
      iteration++

      // Try moving each node to neighboring clusters
      for (const node of this.data.nodes) {
        const currentCluster = nodeClusters.get(node.id)!
        const neighborClusters = new Map<number, number>()

        // Calculate modularity gain for each neighboring cluster
        const neighbors = this.adjacencyList.get(node.id) || new Map()
        neighbors.forEach((edge, neighborId) => {
          const neighborCluster = nodeClusters.get(neighborId)!
          neighborClusters.set(
            neighborCluster,
            (neighborClusters.get(neighborCluster) || 0) + edge.weight
          )
        })

        // Find best cluster to move to
        let bestCluster = currentCluster
        let bestGain = 0

        neighborClusters.forEach((weight, clusterId) => {
          if (clusterId !== currentCluster) {
            const gain = this.calculateModularityGain(
              node.id,
              currentCluster,
              clusterId,
              nodeClusters,
              resolution
            )
            if (gain > bestGain) {
              bestGain = gain
              bestCluster = clusterId
            }
          }
        })

        if (bestCluster !== currentCluster) {
          nodeClusters.set(node.id, bestCluster)
          improved = true
        }
      }
    }

    // Build cluster information
    const clusters = new Map<number, Set<string>>()
    nodeClusters.forEach((clusterId, nodeId) => {
      if (!clusters.has(clusterId)) {
        clusters.set(clusterId, new Set())
      }
      clusters.get(clusterId)!.add(nodeId)
    })

    const clusterInfos: ClusterInfo[] = []
    clusters.forEach((nodes, clusterId) => {
      const clusterNodes = Array.from(nodes).map(id => this.nodeMap.get(id)!).filter(Boolean)
      const keywords = this.extractClusterKeywords(clusterNodes)
      const centroid = this.calculateCentroid(clusterNodes)
      const weight = clusterNodes.reduce((sum, node) => sum + node.weight, 0)

      clusterInfos.push({
        id: clusterId,
        nodes: Array.from(nodes),
        centroid,
        keywords,
        weight
      })
    })

    this.clusters = new Map(clusterInfos.map(c => [c.id, c]))
    return clusterInfos.sort((a, b) => b.weight - a.weight)
  }

  private calculateModularityGain(
    nodeId: string,
    currentCluster: number,
    targetCluster: number,
    nodeClusters: Map<string, number>,
    resolution: number
  ): number {
    // Simplified modularity calculation
    const neighbors = this.adjacencyList.get(nodeId) || new Map()
    let intraClusterWeight = 0
    let totalWeight = 0

    neighbors.forEach((edge, neighborId) => {
      totalWeight += edge.weight
      if (nodeClusters.get(neighborId) === targetCluster) {
        intraClusterWeight += edge.weight
      }
    })

    return (intraClusterWeight - totalWeight * 0.1) * resolution
  }

  private extractClusterKeywords(nodes: GraphNode[]): string[] {
    const keywordCounts = new Map<string, number>()

    nodes.forEach(node => {
      const words = node.label.toLowerCase().split(/\s+/)
      words.forEach(word => {
        if (word.length > 3) {
          keywordCounts.set(word, (keywordCounts.get(word) || 0) + 1)
        }
      })
    })

    return Array.from(keywordCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([word]) => word)
  }

  private calculateCentroid(nodes: GraphNode[]): { x: number; y: number } {
    if (nodes.length === 0) return { x: 0, y: 0 }

    const sumX = nodes.reduce((sum, node) => sum + (node.x || 0), 0)
    const sumY = nodes.reduce((sum, node) => sum + (node.y || 0), 0)

    return {
      x: sumX / nodes.length,
      y: sumY / nodes.length
    }
  }

  /**
   * Calculate various graph metrics
   */
  calculateMetrics(): {
    nodeCount: number
    edgeCount: number
    density: number
    avgDegree: number
    clustering: number
    components: number
  } {
    if (!this.initialized) {
      return {
        nodeCount: 0,
        edgeCount: 0,
        density: 0,
        avgDegree: 0,
        clustering: 0,
        components: 0
      }
    }

    const nodeCount = this.data.nodes.length
    const edgeCount = this.data.edges.length
    const maxPossibleEdges = (nodeCount * (nodeCount - 1)) / 2
    const density = maxPossibleEdges > 0 ? edgeCount / maxPossibleEdges : 0

    // Calculate average degree
    let totalDegree = 0
    this.adjacencyList.forEach(neighbors => {
      totalDegree += neighbors.size
    })
    const avgDegree = nodeCount > 0 ? totalDegree / nodeCount : 0

    // Calculate clustering coefficient (simplified)
    let clusteringSum = 0
    this.data.nodes.forEach(node => {
      const neighbors = this.adjacencyList.get(node.id) || new Map()
      if (neighbors.size < 2) return

      let neighborEdges = 0
      neighbors.forEach((_, neighborId) => {
        const neighborNeighbors = this.adjacencyList.get(neighborId) || new Map()
        neighborEdges += Array.from(neighbors.keys()).filter(id =>
          id !== neighborId && neighborNeighbors.has(id)
        ).length
      })

      clusteringSum += neighborEdges / (neighbors.size * (neighbors.size - 1))
    })
    const clustering = nodeCount > 0 ? clusteringSum / nodeCount : 0

    // Count connected components
    const components = this.countConnectedComponents()

    return {
      nodeCount,
      edgeCount,
      density,
      avgDegree,
      clustering,
      components
    }
  }

  private countConnectedComponents(): number {
    if (this.data.nodes.length === 0) return 0

    const visited = new Set<string>()
    let components = 0

    const dfs = (nodeId: string): void => {
      if (visited.has(nodeId)) return
      visited.add(nodeId)

      const neighbors = this.adjacencyList.get(nodeId) || new Map()
      neighbors.forEach((_, neighborId) => {
        dfs(neighborId)
      })
    }

    for (const node of this.data.nodes) {
      if (!visited.has(node.id)) {
        components++
        dfs(node.id)
      }
    }

    return components
  }

  /**
   * Find nodes matching a search query
   */
  searchNodes(query: string, filters?: {
    type?: string[]
    cluster?: number
    minWeight?: number
  }): GraphNode[] {
    if (!this.initialized) return []

    const searchTerm = query.toLowerCase()
    let results = this.data.nodes.filter(node =>
      node.label.toLowerCase().includes(searchTerm) ||
      Object.values(node.metadata).some(value =>
        String(value).toLowerCase().includes(searchTerm)
      )
    )

    if (filters) {
      if (filters.type && filters.type.length > 0) {
        results = results.filter(node => filters.type!.includes(node.type))
      }
      if (filters.cluster !== undefined) {
        results = results.filter(node => node.cluster === filters.cluster)
      }
      if (filters.minWeight !== undefined) {
        results = results.filter(node => node.weight >= filters.minWeight!)
      }
    }

    return results.sort((a, b) => b.weight - a.weight)
  }

  /**
   * Get cluster information
   */
  getCluster(clusterId: number): ClusterInfo | null {
    return this.clusters.get(clusterId) || null
  }

  /**
   * Get all clusters
   */
  getAllClusters(): ClusterInfo[] {
    return Array.from(this.clusters.values()).sort((a, b) => b.weight - a.weight)
  }

  /**
   * Update graph data
   */
  updateData(newData: GraphData): void {
    this.data = newData
    this.clusters.clear()
    this.initialize()
  }

  /**
   * Get neighbors of a node
   */
  getNeighbors(nodeId: string): { nodes: GraphNode[]; edges: GraphEdge[] } {
    if (!this.initialized || !this.adjacencyList.has(nodeId)) {
      return { nodes: [], edges: [] }
    }

    const neighbors = this.adjacencyList.get(nodeId)!
    const neighborNodes: GraphNode[] = []
    const neighborEdges: GraphEdge[] = []

    neighbors.forEach((edge, neighborId) => {
      const node = this.nodeMap.get(neighborId)
      if (node) {
        neighborNodes.push(node)
        neighborEdges.push(edge)
      }
    })

    return { nodes: neighborNodes, edges: neighborEdges }
  }

  /**
   * Calculate node importance (centrality)
   */
  calculateNodeImportance(): Map<string, number> {
    const importance = new Map<string, number>()

    // Simple degree centrality with weight consideration
    this.data.nodes.forEach(node => {
      const neighbors = this.adjacencyList.get(node.id) || new Map()
      let totalWeight = 0
      neighbors.forEach(edge => {
        totalWeight += edge.weight
      })
      importance.set(node.id, totalWeight)
    })

    return importance
  }
}