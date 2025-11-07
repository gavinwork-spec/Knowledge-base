/**
 * Sample data generator for knowledge graph visualization
 * Provides realistic test data with various node types and relationships
 */

import { GraphData, GraphNode, GraphEdge } from './graph-engine'

/**
 * Generate sample knowledge graph data for testing and demonstration
 */
export function generateSampleData(nodeCount: number = 50): GraphData {
  const nodes: GraphNode[] = []
  const edges: GraphEdge[] = []
  const nodeTypes: GraphNode['type'][] = ['document', 'concept', 'entity', 'topic']

  // Categories for generating meaningful content
  const categories = {
    technology: ['Artificial Intelligence', 'Machine Learning', 'Neural Networks', 'Deep Learning', 'Python', 'JavaScript', 'React', 'TypeScript'],
    science: ['Physics', 'Chemistry', 'Biology', 'Mathematics', 'Statistics', 'Quantum Mechanics', 'Genetics', 'Evolution'],
    business: ['Marketing', 'Finance', 'Strategy', 'Leadership', 'Innovation', 'Entrepreneurship', 'Management', 'Economics'],
    philosophy: ['Ethics', 'Logic', 'Metaphysics', 'Epistemology', 'Aesthetics', 'Political Philosophy', 'Philosophy of Mind', 'Existentialism']
  }

  const categoryKeys = Object.keys(categories) as Array<keyof typeof categories>

  // Generate nodes
  for (let i = 0; i < nodeCount; i++) {
    const category = categoryKeys[Math.floor(Math.random() * categoryKeys.length)]
    const categoryItems = categories[category]
    const baseItem = categoryItems[Math.floor(Math.random() * categoryItems.length)]
    const suffix = Math.random() > 0.7 ? ` ${Math.floor(Math.random() * 100)}` : ''

    const node: GraphNode = {
      id: `node-${i}`,
      label: baseItem + suffix,
      type: nodeTypes[Math.floor(Math.random() * nodeTypes.length)],
      weight: Math.random() * 0.8 + 0.2, // Weight between 0.2 and 1.0
      metadata: {
        category,
        created: generateRandomDate(),
        author: generateRandomAuthor(),
        tags: generateRandomTags(category),
        description: generateRandomDescription(baseItem),
        connections: 0
      }
    }
    nodes.push(node)
  }

  // Generate edges with realistic relationship patterns
  const edgeCount = Math.floor(nodeCount * 1.5) // Average 1.5 edges per node

  for (let i = 0; i < edgeCount; i++) {
    const sourceIndex = Math.floor(Math.random() * nodeCount)
    let targetIndex = Math.floor(Math.random() * nodeCount)

    // Avoid self-loops
    while (targetIndex === sourceIndex) {
      targetIndex = Math.floor(Math.random() * nodeCount)
    }

    const sourceNode = nodes[sourceIndex]
    const targetNode = nodes[targetIndex]

    // Determine edge type based on node types and categories
    const edgeType = determineEdgeType(sourceNode, targetNode)
    const weight = calculateEdgeWeight(sourceNode, targetNode, edgeType)

    const edge: GraphEdge = {
      source: sourceNode.id,
      target: targetNode.id,
      weight,
      type: edgeType,
      metadata: {
        strength: weight > 0.7 ? 'strong' : weight > 0.4 ? 'moderate' : 'weak',
        created: generateRandomDate(),
        context: generateEdgeContext(sourceNode, targetNode, edgeType)
      }
    }
    edges.push(edge)
  }

  // Update node connection counts
  edges.forEach(edge => {
    const sourceId = typeof edge.source === 'string' ? edge.source : edge.source.id
    const targetId = typeof edge.target === 'string' ? edge.target : edge.target.id

    const sourceNode = nodes.find(n => n.id === sourceId)
    const targetNode = nodes.find(n => n.id === targetId)

    if (sourceNode) sourceNode.metadata.connections++
    if (targetNode) targetNode.metadata.connections++
  })

  return { nodes, edges }
}

/**
 * Generate time-series data for evolution visualization
 */
export function generateTimeSeriesData(baseData: GraphData, timeSteps: number = 10): {
  [timestamp: string]: GraphData
} {
  const timeSeries: { [timestamp: string]: GraphData } = {}
  const startDate = new Date('2024-01-01')

  // Start with a subset of nodes
  let currentData: GraphData = {
    nodes: baseData.nodes.slice(0, Math.floor(baseData.nodes.length * 0.2)),
    edges: []
  }

  for (let step = 0; step < timeSteps; step++) {
    const currentDate = new Date(startDate)
    currentDate.setDate(startDate.getDate() + (step * 30)) // Each step represents a month

    const timestamp = currentDate.toISOString().split('T')[0]

    if (step === 0) {
      timeSeries[timestamp] = currentData
    } else {
      // Add new nodes and edges progressively
      const nodesToAdd = Math.floor(baseData.nodes.length * 0.1)
      const startIndex = step * nodesToAdd
      const endIndex = Math.min(startIndex + nodesToAdd, baseData.nodes.length)

      const newNodes = baseData.nodes.slice(startIndex, endIndex).map(node => ({
        ...node,
        metadata: {
          ...node.metadata,
          created: timestamp
        }
      }))

      currentData = {
        nodes: [...currentData.nodes, ...newNodes],
        edges: baseData.edges.filter(edge => {
          const sourceId = typeof edge.source === 'string' ? edge.source : edge.source.id
          const targetId = typeof edge.target === 'string' ? edge.target : edge.target.id

          return currentData.nodes.some(n => n.id === sourceId) &&
                 currentData.nodes.some(n => n.id === targetId)
        })
      }

      timeSeries[timestamp] = currentData
    }
  }

  return timeSeries
}

/**
 * Generate data with specific clustering patterns
 */
export function generateClusteredData(clusterCount: number = 3, nodesPerCluster: number = 15): GraphData {
  const nodes: GraphNode[] = []
  const edges: GraphEdge[] = []
  const clusterTopics = [
    ['Machine Learning', 'Neural Networks', 'Deep Learning', 'AI', 'Python', 'TensorFlow'],
    ['Web Development', 'React', 'JavaScript', 'TypeScript', 'Frontend', 'CSS'],
    ['Data Science', 'Statistics', 'Analytics', 'Visualization', 'R', 'SQL']
  ]

  // Generate clustered nodes
  for (let cluster = 0; cluster < clusterCount; cluster++) {
    const topics = clusterTopics[cluster % clusterTopics.length]

    for (let i = 0; i < nodesPerCluster; i++) {
      const topic = topics[Math.floor(Math.random() * topics.length)]
      const suffix = Math.random() > 0.7 ? ` ${i + 1}` : ''

      const node: GraphNode = {
        id: `cluster-${cluster}-node-${i}`,
        label: topic + suffix,
        type: ['concept', 'document', 'entity'][Math.floor(Math.random() * 3)] as GraphNode['type'],
        weight: Math.random() * 0.5 + 0.5,
        cluster,
        metadata: {
          cluster,
          category: topic,
          created: generateRandomDate(),
          importance: Math.random()
        }
      }
      nodes.push(node)
    }
  }

  // Generate strong intra-cluster edges
  for (let cluster = 0; cluster < clusterCount; cluster++) {
    const clusterNodes = nodes.filter(n => n.cluster === cluster)

    for (let i = 0; i < clusterNodes.length * 2; i++) {
      const node1 = clusterNodes[Math.floor(Math.random() * clusterNodes.length)]
      const node2 = clusterNodes[Math.floor(Math.random() * clusterNodes.length)]

      if (node1.id !== node2.id) {
        const edge: GraphEdge = {
          source: node1.id,
          target: node2.id,
          weight: Math.random() * 0.4 + 0.6, // Strong weights for intra-cluster edges
          type: 'semantic',
          metadata: {
            cluster,
            strength: 'strong'
          }
        }
        edges.push(edge)
      }
    }
  }

  // Generate weak inter-cluster edges
  for (let i = 0; i < clusterCount; i++) {
    for (let j = i + 1; j < clusterCount; j++) {
      const cluster1 = nodes.filter(n => n.cluster === i)
      const cluster2 = nodes.filter(n => n.cluster === j)

      if (cluster1.length > 0 && cluster2.length > 0) {
        const node1 = cluster1[Math.floor(Math.random() * cluster1.length)]
        const node2 = cluster2[Math.floor(Math.random() * cluster2.length)]

        const edge: GraphEdge = {
          source: node1.id,
          target: node2.id,
          weight: Math.random() * 0.3 + 0.1, // Weak weights for inter-cluster edges
          type: 'reference',
          metadata: {
            strength: 'weak',
            bridge: true
          }
        }
        edges.push(edge)
      }
    }
  }

  return { nodes, edges }
}

function determineEdgeType(sourceNode: GraphNode, targetNode: GraphNode): GraphEdge['type'] {
  // Same category -> semantic relationship
  if (sourceNode.metadata.category === targetNode.metadata.category) {
    return 'semantic'
  }

  // Document -> Concept -> Entity hierarchy
  if (sourceNode.type === 'document' && targetNode.type === 'concept') return 'hierarchy'
  if (sourceNode.type === 'concept' && targetNode.type === 'entity') return 'hierarchy'
  if (sourceNode.type === 'entity' && targetNode.type === 'document') return 'reference'

  // Temporal relationship based on creation dates
  const sourceDate = new Date(sourceNode.metadata.created)
  const targetDate = new Date(targetNode.metadata.created)
  const daysDiff = Math.abs((targetDate.getTime() - sourceDate.getTime()) / (1000 * 60 * 60 * 24))

  if (daysDiff < 30) return 'temporal'

  // Default to reference
  return 'reference'
}

function calculateEdgeWeight(sourceNode: GraphNode, targetNode: GraphNode, edgeType: GraphEdge['type']): number {
  let baseWeight = 0.5

  // Adjust based on edge type
  switch (edgeType) {
    case 'semantic':
      baseWeight = 0.7
      break
    case 'hierarchy':
      baseWeight = 0.8
      break
    case 'temporal':
      baseWeight = 0.6
      break
    case 'reference':
      baseWeight = 0.4
      break
  }

  // Adjust based on node weights
  const avgNodeWeight = (sourceNode.weight + targetNode.weight) / 2
  baseWeight *= (0.5 + avgNodeWeight)

  // Add some randomness
  baseWeight += (Math.random() - 0.5) * 0.2

  // Clamp between 0.1 and 1.0
  return Math.max(0.1, Math.min(1.0, baseWeight))
}

function generateRandomDate(): string {
  const start = new Date('2024-01-01')
  const end = new Date('2024-12-31')
  const date = new Date(start.getTime() + Math.random() * (end.getTime() - start.getTime()))
  return date.toISOString().split('T')[0]
}

function generateRandomAuthor(): string {
  const authors = ['Alice Johnson', 'Bob Smith', 'Carol Davis', 'David Wilson', 'Emma Brown', 'Frank Miller']
  return authors[Math.floor(Math.random() * authors.length)]
}

function generateRandomTags(category: string): string[] {
  const tagSets = {
    technology: ['AI', 'ML', 'Programming', 'Algorithms', 'Data'],
    science: ['Research', 'Theory', 'Experiment', 'Analysis', 'Methodology'],
    business: ['Strategy', 'Management', 'Marketing', 'Finance', 'Innovation'],
    philosophy: ['Logic', 'Ethics', 'Theory', 'Analysis', 'Critical Thinking']
  }

  const availableTags = tagSets[category as keyof typeof tagSets] || tagSets.technology
  const numTags = Math.floor(Math.random() * 3) + 1
  const tags: string[] = []

  for (let i = 0; i < numTags; i++) {
    const tag = availableTags[Math.floor(Math.random() * availableTags.length)]
    if (!tags.includes(tag)) {
      tags.push(tag)
    }
  }

  return tags
}

function generateRandomDescription(topic: string): string {
  const templates = [
    `A comprehensive overview of ${topic} and its applications.`,
    `An in-depth analysis of ${topic} with practical examples.`,
    `Exploring the fundamental concepts of ${topic}.`,
    `Advanced techniques and methodologies in ${topic}.`,
    `A detailed examination of ${topic} principles and practices.`
  ]

  return templates[Math.floor(Math.random() * templates.length)]
}

function generateEdgeContext(sourceNode: GraphNode, targetNode: GraphNode, edgeType: GraphEdge['type']): string {
  const contexts = {
    semantic: `Strong conceptual relationship between ${sourceNode.label} and ${targetNode.label}`,
    hierarchy: `Hierarchical relationship: ${sourceNode.label} encompasses ${targetNode.label}`,
    temporal: `Temporal connection between ${sourceNode.label} and ${targetNode.label}`,
    reference: `${sourceNode.label} references ${targetNode.label}`
  }

  return contexts[edgeType]
}

/**
 * Export sample data generators for different use cases
 */
export const sampleDataGenerators = {
  random: generateSampleData,
  timeSeries: generateTimeSeriesData,
  clustered: generateClusteredData
}

/**
 * Pre-configured sample datasets for demonstration
 */
export const sampleDatasets = {
  small: () => generateSampleData(20),
  medium: () => generateSampleData(50),
  large: () => generateSampleData(200),
  clustered: () => generateClusteredData(4, 20),
  timeline: () => {
    const base = generateSampleData(100)
    return generateTimeSeriesData(base, 8)
  }
}