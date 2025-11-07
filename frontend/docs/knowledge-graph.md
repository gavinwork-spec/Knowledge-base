# Knowledge Graph Visualization

A comprehensive interactive knowledge graph visualization system built with React, TypeScript, and D3.js. This system provides powerful tools for exploring relationships between documents, concepts, and entities in your knowledge base.

## Features

### Core Visualization
- **Interactive Graph View**: Force-directed layout with smooth animations
- **Node Types**: Documents, concepts, entities, and topics with distinct visual styles
- **Edge Types**: Semantic, reference, hierarchy, and temporal relationships
- **Drag & Drop**: Manually position nodes for custom layouts
- **Zoom & Pan**: Smooth navigation controls for large graphs

### Advanced Analytics
- **Path Finding**: Dijkstra's algorithm for shortest path calculation
- **Multiple Paths**: Find all possible paths between nodes
- **Node Clustering**: Louvain method for community detection
- **Graph Metrics**: Density, clustering coefficient, connected components
- **Node Importance**: Centrality measures for key node identification

### Filtering & Search
- **Type Filtering**: Filter nodes by type (document, concept, entity, topic)
- **Weight Filtering**: Show only high-importance nodes
- **Cluster Filtering**: Explore specific communities
- **Text Search**: Find nodes by label or metadata
- **Real-time Updates**: Instant filtering with smooth transitions

### Temporal Evolution
- **Timeline Visualization**: Track graph evolution over time
- **Playback Controls**: Play, pause, and adjust playback speed
- **Time Scrubber**: Navigate to specific time periods
- **Growth Tracking**: See how nodes and connections are added

### User Interface
- **Responsive Design**: Works seamlessly on desktop and mobile
- **Dark/Light Themes**: Automatic theme switching
- **Accessibility**: Full keyboard navigation and screen reader support
- **Performance Optimized**: Efficient rendering for large graphs
- **Export Options**: Export graph data as JSON or images

## Architecture

### Components

#### GraphVisualization (`components/graph/graph-visualization.tsx`)
The main visualization component that orchestrates the entire graph display:

```typescript
interface GraphVisualizationProps {
  data: GraphData
  width?: number
  height?: number
  onNodeClick?: (node: GraphNode) => void
  onEdgeClick?: (edge: GraphEdge) => void
  className?: string
}
```

**Key Features:**
- D3.js force simulation with collision detection
- Interactive controls for zoom, pan, and selection
- Real-time filtering and clustering
- Path finding visualization
- Timeline playback controls

#### GraphEngine (`lib/graph-engine.ts`)
High-performance graph algorithms and data structures:

```typescript
class GraphEngine {
  // Core algorithms
  findShortestPath(sourceId: string, targetId: string): PathResult | null
  findAllPaths(sourceId: string, targetId: string, maxDepth?: number): PathResult[]
  performClustering(resolution?: number): ClusterInfo[]

  // Analysis methods
  calculateMetrics(): GraphMetrics
  calculateNodeImportance(): Map<string, number>
  searchNodes(query: string, filters?: SearchFilters): GraphNode[]
}
```

**Performance Optimizations:**
- Efficient adjacency list representation
- Priority queue for shortest path calculations
- Memoized distance calculations
- Lazy clustering evaluation

### Data Structures

#### GraphNode
```typescript
interface GraphNode {
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
}
```

#### GraphEdge
```typescript
interface GraphEdge {
  source: string | GraphNode
  target: string | GraphNode
  weight: number
  type: 'semantic' | 'reference' | 'hierarchy' | 'temporal'
  metadata: Record<string, any>
}
```

#### GraphData
```typescript
interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}
```

## Usage

### Basic Implementation

```typescript
import { KnowledgeGraphVisualization } from '@/components/graph/graph-visualization'
import { GraphData } from '@/lib/graph-engine'

const MyComponent = () => {
  const graphData: GraphData = {
    nodes: [
      {
        id: '1',
        label: 'Machine Learning',
        type: 'concept',
        weight: 0.9,
        metadata: { category: 'AI', created: '2024-01-01' }
      },
      // ... more nodes
    ],
    edges: [
      {
        source: '1',
        target: '2',
        weight: 0.8,
        type: 'semantic',
        metadata: { strength: 'strong' }
      },
      // ... more edges
    ]
  }

  return (
    <KnowledgeGraphVisualization
      data={graphData}
      width={800}
      height={600}
      onNodeClick={(node) => console.log('Node clicked:', node)}
      onEdgeClick={(edge) => console.log('Edge clicked:', edge)}
    />
  )
}
```

### Advanced Usage with Clustering

```typescript
import { GraphEngine } from '@/lib/graph-engine'

const MyAdvancedComponent = () => {
  const graphEngine = new GraphEngine(graphData)

  // Perform clustering
  const clusters = graphEngine.performClustering(1.0)

  // Find shortest path
  const path = graphEngine.findShortestPath('node1', 'node5')

  // Search for nodes
  const results = graphEngine.searchNodes('machine learning', {
    type: ['concept'],
    minWeight: 0.7
  })

  return (
    <KnowledgeGraphVisualization
      data={graphData}
      clusters={clusters}
      selectedPath={path}
      selectedNodes={results}
    />
  )
}
```

### Integration with Knowledge Page

The knowledge graph is integrated into the main knowledge page:

```typescript
// app/knowledge/page.tsx
import KnowledgeGraphVisualization from '@/components/graph/graph-visualization'

export default function KnowledgeGraphPage() {
  const [selectedView, setSelectedView] = useState<'graph' | 'timeline' | 'analytics'>('graph')

  return (
    <Layout>
      {/* View toggle buttons */}
      <div className="flex space-x-2 mb-6">
        <Button onClick={() => setSelectedView('graph')}>Graph View</Button>
        <Button onClick={() => setSelectedView('timeline')}>Timeline</Button>
        <Button onClick={() => setSelectedView('analytics')}>Analytics</Button>
      </div>

      {/* Graph visualization */}
      {selectedView === 'graph' && (
        <KnowledgeGraphVisualization data={graphData} />
      )}

      {/* Timeline and analytics views */}
      {/* ... */}
    </Layout>
  )
}
```

## Performance Considerations

### Large Graph Optimization

For graphs with thousands of nodes, the system includes several optimizations:

1. **Level of Detail (LOD)**
   ```typescript
   const zoomScale = transform.k
   const showLabels = zoomScale > 0.8
   const nodeSize = Math.max(3, Math.min(12, zoomScale * 8))
   ```

2. **Virtualization**
   - Only render visible nodes in the viewport
   - Use quadtree for spatial indexing
   - Batch edge rendering

3. **Web Workers**
   ```typescript
   // Perform heavy calculations in web worker
   const worker = new Worker('/graph-worker.js')
   worker.postMessage({ type: 'CLUSTER', data: graphData })
   ```

4. **Memory Management**
   - Dispose of unused simulations
   - Clear event listeners on unmount
   - Use object pooling for frequent objects

### Animation Performance

```typescript
// Use requestAnimationFrame for smooth animations
const animate = () => {
  if (simulation.alpha() > 0.01) {
    simulation.tick()
    requestAnimationFrame(animate)
  }
}

// Debounce rapid interactions
const debouncedFilter = useMemo(
  () => debounce((filters) => applyFilters(filters), 300),
  []
)
```

## Styling and Theming

### CSS Custom Properties

The graph visualization uses CSS custom properties for theming:

```css
:root {
  --graph-bg: #ffffff;
  --graph-node-default: #3b82f6;
  --graph-node-concept: #10b981;
  --graph-node-entity: #f59e0b;
  --graph-node-topic: #8b5cf6;
  --graph-edge-default: #94a3b8;
  --graph-edge-highlight: #ef4444;
}

[data-theme="dark"] {
  --graph-bg: #0f172a;
  --graph-node-default: #60a5fa;
  --graph-node-concept: #34d399;
  --graph-node-entity: #fbbf24;
  --graph-node-topic: #a78bfa;
  --graph-edge-default: #64748b;
  --graph-edge-highlight: #f87171;
}
```

### SVG Styling

```css
.graph-node {
  cursor: pointer;
  transition: all 0.2s ease;
}

.graph-node:hover {
  filter: brightness(1.2);
  stroke-width: 2px;
}

.graph-edge {
  pointer-events: stroke;
  stroke-width: 1.5;
  opacity: 0.6;
  transition: opacity 0.2s ease;
}

.graph-edge:hover {
  opacity: 1;
  stroke-width: 2px;
}
```

## Accessibility

### Keyboard Navigation

- **Tab**: Navigate between graph elements
- **Enter/Space**: Select focused node or edge
- **Arrow Keys**: Pan the graph view
- **+/-**: Zoom in/out
- **Esc**: Clear selection

### Screen Reader Support

```typescript
// ARIA labels for graph elements
const nodeElement = svg.append('g')
  .attr('role', 'button')
  .attr('aria-label', `Node: ${node.label}, Type: ${node.type}`)
  .attr('aria-describedby', `node-${node.id}-description`)

// Description element
svg.append('desc')
  .attr('id', `node-${node.id}-description`)
  .text(`${node.label} is a ${node.type} with weight ${node.weight}`)
```

### High Contrast Mode

```css
@media (prefers-contrast: high) {
  .graph-edge {
    stroke-width: 3px;
    opacity: 1;
  }

  .graph-node {
    stroke: #000000;
    stroke-width: 2px;
  }
}
```

## API Reference

### GraphVisualization Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `data` | `GraphData` | Required | Graph data to visualize |
| `width` | `number` | `800` | Width of the visualization |
| `height` | `number` | `600` | Height of the visualization |
| `onNodeClick` | `(node: GraphNode) => void` | `undefined` | Callback when node is clicked |
| `onEdgeClick` | `(edge: GraphEdge) => void` | `undefined` | Callback when edge is clicked |
| `className` | `string` | `''` | Additional CSS classes |
| `showControls` | `boolean` | `true` | Show control panel |
| `enableClustering` | `boolean` | `true` | Enable clustering features |
| `showTimeline` | `boolean` | `true` | Show timeline controls |

### GraphEngine Methods

#### Path Finding
- `findShortestPath(sourceId, targetId)` - Find shortest path using Dijkstra
- `findAllPaths(sourceId, targetId, maxDepth)` - Find all paths up to max depth

#### Clustering
- `performClustering(resolution)` - Perform Louvain clustering
- `getCluster(clusterId)` - Get specific cluster information
- `getAllClusters()` - Get all clusters sorted by weight

#### Search & Filter
- `searchNodes(query, filters)` - Search nodes with optional filters
- `getNeighbors(nodeId)` - Get neighboring nodes and edges

#### Analysis
- `calculateMetrics()` - Calculate graph-wide metrics
- `calculateNodeImportance()` - Calculate node centrality scores

## Troubleshooting

### Common Issues

1. **Graph Not Rendering**
   - Check that data structure is correct
   - Verify D3.js is properly imported
   - Ensure container has valid dimensions

2. **Performance Issues**
   - Reduce number of nodes rendered simultaneously
   - Enable virtualization for large graphs
   - Use web workers for heavy calculations

3. **Memory Leaks**
   - Properly clean up D3 simulations on unmount
   - Remove event listeners
   - Dispose of timers and intervals

### Debug Mode

Enable debug mode for detailed logging:

```typescript
const debugMode = process.env.NODE_ENV === 'development'

if (debugMode) {
  console.log('Graph data:', data)
  console.log('Simulation state:', simulation.alpha())
}
```

## Contributing

### Development Setup

```bash
npm install
npm run dev
npm run test
npm run lint
```

### Testing

```bash
# Unit tests for graph algorithms
npm run test:unit

# Integration tests for components
npm run test:integration

# Visual regression tests
npm run test:visual
```

### Code Style

The project uses:
- TypeScript for type safety
- ESLint for linting
- Prettier for formatting
- Husky for pre-commit hooks

## License

MIT License - see LICENSE file for details.