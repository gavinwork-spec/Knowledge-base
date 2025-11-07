# Knowledge Base API TypeScript/JavaScript SDK 🚀

A comprehensive, type-safe TypeScript/JavaScript SDK for the Knowledge Base API Suite. Provides intuitive interfaces for search, document management, user personalization, and real-time WebSocket interactions.

## ✨ Features

- 🔍 **Hybrid Search**: Semantic, keyword, and knowledge graph search
- 📚 **Document Management**: Index and manage documents with metadata
- 👤 **Personalized Search**: Privacy-first personalization with GDPR/CCPA compliance
- 🔄 **Real-time WebSocket**: Live search and suggestion updates
- 🎯 **Type Safety**: Full TypeScript support with comprehensive type definitions
- 🛡️ **Error Handling**: Comprehensive error handling with custom exceptions
- 📊 **Analytics**: Built-in analytics and performance monitoring
- 🔐 **Authentication**: JWT and API key support

## 🚀 Quick Start

### Installation

```bash
# npm
npm install @knowledge-base/sdk

# yarn
yarn add @knowledge-base/sdk

# pnpm
pnpm add @knowledge-base/sdk
```

### Basic Usage

```typescript
import { KnowledgeBaseClient } from '@knowledge-base/sdk';

// Initialize client
const client = new KnowledgeBaseClient({
  baseURL: 'http://localhost:8000',
  apiKey: 'your-api-key'
});

// Perform unified search
const results = await client.search.unified({
  query: 'machine learning algorithms',
  searchStrategy: 'unified',
  topK: 10
});

console.log(`Found ${results.results.length} results`);
results.results.forEach(result => {
  console.log(`- ${result.title} (Score: ${result.score})`);
});

// Index a document
await client.documents.index({
  documentId: 'doc_123',
  title: 'Introduction to Neural Networks',
  content: 'Neural networks are computing systems inspired by biological neural networks...',
  metadata: {
    author: 'AI Research Team',
    category: 'Machine Learning',
    tags: ['neural networks', 'deep learning', 'AI']
  }
});
```

### Personalized Search

```typescript
import { KnowledgeBaseClient } from '@knowledge-base/sdk';

const client = new KnowledgeBaseClient({
  apiKey: 'your-api-key'
});

// Set up user preferences
await client.users.setPrivacyPreferences({
  userId: 'user_123',
  trackingEnabled: true,
  personalizationEnabled: true,
  expertiseLearningEnabled: true
});

// Give consent for personalized search
await client.users.giveConsent({
  userId: 'user_123',
  consentGiven: true,
  consentText: 'I consent to personalized search and data processing',
  dataPurposes: ['personalization', 'analytics']
});

// Perform personalized search
const results = await client.search.personalized({
  query: 'best practices for API design',
  userId: 'user_123',
  personalizationLevel: 0.8,
  boostExpertise: true,
  boostHistory: true
});

console.log(`Personalization applied: ${results.personalizationApplied}`);
console.log(`User expertise domains:`, results.userExpertiseDomains);
```

### Real-time WebSocket Search

```typescript
import { KnowledgeBaseClient } from '@knowledge-base/sdk';

const client = new KnowledgeBaseClient({ apiKey: 'your-api-key' });

// Set up WebSocket event handlers
client.websocket.on('results', (data) => {
  console.log(`Search results: ${data.results.length} items`);
  data.results.forEach(result => {
    console.log(`- ${result.title} (Score: ${result.score})`);
  });
});

client.websocket.on('progress', (data) => {
  console.log(`Search progress: ${data.status} - ${data.message}`);
});

client.websocket.on('error', (error) => {
  console.error(`Search error: ${error.message}`);
});

// Start WebSocket connections
await client.startWebSockets();

// Send search request
await client.websocket.sendSearchRequest({
  query: 'real-time search implementation',
  strategy: 'unified',
  topK: 15
});

// Keep running to receive results
// In a real application, you'd handle this differently
setTimeout(async () => {
  await client.stopWebSockets();
}, 10000);
```

## 📚 Advanced Usage

### Search Strategies

```typescript
// Semantic search
const semanticResults = await client.search.semantic({
  query: 'neural networks and deep learning',
  topK: 15,
  similarityThreshold: 0.8,
  filters: {
    category: 'AI/ML',
    date_range: 'last_6_months'
  }
});

// Keyword search
const keywordResults = await client.search.keyword({
  query: 'API design patterns REST',
  topK: 10
});

// Knowledge graph search
const graphResults = await client.search.knowledgeGraph({
  entityName: 'Artificial Intelligence',
  relationType: 'related_to',
  direction: 'both',
  maxDepth: 3,
  topK: 10
});
```

### Document Management

```typescript
// Index multiple documents
const documents = [
  {
    documentId: 'doc_124',
    title: 'Deep Learning Fundamentals',
    content: 'Deep learning is a subset of machine learning...',
    metadata: {
      author: 'ML Team',
      category: 'Deep Learning',
      tags: ['deep learning', 'neural networks']
    }
  },
  {
    documentId: 'doc_125',
    title: 'Computer Vision Applications',
    content: 'Computer vision enables machines to interpret visual information...',
    metadata: {
      author: 'CV Team',
      category: 'Computer Vision',
      tags: ['computer vision', 'image processing']
    }
  }
];

const batchResults = await client.documents.batchIndex({ documents });
console.log(`Successfully indexed ${batchResults.successfulIndexed} documents`);

// Remove a document
await client.documents.remove('doc_123');
```

### User Management and Analytics

```typescript
// Track user feedback
await client.users.trackFeedback({
  userId: 'user_123',
  sessionId: 'sess_456',
  resultId: 'result_789',
  feedbackType: 'click',
  dwellTime: 45.2,
  satisfactionScore: 0.8
});

// Get user expertise profile
const expertise = await client.users.getExpertiseProfile('user_123');
console.log(`Technical level: ${expertise.technicalLevel}`);
console.log(`Expertise domains: ${expertise.expertiseDomains}`);

// Get analytics
const analytics = await client.analytics.getSearchAnalytics();
console.log(`Total sessions: ${analytics.totalSessions}`);
console.log(`Active connections: ${analytics.activeConnections}`);
```

## 🔧 Configuration

### Client Configuration

```typescript
const client = new KnowledgeBaseClient({
  baseURL: 'https://api.knowledgebase.com',
  apiKey: 'your-api-key',
  timeout: 30000,           // Request timeout in milliseconds
  maxRetries: 3,            // Maximum retry attempts
  retryDelay: 1000,         // Delay between retries in milliseconds
  headers: {                // Additional headers
    'X-Client-Version': '1.0.0'
  }
});
```

### WebSocket Configuration

```typescript
import { createSearchWebSocket } from '@knowledge-base/sdk';

const ws = createSearchWebSocket({
  baseURL: 'wss://api.knowledgebase.com',
  personalizedURL: 'wss://api.knowledgebase.com',
  apiKey: 'your-api-key',
  pingInterval: 20000,           // Ping interval in milliseconds
  pingTimeout: 20000,            // Ping timeout in milliseconds
  maxReconnectAttempts: 5,       // Maximum reconnection attempts
  reconnectDelay: 1000,          // Delay between reconnections
});
```

## 🚨 Error Handling

The SDK provides comprehensive error handling with specific exception types:

```typescript
import {
  KnowledgeBaseError,
  AuthenticationError,
  RateLimitError,
  ValidationError,
  NotFoundError,
  isAPIError,
  isValidationError
} from '@knowledge-base/sdk';

try {
  const results = await client.search.unified({
    query: 'machine learning',
    topK: 10
  });
} catch (error) {
  if (error instanceof AuthenticationError) {
    console.error(`Authentication failed: ${error.message}`);
  } else if (error instanceof RateLimitError) {
    console.error(`Rate limit exceeded. Retry after ${error.retryAfter} seconds`);
  } else if (isValidationError(error)) {
    console.error(`Validation failed:`, error.validationErrors);
  } else if (error instanceof NotFoundError) {
    console.error(`Resource not found: ${error.resourceType} ${error.resourceId}`);
  } else if (isAPIError(error)) {
    console.error(`API error: ${error.message} (Status: ${error.status})`);
  } else {
    console.error(`Unknown error:`, error);
  }
}
```

## 📖 API Reference

### Search Manager

- `unified(request)`: Perform unified hybrid search
- `semantic(request)`: Perform semantic search
- `keyword(request)`: Perform keyword search
- `knowledgeGraph(request)`: Perform knowledge graph search
- `personalized(request)`: Perform personalized search
- `suggestions(request)`: Get query suggestions

### Document Manager

- `index(request)`: Index a single document
- `batchIndex(request)`: Index multiple documents
- `remove(documentId)`: Remove a document

### User Manager

- `giveConsent(request)`: Record user consent
- `setPrivacyPreferences(request)`: Set privacy preferences
- `getPrivacyPreferences(userId)`: Get user privacy preferences
- `trackFeedback(request)`: Track user feedback
- `getExpertiseProfile(userId)`: Get user expertise profile

### Analytics Manager

- `getSearchAnalytics()`: Get search analytics
- `getPersonalizedAnalytics()`: Get personalized analytics
- `getCustomers()`: Get customer list
- `searchCustomers()`: Search customers

### Knowledge Manager

- `getEntries()`: Get knowledge entries
- `getEntry(entryId)`: Get specific knowledge entry
- `createEntry(request)`: Create knowledge entry
- `updateEntry(entryId, request)`: Update knowledge entry
- `deleteEntry(entryId)`: Delete knowledge entry

### WebSocket Client

- `start()`: Start WebSocket connections
- `stop()`: Stop WebSocket connections
- `sendSearchRequest()`: Send search request
- `sendSuggestionRequest()`: Send suggestion request
- `on(event, handler)`: Register event handler
- `off(event, handler)`: Remove event handler

## 🧪 Development

### Building

```bash
# Install dependencies
npm install

# Build the project
npm run build

# Watch for changes and rebuild
npm run dev
```

### Testing

```bash
# Run tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage
```

### Code Quality

```bash
# Type checking
npm run typecheck

# Linting
npm run lint

# Fix linting issues
npm run lint:fix

# Format code
npm run format
```

## 🔧 TypeScript Support

This SDK is written in TypeScript and provides comprehensive type definitions:

```typescript
import {
  UnifiedSearchRequest,
  SearchResponse,
  KnowledgeBaseClientConfig
} from '@knowledge-base/sdk';

// Type-safe request creation
const request: UnifiedSearchRequest = {
  query: 'machine learning algorithms',
  searchStrategy: 'semantic',
  topK: 10,
  similarityThreshold: 0.8,
  filters: {
    category: 'AI/ML'
  }
};

// Type-safe response handling
const response: SearchResponse = await client.search.unified(request);
console.log(`Found ${response.results.length} results`);

// Type-safe configuration
const config: KnowledgeBaseClientConfig = {
  baseURL: 'https://api.knowledgebase.com',
  apiKey: 'your-api-key',
  timeout: 30000
};
```

## 🌐 Browser Usage

The SDK works in both Node.js and browser environments:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Knowledge Base SDK Example</title>
</head>
<body>
    <div id="results"></div>

    <script type="module">
        import { KnowledgeBaseClient } from 'https://cdn.skypack.dev/@knowledge-base/sdk';

        const client = new KnowledgeBaseClient({
            baseURL: 'https://api.knowledgebase.com',
            apiKey: 'your-api-key'
        });

        // Perform search
        const results = await client.search.unified({
            query: 'machine learning',
            topK: 10
        });

        // Display results
        const resultsDiv = document.getElementById('results');
        results.results.forEach(result => {
            const div = document.createElement('div');
            div.textContent = `${result.title} (Score: ${result.score})`;
            resultsDiv.appendChild(div);
        });
    </script>
</body>
</html>
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

- 📖 [Documentation](https://docs.knowledgebase.com/sdk/typescript/)
- 🐛 [Issue Tracker](https://github.com/knowledge-base/sdk-typescript/issues)
- 💬 [Discussions](https://github.com/knowledge-base/sdk-typescript/discussions)
- 📧 [Email Support](mailto:support@knowledgebase.com)

---

**Knowledge Base API TypeScript/JavaScript SDK** v1.0.0 - Your intelligent knowledge management companion 🤖✨