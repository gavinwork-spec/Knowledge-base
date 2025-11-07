# Knowledge Base API Python SDK 🚀

A comprehensive, type-safe Python SDK for the Knowledge Base API Suite. Provides intuitive interfaces for search, document management, user personalization, and real-time WebSocket interactions.

## ✨ Features

- 🔍 **Hybrid Search**: Semantic, keyword, and knowledge graph search
- 📚 **Document Management**: Index and manage documents with metadata
- 👤 **Personalized Search**: Privacy-first personalization with GDPR/CCPA compliance
- 🔄 **Real-time WebSocket**: Live search and suggestion updates
- 🎯 **Type Safety**: Full Pydantic model support with type hints
- 🛡️ **Error Handling**: Comprehensive error handling with custom exceptions
- 📊 **Analytics**: Built-in analytics and performance monitoring
- 🔐 **Authentication**: JWT and API key support

## 🚀 Quick Start

### Installation

```bash
pip install knowledge-base-sdk
```

### Basic Usage

```python
from knowledge_base_sdk import KnowledgeBaseClient

# Initialize client
client = KnowledgeBaseClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Perform unified search
results = client.search.unified(
    query="machine learning algorithms",
    search_strategy="unified",
    top_k=10
)

print(f"Found {len(results.results)} results")
for result in results.results:
    print(f"- {result.title} (Score: {result.score:.2f})")

# Index a document
client.documents.index(
    document_id="doc_123",
    title="Introduction to Neural Networks",
    content="Neural networks are computing systems inspired by biological neural networks...",
    metadata={
        "author": "AI Research Team",
        "category": "Machine Learning",
        "tags": ["neural networks", "deep learning", "AI"]
    }
)
```

### Personalized Search

```python
import knowledge_base_sdk as kb

# Set up user preferences
client.users.set_privacy_preferences(
    user_id="user_123",
    tracking_enabled=True,
    personalization_enabled=True,
    expertise_learning_enabled=True
)

# Give consent for personalized search
client.users.give_consent(
    user_id="user_123",
    consent_given=True,
    consent_text="I consent to personalized search and data processing",
    data_purposes=["personalization", "analytics"]
)

# Perform personalized search
results = client.search.personalized(
    query="best practices for API design",
    user_id="user_123",
    personalization_level=0.8,
    boost_expertise=True,
    boost_history=True
)

print(f"Personalization applied: {results.personalization_applied}")
print(f"User expertise domains: {results.user_expertise_domains}")
```

### Real-time WebSocket Search

```python
import asyncio
from knowledge_base_sdk import create_search_websocket

async def realtime_search():
    # Create WebSocket client
    ws = create_search_websocket(api_key="your-api-key")

    # Set up event handlers
    @ws.on("results")
    def handle_results(data):
        print(f"Search results: {len(data['results'])} items")
        for result in data["results"]:
            print(f"- {result['title']} (Score: {result['score']})")

    @ws.on("progress")
    def handle_progress(data):
        print(f"Search progress: {data['status']} - {data['message']}")

    @ws.on("error")
    def handle_error(error):
        print(f"Search error: {error}")

    # Start WebSocket client
    await ws.start()

    # Send search request
    await ws.send_search_request(
        query="real-time search implementation",
        strategy="unified",
        top_k=15
    )

    # Keep running to receive results
    await asyncio.sleep(10)
    await ws.stop()

# Run the example
asyncio.run(realtime_search())
```

## 📚 Advanced Usage

### Search Strategies

```python
# Semantic search
semantic_results = client.search.semantic({
    "query": "neural networks and deep learning",
    "top_k": 15,
    "similarity_threshold": 0.8,
    "filters": {
        "category": "AI/ML",
        "date_range": "last_6_months"
    }
})

# Keyword search
keyword_results = client.search.keyword({
    "query": "API design patterns REST",
    "top_k": 10
})

# Knowledge graph search
graph_results = client.search.knowledge_graph({
    "entity_name": "Artificial Intelligence",
    "relation_type": "related_to",
    "direction": "both",
    "max_depth": 3,
    "top_k": 10
})
```

### Document Management

```python
# Index multiple documents
documents = [
    {
        "document_id": "doc_124",
        "title": "Deep Learning Fundamentals",
        "content": "Deep learning is a subset of machine learning...",
        "metadata": {
            "author": "ML Team",
            "category": "Deep Learning",
            "tags": ["deep learning", "neural networks"]
        }
    },
    {
        "document_id": "doc_125",
        "title": "Computer Vision Applications",
        "content": "Computer vision enables machines to interpret visual information...",
        "metadata": {
            "author": "CV Team",
            "category": "Computer Vision",
            "tags": ["computer vision", "image processing"]
        }
    }
]

batch_results = client.documents.batch_index(documents)
print(f"Successfully indexed {batch_results.successful_indexed} documents")

# Remove a document
client.documents.remove("doc_123")
```

### User Management and Analytics

```python
# Track user feedback
client.users.track_feedback({
    "user_id": "user_123",
    "session_id": "sess_456",
    "result_id": "result_789",
    "feedback_type": "click",
    "dwell_time": 45.2,
    "satisfaction_score": 0.8
})

# Get user expertise profile
expertise = client.users.get_expertise_profile("user_123")
print(f"Technical level: {expertise['technical_level']}")
print(f"Expertise domains: {expertise['expertise_domains']}")

# Get analytics
analytics = client.analytics.get_search_analytics()
print(f"Total sessions: {analytics.total_sessions}")
print(f"Active connections: {analytics.active_connections}")
```

## 🔧 Configuration

### Client Configuration

```python
client = KnowledgeBaseClient(
    base_url="https://api.knowledgebase.com",
    api_key="your-api-key",
    timeout=30.0,           # Request timeout in seconds
    max_retries=3,          # Maximum retry attempts
    retry_delay=1.0,        # Delay between retries
    headers={               # Additional headers
        "X-Client-Version": "1.0.0"
    }
)
```

### WebSocket Configuration

```python
from knowledge_base_sdk import WebSocketClient

ws = WebSocketClient(
    base_url="wss://api.knowledgebase.com",
    personalized_url="wss://api.knowledgebase.com",
    api_key="your-api-key",
    ping_interval=20,           # Ping interval in seconds
    ping_timeout=20,            # Ping timeout in seconds
    max_reconnect_attempts=5,   # Maximum reconnection attempts
    reconnect_delay=1.0,        # Delay between reconnections
)
```

## 🚨 Error Handling

The SDK provides comprehensive error handling with specific exception types:

```python
from knowledge_base_sdk import (
    KnowledgeBaseError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    NotFoundError
)

try:
    results = client.search.unified({
        "query": "machine learning",
        "top_k": 10
    })
except AuthenticationError as e:
    print(f"Authentication failed: {e.message}")
except RateLimitError as e:
    print(f"Rate limit exceeded. Retry after {e.retry_after} seconds")
except ValidationError as e:
    print(f"Validation failed: {e.validation_errors}")
except NotFoundError as e:
    print(f"Resource not found: {e.resource_type} {e.resource_id}")
except KnowledgeBaseError as e:
    print(f"API error: {e.message} (Status: {e.status_code})")
```

## 📖 API Reference

### Search Manager

- `unified(request)`: Perform unified hybrid search
- `semantic(request)`: Perform semantic search
- `keyword(request)`: Perform keyword search
- `knowledge_graph(request)`: Perform knowledge graph search
- `personalized(request)`: Perform personalized search
- `suggestions(request)`: Get query suggestions

### Document Manager

- `index(request)`: Index a single document
- `batch_index(request)`: Index multiple documents
- `remove(document_id)`: Remove a document

### User Manager

- `give_consent(request)`: Record user consent
- `set_privacy_preferences(request)`: Set privacy preferences
- `get_privacy_preferences(user_id)`: Get user privacy preferences
- `track_feedback(request)`: Track user feedback
- `get_expertise_profile(user_id)`: Get user expertise profile

### Analytics Manager

- `get_search_analytics()`: Get search analytics
- `get_personalized_analytics()`: Get personalized analytics
- `get_customers()`: Get customer list
- `search_customers()`: Search customers

### Knowledge Manager

- `get_entries()`: Get knowledge entries
- `get_entry(entry_id)`: Get specific knowledge entry
- `create_entry(request)`: Create knowledge entry
- `update_entry(entry_id, request)`: Update knowledge entry
- `delete_entry(entry_id)`: Delete knowledge entry

## 🧪 Development

### Running Tests

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
pytest

# Run tests with coverage
pytest --cov=knowledge_base_sdk

# Run type checking
mypy knowledge_base_sdk
```

### Code Formatting

```bash
# Format code
black knowledge_base_sdk
isort knowledge_base_sdk

# Lint code
flake8 knowledge_base_sdk
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

- 📖 [Documentation](https://docs.knowledgebase.com/sdk/python/)
- 🐛 [Issue Tracker](https://github.com/knowledge-base/sdk-python/issues)
- 💬 [Discussions](https://github.com/knowledge-base/sdk-python/discussions)
- 📧 [Email Support](mailto:support@knowledgebase.com)

---

**Knowledge Base API Python SDK** v1.0.0 - Your intelligent knowledge management companion 🤖✨