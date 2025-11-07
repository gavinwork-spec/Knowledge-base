"""
Interactive API Documentation Server

Serves the OpenAPI specification with interactive Swagger UI and ReDoc documentation.
Based on FastAPI's automatic documentation generation capabilities.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
import uvicorn


class APIDocumentationServer:
    """Interactive API documentation server with multiple UI options"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="API Documentation Server",
            description="Interactive documentation for Knowledge Base API Suite",
            version="1.0.0"
        )
        self.openapi_spec_path = Path(__file__).parent / "openapi_spec.yaml"

        self._setup_middleware()
        self._setup_routes()
        self._load_openapi_spec()

    def _setup_middleware(self):
        """Setup CORS middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Setup documentation routes"""

        @self.app.get("/", response_class=HTMLResponse)
        async def documentation_home():
            """Documentation home page with UI options"""
            return HTMLResponse("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Knowledge Base API Documentation</title>
                <style>
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        margin: 0;
                        padding: 40px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                    }
                    .container {
                        max-width: 1200px;
                        margin: 0 auto;
                        background: white;
                        border-radius: 12px;
                        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                        overflow: hidden;
                    }
                    .header {
                        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                        color: white;
                        padding: 40px;
                        text-align: center;
                    }
                    .header h1 {
                        margin: 0;
                        font-size: 2.5em;
                        font-weight: 300;
                    }
                    .header p {
                        margin: 10px 0 0 0;
                        opacity: 0.8;
                        font-size: 1.2em;
                    }
                    .content {
                        padding: 40px;
                    }
                    .docs-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                        gap: 30px;
                        margin-bottom: 40px;
                    }
                    .doc-card {
                        border: 1px solid #e1e8ed;
                        border-radius: 8px;
                        padding: 30px;
                        transition: all 0.3s ease;
                        text-decoration: none;
                        color: inherit;
                        display: block;
                    }
                    .doc-card:hover {
                        transform: translateY(-5px);
                        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
                        border-color: #667eea;
                    }
                    .doc-card h3 {
                        margin: 0 0 10px 0;
                        color: #2c3e50;
                        font-size: 1.4em;
                    }
                    .doc-card p {
                        margin: 0;
                        color: #7f8c8d;
                        line-height: 1.6;
                    }
                    .features {
                        background: #f8f9fa;
                        border-radius: 8px;
                        padding: 30px;
                        margin-top: 40px;
                    }
                    .features h3 {
                        margin: 0 0 20px 0;
                        color: #2c3e50;
                    }
                    .feature-list {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 15px;
                    }
                    .feature-item {
                        display: flex;
                        align-items: center;
                        gap: 10px;
                    }
                    .feature-icon {
                        width: 24px;
                        height: 24px;
                        background: #667eea;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: white;
                        font-weight: bold;
                    }
                    .badge {
                        display: inline-block;
                        background: #e74c3c;
                        color: white;
                        padding: 4px 8px;
                        border-radius: 4px;
                        font-size: 0.8em;
                        margin-left: 10px;
                    }
                    .badge.new { background: #27ae60; }
                    .badge.popular { background: #f39c12; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🚀 Knowledge Base API Suite</h1>
                        <p>Interactive Documentation & SDK Hub</p>
                    </div>

                    <div class="content">
                        <div class="docs-grid">
                            <a href="/docs" class="doc-card">
                                <h3>📚 Swagger UI</h3>
                                <p>Interactive API explorer with live testing capabilities. Try endpoints directly from your browser.</p>
                            </a>

                            <a href="/redoc" class="doc-card">
                                <h3>📖 ReDoc</h3>
                                <p>Beautiful, three-panel API documentation with rich examples and detailed descriptions.</p>
                            </a>

                            <a href="/openapi.json" class="doc-card">
                                <h3>🔧 OpenAPI Spec</h3>
                                <p>Raw OpenAPI 3.0 specification in JSON format for programmatic access and tools.</p>
                            </a>

                            <a href="/openapi.yaml" class="doc-card">
                                <h3>📄 YAML Spec</h3>
                                <p>Human-readable OpenAPI specification in YAML format for easy editing and version control.</p>
                            </a>

                            <a href="/sdk-examples" class="doc-card">
                                <h3>💻 SDK Examples <span class="badge new">New</span></h3>
                                <p>Code examples and SDK usage patterns for Python, JavaScript, TypeScript, and Go.</p>
                            </a>

                            <a href="/postman-collection" class="doc-card">
                                <h3>📮 Postman Collection <span class="badge popular">Popular</span></h3>
                                <p>Ready-to-import Postman collection with all endpoints, examples, and authentication setup.</p>
                            </a>
                        </div>

                        <div class="features">
                            <h3>🎯 Key Features</h3>
                            <div class="feature-list">
                                <div class="feature-item">
                                    <div class="feature-icon">✓</div>
                                    <span>Hybrid Search (Semantic + Keyword + Knowledge Graph)</span>
                                </div>
                                <div class="feature-item">
                                    <div class="feature-icon">✓</div>
                                    <span>Real-time WebSocket Support</span>
                                </div>
                                <div class="feature-item">
                                    <div class="feature-icon">✓</div>
                                    <span>GDPR/CCPA Compliant Personalization</span>
                                </div>
                                <div class="feature-item">
                                    <div class="feature-icon">✓</div>
                                    <span>Multi-language SDKs</span>
                                </div>
                                <div class="feature-item">
                                    <div class="feature-icon">✓</div>
                                    <span>Advanced Analytics & Monitoring</span>
                                </div>
                                <div class="feature-item">
                                    <div class="feature-icon">✓</div>
                                    <span>Batch Document Processing</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """)

        @self.app.get("/sdk-examples", response_class=HTMLResponse)
        async def sdk_examples():
            """SDK examples and code snippets"""
            return HTMLResponse("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>SDK Examples - Knowledge Base API</title>
                <style>
                    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
                    .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                    .header { background: #2c3e50; color: white; padding: 30px; border-radius: 8px 8px 0 0; }
                    .content { padding: 30px; }
                    .tabs { display: flex; border-bottom: 1px solid #e1e8ed; margin-bottom: 30px; }
                    .tab { padding: 15px 25px; cursor: pointer; border: none; background: none; font-size: 16px; border-bottom: 3px solid transparent; }
                    .tab.active { border-bottom-color: #667eea; color: #667eea; font-weight: 600; }
                    .tab-content { display: none; }
                    .tab-content.active { display: block; }
                    .code-block { background: #2d3748; color: #e2e8f0; padding: 20px; border-radius: 8px; overflow-x: auto; margin: 15px 0; }
                    .code-block pre { margin: 0; font-family: 'Monaco', 'Menlo', monospace; font-size: 14px; line-height: 1.6; }
                    .install-cmd { background: #1a202c; padding: 15px; border-radius: 6px; margin: 10px 0; font-family: monospace; }
                    .back-link { color: #667eea; text-decoration: none; display: inline-block; margin-bottom: 20px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>💻 SDK Examples & Code Snippets</h1>
                        <p>Ready-to-use code examples for multiple programming languages</p>
                    </div>
                    <div class="content">
                        <a href="/" class="back-link">← Back to Documentation</a>

                        <div class="tabs">
                            <button class="tab active" onclick="showTab('python')">Python</button>
                            <button class="tab" onclick="showTab('javascript')">JavaScript/TypeScript</button>
                            <button class="tab" onclick="showTab('go')">Go</button>
                            <button class="tab" onclick="showTab('curl')">cURL</button>
                        </div>

                        <div id="python" class="tab-content active">
                            <h3>🐍 Python SDK</h3>
                            <div class="install-cmd">pip install knowledge-base-sdk</div>

                            <h4>Basic Usage</h4>
                            <div class="code-block"><pre>from knowledge_base_sdk import KnowledgeBaseClient

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
    print(f"- {result.title} (Score: {result.score:.2f})")</pre></div>

                            <h4>Document Indexing</h4>
                            <div class="code-block"><pre># Index a single document
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

# Batch index multiple documents
documents = [
    {
        "document_id": "doc_124",
        "title": "Deep Learning Fundamentals",
        "content": "Deep learning is a subset of machine learning..."
    },
    {
        "document_id": "doc_125",
        "title": "Computer Vision Applications",
        "content": "Computer vision enables machines to interpret visual information..."
    }
]

results = client.documents.batch_index(documents)
print(f"Successfully indexed {sum(results)} documents")</pre></div>

                            <h4>Personalized Search</h4>
                            <div class="code-block"><pre># Set up user preferences
client.users.set_privacy_preferences(
    user_id="user_123",
    tracking_enabled=True,
    personalization_enabled=True,
    expertise_learning_enabled=True
)

# Give consent for personalized search
client.users.give_consent(
    user_id="user_123",
    consent_text="I consent to personalized search",
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
print(f"User expertise domains: {results.user_expertise_domains}")</pre></div>
                        </div>

                        <div id="javascript" class="tab-content">
                            <h3>📜 JavaScript/TypeScript SDK</h3>
                            <div class="install-cmd">npm install @knowledge-base/sdk</div>

                            <h4>Basic Usage</h4>
                            <div class="code-block"><pre>import { KnowledgeBaseClient } from '@knowledge-base/sdk';

// Initialize client
const client = new KnowledgeBaseClient({
  baseURL: 'http://localhost:8000',
  apiKey: 'your-api-key'
});

// Perform semantic search
const results = await client.search.semantic({
  query: 'neural networks',
  topK: 10,
  threshold: 0.7,
  filters: {
    category: 'AI/ML'
  }
});

console.log(`Found ${results.results.length} results`);
results.results.forEach(result => {
  console.log(`- ${result.title} (Score: ${result.score})`);
});</pre></div>

                            <h4>Real-time WebSocket Search</h4>
                            <div class="code-block"><pre>// Create WebSocket connection
const ws = client.websocket.search();

ws.on('open', () => {
  console.log('Connected to search WebSocket');
});

ws.on('results', (data) => {
  console.log('Search results:', data.results);
  console.log('Execution time:', data.execution_time);
});

ws.on('progress', (data) => {
  console.log('Search progress:', data.status, data.message);
});

// Send search request
ws.send({
  type: 'search',
  query: 'real-time search implementation',
  strategy: 'unified',
  topK: 15
});

// Get query suggestions
ws.send({
  type: 'suggestions',
  query: 'machine lea',
  user_id: 'user_123'
});</pre></div>

                            <h4>TypeScript Support</h4>
                            <div class="code-block"><pre>import {
  KnowledgeBaseClient,
  SearchRequest,
  SearchResult,
  PersonalizedSearchRequest
} from '@knowledge-base/sdk';

// Type-safe search requests
const searchRequest: SearchRequest = {
  query: 'API design patterns',
  searchStrategy: 'semantic',
  topK: 10,
  similarityThreshold: 0.8,
  filters: {
    category: 'Software Engineering'
  }
};

const results: SearchResult[] = await client.search.unified(searchRequest);

// Personalized search with types
const personalizedRequest: PersonalizedSearchRequest = {
  query: 'microservices architecture',
  userId: 'user_123',
  personalizationLevel: 0.9,
  boostExpertise: true,
  boostHistory: true
};</pre></div>
                        </div>

                        <div id="go" class="tab-content">
                            <h3>🐹 Go SDK</h3>
                            <div class="install-cmd">go get github.com/knowledge-base/sdk-go</div>

                            <h4>Basic Usage</h4>
                            <div class="code-block"><pre>package main

import (
    "context"
    "fmt"
    "log"

    kb "github.com/knowledge-base/sdk-go"
    "github.com/knowledge-base/sdk-go/search"
)

func main() {
    // Initialize client
    client, err := kb.NewClient(&kb.Config{
        BaseURL: "http://localhost:8000",
        APIKey: "your-api-key",
    })
    if err != nil {
        log.Fatal(err)
    }

    // Perform unified search
    results, err := client.Search.Unified(context.Background(), &search.UnifiedRequest{
        Query:          "golang best practices",
        SearchStrategy: search.StrategySemantic,
        TopK:           10,
        Threshold:      0.7,
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Found %d results\\n", len(results.Results))
    for _, result := range results.Results {
        fmt.Printf("- %s (Score: %.2f)\\n", result.Title, result.Score)
    }

    // Index a document
    err = client.Documents.Index(context.Background(), &search.IndexRequest{
        DocumentID: "doc_123",
        Title:      "Go Concurrency Patterns",
        Content:    "Go provides powerful concurrency primitives...",
        Metadata: map[string]interface{}{
            "author":   "Go Team",
            "category": "Programming",
            "tags":     []string{"go", "concurrency", "goroutines"},
        },
    })
    if err != nil {
        log.Fatal(err)
    }
}</pre></div>
                        </div>

                        <div id="curl" class="tab-content">
                            <h3>🔧 cURL Examples</h3>

                            <h4>Basic Search</h4>
                            <div class="code-block"><pre># Unified search
curl -X POST "http://localhost:8006/api/v1/search/unified" \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "machine learning algorithms",
    "search_strategy": "unified",
    "top_k": 10,
    "similarity_threshold": 0.7
  }'

# Semantic search
curl -X POST "http://localhost:8006/api/v1/search/semantic" \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "neural networks",
    "top_k": 15,
    "similarity_threshold": 0.8
  }'

# Knowledge graph search
curl -X POST "http://localhost:8006/api/v1/search/knowledge-graph" \\
  -H "Content-Type: application/json" \\
  -d '{
    "entity_name": "Artificial Intelligence",
    "relation_type": "related_to",
    "max_depth": 3,
    "top_k": 10
  }'</pre></div>

                            <h4>Document Management</h4>
                            <div class="code-block"><pre># Index a document
curl -X POST "http://localhost:8006/api/v1/documents/index" \\
  -H "Content-Type: application/json" \\
  -d '{
    "document_id": "doc_123",
    "title": "API Design Patterns",
    "content": "RESTful API design patterns and best practices...",
    "metadata": {
      "author": "API Team",
      "category": "Software Engineering",
      "tags": ["api", "rest", "design"]
    }
  }'

# Batch index documents
curl -X POST "http://localhost:8006/api/v1/documents/batch-index" \\
  -H "Content-Type: application/json" \\
  -d '{
    "documents": [
      {
        "document_id": "doc_124",
        "title": "Microservices Guide",
        "content": "Building microservices with containerization..."
      },
      {
        "document_id": "doc_125",
        "title": "Database Optimization",
        "content": "Techniques for optimizing database performance..."
      }
    ]
  }'

# Delete a document
curl -X DELETE "http://localhost:8006/api/v1/documents/doc_123"</pre></div>

                            <h4>Personalized Search</h4>
                            <div class="code-block"><pre># Set user consent
curl -X POST "http://localhost:8007/api/v1/user/consent" \\
  -H "Content-Type: application/json" \\
  -d '{
    "user_id": "user_123",
    "consent_given": true,
    "consent_text": "I consent to personalized search",
    "data_purposes": ["personalization", "analytics"]
  }'

# Personalized search
curl -X POST "http://localhost:8007/api/v1/search/personalized" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer your-jwt-token" \\
  -d '{
    "query": "advanced machine learning",
    "user_id": "user_123",
    "personalization_level": 0.8,
    "boost_expertise": true,
    "boost_history": true
  }'

# Get user expertise profile
curl -X GET "http://localhost:8007/api/v1/user/expertise/user_123" \\
  -H "Authorization: Bearer your-jwt-token"</pre></div>
                        </div>
                    </div>
                </div>

                <script>
                    function showTab(tabName) {
                        // Hide all tab contents
                        const tabContents = document.querySelectorAll('.tab-content');
                        tabContents.forEach(content => content.classList.remove('active'));

                        // Remove active class from all tabs
                        const tabs = document.querySelectorAll('.tab');
                        tabs.forEach(tab => tab.classList.remove('active'));

                        // Show selected tab content
                        document.getElementById(tabName).classList.add('active');

                        // Add active class to clicked tab
                        event.target.classList.add('active');
                    }
                </script>
            </body>
            </html>
            """)

        @self.app.get("/openapi.json")
        async def get_openapi_json():
            """Get OpenAPI specification as JSON"""
            if not self.openapi_spec:
                raise HTTPException(status_code=404, detail="OpenAPI specification not found")
            return JSONResponse(self.openapi_spec)

        @self.app.get("/openapi.yaml")
        async def get_openapi_yaml():
            """Get OpenAPI specification as YAML"""
            if not self.openapi_spec_path.exists():
                raise HTTPException(status_code=404, detail="OpenAPI YAML file not found")

            return HTMLResponse(
                content=self.openapi_spec_path.read_text(),
                media_type="text/yaml"
            )

        @self.app.get("/postman-collection")
        async def get_postman_collection():
            """Generate and return Postman collection"""
            collection = self._generate_postman_collection()
            return JSONResponse(collection)

    def _load_openapi_spec(self):
        """Load OpenAPI specification from YAML file"""
        try:
            if self.openapi_spec_path.exists():
                with open(self.openapi_spec_path, 'r') as f:
                    self.openapi_spec = yaml.safe_load(f)
            else:
                # Generate from FastAPI if file doesn't exist
                self.openapi_spec = get_openapi(
                    title="Knowledge Base API Suite",
                    version="1.0.0",
                    description="Comprehensive API suite for knowledge base management",
                    routes=self.app.routes,
                )
        except Exception as e:
            print(f"Error loading OpenAPI spec: {e}")
            self.openapi_spec = {}

    def _generate_postman_collection(self) -> Dict[str, Any]:
        """Generate Postman collection from OpenAPI spec"""
        if not self.openapi_spec:
            return {}

        collection = {
            "info": {
                "name": "Knowledge Base API Suite",
                "description": "Complete collection for Knowledge Base API testing",
                "version": "1.0.0",
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
            },
            "auth": {
                "type": "bearer",
                "bearer": [
                    {
                        "key": "token",
                        "value": "{{access_token}}",
                        "type": "string"
                    }
                ]
            },
            "variable": [
                {
                    "key": "base_url",
                    "value": "http://localhost:8000"
                },
                {
                    "key": "access_token",
                    "value": "your-api-key-or-jwt-token"
                }
            ],
            "item": []
        }

        # Add folders for different API sections
        folders = {
            "Chat Interface": [
                {
                    "name": "Natural Language Query",
                    "request": {
                        "method": "POST",
                        "header": [
                            {
                                "key": "Content-Type",
                                "value": "application/json"
                            }
                        ],
                        "body": {
                            "mode": "raw",
                            "raw": json.dumps({
                                "query": "Show me customers in California",
                                "context": {
                                    "user_id": "user_123",
                                    "session_id": "sess_456"
                                },
                                "filters": {
                                    "status": "active",
                                    "date_range": "last_30_days"
                                }
                            }, indent=2)
                        },
                        "url": {
                            "raw": "{{base_url}}/api/chat/query",
                            "host": ["{{base_url}}"],
                            "path": ["api", "chat", "query"]
                        }
                    }
                }
            ],
            "Search Operations": [
                {
                    "name": "Unified Search",
                    "request": {
                        "method": "POST",
                        "header": [
                            {
                                "key": "Content-Type",
                                "value": "application/json"
                            }
                        ],
                        "body": {
                            "mode": "raw",
                            "raw": json.dumps({
                                "query": "machine learning algorithms",
                                "search_strategy": "unified",
                                "top_k": 10,
                                "similarity_threshold": 0.7,
                                "rerank": True,
                                "include_metadata": True
                            }, indent=2)
                        },
                        "url": {
                            "raw": "{{base_url}}/api/v1/search/unified",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "search", "unified"]
                        }
                    }
                },
                {
                    "name": "Semantic Search",
                    "request": {
                        "method": "POST",
                        "header": [
                            {
                                "key": "Content-Type",
                                "value": "application/json"
                            }
                        ],
                        "body": {
                            "mode": "raw",
                            "raw": json.dumps({
                                "query": "neural networks and deep learning",
                                "top_k": 15,
                                "similarity_threshold": 0.8,
                                "filters": {
                                    "category": "AI/ML"
                                }
                            }, indent=2)
                        },
                        "url": {
                            "raw": "{{base_url}}/api/v1/search/semantic",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "search", "semantic"]
                        }
                    }
                },
                {
                    "name": "Knowledge Graph Search",
                    "request": {
                        "method": "POST",
                        "header": [
                            {
                                "key": "Content-Type",
                                "value": "application/json"
                            }
                        ],
                        "body": {
                            "mode": "raw",
                            "raw": json.dumps({
                                "entity_name": "Artificial Intelligence",
                                "relation_type": "related_to",
                                "direction": "both",
                                "max_depth": 3,
                                "top_k": 10
                            }, indent=2)
                        },
                        "url": {
                            "raw": "{{base_url}}/api/v1/search/knowledge-graph",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "search", "knowledge-graph"]
                        }
                    }
                }
            ],
            "Document Management": [
                {
                    "name": "Index Document",
                    "request": {
                        "method": "POST",
                        "header": [
                            {
                                "key": "Content-Type",
                                "value": "application/json"
                            }
                        ],
                        "body": {
                            "mode": "raw",
                            "raw": json.dumps({
                                "document_id": "doc_123",
                                "title": "Introduction to Machine Learning",
                                "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed...",
                                "metadata": {
                                    "author": "AI Research Team",
                                    "category": "Machine Learning",
                                    "tags": ["machine learning", "AI", "algorithms"],
                                    "published_date": "2024-01-15"
                                }
                            }, indent=2)
                        },
                        "url": {
                            "raw": "{{base_url}}/api/v1/documents/index",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "documents", "index"]
                        }
                    }
                },
                {
                    "name": "Batch Index Documents",
                    "request": {
                        "method": "POST",
                        "header": [
                            {
                                "key": "Content-Type",
                                "value": "application/json"
                            }
                        ],
                        "body": {
                            "mode": "raw",
                            "raw": json.dumps({
                                "documents": [
                                    {
                                        "document_id": "doc_124",
                                        "title": "Deep Learning Fundamentals",
                                        "content": "Deep learning is a type of machine learning that trains a computer to perform human-like tasks..."
                                    },
                                    {
                                        "document_id": "doc_125",
                                        "title": "Natural Language Processing",
                                        "content": "NLP is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language..."
                                    }
                                ]
                            }, indent=2)
                        },
                        "url": {
                            "raw": "{{base_url}}/api/v1/documents/batch-index",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "documents", "batch-index"]
                        }
                    }
                }
            ],
            "Personalized Search": [
                {
                    "name": "Give User Consent",
                    "request": {
                        "method": "POST",
                        "header": [
                            {
                                "key": "Content-Type",
                                "value": "application/json"
                            },
                            {
                                "key": "Authorization",
                                "value": "Bearer {{access_token}}"
                            }
                        ],
                        "body": {
                            "mode": "raw",
                            "raw": json.dumps({
                                "user_id": "user_123",
                                "consent_given": true,
                                "consent_text": "I consent to personalized search and data processing for improved recommendations and analytics.",
                                "data_purposes": ["personalization", "analytics", "improvement"]
                            }, indent=2)
                        },
                        "url": {
                            "raw": "{{base_url}}/api/v1/user/consent",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "user", "consent"]
                        }
                    }
                },
                {
                    "name": "Personalized Search",
                    "request": {
                        "method": "POST",
                        "header": [
                            {
                                "key": "Content-Type",
                                "value": "application/json"
                            },
                            {
                                "key": "Authorization",
                                "value": "Bearer {{access_token}}"
                            }
                        ],
                        "body": {
                            "mode": "raw",
                            "raw": json.dumps({
                                "query": "advanced machine learning techniques",
                                "user_id": "user_123",
                                "personalization_level": 0.8,
                                "boost_expertise": true,
                                "boost_history": true,
                                "boost_preferences": true,
                                "top_k": 10
                            }, indent=2)
                        },
                        "url": {
                            "raw": "{{base_url}}/api/v1/search/personalized",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "search", "personalized"]
                        }
                    }
                }
            ]
        }

        # Convert folders to Postman format
        for folder_name, requests in folders.items():
            folder = {
                "name": folder_name,
                "item": requests
            }
            collection["item"].append(folder)

        return collection

    def run(self):
        """Start the documentation server"""
        print(f"📚 Starting API Documentation Server on {self.host}:{self.port}")
        print(f"🔗 Swagger UI: http://{self.host}:{self.port}/docs")
        print(f"📖 ReDoc: http://{self.host}:{self.port}/redoc")
        print(f"🏠 Home: http://{self.host}:{self.port}/")
        print(f"💻 SDK Examples: http://{self.host}:{self.port}/sdk-examples")
        print(f"📮 Postman Collection: http://{self.host}:{self.port}/postman-collection")

        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )


if __name__ == "__main__":
    server = APIDocumentationServer()
    server.run()