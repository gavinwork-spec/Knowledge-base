# Manufacturing Knowledge Base System

A comprehensive AI-powered knowledge base system specifically designed for manufacturing operations, featuring advanced RAG capabilities, multi-agent orchestration, and real-time observability.

## 🚀 Key Features

### 🤖 Advanced AI Capabilities
- **Advanced RAG System**: State-of-the-art retrieval with LangChain and LlamaIndex integration
- **Multi-Agent Orchestration**: Intelligent agent coordination for complex tasks
- **Multi-Modal Processing**: Handle text, images, tables, and technical drawings
- **Query Decomposition**: Break down complex manufacturing queries
- **Conversation Memory**: Context-aware dialogue management

### 🏭 Manufacturing-Specific Features
- **Quote Management**: Automated quote generation and analysis
- **Quality Control**: Integrated quality assurance workflows
- **Compliance Tracking**: ISO and industry standard compliance
- **Document Processing**: Technical drawing and specification analysis
- **Safety Management**: Safety procedure enforcement and monitoring

### 📊 Comprehensive Observability
- **Real-time Monitoring**: WebSocket-based dashboard with live metrics
- **AI Interaction Tracking**: Detailed logging with LangFuse patterns
- **Cost Analysis**: API call cost breakdown and forecasting
- **User Analytics**: Behavior pattern recognition and insights
- **Intelligent Alerting**: Proactive anomaly detection and notification

### 🔍 Advanced Search & Retrieval
- **Hybrid Search Engine**: Multiple search strategies combined
- **Personalized Search**: User-adaptive search results
- **Semantic Search**: Concept-based understanding and matching
- **Cross-Modal Retrieval**: Search across different content types
- **Citation Tracking**: Source verification and trust scoring

## 📁 Project Structure

```
├── rag/                          # Advanced RAG System
├── multi_agent_system/           # Multi-Agent Architecture
├── observability/                # Comprehensive Monitoring System
├── github-frontend/              # Modern React Frontend
├── python_sdk/                   # Python Client SDK
├── microservices/                # Microservices Architecture
├── frontend/                     # Legacy Frontend
├── docs/                         # Documentation
├── tests/                        # Test Files
├── config/                       # Configuration Files
├── data/                         # Data and Reports
└── scripts/                      # Utility Scripts
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Node.js 16+ (for frontend)
- SQLite 3
- Docker (optional)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd manufacturing-knowledge-base
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**
   ```bash
   cd github-frontend
   npm install
   ```

4. **Initialize the database**
   ```bash
   python setup_models.py
   ```

5. **Start the system**
   ```bash
   # Start the main API server
   python api_server_knowledge.py --port 8001

   # Start the chat interface
   python api_chat_interface.py --port 8002

   # Start the frontend (optional)
   cd github-frontend && npm start
   ```

## 📖 Usage Examples

### Basic RAG Query
```python
from rag.advanced_rag_system import create_advanced_rag_system

# Initialize RAG system
rag_system = await create_advanced_rag_system()
await rag_system.initialize()

# Query the system
response = await rag_system.query(
    "What are the safety procedures for HAAS VF-2 CNC machines?"
)

print(response.answer)
```

### Multi-Agent Orchestration
```python
from multi_agent_system import create_multi_agent_orchestrator

# Initialize agent system
orchestrator = await create_multi_agent_orchestrator()

# Process complex manufacturing query
result = await orchestrator.process_query(
    "Analyze quote trends for titanium aerospace parts"
)
```

### Observability Integration
```python
from observability import create_observability_orchestrator

# Initialize observability
observability = await create_observability_orchestrator()

# Track AI interactions
await observability.log_ai_interaction(
    session_id="session_001",
    user_id="user_123",
    query="Manufacturing safety procedures",
    response="Detailed safety guidelines...",
    performance_data={"response_time_ms": 1200}
)
```

## 🔧 Configuration

### Environment Variables
```bash
# Database
DATABASE_PATH=knowledge_base.db

# AI Services (Optional)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# LangFuse (Optional)
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
```

### Advanced Configuration
See `config/` directory for detailed configuration options.

## 📊 Dashboard & Monitoring

### Real-time Dashboard
- **WebSocket Connection**: `ws://localhost:8765`
- **System Health**: CPU, memory, API performance
- **Manufacturing KPIs**: Quote accuracy, quality metrics, customer satisfaction
- **User Analytics**: Behavior patterns and knowledge gaps

### Monitoring Features
- **AI Interaction Tracking**: Complete audit trail
- **Cost Analysis**: Per-operation cost breakdown
- **Performance Metrics**: Real-time system performance
- **Alert Management**: Intelligent anomaly detection
- **User Insights**: Behavior analytics and recommendations

## 🚀 Deployment

### Docker Deployment
```bash
# Build and start all services
docker-compose up -d
```

### Production Setup
1. Configure environment variables
2. Set up monitoring and alerting
3. Configure database backups
4. Set up SSL/TLS certificates
5. Configure load balancing

## 📚 Documentation

- [API Documentation](docs/API_DESIGN.md)
- [Multi-Agent System](docs/MULTI_AGENT_SYSTEM_DOCUMENTATION.md)
- [Advanced RAG System](docs/ADVANCED_RAG_SYSTEM_DOCUMENTATION.md)
- [Observability Guide](docs/OBSERVABILITY_GUIDE.md)
- [Microservices Architecture](docs/MICROSERVICES_README.md)

## 🔍 Manufacturing Use Cases

### Quote Management
- Automated quote generation with cost analysis
- Accuracy tracking and improvement
- Customer preference learning
- Competitive analysis integration

### Quality Control
- Document classification and processing
- Quality procedure enforcement
- Compliance tracking and reporting
- Defect analysis and prevention

### Document Processing
- Technical drawing analysis
- Specification extraction
- Cross-reference linking
- Version control management

### Customer Service
- Intelligent query routing
- Personalized response generation
- Feedback integration and analysis
- Satisfaction tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the documentation
- Review the examples in the `examples/` directory

---

Built with ❤️ for Advanced Manufacturing Knowledge Management

This system combines state-of-the-art AI technology with manufacturing domain expertise to create a comprehensive knowledge management solution.
