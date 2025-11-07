# Advanced RAG System for Manufacturing Knowledge Base

A sophisticated Retrieval-Augmented Generation (RAG) system inspired by LangChain and LlamaIndex, specifically optimized for manufacturing knowledge management and query processing.

## 🚀 Features

### Core RAG Components

#### 1. **Hierarchical Document Chunking**
- **Multiple Strategies**: Semantic, recursive, fixed-size, and sliding window chunking
- **Context Preservation**: Maintains document structure and relationships
- **Manufacturing-Specific**: Handles technical specifications, procedures, and standards
- **Multi-Modal Support**: Processes text, images, tables, and charts from PDF/Excel documents

#### 2. **Advanced Multi-Modal Retrieval**
- **Hybrid Search**: Combines semantic, keyword, and entity-based retrieval
- **Cross-Modal Enhancement**: Text queries can retrieve image/table content
- **Intelligent Ranking**: Temporal, popularity, and relevance-based scoring
- **Manufacturing Context**: Entity recognition for equipment, standards, and procedures

#### 3. **Conversation Memory System**
- **Context-Aware Processing**: Maintains conversation history and context
- **Entity Tracking**: Follows manufacturing entities across conversations
- **Session Management**: Handles multiple user sessions with timeout
- **Smart Summarization**: Maintains relevant context without memory bloat

#### 4. **Query Decomposition**
- **Complex Query Handling**: Breaks down complex manufacturing queries
- **Intent Recognition**: Identifies query types and user intent
- **Entity Extraction**: Extracts manufacturing-specific entities
- **Relationship Mapping**: Understands connections between concepts

#### 5. **Citation Tracking & Source Verification**
- **Comprehensive Citation System**: Tracks all sources used in responses
- **Trust Scoring**: Evaluates source credibility and reliability
- **Cross-Verification**: Checks for conflicts and correlations between sources
- **Manufacturing Standards**: Prioritizes official documentation and standards

#### 6. **SQLite Database Integration**
- **Backward Compatibility**: Maintains existing database structure
- **Seamless Migration**: Automatically migrates legacy data
- **Performance Optimized**: Efficient indexing and retrieval
- **Data Integrity**: Comprehensive backup and recovery

## 📁 Project Structure

```
rag/
├── README.md                    # This file
├── advanced_rag_system.py      # Main RAG system orchestrator
└── core/
    ├── document_chunker.py     # Hierarchical document chunking
    ├── conversation_memory.py  # Conversation memory and context
    ├── multi_modal_retriever.py # Multi-modal retrieval system
    ├── query_decomposer.py     # Query decomposition engine
    ├── citation_tracker.py     # Citation tracking and verification
    └── database_integration.py # SQLite database integration layer
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- SQLite 3
- Sufficient disk space for document storage

### Required Dependencies

```bash
# Core dependencies
pip3 install langchain==0.1.0 langchain-community==0.0.20
pip3 install chromadb sentence-transformers rank-bm25 faiss-cpu
pip3 install numpy scikit-learn pandas

# Optional: For advanced NLP features
pip3 install spacy nltk transformers torch

# Optional: For image processing
pip3 install pillow opencv-python pytesseract

# Optional: For PDF processing
pip3 install PyPDF2 pdfplumber
```

### Setup

1. **Clone or navigate to the project directory**
   ```bash
   cd /path/to/knowledge/base/rag
   ```

2. **Install dependencies**
   ```bash
   pip3 install -r requirements.txt  # if available
   # or install individually using commands above
   ```

3. **Initialize the system**
   ```python
   from rag.advanced_rag_system import create_advanced_rag_system

   # Create RAG system with default configuration
   rag_system = create_advanced_rag_system()

   # Initialize the system (handles database setup and migration)
   await rag_system.initialize()
   ```

## 📖 Usage Examples

### Basic Query Processing

```python
from rag.advanced_rag_system import create_advanced_rag_system, RAGQuery, ContentType
import asyncio

async def basic_query_example():
    # Initialize RAG system
    rag_system = create_advanced_rag_system()
    await rag_system.initialize()

    # Create a query
    query = RAGQuery(
        query_id="example_001",
        text="What are the safety procedures for CNC machine operation?",
        max_results=5,
        include_citations=True
    )

    # Process the query
    response = await rag_system.query(query)

    print(f"Answer: {response.answer}")
    print(f"Sources: {len(response.sources)}")
    print(f"Citations: {len(response.citations)}")
    print(f"Confidence: {response.confidence_score:.2f}")

# Run the example
asyncio.run(basic_query_example())
```

### Adding Documents

```python
async def add_document_example():
    rag_system = create_advanced_rag_system()
    await rag_system.initialize()

    # Add a text document
    doc_id = await rag_system.add_document(
        content="CNC Machine Safety Procedures:\n1. Always wear appropriate PPE...\n2. Ensure machine guards are in place...",
        content_type=ContentType.TEXT,
        metadata={
            "title": "CNC Safety Guidelines",
            "department": "Safety",
            "version": "2.1",
            "author": "Safety Team"
        }
    )

    print(f"Document added with ID: {doc_id}")
```

### Conversation with Context

```python
async def conversation_example():
    rag_system = create_advanced_rag_system()
    await rag_system.initialize()

    session_id = "user_session_001"

    # First query
    query1 = RAGQuery(
        query_id="conv_001",
        text="What are the maintenance requirements for our primary CNC machines?",
        session_id=session_id
    )

    response1 = await rag_system.query(query1)
    print(f"Response 1: {response1.answer[:100]}...")

    # Follow-up query (will use conversation context)
    query2 = RAGQuery(
        query_id="conv_002",
        text="What about the secondary machines?",
        session_id=session_id
    )

    response2 = await rag_system.query(query2)
    print(f"Response 2: {response2.answer[:100]}...")
```

### Streaming Responses

```python
async def streaming_example():
    rag_system = create_advanced_rag_system()
    await rag_system.initialize()

    query = RAGQuery(
        query_id="stream_001",
        text="Explain the quality control process for manufactured parts",
        session_id="stream_session"
    )

    # Stream the response
    async for chunk in rag_system.stream_query(query):
        data = json.loads(chunk)
        if data['type'] == 'chunk':
            print(data['content'], end='', flush=True)
        elif data['type'] == 'metadata':
            print(f"\n\nSources used: {data['sources_count']}")
        elif data['type'] == 'error':
            print(f"Error: {data['message']}")
```

## 🎯 Advanced Features

### Query Decomposition

The system automatically decomposes complex queries:

```python
# Complex query example
complex_query = RAGQuery(
    query_id="complex_001",
    text="Compare the safety procedures and maintenance schedules between Haas VF-2 and DMG MORI DMU 50 CNC machines, focusing on monthly inspections and emergency protocols"
)

response = await rag_system.query(complex_query)
# The system will break this down into sub-queries about:
# - Haas VF-2 safety procedures
# - Haas VF-2 maintenance schedules
# - DMG MORI DMU 50 safety procedures
# - DMG MORI DMU 50 maintenance schedules
# - Monthly inspection requirements
# - Emergency protocols
# - Comparative analysis
```

### Multi-Modal Retrieval

```python
# Query that can retrieve from multiple content types
multi_modal_query = RAGQuery(
    query_id="multimodal_001",
    text="Show me the wiring diagrams and torque specifications for the machine",
    content_types=[ContentType.TEXT, ContentType.IMAGE, ContentType.TABLE]
)

response = await rag_system.query(multi_modal_query)
# Can retrieve text descriptions, images, and tabular data
```

### Citation and Source Verification

```python
# Query with citation requirements
cited_query = RAGQuery(
    query_id="cited_001",
    text="What are the ISO 9001 requirements for quality management?",
    include_citations=True
)

response = await rag_system.query(cited_query)

# Access citations
for citation in response.citations:
    print(f"Source: {citation.source_id}")
    print(f"Content: {citation.content_snippet}")
    print(f"Confidence: {citation.confidence_score}")
    print(f"Type: {citation.citation_type.value}")
```

## ⚙️ Configuration

The system can be configured with various options:

```python
from rag.advanced_rag_system import RAGSystemConfig, RetrievalStrategy

config = RAGSystemConfig(
    db_path="custom_knowledge_base.db",
    default_max_results=15,
    default_retrieval_strategy=RetrievalStrategy.SEMANTIC,
    enable_multi_modal=True,
    enable_conversation_memory=True,
    enable_query_decomposition=True,
    enable_citation_tracking=True,
    cache_ttl_hours=48,
    log_level="DEBUG"
)

rag_system = create_advanced_rag_system(config)
```

### Retrieval Strategies

- **SEMANTIC**: Vector similarity-based retrieval
- **KEYWORD**: Traditional keyword search
- **ENTITY**: Entity-based retrieval for manufacturing concepts
- **HYBRID**: Combines multiple strategies (default)
- **MULTI_MODAL**: Cross-modal content retrieval

## 📊 System Monitoring

### Get System Statistics

```python
stats = await rag_system.get_system_stats()

print(f"Database health: {stats['system_health']}")
print(f"Total documents: {stats['database']['rag_components']['document_chunks']}")
print(f"Active sessions: {stats['sessions']['active_sessions']}")
print(f"Average response time: {stats['performance']['avg_response_time_ms']:.2f}ms")
```

### Database Compatibility Check

```python
from rag.core.database_integration import create_database_integration

db_integration = create_database_integration()
compatibility = db_integration.perform_compatibility_check()

if compatibility['status'] == 'passed':
    print("Database is fully compatible")
elif compatibility['status'] == 'warning':
    print("Database needs attention:")
    for recommendation in compatibility['recommendations']:
        print(f"  - {recommendation}")
```

## 🔧 Database Integration

### Automatic Migration

The system automatically handles migration from legacy databases:

```python
# The initialize() method automatically checks and migrates
await rag_system.initialize()

# Or manually trigger migration
migration_result = db_integration.migrate_legacy_data()
print(f"Migration status: {migration_result.success}")
print(f"Records migrated: {migration_result.records_migrated}")
```

### Backup and Recovery

```python
# Create backup
backup_path = db_integration.backup_database()
print(f"Backup created at: {backup_path}")

# Restore from backup
success = db_integration.restore_database(backup_path)
print(f"Restore successful: {success}")
```

## 🎯 Manufacturing-Specific Features

### Entity Recognition

The system recognizes manufacturing-specific entities:

- **Equipment Models**: HAAS-VM3, DMG-MORI-DMU50, FANUC-0iF
- **Standards**: ISO-9001, ANSI-AWS-D17.1, API-1104
- **Measurements**: 2500-psi, 15-kg, 0.001-inch
- **Procedures**: Maintenance-Step-3, Quality-Inspection-2A
- **Safety**: Risk-Level-High, Lockout-Tagout, PPE-Required

### Content Type Handling

- **PDF Documents**: Technical manuals, safety guidelines, standards
- **Excel Files**: Maintenance logs, quality records, inventory data
- **Images**: Diagrams, photos, schematics
- **Text Documents**: Procedures, reports, specifications

### Manufacturing Context

```python
# Manufacturing-specific query example
manufacturing_query = RAGQuery(
    query_id="mfg_001",
    text="According to ISO 9001:2015 section 8.5.1, what are the requirements for production process validation?",
    context={
        "department": "Quality Assurance",
        "standards": ["ISO 9001:2015"],
        "equipment_types": ["CNC", "CMM", "Coordinate Measuring Machine"]
    }
)

response = await rag_system.query(manufacturing_query)
# The system will understand the specific manufacturing context and standards
```

## 🚨 Error Handling

The system includes comprehensive error handling:

```python
try:
    response = await rag_system.query(query)
except Exception as e:
    print(f"Query processing failed: {e}")
    # The system returns graceful error responses even when exceptions occur
```

### Common Error Scenarios

1. **Database Connection Issues**: Automatic retry and fallback mechanisms
2. **Insufficient Sources**: Graceful degradation with partial answers
3. **Memory Limitations**: Automatic context truncation and summarization
4. **Invalid Content Types**: Clear error messages with suggestions

## 🔍 Logging and Debugging

Enable detailed logging:

```python
config = RAGSystemConfig(
    log_level="DEBUG",
    enable_performance_monitoring=True
)

rag_system = create_advanced_rag_system(config)
```

The system provides detailed logs for:
- Query processing steps
- Performance metrics
- Database operations
- Component initialization
- Error conditions

## 📈 Performance Optimization

### Caching

- **Query Cache**: Stores frequent query results
- **Embedding Cache**: Caches document embeddings
- **Session Cache**: Maintains conversation context efficiently

### Indexing

- **Search Indexes**: Optimized for different content types
- **Temporal Indexes**: Time-based retrieval optimization
- **Entity Indexes**: Fast entity-based lookups

### Scalability

- **Chunked Processing**: Handles large documents efficiently
- **Parallel Retrieval**: Concurrent multi-source retrieval
- **Memory Management**: Intelligent context window management

## 🤝 Contributing

### Adding New Content Types

1. Extend the `ContentType` enum in `document_chunker.py`
2. Add processing logic in the document chunker
3. Update the multi-modal retriever
4. Add tests for the new content type

### Adding New Retrieval Strategies

1. Extend the `RetrievalStrategy` enum
2. Implement the strategy in `multi_modal_retriever.py`
3. Add configuration options
4. Test with various query types

### Adding New Entity Types

1. Update entity extraction patterns
2. Add entity validation rules
3. Update the citation tracker for new entity types
4. Test with manufacturing-specific content

## 📄 License

This project is part of the Manufacturing Knowledge Base system and follows the same licensing terms as the main project.

## 🔗 Related Components

- **Database Integration**: Seamless SQLite integration with migration support
- **API Integration**: Compatible with existing REST API endpoints
- **Frontend Integration**: Works with the modern React frontend
- **Monitoring**: Integrates with the observability system

---

Built with ❤️ for Advanced Manufacturing Knowledge Management

This advanced RAG system provides enterprise-grade capabilities for manufacturing knowledge management, combining state-of-the-art retrieval techniques with manufacturing-specific optimizations and comprehensive source verification.