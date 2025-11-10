#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Knowledge API Server with LangChain RAG and LangFuse Observability
基于LangChain RAG和LangFuse可观测性的增强知识库API服务器

This server provides advanced RAG capabilities using LangChain with comprehensive
logging and observability through LangFuse for manufacturing knowledge base.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import os
import sys
import asyncio
from pathlib import Path

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.document_loaders import TextLoader, PDFMinerLoader, UnstructuredMarkdownLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

# LangFuse imports for observability
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler('data/processed/enhanced_knowledge_api.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
DB_PATH = "knowledge_base.db"
VECTOR_STORE_PATH = "data/processed/vector_store"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LangFuse configuration (set environment variables: LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_HOST)
try:
    langfuse = Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-..."),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-..."),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )
    langfuse_handler = CallbackHandler()
    logger.info("✅ LangFuse observability enabled")
except Exception as e:
    logger.warning(f"⚠️ LangFuse initialization failed: {e}")
    langfuse = None
    langfuse_handler = None

class EnhancedRAGSystem:
    """Enhanced RAG system with LangChain and observability"""

    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None
        self.conversation_chain = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def initialize_embeddings(self):
        """Initialize sentence transformer embeddings"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDINGS_MODEL,
                cache_folder="data/processed/embeddings_cache"
            )
            logger.info(f"✅ Initialized embeddings: {EMBEDDINGS_MODEL}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize embeddings: {e}")
            return False

    def create_vector_store(self, documents: List[Document]) -> bool:
        """Create FAISS vector store from documents"""
        try:
            if not self.embeddings:
                self.initialize_embeddings()

            self.vector_store = FAISS.from_documents(documents, self.embeddings)

            # Save vector store
            os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
            self.vector_store.save_local(VECTOR_STORE_PATH)

            logger.info(f"✅ Created vector store with {len(documents)} documents")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create vector store: {e}")
            return False

    def load_vector_store(self) -> bool:
        """Load existing vector store"""
        try:
            if not self.embeddings:
                self.initialize_embeddings()

            if os.path.exists(VECTOR_STORE_PATH):
                self.vector_store = FAISS.load_local(
                    VECTOR_STORE_PATH,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("✅ Loaded existing vector store")
                return True
            else:
                logger.warning("⚠️ No existing vector store found")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to load vector store: {e}")
            return False

    def create_qa_chain(self, llm_type: str = "openai") -> bool:
        """Create QA chain with specified LLM"""
        try:
            if llm_type == "openai" and os.getenv("OPENAI_API_KEY"):
                llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=0,
                    callbacks=[langfuse_handler] if langfuse_handler else None
                )
            else:
                # Fallback to a basic LLM or mock
                from langchain.llms import HuggingFaceHub
                llm = HuggingFaceHub(
                    repo_id="google/flan-t5-large",
                    model_kwargs={"temperature": 0.1}
                )

            if self.vector_store:
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=self.vector_store.as_retriever(
                        search_kwargs={"k": 5}
                    ),
                    return_source_documents=True,
                    callbacks=[langfuse_handler] if langfuse_handler else None
                )
                logger.info(f"✅ Created QA chain with {llm_type}")
                return True
            else:
                logger.error("❌ Vector store not initialized")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to create QA chain: {e}")
            return False

    def create_conversation_chain(self) -> bool:
        """Create conversational RAG chain"""
        try:
            if not self.qa_chain:
                self.create_qa_chain()

            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                callbacks=[langfuse_handler] if langfuse_handler else None
            )

            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.qa_chain.llm,
                retriever=self.vector_store.as_retriever(),
                memory=memory,
                callbacks=[langfuse_handler] if langfuse_handler else None
            )

            logger.info("✅ Created conversation chain")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create conversation chain: {e}")
            return False

    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to vector store"""
        try:
            if not self.vector_store:
                return self.create_vector_store(documents)

            self.vector_store.add_documents(documents)
            self.vector_store.save_local(VECTOR_STORE_PATH)
            logger.info(f"✅ Added {len(documents)} documents to vector store")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to add documents: {e}")
            return False

    async def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        try:
            if not self.vector_store:
                return []

            docs = self.vector_store.similarity_search(query, k=k)
            results = []
            for i, doc in enumerate(docs):
                results.append({
                    "id": i,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": 0.8  # Placeholder
                })

            return results
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []

    async def ask_question(self, question: str, use_conversation: bool = False) -> Dict:
        """Ask a question with RAG"""
        try:
            if use_conversation and self.conversation_chain:
                result = self.conversation_chain({"question": question})
            elif self.qa_chain:
                result = self.qa_chain({"query": question})
            else:
                return {"error": "QA chain not initialized"}

            return {
                "answer": result.get("result", ""),
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    } for doc in result.get("source_documents", [])
                ],
                "query": question,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Question answering failed: {e}")
            return {"error": str(e)}

class EnhancedKnowledgeAPI:
    """Enhanced Knowledge API with RAG capabilities"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = None
        self.rag_system = EnhancedRAGSystem()

    def connect(self):
        """Connect to database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.execute("PRAGMA foreign_keys = ON")
            logger.info(f"✅ Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"❌ Database connection failed: {e}")

    def disconnect(self):
        """Disconnect from database"""
        if self.conn:
            self.conn.close()
            logger.info("✅ Disconnected from database")

    def initialize_rag(self):
        """Initialize RAG system"""
        logger.info("🚀 Initializing RAG system...")

        # Load existing vector store
        if not self.rag_system.load_vector_store():
            logger.warning("⚠️ No existing vector store, will need to create one")

        # Create QA chain
        if not self.rag_system.create_qa_chain():
            logger.error("❌ Failed to create QA chain")

        return True

# Initialize API
api = EnhancedKnowledgeAPI()
api.connect()
api.initialize_rag()

# Manufacturing-specific prompt templates
MANUFACTURING_QA_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a knowledgeable manufacturing expert assistant. Based on the following context from manufacturing documentation, please provide a comprehensive and accurate answer.

Context:
{context}

Question: {question}

Please provide a detailed answer that includes:
1. Direct response to the question
2. Relevant manufacturing standards or procedures (if applicable)
3. Safety considerations (if relevant)
4. Quality requirements (if applicable)

Answer:
"""
)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Enhanced health check with RAG status"""
    try:
        rag_status = {
            "initialized": api.rag_system.vector_store is not None,
            "qa_chain_ready": api.rag_system.qa_chain is not None,
            "conversation_ready": api.rag_system.conversation_chain is not None,
            "embeddings_ready": api.rag_system.embeddings is not None,
            "langfuse_enabled": langfuse is not None
        }

        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0-enhanced",
            "rag_capabilities": rag_status
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/knowledge/search', methods=['GET'])
async def enhanced_search():
    """Enhanced search with RAG"""
    try:
        query = request.args.get('q', '').strip()
        k = int(request.args.get('k', 5))
        strategy = request.args.get('strategy', 'rag')

        if not query:
            return jsonify({"error": "Query parameter 'q' is required"}), 400

        # Log search with LangFuse
        if langfuse:
            trace = langfuse.trace(
                name="knowledge_search",
                inputs={"query": query, "k": k, "strategy": strategy}
            )

        if strategy == 'rag' and api.rag_system.vector_store:
            # Use RAG search
            results = await api.rag_system.search(query, k)

            if langfuse:
                trace.update(
                    output={"results_count": len(results)},
                    metadata={"strategy": "rag"}
                )
        else:
            # Fallback to traditional database search
            api.connect()
            cursor = api.conn.cursor()
            cursor.execute("""
                SELECT id, title, content, metadata, relevance_score
                FROM knowledge_entries
                WHERE content LIKE ? OR title LIKE ?
                ORDER BY relevance_score DESC
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", k))

            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "title": row[1],
                    "content": row[2],
                    "metadata": json.loads(row[3]) if row[3] else {},
                    "relevance_score": row[4]
                })

            if langfuse:
                trace.update(
                    output={"results_count": len(results)},
                    metadata={"strategy": "database"}
                })

        return jsonify({
            "results": results,
            "query": query,
            "strategy": strategy,
            "total_results": len(results)
        })

    except Exception as e:
        logger.error(f"❌ Enhanced search failed: {e}")
        if langfuse:
            trace.update(output={"error": str(e)}, level="ERROR")
        return jsonify({"error": str(e)}), 500

@app.route('/api/knowledge/ask', methods=['POST'])
async def ask_question():
    """Ask question with RAG"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        use_conversation = data.get('use_conversation', False)

        if not question:
            return jsonify({"error": "Question is required"}), 400

        # Log question with LangFuse
        if langfuse:
            trace = langfuse.trace(
                name="ask_question",
                inputs={"question": question, "use_conversation": use_conversation}
            )

        # Get answer from RAG system
        result = await api.rag_system.ask_question(question, use_conversation)

        if "error" in result:
            if langfuse:
                trace.update(output={"error": result["error"]}, level="ERROR")
            return jsonify(result), 500

        # Log result with LangFuse
        if langfuse:
            trace.update(
                output={
                    "answer": result["answer"],
                    "sources_count": len(result.get("source_documents", []))
                }
            )

        return jsonify(result)

    except Exception as e:
        logger.error(f"❌ Question answering failed: {e}")
        if langfuse:
            trace.update(output={"error": str(e)}, level="ERROR")
        return jsonify({"error": str(e)}), 500

@app.route('/api/knowledge/documents', methods=['POST'])
def add_document():
    """Add document to RAG system"""
    try:
        data = request.get_json()
        content = data.get('content', '').strip()
        title = data.get('title', 'Untitled Document')
        metadata = data.get('metadata', {})

        if not content:
            return jsonify({"error": "Content is required"}), 400

        # Create document
        document = Document(
            page_content=content,
            metadata={
                "title": title,
                "source": "api_upload",
                "timestamp": datetime.now().isoformat(),
                **metadata
            }
        )

        # Add to RAG system
        success = api.rag_system.add_documents([document])

        if success:
            return jsonify({
                "message": "Document added successfully",
                "title": title,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Failed to add document"}), 500

    except Exception as e:
        logger.error(f"❌ Document addition failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/knowledge/rebuild-index', methods=['POST'])
def rebuild_vector_index():
    """Rebuild vector index from database"""
    try:
        api.connect()
        cursor = api.conn.cursor()
        cursor.execute("SELECT id, title, content, metadata FROM knowledge_entries")

        documents = []
        for row in cursor.fetchall():
            metadata = json.loads(row[3]) if row[3] else {}
            metadata.update({
                "id": row[0],
                "title": row[1],
                "database_source": True
            })

            documents.append(Document(
                page_content=row[2],
                metadata=metadata
            ))

        # Create new vector store
        success = api.rag_system.create_vector_store(documents)

        if success:
            return jsonify({
                "message": f"Vector index rebuilt with {len(documents)} documents",
                "documents_count": len(documents),
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Failed to rebuild vector index"}), 500

    except Exception as e:
        logger.error(f"❌ Index rebuild failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/observability/metrics', methods=['GET'])
def get_observability_metrics():
    """Get LangFuse observability metrics"""
    try:
        if not langfuse:
            return jsonify({"error": "LangFuse not available"}), 503

        # Get recent traces (this is a simplified example)
        # In practice, you'd use LangFuse's query API
        metrics = {
            "langfuse_enabled": True,
            "total_traces": 0,  # Would query LangFuse API
            "avg_response_time": 0,
            "success_rate": 0,
            "recent_queries": []
        }

        return jsonify(metrics)

    except Exception as e:
        logger.error(f"❌ Metrics retrieval failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Enhanced Knowledge API Server with LangChain RAG and LangFuse observability...")

    # Print system status
    logger.info("📊 System Status:")
    logger.info(f"   - Vector Store: {'✅ Ready' if api.rag_system.vector_store else '❌ Not initialized'}")
    logger.info(f"   - QA Chain: {'✅ Ready' if api.rag_system.qa_chain else '❌ Not initialized'}")
    logger.info(f"   - LangFuse: {'✅ Enabled' if langfuse else '❌ Disabled'}")
    logger.info(f"   - Database: {'✅ Connected' if api.conn else '❌ Not connected'}")

    app.run(
        host='0.0.0.0',
        port=8001,
        debug=False,
        threaded=True
    )