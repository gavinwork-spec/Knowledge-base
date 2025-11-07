#!/usr/bin/env python3
"""
Advanced RAG API Server
Integrates the advanced RAG system with the existing knowledge base API endpoints.
Provides RESTful API for multimodal document processing and intelligent Q&A.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import base64
import io
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import redis
from contextlib import asynccontextmanager

# Import RAG systems
from advanced_rag_system import AdvancedRAGSystem, RetrievalStrategy
from rag_multimodal_integration import MultimodalRAGIntegrator, MultimodalQuery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for systems
rag_system = None
multimodal_integrator = None
redis_client = None

# Pydantic models for API
class DocumentUploadResponse(BaseModel):
    document_id: str
    chunk_count: int
    modalities: List[str]
    processing_time: str
    status: str

class QueryRequest(BaseModel):
    query: str = Field(..., description="The query text")
    session_id: Optional[str] = Field(None, description="Session ID for conversation memory")
    user_id: Optional[str] = Field("default", description="User ID")
    strategy: Optional[str] = Field("context_aware", description="Retrieval strategy")
    top_k: Optional[int] = Field(10, description="Number of top results to retrieve")
    modality_preference: Optional[str] = Field("auto", description="Modality preference")
    cross_modal_weight: Optional[float] = Field(0.5, description="Cross-modal search weight")
    image_query: Optional[str] = Field(None, description="Base64 encoded image for query")

class MultimodalQueryRequest(BaseModel):
    text_query: str = Field(..., description="Text query")
    image_query: Optional[str] = Field(None, description="Base64 encoded image")
    table_query: Optional[Dict[str, Any]] = Field(None, description="Table query parameters")
    modality_preference: Optional[str] = Field("auto", description="Modality preference")
    cross_modal_weight: Optional[float] = Field(0.5, description="Cross-modal weight")
    session_id: Optional[str] = Field(None, description="Session ID")
    user_id: Optional[str] = Field("default", description="User ID")

class AnswerResponse(BaseModel):
    answer: str
    confidence: float
    citations: List[Dict[str, Any]]
    sources_used: List[str]
    generation_metadata: Dict[str, Any]
    session_id: Optional[str] = None

class ConversationHistoryResponse(BaseModel):
    session_id: str
    user_id: str
    conversation_history: List[Dict[str, Any]]
    relevant_chunks_count: int
    entity_memory: Dict[str, List[str]]
    last_updated: str

class SystemStatsResponse(BaseModel):
    total_chunks: int
    index_size: int
    active_contexts: int
    multimodal_documents_processed: int
    supported_modalities: List[str]
    modality_distribution: Dict[str, int]
    system_type: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, str]
    version: str = "1.0.0"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup RAG systems"""
    global rag_system, multimodal_integrator, redis_client

    logger.info("Initializing Advanced RAG API Server...")

    try:
        # Initialize Redis client
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        redis_client.ping()
        logger.info("Redis client initialized")

        # Initialize RAG system
        rag_system = AdvancedRAGSystem(
            redis_client=redis_client,
            openai_api_key="your-openai-api-key"  # Replace with actual key
        )
        logger.info("Advanced RAG system initialized")

        # Initialize multimodal integrator
        multimodal_integrator = MultimodalRAGIntegrator(
            redis_client=redis_client
        )
        logger.info("Multimodal RAG integrator initialized")

        logger.info("Advanced RAG API Server ready!")

        yield

    except Exception as e:
        logger.error(f"Failed to initialize RAG systems: {e}")
        raise

    finally:
        # Cleanup
        logger.info("Shutting down Advanced RAG API Server...")
        if redis_client:
            redis_client.close()

# Create FastAPI app
app = FastAPI(
    title="Advanced RAG API",
    description="Advanced Retrieval-Augmented Generation API with multimodal capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper functions
def get_strategy_from_string(strategy_str: str) -> RetrievalStrategy:
    """Convert string to RetrievalStrategy enum"""
    strategy_map = {
        "semantic": RetrievalStrategy.SEMANTIC,
        "keyword": RetrievalStrategy.KEYWORD,
        "hybrid": RetrievalStrategy.HYBRID,
        "multi_hop": RetrievalStrategy.MULTI_HOP,
        "context_aware": RetrievalStrategy.CONTEXT_AWARE
    }
    return strategy_map.get(strategy_str.lower(), RetrievalStrategy.CONTEXT_AWARE)

async def process_document_background(file_path: str, document_id: str):
    """Background task for document processing"""
    try:
        await multimodal_integrator.process_multimodal_document(file_path, document_id)
        logger.info(f"Background processing completed for document {document_id}")
    except Exception as e:
        logger.error(f"Background processing failed for document {document_id}: {e}")

# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    components = {
        "rag_system": "ok" if rag_system else "not_initialized",
        "multimodal_integrator": "ok" if multimodal_integrator else "not_initialized",
        "redis": "ok" if redis_client and redis_client.ping() else "not_connected"
    }

    overall_status = "healthy" if all(status == "ok" for status in components.values()) else "unhealthy"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        components=components
    )

@app.post("/api/v1/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    process_immediately: bool = Query(True, description="Process document immediately")
):
    """Upload and process a document with multimodal capabilities"""
    if not multimodal_integrator:
        raise HTTPException(status_code=503, detail="Multimodal integrator not initialized")

    try:
        # Generate document ID
        document_id = str(uuid.uuid4())

        # Save uploaded file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / f"{document_id}_{file.filename}"

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"Document uploaded: {file.filename} -> {document_id}")

        if process_immediately:
            # Process document in background
            background_tasks.add_task(process_document_background, str(file_path), document_id)

            return DocumentUploadResponse(
                document_id=document_id,
                chunk_count=0,  # Will be updated after processing
                modalities=[],
                processing_time=datetime.now().isoformat(),
                status="processing"
            )
        else:
            # Return immediately for later processing
            return DocumentUploadResponse(
                document_id=document_id,
                chunk_count=0,
                modalities=[],
                processing_time=datetime.now().isoformat(),
                status="uploaded"
            )

    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.post("/api/v1/query", response_model=AnswerResponse)
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base using advanced RAG"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        # Convert strategy string to enum
        strategy = get_strategy_from_string(request.strategy)

        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Process query
        answer = await rag_system.query(
            query=request.query,
            session_id=session_id,
            user_id=request.user_id,
            strategy=strategy,
            top_k=request.top_k
        )

        # Convert citations to dict format
        citations_dict = []
        for citation in answer.citations:
            citations_dict.append({
                "chunk_id": citation.chunk_id,
                "content_snippet": citation.content_snippet,
                "relevance_score": citation.relevance_score,
                "page_number": citation.page_number,
                "source_document": citation.source_document,
                "confidence": citation.confidence
            })

        return AnswerResponse(
            answer=answer.answer,
            confidence=answer.confidence,
            citations=citations_dict,
            sources_used=answer.sources_used,
            generation_metadata=answer.generation_metadata,
            session_id=session_id
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/api/v1/multimodal-query", response_model=AnswerResponse)
async def multimodal_query(request: MultimodalQueryRequest):
    """Query with multimodal capabilities"""
    if not multimodal_integrator:
        raise HTTPException(status_code=503, detail="Multimodal integrator not initialized")

    try:
        # Create multimodal query
        multimodal_query_obj = MultimodalQuery(
            text_query=request.text_query,
            image_query=request.image_query,
            table_query=request.table_query,
            modality_preference=request.modality_preference,
            cross_modal_weight=request.cross_modal_weight
        )

        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Process multimodal query
        answer = await multimodal_integrator.multimodal_query(
            query=multimodal_query_obj,
            session_id=session_id,
            user_id=request.user_id,
            top_k=10
        )

        # Convert citations to dict format
        citations_dict = []
        for citation in answer.citations:
            if isinstance(citation, dict):
                citations_dict.append(citation)
            else:
                citations_dict.append({
                    "chunk_id": citation.chunk_id,
                    "content_snippet": citation.content_snippet,
                    "relevance_score": citation.relevance_score,
                    "page_number": citation.page_number,
                    "source_document": citation.source_document,
                    "confidence": citation.confidence
                })

        return AnswerResponse(
            answer=answer.answer,
            confidence=answer.confidence,
            citations=citations_dict,
            sources_used=answer.sources_used,
            generation_metadata=answer.generation_metadata,
            session_id=session_id
        )

    except Exception as e:
        logger.error(f"Error processing multimodal query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing multimodal query: {str(e)}")

@app.get("/api/v1/conversation/{session_id}/history", response_model=ConversationHistoryResponse)
async def get_conversation_history(session_id: str, user_id: str = Query("default")):
    """Get conversation history for a session"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        context = await rag_system.memory.get_context(session_id, user_id)

        return ConversationHistoryResponse(
            session_id=context.session_id,
            user_id=context.user_id,
            conversation_history=context.conversation_history,
            relevant_chunks_count=len(context.relevant_chunks),
            entity_memory=context.entity_memory,
            last_updated=context.last_updated.isoformat()
        )

    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting conversation history: {str(e)}")

@app.delete("/api/v1/conversation/{session_id}")
async def clear_conversation_history(session_id: str, user_id: str = Query("default")):
    """Clear conversation history for a session"""
    if not rag_system or not rag_system.memory:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        # Remove from cache
        cache_key = f"context:{session_id}:{user_id}"
        if cache_key in rag_system.memory.context_cache:
            del rag_system.memory.context_cache[cache_key]

        # Remove from Redis
        if rag_system.memory.redis_client:
            rag_system.memory.redis_client.delete(cache_key)

        return {"message": "Conversation history cleared successfully"}

    except Exception as e:
        logger.error(f"Error clearing conversation history: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing conversation history: {str(e)}")

@app.post("/api/v1/documents/process/{document_id}")
async def process_existing_document(document_id: str):
    """Process an uploaded document"""
    if not multimodal_integrator:
        raise HTTPException(status_code=503, detail="Multimodal integrator not initialized")

    try:
        # Find document in uploads directory
        upload_dir = Path("uploads")
        document_files = list(upload_dir.glob(f"{document_id}_*"))

        if not document_files:
            raise HTTPException(status_code=404, detail="Document not found")

        file_path = document_files[0]

        # Process document
        result = await multimodal_integrator.process_multimodal_document(str(file_path), document_id)

        return {
            "document_id": document_id,
            "status": "completed",
            "chunk_count": result['chunk_count'],
            "modalities": result['modalities'],
            "processing_time": result['processing_time']
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.get("/api/v1/statistics", response_model=SystemStatsResponse)
async def get_system_statistics():
    """Get comprehensive system statistics"""
    if not rag_system or not multimodal_integrator:
        raise HTTPException(status_code=503, detail="Systems not initialized")

    try:
        # Get multimodal statistics
        stats = await multimodal_integrator.get_multimodal_statistics()

        return SystemStatsResponse(
            total_chunks=stats.get('total_chunks', 0),
            index_size=stats.get('index_size', 0),
            active_contexts=stats.get('active_contexts', 0),
            multimodal_documents_processed=stats.get('multimodal_documents_processed', 0),
            supported_modalities=stats.get('supported_modalities', []),
            modality_distribution=stats.get('modality_distribution', {}),
            system_type=stats.get('system_type', 'advanced_multimodal_rag')
        )

    except Exception as e:
        logger.error(f"Error getting system statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting system statistics: {str(e)}")

@app.post("/api/v1/batch-query")
async def batch_query(
    queries: List[str] = Body(..., description="List of queries to process"),
    session_id: Optional[str] = Body(None, description="Session ID"),
    user_id: Optional[str] = Body("default", description="User ID")
):
    """Process multiple queries in batch"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        # Generate session ID if not provided
        session_id = session_id or str(uuid.uuid4())

        # Process batch queries
        answers = await rag_system.batch_query(queries, session_id, user_id)

        # Convert to response format
        results = []
        for i, answer in enumerate(answers):
            citations_dict = []
            for citation in answer.citations:
                citations_dict.append({
                    "chunk_id": citation.chunk_id,
                    "content_snippet": citation.content_snippet,
                    "relevance_score": citation.relevance_score,
                    "page_number": citation.page_number,
                    "source_document": citation.source_document,
                    "confidence": citation.confidence
                })

            results.append({
                "query": queries[i],
                "answer": answer.answer,
                "confidence": answer.confidence,
                "citations": citations_dict,
                "sources_used": answer.sources_used,
                "generation_metadata": answer.generation_metadata
            })

        return {
            "session_id": session_id,
            "batch_id": str(uuid.uuid4()),
            "results": results,
            "total_queries": len(queries),
            "processing_time": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error processing batch query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing batch query: {str(e)}")

@app.get("/api/v1/retrieval-strategies")
async def get_retrieval_strategies():
    """Get available retrieval strategies"""
    strategies = {
        "semantic": {
            "name": "Semantic Search",
            "description": "Uses vector embeddings for semantic similarity",
            "best_for": ["Conceptual queries", "Synonym-rich queries", "Context understanding"]
        },
        "keyword": {
            "name": "Keyword Search",
            "description": "Uses BM25 algorithm for keyword matching",
            "best_for": ["Specific terms", "Exact matches", "Technical queries"]
        },
        "hybrid": {
            "name": "Hybrid Search",
            "description": "Combines semantic and keyword search",
            "best_for": ["General queries", "Balanced approach", "Most use cases"]
        },
        "multi_hop": {
            "name": "Multi-hop Search",
            "description": "Performs multiple retrieval steps for complex queries",
            "best_for": ["Complex questions", "Multi-faceted queries", "Research tasks"]
        },
        "context_aware": {
            "name": "Context-Aware Search",
            "description": "Considers conversation history and context",
            "best_for": ["Conversational queries", "Follow-up questions", "Personalized results"]
        }
    }

    return {"strategies": strategies}

@app.get("/api/v1/supported-modalities")
async def get_supported_modalities():
    """Get supported content modalities"""
    modalities = {
        "text": {
            "name": "Text Content",
            "description": "Plain text, paragraphs, headings",
            "supported_formats": ["txt", "md", "docx", "pdf"]
        },
        "table": {
            "name": "Tabular Data",
            "description": "Structured tables with rows and columns",
            "supported_formats": ["xlsx", "xls", "csv", "pdf"]
        },
        "chart": {
            "name": "Charts & Graphs",
            "description": "Data visualizations, graphs, plots",
            "supported_formats": ["pdf", "png", "jpg", "svg"]
        },
        "image": {
            "name": "Images",
            "description": "Photographs, diagrams, illustrations",
            "supported_formats": ["png", "jpg", "jpeg", "gif", "bmp"]
        },
        "mixed": {
            "name": "Mixed Content",
            "description": "Documents with multiple content types",
            "supported_formats": ["pdf", "docx", "html"]
        }
    }

    return {"modalities": modalities}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.now().isoformat()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "timestamp": datetime.now().isoformat()
            }
        }
    )

# Main execution
if __name__ == "__main__":
    uvicorn.run(
        "advanced_rag_api:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )