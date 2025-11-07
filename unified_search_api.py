"""
Unified Search API Server for Hybrid Search Engine

Provides RESTful endpoints and WebSocket support for the hybrid search system
that combines semantic search, keyword search, and knowledge graph traversal.
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dataclasses import asdict
import uvicorn

from hybrid_search_engine import (
    HybridSearchEngine,
    SearchRequest,
    SearchResult,
    SemanticSearchRequest,
    KeywordSearchRequest,
    KnowledgeGraphSearchRequest,
    DocumentInfo,
    QueryExpansion,
    SearchAnalytics
)
from search_optimization import (
    SearchOptimizer,
    OptimizationConfig,
    PerformanceMonitor,
    CacheConfig
)


class SearchAPIServer:
    """Unified API server for hybrid search engine"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8006):
        self.host = host
        self.port = port
        self.search_engine: Optional[HybridSearchEngine] = None
        self.optimizer: Optional[SearchOptimizer] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.app: Optional[FastAPI] = None
        self.websocket_connections: List[WebSocket] = []
        self.search_sessions: Dict[str, Dict] = {}

        self._setup_server()

    def _setup_server(self):
        """Setup FastAPI application with lifecycle management"""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Initialize and cleanup search engine"""
            print("🔍 Initializing Hybrid Search Engine...")
            try:
                await self._initialize_search_engine()
                print("✅ Hybrid Search Engine initialized successfully")
                yield
            except Exception as e:
                print(f"❌ Failed to initialize search engine: {e}")
                raise
            finally:
                print("🔄 Shutting down Hybrid Search Engine...")
                if self.performance_monitor:
                    await self.performance_monitor.stop_monitoring()
                if self.optimizer:
                    await self.optimizer.cleanup()
                if self.search_engine:
                    await self.search_engine.aclose()

        self.app = FastAPI(
            title="Hybrid Search API",
            description="Unified search API combining semantic, keyword, and knowledge graph search",
            version="1.0.0",
            lifespan=lifespan
        )

        self._setup_middleware()
        self._setup_routes()

    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.get("/", response_model=Dict[str, Any])
        async def root():
            """Health check endpoint"""
            return {
                "service": "Hybrid Search API",
                "status": "active",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            }

        @self.app.post("/api/v1/search/unified")
        async def unified_search(request: UnifiedSearchRequest) -> Dict[str, Any]:
            """Unified search endpoint that combines all search strategies"""
            try:
                search_id = str(uuid.uuid4())
                start_time = time.time()

                # Create search request
                search_request = SearchRequest(
                    query=request.query,
                    search_strategy=request.search_strategy,
                    top_k=request.top_k,
                    similarity_threshold=request.similarity_threshold,
                    rerank=request.rerank,
                    include_metadata=request.include_metadata,
                    filters=request.filters or {}
                )

                # Execute optimized search
                if self.optimizer:
                    results = await self.optimizer.optimize_search_request(
                        search_request,
                        self.search_engine.search
                    )
                else:
                    results = await self.search_engine.search(search_request)

                execution_time = time.time() - start_time

                # Store session
                self.search_sessions[search_id] = {
                    "query": request.query,
                    "strategy": request.search_strategy,
                    "results_count": len(results.results),
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat()
                }

                return {
                    "search_id": search_id,
                    "query": request.query,
                    "strategy": request.search_strategy,
                    "results": [result.to_dict() for result in results.results],
                    "query_expansions": [exp.to_dict() for exp in results.query_expansions],
                    "aggregated_result": results.aggregated_result.to_dict() if results.aggregated_result else None,
                    "execution_time": execution_time,
                    "analytics": results.analytics.to_dict(),
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/v1/search/semantic")
        async def semantic_search(request: SemanticSearchAPIRequest) -> Dict[str, Any]:
            """Semantic search endpoint"""
            try:
                search_request = SemanticSearchRequest(
                    query=request.query,
                    top_k=request.top_k,
                    similarity_threshold=request.similarity_threshold,
                    filters=request.filters or {}
                )

                results = await self.search_engine.semantic_search_engine.search(search_request)

                return {
                    "results": [result.to_dict() for result in results],
                    "search_type": "semantic",
                    "query": request.query,
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/v1/search/keyword")
        async def keyword_search(request: KeywordSearchAPIRequest) -> Dict[str, Any]:
            """Keyword search endpoint"""
            try:
                search_request = KeywordSearchRequest(
                    query=request.query,
                    top_k=request.top_k,
                    similarity_threshold=request.similarity_threshold,
                    filters=request.filters or {}
                )

                results = await self.search_engine.keyword_search_engine.search(search_request)

                return {
                    "results": [result.to_dict() for result in results],
                    "search_type": "keyword",
                    "query": request.query,
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/v1/search/knowledge-graph")
        async def knowledge_graph_search(request: KnowledgeGraphSearchAPIRequest) -> Dict[str, Any]:
            """Knowledge graph search endpoint"""
            try:
                search_request = KnowledgeGraphSearchRequest(
                    entity_name=request.entity_name,
                    relation_type=request.relation_type,
                    direction=request.direction,
                    max_depth=request.max_depth,
                    top_k=request.top_k
                )

                results = await self.search_engine.knowledge_graph_search_engine.search(search_request)

                return {
                    "results": [result.to_dict() for result in results],
                    "search_type": "knowledge_graph",
                    "entity": request.entity_name,
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/v1/documents/index")
        async def index_document(request: IndexDocumentRequest) -> Dict[str, Any]:
            """Index a document in the hybrid search engine"""
            try:
                doc_info = DocumentInfo(
                    id=request.document_id,
                    title=request.title,
                    content=request.content,
                    metadata=request.metadata or {},
                    timestamp=datetime.now()
                )

                success = await self.search_engine.add_document(doc_info)

                return {
                    "success": success,
                    "document_id": request.document_id,
                    "message": "Document indexed successfully" if success else "Failed to index document",
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/v1/documents/batch-index")
        async def batch_index_documents(request: BatchIndexDocumentsRequest) -> Dict[str, Any]:
            """Index multiple documents in batch"""
            try:
                documents = []
                for doc in request.documents:
                    doc_info = DocumentInfo(
                        id=doc.document_id,
                        title=doc.title,
                        content=doc.content,
                        metadata=doc.metadata or {},
                        timestamp=datetime.now()
                    )
                    documents.append(doc_info)

                results = await self.search_engine.add_documents_batch(documents)

                return {
                    "results": results,
                    "total_documents": len(documents),
                    "successful_indexed": sum(1 for r in results if r),
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/api/v1/documents/{document_id}")
        async def remove_document(document_id: str) -> Dict[str, Any]:
            """Remove a document from the search index"""
            try:
                success = await self.search_engine.remove_document(document_id)

                return {
                    "success": success,
                    "document_id": document_id,
                    "message": "Document removed successfully" if success else "Failed to remove document",
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/search/suggestions")
        async def get_search_suggestions(q: str, limit: int = 5) -> Dict[str, Any]:
            """Get search suggestions based on query"""
            try:
                suggestions = await self.search_engine.get_search_suggestions(q, limit)

                return {
                    "query": q,
                    "suggestions": suggestions,
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/analytics")
        async def get_search_analytics() -> Dict[str, Any]:
            """Get search analytics and performance metrics"""
            try:
                analytics = await self.search_engine.get_analytics()
                recent_sessions = list(self.search_sessions.values())[-10:]  # Last 10 sessions

                return {
                    "engine_analytics": analytics,
                    "recent_sessions": recent_sessions,
                    "active_connections": len(self.websocket_connections),
                    "total_sessions": len(self.search_sessions),
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/optimization/report")
        async def get_optimization_report() -> Dict[str, Any]:
            """Get comprehensive optimization report"""
            try:
                if not self.optimizer:
                    raise HTTPException(status_code=503, detail="Optimizer not initialized")

                report = await self.optimizer.get_optimization_report()
                return report

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/v1/optimization/clear-cache")
        async def clear_optimization_cache() -> Dict[str, Any]:
            """Clear optimization caches"""
            try:
                if not self.optimizer:
                    raise HTTPException(status_code=503, detail="Optimizer not initialized")

                await self.optimizer.clear_caches()

                return {
                    "message": "Optimization caches cleared successfully",
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.put("/api/v1/optimization/config")
        async def update_optimization_config(config: OptimizationConfigUpdate) -> Dict[str, Any]:
            """Update optimization configuration"""
            try:
                if not self.optimizer:
                    raise HTTPException(status_code=503, detail="Optimizer not initialized")

                # Convert to OptimizationConfig object
                new_config = OptimizationConfig(
                    cache_config=CacheConfig(
                        enabled=config.cache_enabled,
                        max_size=config.cache_max_size,
                        ttl_seconds=config.cache_ttl_seconds,
                        cleanup_interval=config.cache_cleanup_interval,
                        memory_limit_mb=config.cache_memory_limit_mb
                    ),
                    enable_parallel_search=config.enable_parallel_search,
                    max_workers=config.max_workers,
                    enable_result_caching=config.enable_result_caching,
                    enable_query_cache=config.enable_query_cache,
                    enable_compression=config.enable_compression,
                    batch_size=config.batch_size,
                    prefetch_enabled=config.prefetch_enabled,
                    prefetch_count=config.prefetch_count
                )

                await self.optimizer.update_config(new_config)

                return {
                    "message": "Optimization configuration updated successfully",
                    "config": asdict(new_config),
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/health")
        async def health_check() -> Dict[str, Any]:
            """Detailed health check"""
            try:
                if not self.search_engine:
                    return {"status": "unhealthy", "reason": "Search engine not initialized"}

                # Check if search engine is responsive
                test_query = SearchRequest(
                    query="health check",
                    search_strategy="unified",
                    top_k=1
                )

                start_time = time.time()
                await self.search_engine.search(test_query)
                response_time = time.time() - start_time

                return {
                    "status": "healthy",
                    "response_time": response_time,
                    "document_count": len(self.search_engine.documents),
                    "active_connections": len(self.websocket_connections),
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                return {
                    "status": "unhealthy",
                    "reason": str(e),
                    "timestamp": datetime.now().isoformat()
                }

        @self.app.websocket("/ws/search")
        async def websocket_search(websocket: WebSocket):
            """WebSocket endpoint for real-time search"""
            await websocket.accept()
            self.websocket_connections.append(websocket)

            try:
                while True:
                    # Receive search request
                    data = await websocket.receive_json()

                    # Process search request
                    if data.get("type") == "search":
                        search_id = str(uuid.uuid4())
                        start_time = time.time()

                        # Create search request
                        search_request = SearchRequest(
                            query=data.get("query"),
                            search_strategy=data.get("strategy", "unified"),
                            top_k=data.get("top_k", 10),
                            similarity_threshold=data.get("threshold", 0.7)
                        )

                        # Send progress update
                        await websocket.send_json({
                            "type": "progress",
                            "search_id": search_id,
                            "status": "searching",
                            "message": f"Searching for: {search_request.query}"
                        })

                        # Execute search
                        results = await self.search_engine.search(search_request)
                        execution_time = time.time() - start_time

                        # Send results
                        await websocket.send_json({
                            "type": "results",
                            "search_id": search_id,
                            "results": [result.to_dict() for result in results.results],
                            "execution_time": execution_time,
                            "analytics": results.analytics.to_dict()
                        })

                    elif data.get("type") == "suggestions":
                        query = data.get("query", "")
                        suggestions = await self.search_engine.get_search_suggestions(query, 5)

                        await websocket.send_json({
                            "type": "suggestions",
                            "query": query,
                            "suggestions": suggestions
                        })

            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
                self.websocket_connections.remove(websocket)

    async def _initialize_search_engine(self):
        """Initialize the hybrid search engine with optimization"""
        try:
            # Initialize search engine
            self.search_engine = HybridSearchEngine()
            await self.search_engine.initialize()
            print("🔧 Hybrid Search Engine components initialized")

            # Initialize optimizer
            self.optimizer = SearchOptimizer()
            print("⚡ Search optimizer initialized")

            # Initialize performance monitor
            self.performance_monitor = PerformanceMonitor(self.optimizer)
            await self.performance_monitor.start_monitoring()
            print("📊 Performance monitoring started")

        except Exception as e:
            print(f"❌ Failed to initialize search engine: {e}")
            raise

    async def start(self):
        """Start the API server"""
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)

        print(f"🚀 Starting Hybrid Search API Server on {self.host}:{self.port}")
        await server.serve()

    def run(self):
        """Run the API server"""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )


# Pydantic models for API requests
class UnifiedSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    search_strategy: str = Field("unified", description="Search strategy: unified, semantic, keyword, graph, auto")
    top_k: int = Field(10, description="Number of results to return", ge=1, le=100)
    similarity_threshold: float = Field(0.7, description="Similarity threshold", ge=0.0, le=1.0)
    rerank: bool = Field(True, description="Whether to apply reranking")
    include_metadata: bool = Field(True, description="Whether to include metadata in results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")


class SemanticSearchAPIRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(10, description="Number of results to return", ge=1, le=100)
    similarity_threshold: float = Field(0.7, description="Similarity threshold", ge=0.0, le=1.0)
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")


class KeywordSearchAPIRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(10, description="Number of results to return", ge=1, le=100)
    similarity_threshold: float = Field(0.7, description="Similarity threshold", ge=0.0, le=1.0)
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")


class KnowledgeGraphSearchAPIRequest(BaseModel):
    entity_name: str = Field(..., description="Entity name to search for")
    relation_type: Optional[str] = Field(None, description="Type of relation to follow")
    direction: str = Field("both", description="Search direction: forward, backward, both")
    max_depth: int = Field(3, description="Maximum depth for graph traversal", ge=1, le=10)
    top_k: int = Field(10, description="Number of results to return", ge=1, le=100)


class IndexDocumentRequest(BaseModel):
    document_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")


class DocumentIndexRequest(BaseModel):
    document_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")


class BatchIndexDocumentsRequest(BaseModel):
    documents: List[DocumentIndexRequest] = Field(..., description="List of documents to index")


class OptimizationConfigUpdate(BaseModel):
    """Request model for updating optimization configuration"""
    cache_enabled: bool = Field(True, description="Enable caching")
    cache_max_size: int = Field(1000, description="Maximum cache size", ge=1, le=10000)
    cache_ttl_seconds: int = Field(3600, description="Cache TTL in seconds", ge=60, le=86400)
    cache_cleanup_interval: int = Field(300, description="Cache cleanup interval", ge=60, le=3600)
    cache_memory_limit_mb: int = Field(512, description="Cache memory limit in MB", ge=64, le=4096)
    enable_parallel_search: bool = Field(True, description="Enable parallel search")
    max_workers: int = Field(4, description="Maximum worker threads", ge=1, le=16)
    enable_result_caching: bool = Field(True, description="Enable result caching")
    enable_query_cache: bool = Field(True, description="Enable query cache")
    enable_compression: bool = Field(True, description="Enable compression")
    batch_size: int = Field(100, description="Batch processing size", ge=10, le=1000)
    prefetch_enabled: bool = Field(True, description="Enable result prefetching")
    prefetch_count: int = Field(5, description="Number of results to prefetch", ge=1, le=20)


async def main():
    """Main function to run the server"""
    server = SearchAPIServer()
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())