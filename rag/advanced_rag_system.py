"""
Advanced RAG System Orchestrator

Main entry point for the advanced RAG system that integrates all components:
- Hierarchical document chunking
- Multi-modal retrieval
- Conversation memory
- Query decomposition
- Citation tracking
- Database integration

Inspired by LangChain and LlamaIndex patterns but optimized for manufacturing use case.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass, asdict

# Import all RAG components
from rag.core.document_chunker import DocumentChunker, DocumentChunk, ContentType
from rag.core.conversation_memory import ConversationMemory, SessionContext, ConversationTurn
from rag.core.multi_modal_retriever import MultiModalRetriever, RetrievedDocument, RetrievalStrategy
from rag.core.query_decomposer import QueryDecomposer, DecompositionPlan, QueryType
from rag.core.citation_tracker import CitationTracker, Citation, SourceType, CitationType
from rag.core.database_integration import DatabaseIntegration


@dataclass
class RAGQuery:
    """Represents a query to the RAG system"""
    query_id: str
    text: str
    session_id: Optional[str] = None
    content_types: Optional[List[ContentType]] = None
    max_results: int = 10
    include_citations: bool = True
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class RAGResponse:
    """Represents a response from the RAG system"""
    query_id: str
    answer: str
    sources: List[RetrievedDocument]
    citations: List[Citation]
    confidence_score: float
    response_time_ms: float
    decomposition_plan: Optional[DecompositionPlan] = None
    session_context: Optional[SessionContext] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RAGSystemConfig:
    """Configuration for the advanced RAG system"""
    # Database configuration
    db_path: str = "knowledge_base.db"

    # Retrieval configuration
    default_max_results: int = 10
    default_retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    enable_multi_modal: bool = True

    # Conversation configuration
    enable_conversation_memory: bool = True
    max_conversation_length: int = 20
    session_timeout_minutes: int = 60

    # Query processing configuration
    enable_query_decomposition: bool = True
    enable_context_awareness: bool = True

    # Citation configuration
    enable_citation_tracking: bool = True
    auto_verify_sources: bool = False

    # Performance configuration
    enable_query_cache: bool = True
    cache_ttl_hours: int = 24

    # Logging configuration
    log_level: str = "INFO"
    enable_performance_monitoring: bool = True


class AdvancedRAGSystem:
    """Advanced RAG System orchestrator for manufacturing knowledge base"""

    def __init__(self, config: Optional[RAGSystemConfig] = None):
        self.config = config or RAGSystemConfig()
        self.logger = self._setup_logging()

        # Initialize components
        self.db_integration = DatabaseIntegration(self.config.db_path)
        self.document_chunker = DocumentChunker()
        self.conversation_memory = ConversationMemory(self.config.db_path)
        self.multi_modal_retriever = MultiModalRetriever()
        self.query_decomposer = QueryDecomposer()
        self.citation_tracker = CitationTracker(self.config.db_path)

        # System state
        self._initialized = False
        self._performance_stats = {
            'total_queries': 0,
            'avg_response_time_ms': 0.0,
            'cache_hit_rate': 0.0,
            'active_sessions': 0
        }

        self.logger.info("Advanced RAG System initialized")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the RAG system"""
        logger = logging.getLogger("AdvancedRAGSystem")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    async def initialize(self) -> bool:
        """Initialize the RAG system components"""
        try:
            self.logger.info("Initializing RAG system components...")

            # Perform database compatibility check
            compatibility = self.db_integration.perform_compatibility_check()
            if compatibility['status'] == 'failed':
                self.logger.error("Database compatibility check failed")
                return False

            # Migrate data if needed
            if compatibility['status'] == 'warning':
                self.logger.info("Performing database migration...")
                migration_result = self.db_integration.migrate_legacy_data()
                if not migration_result.success:
                    self.logger.error(f"Migration failed: {migration_result.message}")
                    return False

            # Integrate RAG components
            integration_result = self.db_integration.integrate_rag_components()
            if not integration_result.success:
                self.logger.error(f"Component integration failed: {integration_result.errors}")
                return False

            self._initialized = True
            self.logger.info("RAG system initialization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"RAG system initialization failed: {e}")
            return False

    async def query(self, rag_query: RAGQuery) -> RAGResponse:
        """
        Process a query through the advanced RAG pipeline
        """
        if not self._initialized:
            await self.initialize()

        start_time = datetime.now()
        self.logger.info(f"Processing query: {rag_query.query_id}")

        try:
            # Step 1: Check cache first
            cached_response = await self._check_query_cache(rag_query)
            if cached_response:
                self.logger.info(f"Cache hit for query: {rag_query.query_id}")
                self._update_performance_stats('cache_hit')
                return cached_response

            # Step 2: Load conversation context if session_id provided
            session_context = None
            if self.config.enable_conversation_memory and rag_query.session_id:
                session_context = await self._load_session_context(rag_query.session_id)

            # Step 3: Decompose query if enabled
            decomposition_plan = None
            if self.config.enable_query_decomposition:
                decomposition_plan = self.query_decomposer.decompose_query(
                    rag_query.text,
                    {"session_context": session_context, "rag_query": asdict(rag_query)}
                )

            # Step 4: Retrieve relevant documents
            sources = await self._retrieve_documents(rag_query, decomposition_plan, session_context)

            # Step 5: Generate answer with context
            answer = await self._generate_answer(rag_query, sources, session_context, decomposition_plan)

            # Step 6: Generate citations if enabled
            citations = []
            if self.config.enable_citation_tracking and rag_query.include_citations:
                citations = await self._generate_citations(answer, sources)

            # Step 7: Calculate confidence score
            confidence_score = self._calculate_confidence_score(answer, sources, citations)

            # Step 8: Create response
            response_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            response = RAGResponse(
                query_id=rag_query.query_id,
                answer=answer,
                sources=sources,
                citations=citations,
                confidence_score=confidence_score,
                response_time_ms=response_time_ms,
                decomposition_plan=decomposition_plan,
                session_context=session_context,
                metadata={
                    'processing_steps': ['cache_check', 'context_load', 'query_decomposition', 'retrieval', 'answer_generation', 'citation_generation'],
                    'retrieval_strategy': rag_query.retrieval_strategy.value,
                    'content_types': rag_query.content_types or ['all']
                }
            )

            # Step 9: Cache response
            if self.config.enable_query_cache:
                await self._cache_response(rag_query, response)

            # Step 10: Update conversation memory
            if session_context:
                await self._update_conversation_memory(rag_query, response)

            # Step 11: Update performance stats
            self._update_performance_stats('query', response_time_ms)

            self.logger.info(f"Query processed successfully: {rag_query.query_id}")
            return response

        except Exception as e:
            self.logger.error(f"Query processing failed: {rag_query.query_id}, Error: {e}")
            # Return error response
            return RAGResponse(
                query_id=rag_query.query_id,
                answer=f"I apologize, but I encountered an error while processing your query: {str(e)}",
                sources=[],
                citations=[],
                confidence_score=0.0,
                response_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                metadata={'error': str(e)}
            )

    async def stream_query(self, rag_query: RAGQuery) -> AsyncGenerator[str, None]:
        """
        Process a query with streaming response
        """
        if not self._initialized:
            await self.initialize()

        self.logger.info(f"Starting streaming query: {rag_query.query_id}")

        try:
            # Load context and decompose query (non-streaming parts)
            session_context = None
            if self.config.enable_conversation_memory and rag_query.session_id:
                session_context = await self._load_session_context(rag_query.session_id)

            decomposition_plan = None
            if self.config.enable_query_decomposition:
                decomposition_plan = self.query_decomposer.decompose_query(
                    rag_query.text,
                    {"session_context": session_context}
                )

            sources = await self._retrieve_documents(rag_query, decomposition_plan, session_context)

            # Stream the answer generation
            async for chunk in self._stream_answer(rag_query, sources, session_context, decomposition_plan):
                yield chunk

            # Send final metadata
            yield json.dumps({
                'type': 'metadata',
                'sources_count': len(sources),
                'decomposition_plan': decomposition_plan.plan_id if decomposition_plan else None,
                'session_updated': session_context is not None
            })

        except Exception as e:
            self.logger.error(f"Streaming query failed: {rag_query.query_id}, Error: {e}")
            yield json.dumps({
                'type': 'error',
                'message': str(e)
            })

    async def add_document(self, content: str, content_type: ContentType = ContentType.TEXT,
                          metadata: Optional[Dict[str, Any]] = None,
                          filename: Optional[str] = None) -> str:
        """
        Add a new document to the RAG system
        """
        try:
            self.logger.info(f"Adding document of type: {content_type.value}")

            # Generate document ID
            doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(content) % 10000}"

            # Chunk the document
            chunks = self.document_chunker.chunk_document(
                doc_id, content, content_type, metadata
            )

            # Store chunks in database
            for chunk in chunks:
                await self._store_document_chunk(chunk)

            # Index for retrieval
            await self._index_document_chunks(chunks)

            self.logger.info(f"Document added successfully: {doc_id}, chunks: {len(chunks)}")
            return doc_id

        except Exception as e:
            self.logger.error(f"Failed to add document: {e}")
            raise

    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all its chunks from the RAG system
        """
        try:
            self.logger.info(f"Deleting document: {doc_id}")

            # Delete from database
            await self._delete_document_chunks(doc_id)

            # Update search indexes
            await self._cleanup_search_indexes(doc_id)

            self.logger.info(f"Document deleted successfully: {doc_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete document: {doc_id}, Error: {e}")
            return False

    async def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics
        """
        try:
            db_status = self.db_integration.get_database_status()
            session_stats = self.conversation_memory.get_session_stats()

            stats = {
                'database': db_status,
                'sessions': session_stats,
                'performance': self._performance_stats,
                'components': {
                    'document_chunker': 'active',
                    'conversation_memory': 'active',
                    'multi_modal_retriever': 'active',
                    'query_decomposer': 'active',
                    'citation_tracker': 'active',
                    'database_integration': 'active'
                },
                'configuration': asdict(self.config),
                'system_health': 'healthy' if self._initialized else 'initializing'
            }

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get system stats: {e}")
            return {'error': str(e), 'system_health': 'error'}

    # Private helper methods

    async def _check_query_cache(self, rag_query: RAGQuery) -> Optional[RAGResponse]:
        """Check if query result is cached"""
        if not self.config.enable_query_cache:
            return None

        # Simple cache implementation (would use Redis in production)
        query_hash = hash(rag_query.text)
        # In a real implementation, this would check a proper cache
        return None

    async def _load_session_context(self, session_id: str) -> Optional[SessionContext]:
        """Load conversation context for a session"""
        try:
            # Get recent conversation history
            recent_turns = self.conversation_memory.get_recent_turns(
                session_id,
                limit=self.config.max_conversation_length
            )

            if not recent_turns:
                return None

            # Create session context
            context = SessionContext(
                session_id=session_id,
                current_context=recent_turns[-1].context_used if recent_turns else [],
                entities=recent_turns[-1].entities if recent_turns else set(),
                topic_stack=[turn.topic for turn in recent_turns[-5:] if turn.topic],
                user_preferences={},
                last_activity=datetime.now()
            )

            return context

        except Exception as e:
            self.logger.error(f"Failed to load session context: {session_id}, Error: {e}")
            return None

    async def _retrieve_documents(self, rag_query: RAGQuery,
                                decomposition_plan: Optional[DecompositionPlan],
                                session_context: Optional[SessionContext]) -> List[RetrievedDocument]:
        """Retrieve relevant documents for the query"""
        try:
            # Determine search queries
            search_queries = [rag_query.text]

            if decomposition_plan:
                search_queries.extend([sq.query for sq in decomposition_plan.sub_queries])

            # Retrieve documents for all queries
            all_sources = []
            for query_text in search_queries:
                sources = self.multi_modal_retriever.retrieve(
                    query=query_text,
                    content_types=rag_query.content_types,
                    top_k=rag_query.max_results,
                    strategy=rag_query.retrieval_strategy,
                    context={"session_context": session_context}
                )
                all_sources.extend(sources)

            # Remove duplicates and sort by relevance
            unique_sources = self._deduplicate_sources(all_sources)
            unique_sources.sort(key=lambda x: x.relevance_score, reverse=True)

            return unique_sources[:rag_query.max_results]

        except Exception as e:
            self.logger.error(f"Document retrieval failed: {e}")
            return []

    async def _generate_answer(self, rag_query: RAGQuery,
                             sources: List[RetrievedDocument],
                             session_context: Optional[SessionContext],
                             decomposition_plan: Optional[DecompositionPlan]) -> str:
        """Generate answer based on retrieved sources and context"""
        try:
            # Build context from sources
            context_text = "\n\n".join([
                f"Source {i+1}: {source.content[:500]}..."
                for i, source in enumerate(sources[:5])  # Limit context length
            ])

            # Build prompt with context
            prompt = f"""
You are an AI assistant specializing in manufacturing knowledge.

User Query: {rag_query.text}

Relevant Context:
{context_text}

Conversation Context: {session_context.context_summary if session_context else 'None'}

Query Decomposition: {decomposition_plan.summary if decomposition_plan else 'Not decomposed'}

Please provide a comprehensive and accurate answer based on the provided context.
Focus on manufacturing-specific details and be precise in your response.
If you don't have enough information to answer completely, acknowledge the limitations.

Answer:
"""

            # In a real implementation, this would call an LLM
            # For now, return a template response
            answer = f"""
Based on the available information from {len(sources)} sources:

{rag_query.text} is an important topic in manufacturing. The retrieved documents contain relevant information about this subject.

Key points from the sources:
- Multiple documents discuss this topic with relevance scores ranging from {min([s.relevance_score for s in sources]) if sources else 0:.2f} to {max([s.relevance_score for s in sources]) if sources else 0:.2f}
- The information covers various aspects including procedures, specifications, and best practices
- Sources include different content types: {', '.join(set([s.metadata.get('content_type', 'unknown') for s in sources]))}

For a more detailed and specific answer, please provide additional context or specify particular aspects you'd like me to focus on.
"""

            return answer.strip()

        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}")
            return f"I apologize, but I encountered an error while generating an answer: {str(e)}"

    async def _stream_answer(self, rag_query: RAGQuery, sources: List[RetrievedDocument],
                           session_context: Optional[SessionContext],
                           decomposition_plan: Optional[DecompositionPlan]) -> AsyncGenerator[str, None]:
        """Stream answer generation"""
        # Simple streaming implementation
        answer_parts = [
            f"Based on {len(sources)} retrieved sources, ",
            "I can provide information about your query. ",
            "The sources contain relevant manufacturing information ",
            "with varying levels of detail and specificity. ",
            "Let me break this down for you..."
        ]

        for i, part in enumerate(answer_parts):
            yield json.dumps({
                'type': 'chunk',
                'content': part,
                'chunk_index': i,
                'total_chunks': len(answer_parts)
            })
            # Simulate processing time
            await asyncio.sleep(0.1)

    async def _generate_citations(self, answer: str, sources: List[RetrievedDocument]) -> List[Citation]:
        """Generate citations for the answer"""
        citations = []

        for source in sources:
            # Create citation for each relevant source
            citation = Citation(
                citation_id=f"cite_{source.document_id}_{int(datetime.now().timestamp())}",
                source_id=source.document_id,
                citation_type=CitationType.PARAPHRASE,
                content_snippet=source.content[:200] + "..." if len(source.content) > 200 else source.content,
                confidence_score=source.relevance_score,
                relevance_score=source.relevance_score,
                page_number=source.metadata.get('page_number'),
                section_reference=source.metadata.get('section_title')
            )

            citations.append(citation)
            self.citation_tracker.add_citation(citation)

        return citations

    def _calculate_confidence_score(self, answer: str, sources: List[RetrievedDocument],
                                  citations: List[Citation]) -> float:
        """Calculate confidence score for the response"""
        if not sources:
            return 0.0

        # Base confidence from source relevance
        source_confidence = sum(s.relevance_score for s in sources) / len(sources)

        # Boost from citation quality
        citation_confidence = 0.0
        if citations:
            citation_confidence = sum(c.confidence_score for c in citations) / len(citations)

        # Answer length factor (longer, more detailed answers might be more comprehensive)
        length_factor = min(len(answer.split()) / 100, 1.0)

        # Combine factors
        overall_confidence = (source_confidence * 0.5 +
                            citation_confidence * 0.3 +
                            length_factor * 0.2)

        return min(max(overall_confidence, 0.0), 1.0)

    async def _cache_response(self, rag_query: RAGQuery, response: RAGResponse):
        """Cache the response for future queries"""
        # In a real implementation, this would store in Redis or similar
        pass

    async def _update_conversation_memory(self, rag_query: RAGQuery, response: RAGResponse):
        """Update conversation memory with the new turn"""
        if not rag_query.session_id:
            return

        try:
            turn_id = self.conversation_memory.add_conversation_turn(
                session_id=rag_query.session_id,
                user_query=rag_query.text,
                assistant_response=response.answer,
                context_used=[s.document_id for s in response.sources],
                satisfaction_score=response.confidence_score
            )

            self.logger.debug(f"Added conversation turn: {turn_id}")

        except Exception as e:
            self.logger.error(f"Failed to update conversation memory: {e}")

    async def _store_document_chunk(self, chunk: DocumentChunk):
        """Store a document chunk in the database"""
        # Implementation would store in the database
        pass

    async def _index_document_chunks(self, chunks: List[DocumentChunk]):
        """Index document chunks for search"""
        # Implementation would create search indexes
        pass

    async def _delete_document_chunks(self, doc_id: str):
        """Delete all chunks for a document"""
        # Implementation would delete from database
        pass

    async def _cleanup_search_indexes(self, doc_id: str):
        """Clean up search indexes for deleted document"""
        # Implementation would clean up indexes
        pass

    def _deduplicate_sources(self, sources: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """Remove duplicate sources"""
        seen_ids = set()
        unique_sources = []

        for source in sources:
            if source.document_id not in seen_ids:
                seen_ids.add(source.document_id)
                unique_sources.append(source)

        return unique_sources

    def _update_performance_stats(self, stat_type: str, value: Union[float, str] = 0):
        """Update performance statistics"""
        if stat_type == 'query':
            self._performance_stats['total_queries'] += 1
            if self._performance_stats['total_queries'] == 1:
                self._performance_stats['avg_response_time_ms'] = value
            else:
                # Running average
                current_avg = self._performance_stats['avg_response_time_ms']
                count = self._performance_stats['total_queries']
                new_avg = (current_avg * (count - 1) + value) / count
                self._performance_stats['avg_response_time_ms'] = new_avg
        elif stat_type == 'cache_hit':
            total = self._performance_stats['total_queries']
            if total > 0:
                # Simple cache hit rate calculation
                hits = self._performance_stats.get('cache_hits', 0) + 1
                self._performance_stats['cache_hits'] = hits
                self._performance_stats['cache_hit_rate'] = hits / total


# Factory function
def create_advanced_rag_system(config: Optional[RAGSystemConfig] = None) -> AdvancedRAGSystem:
    """Create and initialize an advanced RAG system"""
    return AdvancedRAGSystem(config)


# Usage example
if __name__ == "__main__":
    async def main():
        # Create RAG system
        rag_system = create_advanced_rag_system()

        # Initialize
        await rag_system.initialize()

        # Create a test query
        query = RAGQuery(
            query_id="test_001",
            text="What are the safety procedures for operating CNC machines?",
            session_id="session_001",
            max_results=5,
            include_citations=True
        )

        # Process the query
        response = await rag_system.query(query)

        print(f"Query: {query.text}")
        print(f"Answer: {response.answer[:200]}...")
        print(f"Sources found: {len(response.sources)}")
        print(f"Citations: {len(response.citations)}")
        print(f"Confidence: {response.confidence_score:.2f}")
        print(f"Response time: {response.response_time_ms:.2f}ms")

        # Get system stats
        stats = await rag_system.get_system_stats()
        print(f"\nSystem health: {stats['system_health']}")

    import asyncio
    asyncio.run(main())