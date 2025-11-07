#!/usr/bin/env python3
"""
Advanced Retrieval-Augmented Generation (RAG) System
Implements sophisticated RAG capabilities with hierarchical chunking,
context-aware retrieval, query decomposition, answer synthesis with citations,
and conversation memory.
"""

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from rank_bm25 import BM25Okapi
import faiss
import redis
import asyncpg
from fastapi import HTTPException
import openai
from openai import AsyncOpenAI
import pdfplumber
from PIL import Image
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChunkType(Enum):
    """Document chunk types"""
    DOCUMENT = "document"
    SECTION = "section"
    SUBSECTION = "subsection"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"

class RetrievalStrategy(Enum):
    """Retrieval strategies"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    MULTI_HOP = "multi_hop"
    CONTEXT_AWARE = "context_aware"

@dataclass
class DocumentChunk:
    """Represents a chunk of document content"""
    id: str
    content: str
    chunk_type: ChunkType
    level: int
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    source_document: Optional[str] = None
    page_number: Optional[int] = None
    bbox: Optional[List[float]] = None
    tokens: int = 0
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class QueryDecomposition:
    """Decomposed query components"""
    original_query: str
    subqueries: List[str]
    query_type: str
    entities: List[str]
    intent: str
    context_requirements: List[str]
    decomposition_strategy: str

@dataclass
class RetrievalResult:
    """Result from retrieval operation"""
    chunks: List[DocumentChunk]
    scores: List[float]
    retrieval_strategy: RetrievalStrategy
    query_time: float
    total_results: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Citation:
    """Citation information for answer components"""
    chunk_id: str
    content_snippet: str
    relevance_score: float
    page_number: Optional[int] = None
    source_document: Optional[str] = None
    confidence: float = 0.0

@dataclass
class GeneratedAnswer:
    """Generated answer with citations"""
    answer: str
    citations: List[Citation]
    confidence: float
    sources_used: List[str]
    generation_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationContext:
    """Conversation context and memory"""
    session_id: str
    user_id: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    relevant_chunks: List[DocumentChunk] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    entity_memory: Dict[str, List[str]] = field(default_factory=dict)
    query_patterns: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

class HierarchicalDocumentChunker:
    """Advanced hierarchical document chunking system"""

    def __init__(self,
                 max_chunk_tokens: int = 512,
                 min_chunk_tokens: int = 50,
                 overlap_tokens: int = 50,
                 encoding_model: str = "cl100k_base"):
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tiktoken.get_encoding(encoding_model)
        self.nlp = spacy.load("en_core_web_sm")

        # Section detection patterns
        self.section_patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown headers
            r'^[A-Z][^.]*:$',    # Capitalized titles ending with colon
            r'^\d+\.\s+(.+)$',   # Numbered sections
            r'^[IVX]+\.\s+(.+)$', # Roman numeral sections
            r'^[A-Z]+\.\s+(.+)$', # Letter sections
        ]

        # Document structure markers
        self.structure_markers = {
            'table_start': r'<table[^>]*>',
            'table_end': r'</table>',
            'code_start': r'```[\w]*\n?',
            'code_end': r'\n?```',
            'image_start': r'!\[.*?\]\(',
            'list_item': r'^[\s]*[-*+]\s+',
        }

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.tokenizer.encode(text))
        except:
            # Fallback to character-based estimation
            return len(text) // 4

    def detect_structure(self, text: str) -> List[Tuple[str, int, int]]:
        """Detect document structure elements"""
        structures = []

        # Detect headers/sections
        for pattern in self.section_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                structures.append(('section', match.start(), match.end()))

        # Detect tables
        for match in re.finditer(self.structure_markers['table_start'], text, re.IGNORECASE):
            table_end = re.search(self.structure_markers['table_end'], text[match.end():])
            if table_end:
                structures.append(('table', match.start(), match.end() + table_end.end()))

        # Detect code blocks
        for match in re.finditer(self.structure_markers['code_start'], text, re.MULTILINE):
            code_end = re.search(self.structure_markers['code_end'], text[match.end():])
            if code_end:
                structures.append(('code', match.start(), match.end() + code_end.end()))

        # Detect images
        for match in re.finditer(self.structure_markers['image_start'], text):
            # Find end of image markdown
            end_pos = text.find(')', match.end())
            if end_pos != -1:
                structures.append(('image', match.start(), end_pos + 1))

        # Sort by position
        structures.sort(key=lambda x: x[1])
        return structures

    def chunk_document(self,
                      content: str,
                      source_document: str = None,
                      page_number: int = None) -> List[DocumentChunk]:
        """Perform hierarchical document chunking"""
        chunks = []
        structures = self.detect_structure(content)

        # If no clear structure, use semantic chunking
        if not structures:
            return self._semantic_chunking(content, source_document, page_number)

        # Create hierarchical chunks based on detected structures
        current_pos = 0

        for i, (struct_type, start, end) in enumerate(structures):
            # Process content before this structure
            if current_pos < start:
                text_before = content[current_pos:start].strip()
                if text_before:
                    before_chunks = self._semantic_chunking(
                        text_before, source_document, page_number
                    )
                    chunks.extend(before_chunks)

            # Process the structure element
            struct_content = content[start:end].strip()
            if struct_content:
                struct_chunk = DocumentChunk(
                    id=str(uuid.uuid4()),
                    content=struct_content,
                    chunk_type=ChunkType(struct_type),
                    level=0,
                    source_document=source_document,
                    page_number=page_number,
                    tokens=self.count_tokens(struct_content),
                    metadata={
                        'structure_type': struct_type,
                        'position': start,
                        'length': end - start
                    }
                )
                chunks.append(struct_chunk)

            current_pos = end

        # Process remaining content
        if current_pos < len(content):
            remaining_text = content[current_pos:].strip()
            if remaining_text:
                remaining_chunks = self._semantic_chunking(
                    remaining_text, source_document, page_number
                )
                chunks.extend(remaining_chunks)

        # Build hierarchy
        self._build_hierarchy(chunks)

        return chunks

    def _semantic_chunking(self,
                          content: str,
                          source_document: str = None,
                          page_number: int = None) -> List[DocumentChunk]:
        """Perform semantic chunking of content"""
        doc = self.nlp(content)
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_tokens = self.count_tokens(sent_text)

            # If adding this sentence would exceed max tokens, create a chunk
            if current_tokens + sent_tokens > self.max_chunk_tokens and current_chunk:
                chunk = DocumentChunk(
                    id=str(uuid.uuid4()),
                    content=current_chunk.strip(),
                    chunk_type=ChunkType.PARAGRAPH,
                    level=1,
                    source_document=source_document,
                    page_number=page_number,
                    tokens=current_tokens,
                    metadata={'chunking_method': 'semantic'}
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + sent_text
                current_tokens = self.count_tokens(current_chunk)
            else:
                current_chunk += " " + sent_text if current_chunk else sent_text
                current_tokens += sent_tokens

        # Add final chunk if there's remaining content
        if current_chunk.strip():
            chunk = DocumentChunk(
                id=str(uuid.uuid4()),
                content=current_chunk.strip(),
                chunk_type=ChunkType.PARAGRAPH,
                level=1,
                source_document=source_document,
                page_number=page_number,
                tokens=current_tokens,
                metadata={'chunking_method': 'semantic'}
            )
            chunks.append(chunk)

        return chunks

    def _get_overlap_sentences(self, current_chunk: str) -> str:
        """Get overlapping sentences for chunk continuity"""
        doc = self.nlp(current_chunk)
        sentences = list(doc.sents)

        if len(sentences) <= 1:
            return ""

        # Take last few sentences for overlap
        overlap_sent_count = min(2, len(sentences) // 2)
        overlap_sentences = sentences[-overlap_sent_count:]

        return " ".join([sent.text for sent in overlap_sentences])

    def _build_hierarchy(self, chunks: List[DocumentChunk]):
        """Build hierarchical relationships between chunks"""
        # Simple hierarchy based on chunk types and positions
        section_chunks = [c for c in chunks if c.chunk_type == ChunkType.SECTION]
        paragraph_chunks = [c for c in chunks if c.chunk_type == ChunkType.PARAGRAPH]

        # Assign parents to paragraph chunks
        for para_chunk in paragraph_chunks:
            # Find the most recent section chunk
            for section_chunk in reversed(section_chunks):
                if section_chunk.metadata.get('position', 0) < para_chunk.metadata.get('position', float('inf')):
                    para_chunk.parent_id = section_chunk.id
                    section_chunk.children_ids.append(para_chunk.id)
                    break

class ContextAwareRetriever:
    """Context-aware retrieval system"""

    def __init__(self,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 dimension: int = 384,
                 redis_client: redis.Redis = None):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.dimension = dimension
        self.redis_client = redis_client
        self.faiss_index = None
        self.chunk_metadata = {}
        self.bm25_index = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

        # Context weights
        self.context_weights = {
            'recency': 0.2,
            'relevance': 0.4,
            'authority': 0.2,
            'diversity': 0.2
        }

    def build_index(self, chunks: List[DocumentChunk]):
        """Build retrieval indices"""
        if not chunks:
            return

        # Generate embeddings
        embeddings = []
        texts = []

        for chunk in chunks:
            text = chunk.content
            texts.append(text)
            embedding = self.embedding_model.encode(text)
            embeddings.append(embedding)
            self.chunk_metadata[chunk.id] = chunk

        embeddings = np.array(embeddings)

        # Build FAISS index
        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        self.faiss_index.add(embeddings)

        # Build BM25 index
        tokenized_texts = [text.lower().split() for text in texts]
        self.bm25_index = BM25Okapi(tokenized_texts)

        # Build TF-IDF matrix
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

    def retrieve(self,
                query: str,
                context: Optional[ConversationContext] = None,
                strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
                top_k: int = 10) -> RetrievalResult:
        """Retrieve relevant chunks with context awareness"""
        start_time = time.time()

        if strategy == RetrievalStrategy.SEMANTIC:
            results = self._semantic_search(query, top_k)
        elif strategy == RetrievalStrategy.KEYWORD:
            results = self._keyword_search(query, top_k)
        elif strategy == RetrievalStrategy.HYBRID:
            results = self._hybrid_search(query, top_k)
        elif strategy == RetrievalStrategy.CONTEXT_AWARE:
            results = self._context_aware_search(query, context, top_k)
        else:
            results = self._multi_hop_search(query, top_k)

        query_time = time.time() - start_time

        return RetrievalResult(
            chunks=results['chunks'],
            scores=results['scores'],
            retrieval_strategy=strategy,
            query_time=query_time,
            total_results=len(results['chunks']),
            metadata=results.get('metadata', {})
        )

    def _semantic_search(self, query: str, top_k: int) -> Dict[str, Any]:
        """Semantic search using embeddings"""
        if not self.faiss_index:
            return {'chunks': [], 'scores': []}

        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        scores, indices = self.faiss_index.search(query_embedding, min(top_k, self.faiss_index.ntotal))

        chunks = []
        score_list = []

        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index
                chunk_id = list(self.chunk_metadata.keys())[idx]
                chunk = self.chunk_metadata[chunk_id]
                chunks.append(chunk)
                score_list.append(float(score))

        return {
            'chunks': chunks,
            'scores': score_list,
            'metadata': {'search_type': 'semantic'}
        }

    def _keyword_search(self, query: str, top_k: int) -> Dict[str, Any]:
        """Keyword search using BM25"""
        if not self.bm25_index:
            return {'chunks': [], 'scores': []}

        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)

        # Get top-k results
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]

        chunks = []
        score_list = []

        for idx in top_indices:
            if bm25_scores[idx] > 0:
                chunk_id = list(self.chunk_metadata.keys())[idx]
                chunk = self.chunk_metadata[chunk_id]
                chunks.append(chunk)
                score_list.append(float(bm25_scores[idx]))

        return {
            'chunks': chunks,
            'scores': score_list,
            'metadata': {'search_type': 'keyword'}
        }

    def _hybrid_search(self, query: str, top_k: int) -> Dict[str, Any]:
        """Hybrid search combining semantic and keyword"""
        semantic_results = self._semantic_search(query, top_k * 2)
        keyword_results = self._keyword_search(query, top_k * 2)

        # Combine results with scores
        combined_scores = {}

        # Add semantic scores (weight 0.6)
        for chunk, score in zip(semantic_results['chunks'], semantic_results['scores']):
            combined_scores[chunk.id] = score * 0.6

        # Add keyword scores (weight 0.4)
        for chunk, score in zip(keyword_results['chunks'], keyword_results['scores']):
            if chunk.id in combined_scores:
                combined_scores[chunk.id] += score * 0.4
            else:
                combined_scores[chunk.id] = score * 0.4

        # Sort by combined score and get top-k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        chunks = []
        scores = []

        for chunk_id, score in sorted_results:
            chunk = self.chunk_metadata[chunk_id]
            chunks.append(chunk)
            scores.append(score)

        return {
            'chunks': chunks,
            'scores': scores,
            'metadata': {'search_type': 'hybrid'}
        }

    def _context_aware_search(self,
                            query: str,
                            context: Optional[ConversationContext],
                            top_k: int) -> Dict[str, Any]:
        """Context-aware search considering conversation history"""
        # Start with hybrid search
        hybrid_results = self._hybrid_search(query, top_k * 2)

        if not context or not context.relevant_chunks:
            return hybrid_results

        # Adjust scores based on context
        context_boost = defaultdict(float)

        # Boost chunks that appeared in previous interactions
        for chunk in context.relevant_chunks:
            context_boost[chunk.id] += 0.1

        # Boost chunks related to entities mentioned in conversation
        for entity, mentions in context.entity_memory.items():
            if entity.lower() in query.lower():
                # Boost chunks containing related entities
                for chunk in hybrid_results['chunks']:
                    if entity.lower() in chunk.content.lower():
                        context_boost[chunk.id] += 0.2

        # Apply context boosts
        adjusted_scores = []
        for chunk, score in zip(hybrid_results['chunks'], hybrid_results['scores']):
            adjusted_score = score + context_boost[chunk.id]
            adjusted_scores.append((chunk, adjusted_score))

        # Sort and return top-k
        adjusted_scores.sort(key=lambda x: x[1], reverse=True)
        top_chunks = adjusted_scores[:top_k]

        chunks = [item[0] for item in top_chunks]
        scores = [item[1] for item in top_chunks]

        return {
            'chunks': chunks,
            'scores': scores,
            'metadata': {'search_type': 'context_aware', 'context_applied': True}
        }

    def _multi_hop_search(self, query: str, top_k: int) -> Dict[str, Any]:
        """Multi-hop search for complex queries"""
        # Start with initial search
        initial_results = self._hybrid_search(query, top_k)

        # Extract entities from initial results
        entities = self._extract_entities_from_chunks(initial_results['chunks'])

        # If no entities found, return initial results
        if not entities:
            return initial_results

        # Search for related content using entities
        expanded_results = initial_results['chunks'].copy()
        expanded_scores = initial_results['scores'].copy()

        for entity in entities[:3]:  # Limit to top 3 entities
            entity_query = f"{query} {entity}"
            entity_results = self._hybrid_search(entity_query, top_k // 2)

            # Add new results
            for chunk, score in zip(entity_results['chunks'], entity_results['scores']):
                if chunk not in expanded_results:
                    expanded_results.append(chunk)
                    expanded_scores.append(score * 0.8)  # Slightly lower score for hops

        # Sort all results and return top-k
        combined = list(zip(expanded_results, expanded_scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        combined = combined[:top_k]

        chunks = [item[0] for item in combined]
        scores = [item[1] for item in combined]

        return {
            'chunks': chunks,
            'scores': scores,
            'metadata': {'search_type': 'multi_hop', 'entities_found': entities}
        }

    def _extract_entities_from_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        """Extract entities from document chunks"""
        entities = []

        for chunk in chunks:
            doc = self.nlp(chunk.content)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                    entities.append(ent.text)

        # Remove duplicates and return
        return list(set(entities))

class QueryDecomposer:
    """Advanced query decomposition system"""

    def __init__(self, nlp_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(nlp_model)

        # Query patterns
        self.comparison_patterns = [
            r'(compare|vs|versus|difference between)',
            r'(better than|worse than)',
            r'(similar to|different from)'
        ]

        self.temporal_patterns = [
            r'(recent|latest|current)',
            r'(past|history|historical)',
            r'(future|upcoming|planned)'
        ]

        self.aggregate_patterns = [
            r'(total|sum|overall)',
            r'(average|mean)',
            r'(maximum|minimum|best|worst)'
        ]

    def decompose_query(self, query: str, context: Optional[ConversationContext] = None) -> QueryDecomposition:
        """Decompose complex query into simpler subqueries"""
        doc = self.nlp(query)

        # Identify query type
        query_type = self._identify_query_type(query)

        # Extract entities
        entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']]

        # Identify intent
        intent = self._identify_intent(query)

        # Generate subqueries based on query type
        subqueries = self._generate_subqueries(query, query_type, entities, context)

        # Identify context requirements
        context_requirements = self._identify_context_requirements(query, query_type)

        return QueryDecomposition(
            original_query=query,
            subqueries=subqueries,
            query_type=query_type,
            entities=entities,
            intent=intent,
            context_requirements=context_requirements,
            decomposition_strategy="pattern_based"
        )

    def _identify_query_type(self, query: str) -> str:
        """Identify the type of query"""
        query_lower = query.lower()

        # Check for comparison queries
        for pattern in self.comparison_patterns:
            if re.search(pattern, query_lower):
                return "comparison"

        # Check for temporal queries
        for pattern in self.temporal_patterns:
            if re.search(pattern, query_lower):
                return "temporal"

        # Check for aggregate queries
        for pattern in self.aggregate_patterns:
            if re.search(pattern, query_lower):
                return "aggregate"

        # Check for causal queries
        if any(word in query_lower for word in ['why', 'cause', 'reason', 'because']):
            return "causal"

        # Check for procedural queries
        if any(word in query_lower for word in ['how', 'process', 'steps', 'procedure']):
            return "procedural"

        # Default to factual query
        return "factual"

    def _identify_intent(self, query: str) -> str:
        """Identify user intent"""
        query_lower = query.lower()

        intents = {
            'search': ['find', 'search', 'look for', 'show me'],
            'explain': ['explain', 'describe', 'what is', 'tell me about'],
            'compare': ['compare', 'difference', 'versus', 'vs'],
            'analyze': ['analyze', 'examine', 'investigate'],
            'recommend': ['recommend', 'suggest', 'best'],
            'troubleshoot': ['fix', 'solve', 'problem', 'issue'],
            'list': ['list', 'show all', 'enumerate']
        }

        for intent, patterns in intents.items():
            if any(pattern in query_lower for pattern in patterns):
                return intent

        return 'search'

    def _generate_subqueries(self,
                           query: str,
                           query_type: str,
                           entities: List[str],
                           context: Optional[ConversationContext]) -> List[str]:
        """Generate subqueries based on query type"""
        subqueries = [query]  # Always include original query

        if query_type == "comparison":
            # Split comparison into individual queries
            if " vs " in query.lower() or " versus " in query.lower():
                parts = re.split(r'\s+vs\.?\s+|\s+versus\s+', query, flags=re.IGNORECASE)
                if len(parts) >= 2:
                    subqueries.extend([f"information about {parts[0]}", f"information about {parts[1]}"])

        elif query_type == "aggregate":
            # Create specific queries for aggregation
            if "total" in query.lower() or "sum" in query.lower():
                subqueries.append(query.replace("total", "").replace("sum", ""))
            elif "average" in query.lower():
                subqueries.append(query.replace("average", ""))

        # Add entity-specific queries
        for entity in entities[:3]:  # Limit to top 3 entities
            entity_query = f"{entity} {query}"
            subqueries.append(entity_query)

        # Add context-aware queries if context is available
        if context and context.entity_memory:
            for related_entity, mentions in context.entity_memory.items():
                if related_entity.lower() in query.lower():
                    # Add queries for related entities
                    for mention in mentions[:2]:
                        if mention.lower() != related_entity.lower():
                            subqueries.append(f"{mention} {query}")

        return list(set(subqueries))  # Remove duplicates

    def _identify_context_requirements(self, query: str, query_type: str) -> List[str]:
        """Identify what context is needed for the query"""
        requirements = []

        if query_type == "comparison":
            requirements.append("comparison_data")
        elif query_type == "temporal":
            requirements.append("temporal_context")
        elif query_type == "aggregate":
            requirements.append("aggregate_data")
        elif query_type == "causal":
            requirements.append("causal_relationships")
        elif query_type == "procedural":
            requirements.append("procedural_steps")

        # Check if user preferences would help
        if any(word in query.lower() for word in ['best', 'recommend', 'prefer']):
            requirements.append("user_preferences")

        return requirements

class AnswerSynthesizer:
    """Advanced answer synthesis with citations"""

    def __init__(self, openai_client: AsyncOpenAI = None):
        self.openai_client = openai_client
        self.citation_threshold = 0.1  # Minimum relevance score for citation

    async def synthesize_answer(self,
                              query: str,
                              retrieval_result: RetrievalResult,
                              decomposition: Optional[QueryDecomposition] = None,
                              context: Optional[ConversationContext] = None) -> GeneratedAnswer:
        """Synthesize comprehensive answer with citations"""

        # Prepare context for synthesis
        relevant_chunks = retrieval_result.chunks
        context_text = self._prepare_context(relevant_chunks, retrieval_result.scores)

        # Generate answer
        if self.openai_client:
            answer = await self._generate_with_llm(query, context_text, decomposition, context)
        else:
            answer = self._generate_with_rules(query, context_text, decomposition, context)

        # Generate citations
        citations = self._generate_citations(relevant_chunks, retrieval_result.scores, answer)

        # Calculate confidence
        confidence = self._calculate_confidence(retrieval_result, citations, len(answer))

        return GeneratedAnswer(
            answer=answer,
            citations=citations,
            confidence=confidence,
            sources_used=[chunk.source_document for chunk in relevant_chunks if chunk.source_document],
            generation_metadata={
                'chunks_used': len(relevant_chunks),
                'retrieval_strategy': retrieval_result.retrieval_strategy.value,
                'query_time': retrieval_result.query_time,
                'average_score': np.mean(retrieval_result.scores) if retrieval_result.scores else 0
            }
        )

    def _prepare_context(self, chunks: List[DocumentChunk], scores: List[float]) -> str:
        """Prepare context text for answer generation"""
        context_parts = []

        for chunk, score in zip(chunks, scores):
            if score >= self.citation_threshold:
                # Include chunk metadata in context
                context_part = f"[Source: {chunk.source_document or 'Unknown'}]"
                if chunk.page_number:
                    context_part += f" [Page: {chunk.page_number}]"
                context_part += f"\n{chunk.content}\n"
                context_parts.append(context_part)

        return "\n".join(context_parts)

    async def _generate_with_llm(self,
                               query: str,
                               context: str,
                               decomposition: Optional[QueryDecomposition],
                               context_info: Optional[ConversationContext]) -> str:
        """Generate answer using language model"""

        system_prompt = """You are a helpful AI assistant that provides accurate, comprehensive answers based on the provided context.

Guidelines:
1. Use only the information provided in the context
2. Synthesize information from multiple sources when helpful
3. Be specific and provide details from the context
4. If the context doesn't contain relevant information, say so clearly
5. Structure your answer logically with clear sections
6. Include relevant data, numbers, and examples from the context"""

        user_prompt = f"""Query: {query}

Context Information:
{context}

Please provide a comprehensive answer to the query based on the context above."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_with_rules(query, context, decomposition, context_info)

    def _generate_with_rules(self,
                           query: str,
                           context: str,
                           decomposition: Optional[QueryDecomposition],
                           context_info: Optional[ConversationContext]) -> str:
        """Generate answer using rule-based approach"""

        if not context.strip():
            return "I don't have enough information to answer this question based on the available documents."

        # Simple rule-based synthesis
        sentences = context.split('.')
        relevant_sentences = []

        query_words = set(query.lower().split())

        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            if query_words.intersection(sentence_words):
                relevant_sentences.append(sentence.strip())

        if not relevant_sentences:
            # Take first few sentences as fallback
            relevant_sentences = [s.strip() for s in sentences[:3] if s.strip()]

        answer = "Based on the available information:\n\n"
        answer += " ".join(relevant_sentences[:5])  # Limit to 5 sentences

        return answer

    def _generate_citations(self,
                          chunks: List[DocumentChunk],
                          scores: List[float],
                          answer: str) -> List[Citation]:
        """Generate citations for the answer"""
        citations = []

        for chunk, score in zip(chunks, scores):
            if score >= self.citation_threshold:
                # Find relevant snippet from chunk
                snippet = self._extract_relevant_snippet(chunk.content, answer)

                citation = Citation(
                    chunk_id=chunk.id,
                    content_snippet=snippet,
                    relevance_score=score,
                    page_number=chunk.page_number,
                    source_document=chunk.source_document,
                    confidence=min(score, 1.0)
                )
                citations.append(citation)

        return citations

    def _extract_relevant_snippet(self, chunk_content: str, answer: str) -> str:
        """Extract most relevant snippet from chunk content"""
        sentences = chunk_content.split('.')
        answer_words = set(answer.lower().split())

        best_sentence = ""
        best_score = 0

        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(answer_words.intersection(sentence_words))

            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence.strip()

        return best_sentence if best_sentence else chunk_content[:200] + "..."

    def _calculate_confidence(self,
                            retrieval_result: RetrievalResult,
                            citations: List[Citation],
                            answer_length: int) -> float:
        """Calculate confidence score for the answer"""

        # Base confidence from retrieval scores
        if retrieval_result.scores:
            avg_score = np.mean(retrieval_result.scores)
        else:
            avg_score = 0.0

        # Factor in number of citations
        citation_factor = min(len(citations) / 3, 1.0)  # Cap at 3 citations

        # Factor in answer quality (length)
        length_factor = min(answer_length / 200, 1.0)  # Ideal length around 200 chars

        # Combine factors
        confidence = (avg_score * 0.5 + citation_factor * 0.3 + length_factor * 0.2)

        return min(confidence, 1.0)

class ConversationMemory:
    """Advanced conversation memory system"""

    def __init__(self, redis_client: redis.Redis = None):
        self.redis_client = redis_client
        self.context_cache = {}
        self.max_history_length = 20
        self.context_ttl = 3600  # 1 hour

    async def get_context(self, session_id: str, user_id: str) -> ConversationContext:
        """Get or create conversation context"""
        cache_key = f"context:{session_id}:{user_id}"

        # Try to get from cache
        if cache_key in self.context_cache:
            return self.context_cache[cache_key]

        # Try to get from Redis
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    context_data = json.loads(cached_data)
                    context = self._deserialize_context(context_data)
                    self.context_cache[cache_key] = context
                    return context
            except Exception as e:
                logger.error(f"Failed to get context from Redis: {e}")

        # Create new context
        context = ConversationContext(
            session_id=session_id,
            user_id=user_id
        )

        self.context_cache[cache_key] = context
        return context

    async def update_context(self,
                           context: ConversationContext,
                           query: str,
                           answer: GeneratedAnswer,
                           retrieval_result: RetrievalResult):
        """Update conversation context with new interaction"""

        # Add to conversation history
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'answer': answer.answer,
            'chunks_used': len(retrieval_result.chunks),
            'confidence': answer.confidence,
            'retrieval_strategy': retrieval_result.retrieval_strategy.value
        }

        context.conversation_history.append(interaction)

        # Limit history length
        if len(context.conversation_history) > self.max_history_length:
            context.conversation_history = context.conversation_history[-self.max_history_length:]

        # Update relevant chunks
        context.relevant_chunks.extend(retrieval_result.chunks)

        # Limit relevant chunks
        if len(context.relevant_chunks) > 50:
            # Keep most recent and most relevant chunks
            context.relevant_chunks = context.relevant_chunks[-50:]

        # Update entity memory
        self._update_entity_memory(context, query, retrieval_result.chunks)

        # Update query patterns
        self._update_query_patterns(context, query)

        # Update timestamp
        context.last_updated = datetime.now()

        # Save to cache and Redis
        cache_key = f"context:{context.session_id}:{context.user_id}"
        self.context_cache[cache_key] = context

        if self.redis_client:
            try:
                context_data = self._serialize_context(context)
                self.redis_client.setex(
                    cache_key,
                    self.context_ttl,
                    json.dumps(context_data)
                )
            except Exception as e:
                logger.error(f"Failed to save context to Redis: {e}")

    def _update_entity_memory(self, context: ConversationContext, query: str, chunks: List[DocumentChunk]):
        """Update entity memory with new information"""
        # Simple entity extraction (can be enhanced with NER)
        words = query.lower().split()

        for chunk in chunks:
            chunk_words = chunk.content.lower().split()

            # Find overlapping words (potential entities)
            for word in words:
                if len(word) > 3 and word in chunk_words:  # Only consider words longer than 3 chars
                    if word not in context.entity_memory:
                        context.entity_memory[word] = []

                    # Add related words from chunk
                    related_words = [w for w in chunk_words if w != word and len(w) > 3]
                    context.entity_memory[word].extend(related_words[:5])  # Limit related words

        # Remove duplicates and limit entity memory size
        for entity in context.entity_memory:
            context.entity_memory[entity] = list(set(context.entity_memory[entity]))[:10]

    def _update_query_patterns(self, context: ConversationContext, query: str):
        """Update query patterns based on current query"""
        # Extract key terms from query
        words = query.lower().split()
        key_terms = [w for w in words if len(w) > 4]  # Only consider words longer than 4 chars

        context.query_patterns.extend(key_terms)

        # Limit patterns and remove duplicates
        context.query_patterns = list(set(context.query_patterns))[-20:]

    def _serialize_context(self, context: ConversationContext) -> Dict[str, Any]:
        """Serialize context for storage"""
        return {
            'session_id': context.session_id,
            'user_id': context.user_id,
            'conversation_history': context.conversation_history,
            'relevant_chunks': [
                {
                    'id': chunk.id,
                    'content': chunk.content[:200],  # Truncate for storage
                    'source_document': chunk.source_document,
                    'page_number': chunk.page_number
                }
                for chunk in context.relevant_chunks[-20:]  # Only keep last 20 chunks
            ],
            'user_preferences': context.user_preferences,
            'entity_memory': context.entity_memory,
            'query_patterns': context.query_patterns,
            'last_updated': context.last_updated.isoformat()
        }

    def _deserialize_context(self, data: Dict[str, Any]) -> ConversationContext:
        """Deserialize context from storage"""
        context = ConversationContext(
            session_id=data['session_id'],
            user_id=data['user_id'],
            conversation_history=data.get('conversation_history', []),
            user_preferences=data.get('user_preferences', {}),
            entity_memory=data.get('entity_memory', {}),
            query_patterns=data.get('query_patterns', [])
        )

        if 'last_updated' in data:
            context.last_updated = datetime.fromisoformat(data['last_updated'])

        return context

class AdvancedRAGSystem:
    """Main advanced RAG system orchestrator"""

    def __init__(self,
                 openai_api_key: str = None,
                 redis_client: redis.Redis = None,
                 embedding_model: str = "all-MiniLM-L6-v2"):

        # Initialize components
        self.chunker = HierarchicalDocumentChunker()
        self.retriever = ContextAwareRetriever(embedding_model=embedding_model, redis_client=redis_client)
        self.query_decomposer = QueryDecomposer()
        self.memory = ConversationMemory(redis_client=redis_client)

        # Initialize OpenAI client if API key is provided
        self.synthesizer = None
        if openai_api_key:
            openai_client = AsyncOpenAI(api_key=openai_api_key)
            self.synthesizer = AnswerSynthesizer(openai_client)
        else:
            self.synthesizer = AnswerSynthesizer()

        logger.info("Advanced RAG System initialized")

    async def process_document(self,
                             content: str,
                             source_document: str = None,
                             page_number: int = None) -> List[str]:
        """Process document and return chunk IDs"""
        chunks = self.chunker.chunk_document(content, source_document, page_number)

        # Add chunks to retriever index
        if not self.retriever.faiss_index:
            self.retriever.build_index(chunks)
        else:
            # Add to existing index
            self._add_chunks_to_index(chunks)

        logger.info(f"Processed document: {source_document}, created {len(chunks)} chunks")
        return [chunk.id for chunk in chunks]

    def _add_chunks_to_index(self, chunks: List[DocumentChunk]):
        """Add new chunks to existing index"""
        if not chunks:
            return

        # Generate embeddings for new chunks
        embeddings = []
        for chunk in chunks:
            embedding = self.retriever.embedding_model.encode(chunk.content)
            embeddings.append(embedding)
            self.retriever.chunk_metadata[chunk.id] = chunk

        embeddings = np.array(embeddings)
        faiss.normalize_L2(embeddings)

        # Add to FAISS index
        self.retriever.faiss_index.add(embeddings)

    async def query(self,
                   query: str,
                   session_id: str = None,
                   user_id: str = "default",
                   strategy: RetrievalStrategy = RetrievalStrategy.CONTEXT_AWARE,
                   top_k: int = 10) -> GeneratedAnswer:
        """Process query and return generated answer"""

        # Get conversation context
        context = None
        if session_id:
            context = await self.memory.get_context(session_id, user_id)

        # Decompose query
        decomposition = self.query_decomposer.decompose_query(query, context)

        # Retrieve relevant chunks
        retrieval_result = self.retriever.retrieve(
            query,
            context,
            strategy,
            top_k
        )

        # Synthesize answer
        answer = await self.synthesizer.synthesize_answer(
            query,
            retrieval_result,
            decomposition,
            context
        )

        # Update conversation context
        if session_id and context:
            await self.memory.update_context(context, query, answer, retrieval_result)

        return answer

    async def batch_query(self,
                         queries: List[str],
                         session_id: str = None,
                         user_id: str = "default") -> List[GeneratedAnswer]:
        """Process multiple queries in batch"""
        tasks = []
        for query in queries:
            task = self.query(query, session_id, user_id)
            tasks.append(task)

        return await asyncio.gather(*tasks)

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            'total_chunks': len(self.retriever.chunk_metadata),
            'index_size': self.retriever.faiss_index.ntotal if self.retriever.faiss_index else 0,
            'active_contexts': len(self.memory.context_cache),
            'system_initialized': True
        }

        # Add chunk type distribution
        if self.retriever.chunk_metadata:
            chunk_types = defaultdict(int)
            for chunk in self.retriever.chunk_metadata.values():
                chunk_types[chunk.chunk_type.value] += 1
            stats['chunk_distribution'] = dict(chunk_types)

        return stats

# Example usage and testing
if __name__ == "__main__":
    async def test_rag_system():
        """Test the advanced RAG system"""

        # Initialize system
        rag = AdvancedRAGSystem()

        # Sample document
        sample_document = """
        # Advanced RAG System Design

        ## Overview
        The Advanced Retrieval-Augmented Generation (RAG) system represents a cutting-edge approach to information retrieval and synthesis. Unlike traditional search systems that rely solely on keyword matching or semantic similarity, our advanced RAG system incorporates multiple sophisticated techniques to provide more accurate, contextually relevant, and comprehensive answers.

        ## Key Components

        ### Hierarchical Document Chunking
        Our system employs intelligent document chunking that preserves the hierarchical structure of documents. This means that relationships between sections, subsections, and paragraphs are maintained throughout the processing pipeline. The chunking algorithm can identify document structures like headers, tables, and code blocks, ensuring that content is broken down in a semantically meaningful way.

        ### Context-Aware Retrieval
        The retrieval system is designed to understand the context of the conversation and previous queries. By maintaining conversation memory and understanding user preferences, the system can provide more personalized and relevant responses. The context-aware retrieval mechanism adjusts search results based on conversation history, previously accessed documents, and user interaction patterns.

        ### Query Decomposition
        Complex queries are automatically decomposed into simpler, more manageable subqueries. This allows the system to handle multi-faceted questions by breaking them down into components that can be answered individually and then synthesized into a comprehensive response. The decomposition engine identifies query types, extracts entities, and determines the intent behind each query.

        ### Answer Synthesis with Citations
        Responses are generated using advanced language models that can synthesize information from multiple sources. Each answer includes proper citations, allowing users to verify the information and explore the source documents further. The synthesis process ensures that answers are not only accurate but also well-structured and easy to understand.

        ## Performance Metrics
        The system demonstrates significant improvements over traditional approaches:
        - 40-60% improvement in search accuracy
        - 3-5x better context-aware responses
        - Sub-200ms average response time
        - Support for 1000+ concurrent users
        """

        # Process document
        chunk_ids = await rag.process_document(sample_document, "RAG_Design_Doc.md")
        print(f"Created {len(chunk_ids)} chunks")

        # Test queries
        test_queries = [
            "What is hierarchical document chunking?",
            "How does context-aware retrieval work?",
            "What are the performance metrics of the RAG system?",
            "Compare query decomposition with traditional search methods"
        ]

        # Process queries
        for query in test_queries:
            print(f"\nQuery: {query}")
            answer = await rag.query(query, session_id="test_session")
            print(f"Answer: {answer.answer[:200]}...")
            print(f"Confidence: {answer.confidence:.2f}")
            print(f"Citations: {len(answer.citations)}")

        # Print statistics
        stats = rag.get_statistics()
        print(f"\nSystem Statistics: {stats}")

    # Run test
    asyncio.run(test_rag_system())