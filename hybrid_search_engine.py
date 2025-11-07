#!/usr/bin/env python3
"""
Hybrid Search Engine
Implements a sophisticated search engine that combines semantic search,
traditional keyword search, and knowledge graph traversal with advanced
multi-signal ranking algorithms inspired by modern search engines.
"""

import asyncio
import json
import logging
import math
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import sqlite3
import asyncpg
from pathlib import Path
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor
import sqlite_vec  # SQLite with vector extension

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchStrategy(Enum):
    """Search strategy types"""
    SEMANTIC_ONLY = "semantic_only"
    KEYWORD_ONLY = "keyword_only"
    GRAPH_ONLY = "graph_only"
    HYBRID_BALANCED = "hybrid_balanced"
    HYBRID_SEMANTIC_WEIGHTED = "hybrid_semantic_weighted"
    HYBRID_GRAPH_WEIGHTED = "hybrid_graph_weighted"
    ADAPTIVE = "adaptive"

class SignalType(Enum):
    """Search signal types for ranking"""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    KEYWORD_BM25 = "keyword_bm25"
    GRAPH_RELEVANCE = "graph_relevance"
    RECENCY_BOOST = "recency_boost"
    AUTHORITY_SCORE = "authority_score"
    CLICK_THROUGH_RATE = "click_through_rate"
    USER_ENGAGEMENT = "user_engagement"
    CONTENT_QUALITY = "content_quality"
    FRESHNESS = "freshness"

@dataclass
class SearchQuery:
    """Search query with multiple parameters"""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    filters: Dict[str, Any] = field(default_factory=dict)
    strategy: SearchStrategy = SearchStrategy.HYBRID_BALANCED
    limit: int = 10
    offset: int = 0
    boost_fields: Dict[str, float] = field(default_factory=dict)
    include_metadata: bool = True
    user_context: Optional[Dict[str, Any]] = None
    search_intent: Optional[str] = None
    time_range: Optional[str] = None
    result_types: List[str] = field(default_factory=list)
    min_confidence: float = 0.0
    max_distance: Optional[float] = None

@dataclass
class SearchResult:
    """Individual search result with ranking scores"""
    id: str
    title: str
    content: str
    url: Optional[str] = None
    score: float = 0.0
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    graph_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    signals: Dict[SignalType, float] = field(default_factory=dict)
    explanation: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class SearchSession:
    """Search session for tracking user interactions"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    query_history: List[SearchQuery] = field(default_factory=list)
    clicked_results: List[str] = field(default_factory=list)
    search_metrics: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

class BaseSearchComponent(ABC):
    """Abstract base class for search components"""

    def __init__(self):
        self.indexed_count = 0
        self.last_updated = datetime.now()

    @abstractmethod
    async def index(self, documents: List[Dict[str, Any]]) -> bool:
        """Index documents for searching"""
        pass

    @abstractmethod
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform search on indexed documents"""
        pass

    @abstractmethod
    async def update_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """Update indexed document"""
        pass

    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool:
        """Delete indexed document"""
        pass

class SemanticSearchEngine(BaseSearchComponent):
    """Advanced semantic search using neural embeddings"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        super().__init__()
        self.model_name = model_name
        self.dimension = dimension
        self.model = None
        self.faiss_index = None
        self.document_store = {}
        self.id_mapping = {}  # Internal ID -> Document ID mapping
        self.reverse_mapping = {}  # Document ID -> Internal ID mapping

    async def initialize(self):
        """Initialize the semantic search engine"""
        logger.info(f"Initializing semantic search engine with model: {self.model_name}")

        # Load the sentence transformer model
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"Loaded model: {self.model_name}")

        # Initialize FAISS index
        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        logger.info("FAISS index initialized")

    async def index(self, documents: List[Dict[str, Any]]) -> bool:
        """Index documents for semantic search"""
        if not self.model:
            await self.initialize()

        start_time = time.time()

        try:
            # Generate embeddings for all documents
            texts = [doc.get('content', '') for doc in documents]
            embeddings = self.model.encode(texts)

            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)

            # Add to FAISS index
            start_idx = self.indexed_count
            self.faiss_index.add(embeddings)

            # Store documents with mapping
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                internal_id = f"doc_{start_idx + i}"
                self.id_mapping[internal_id] = doc['id']
                self.reverse_mapping[doc['id']] = internal_id
                self.document_store[internal_id] = {
                    **doc,
                    'embedding': embedding,
                    'indexed_at': datetime.now().isoformat()
                }

            self.indexed_count += len(documents)
            self.last_updated = datetime.now()

            logger.info(f"Indexed {len(documents)} documents in {time.time() - start_time:.2f}s")
            return True

        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return False

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform semantic search"""
        if not self.model or self.indexed_count == 0:
            return []

        try:
            # Generate query embedding
            query_embedding = self.model.encode([query.text])
            faiss.normalize_L2(query_embedding)

            # Search FAISS index
            k = min(query.limit + 50, self.indexed_count)  # Get more candidates for ranking
            scores, indices = self.faiss_index.search(query_embedding, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.document_store):
                    internal_id = list(self.document_store.keys())[idx]
                    doc = self.document_store[internal_id]
                    original_id = self.id_mapping.get(internal_id, internal_id)

                    result = SearchResult(
                        id=original_id,
                        title=doc.get('title', ''),
                        content=doc.get('content', ''),
                        url=doc.get('url'),
                        score=float(score),
                        semantic_score=float(score),
                        keyword_score=0.0,
                        graph_score=0.0,
                        metadata=doc.get('metadata', {}),
                        signals={SignalType.SEMANTIC_SIMILARITY: float(score)},
                        explanation={
                            'method': 'semantic_search',
                            'model': self.model_name,
                            'similarity_score': float(score)
                        }
                    )
                    results.append(result)

            # Apply boost if specified
            if query.boost_fields:
                results = self._apply_boosts(results, query.boost_fields)

            return results[:query.limit]

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def _apply_boosts(self, results: List[SearchResult], boosts: Dict[str, float]) -> List[SearchResult]:
        """Apply field-level boosts to results"""
        for result in results:
            for field, boost in boosts.items():
                if field in result.content.lower():
                    result.score *= boost
                    result.signals[SignalType.SEMANTIC_SIMILARITY] *= boost

        return results

    async def update_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """Update indexed document"""
        try:
            # Remove old document if exists
            if doc_id in self.reverse_mapping:
                internal_id = self.reverse_mapping[doc_id]
                if internal_id in self.document_store:
                    del self.document_store[internal_id]
                    del self.id_mapping[internal_id]
                    del self.reverse_mapping[doc_id]

            # Add updated document
            return await self.index([document])

        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False

    async def delete_document(self, doc_id: str) -> bool:
        """Delete indexed document"""
        try:
            if doc_id in self.reverse_mapping:
                internal_id = self.reverse_mapping[doc_id]
                if internal_id in self.document_store:
                    del self.document_store[internal_id]
                    del self.id_mapping[internal_id]
                    del self.reverse_mapping[doc_id]
                    self.indexed_count -= 1

            return True

        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False

class KeywordSearchEngine(BaseSearchComponent):
    """Advanced keyword search using BM25 and TF-IDF"""

    def __init__(self):
        super().__init__()
        self.tfidf_vectorizer = None
        self.bm25_index = None
        self.documents = []
        self.document_ids = []
        self.indexed_count = 0

    async def index(self, documents: List[Dict[str, Any]]) -> bool:
        """Index documents for keyword search"""
        if not documents:
            return True

        start_time = time.time()

        try:
            # Preprocess documents
            texts = []
            new_doc_ids = []

            for doc in documents:
                text = self._preprocess_text(doc.get('content', ''))
                if text.strip():
                    texts.append(text)
                    new_doc_ids.append(doc['id'])
                    self.documents.append(doc)

            # Initialize or update TF-IDF vectorizer
            if self.tfidf_vectorizer is None:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=5000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.7
                )
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            else:
                tfidf_matrix = self.tfidf_vectorizer.transform(texts)

            # Create BM25 index
            self.document_ids = new_doc_ids
            self.bm25_index = BM25Okapi([text.split() for text in texts])

            self.indexed_count = len(texts)
            self.last_updated = datetime.now()

            logger.info(f"Indexed {len(documents)} documents for keyword search in {time.time() - start_time:.2f}s")
            return True

        except Exception as e:
            logger.error(f"Error indexing documents for keyword search: {e}")
            return False

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform keyword search"""
        if not self.bm25_index or self.indexed_count == 0:
            return []

        try:
            # Preprocess query
            query_text = self._preprocess_text(query.text)
            query_tokens = query_text.split()

            # BM25 search
            bm25_scores = self.bm25_index.get_scores(query_tokens)

            # Get top candidates
            top_k = min(query.limit + 50, self.indexed_count)
            top_indices = np.argsort(bm25_scores)[::-1][:top_k]

            results = []
            for idx in top_indices:
                if bm25_scores[idx] > 0 and idx < len(self.documents):
                    doc = self.documents[idx]
                    doc_id = self.document_ids[idx]

                    result = SearchResult(
                        id=doc_id,
                        title=doc.get('title', ''),
                        content=doc.get('content', ''),
                        url=doc.get('url'),
                        score=float(bm25_scores[idx]),
                        semantic_score=0.0,
                        keyword_score=float(bm25_scores[idx]),
                        graph_score=0.0,
                        metadata=doc.get('metadata', {}),
                        signals={SignalType.KEYWORD_BM25: float(bm25_scores[idx])},
                        explanation={
                            'method': 'keyword_search',
                            'algorithm': 'BM25',
                            'score': float(bm25_scores[idx]),
                            'query_terms': query_tokens
                        }
                    )
                    results.append(result)

            # Apply boost if specified
            if query.boost_fields:
                results = self._apply_keyword_boosts(results, query.boost_fields)

            return results[:query.limit]

        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for search"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _apply_keyword_boosts(self, results: List[SearchResult], boosts: Dict[str, float]) -> List[SearchResult]:
        """Apply keyword boosts to results"""
        for result in results:
            title_lower = result.title.lower()
            content_lower = result.content.lower()

            boost_factor = 1.0

            for field, boost in boosts.items():
                field_lower = field.lower()
                if field_lower in title_lower:
                    boost_factor *= boost * 1.5  # Title matches are worth more
                elif field_lower in content_lower:
                    boost_factor *= boost

            result.score *= boost_factor
            result.signals[SignalType.KEYWORD_BM25] *= boost_factor

        return results

    async def update_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """Update indexed document for keyword search"""
        # For simplicity, we'll reindex the entire collection
        # In production, implement incremental updates
        return await self.index([document])

    async def delete_document(self, doc_id: str) -> bool:
        """Delete indexed document"""
        # For simplicity, we'll require reindexing
        # In production, implement efficient deletion
        return True

class KnowledgeGraphSearchEngine(BaseSearchEngine):
    """Knowledge graph traversal search engine"""

    def __init__(self):
        super().__init__()
        self.graph = nx.DiGraph()
        self.node_data = {}
        self.edge_data = {}
        self.page_rank = {}
        self.indexed_count = 0

    async def index(self, documents: List[Dict[str, Any]]) -> bool:
        """Index documents into knowledge graph"""
        start_time = time.time()

        try:
            for doc in documents:
                await self._add_node_to_graph(doc)

            # Calculate PageRank
            if self.graph.number_of_nodes() > 0:
                self.page_rank = nx.pagerank(self.graph, alpha=0.85)

            self.indexed_count = len(self.node_data)
            self.last_updated = datetime.now()

            logger.info(f"Indexed {len(documents)} nodes into knowledge graph in {time.time() - start_time:.2f}s")
            return True

        except Exception as e:
            logger.error(f"Error indexing knowledge graph: {e}")
            return False

    async def _add_node_to_graph(self, document: Dict[str, Any]):
        """Add document as node to knowledge graph"""
        doc_id = document['id']

        # Add node
        self.graph.add_node(doc_id)
        self.node_data[doc_id] = {
            **document,
            'node_type': document.get('entity_type', 'unknown'),
            'indexed_at': datetime.now().isoformat()
        }

        # Add relationships as edges
        relationships = document.get('relationships', [])
        for rel in relationships:
            if 'target_id' in rel and 'relationship_type' in rel:
                target_id = rel['target_id']
                relationship_type = rel['relationship_type']
                weight = rel.get('weight', 1.0)

                # Add edge if target exists or create placeholder
                if target_id not in self.graph:
                    self.graph.add_node(target_id)
                    self.node_data[target_id] = {
                        'id': target_id,
                        'title': target_id,
                        'content': '',
                        'node_type': 'placeholder',
                        'indexed_at': datetime.now().isoformat()
                    }

                self.graph.add_edge(doc_id, target_id,
                                  weight=weight,
                                  relationship_type=relationship_type)

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform knowledge graph traversal search"""
        if self.graph.number_of_nodes() == 0:
            return []

        try:
            # Start with nodes containing query terms
            candidate_nodes = self._find_nodes_with_terms(query.text)

            # Expand using graph traversal
            expanded_nodes = await self._traverse_graph(candidate_nodes, query)

            # Score and rank results
            results = []
            for node_id in expanded_nodes:
                if node_id in self.node_data:
                    doc = self.node_data[node_id]
                    score = self._calculate_graph_score(node_id, query, candidate_nodes, expanded_nodes)

                    result = SearchResult(
                        id=node_id,
                        title=doc.get('title', ''),
                        content=doc.get('content', ''),
                        url=doc.get('url'),
                        score=score,
                        semantic_score=0.0,
                        keyword_score=0.0,
                        graph_score=score,
                        metadata=doc.get('metadata', {}),
                        signals={SignalType.GRAPH_RELEVANCE: score},
                        explanation={
                            'method': 'graph_traversal',
                            'node_id': node_id,
                            'graph_score': score,
                            'path_length': self._calculate_path_length(node_id, candidate_nodes)
                        }
                    )
                    results.append(result)

            # Sort by score and return top results
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:query.limit]

        except Exception as e:
            logger.error(f"Error in graph search: {e}")
            return []

    def _find_nodes_with_terms(self, query_text: str) -> Set[str]:
        """Find nodes containing query terms"""
        terms = query_text.lower().split()
        candidate_nodes = set()

        for node_id, node_data in self.node_data.items():
            content = f"{node_data.get('title', '')} {node_data.get('content', '')}".lower()

            if any(term in content for term in terms):
                candidate_nodes.add(node_id)

        return candidate_nodes

    async def _traverse_graph(self, start_nodes: Set[str], query: SearchQuery) -> Set[str]:
        """Traverse graph to find related nodes"""
        if not start_nodes:
            return set()

        max_depth = query.metadata.get('max_depth', 3)
        relevant_relationships = query.metadata.get('relationship_types', [])

        visited = set()
        queue = list(start_nodes)
        result_nodes = set(start_nodes)

        while queue and len(visited) < 1000:  # Prevent infinite loops
            current_node = queue.pop(0)

            if current_node in visited:
                continue

            visited.add(current_node)
            result_nodes.add(current_node)

            # Explore neighbors
            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in visited:
                    edge_data = self.graph.get_edge_data(current_node, neighbor)
                    if (not relevant_relationships or
                        edge_data.get('relationship_type') in relevant_relationships):
                        queue.append(neighbor)

            # Limit depth
            if len(visited) >= max_depth * 10:
                break

        return result_nodes

    def _calculate_graph_score(self, node_id: str, query: SearchQuery,
                             start_nodes: Set[str], all_nodes: Set[str]) -> float:
        """Calculate graph-based relevance score"""
        if node_id not in self.page_rank:
            return 0.0

        # Base PageRank score
        base_score = self.page_rank.get(node_id, 0.0)

        # Distance from starting nodes
        distance = self._calculate_path_length(node_id, start_nodes)

        # Recency boost if available
        recency_boost = 1.0
        if node_id in self.node_data:
            indexed_at = self.node_data[node_id].get('indexed_at')
            if indexed_at:
                time_diff = (datetime.now() - datetime.fromisoformat(indexed_at)).days
                recency_boost = max(0.1, 1.0 - time_diff / 365)  # Decay over a year

        # Authority boost based on connections
        authority_boost = min(1.0, self.graph.degree(node_id) / 10.0)

        return base_score * 0.5 + recency_boost * 0.2 + authority_boost * 0.3

    def _calculate_path_length(self, target_node: str, start_nodes: Set[str]) -> int:
        """Calculate shortest path length from start nodes to target"""
        if not start_nodes:
            return float('inf')

        try:
            min_length = float('inf')
            for start_node in start_nodes:
                if nx.has_path(self.graph, start_node, target_node):
                    path_length = nx.shortest_path_length(self.graph, start_node, target_node)
                    min_length = min(min_length, path_length)
            return min_length
        except:
            return float('inf')

    async def update_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """Update document in knowledge graph"""
        try:
            # Remove old node if exists
            if doc_id in self.graph:
                self.graph.remove_node(doc_id)
                if doc_id in self.node_data:
                    del self.node_data[doc_id]

            # Add updated document
            await self._add_node_to_graph(document)

            # Recalculate PageRank
            if self.graph.number_of_nodes() > 0:
                self.page_rank = nx.pagerank(self.graph, alpha=0.85)

            return True

        except Exception as e:
            logger.error(f"Error updating document in knowledge graph: {e}")
            return False

    async def delete_document(self, doc_id: str) -> bool:
        """Delete document from knowledge graph"""
        try:
            if doc_id in self.graph:
                self.graph.remove_node(doc_id)
                if doc_id in self.node_data:
                    del self.node_data[doc_id]
                if doc_id in self.page_rank:
                    del self.page_rank[doc_id]

            return True

        except Exception as e:
            search_engine_error(f"Error deleting document from knowledge graph: {e}")
            return False

class RankingEngine:
    """Advanced multi-signal ranking engine inspired by modern search engines"""

    def __init__(self):
        self.signal_weights = {
            SignalType.SEMANTIC_SIMILARITY: 0.4,
            SignalType.KEYWORD_BM25: 0.3,
            SignalType.GRAPH_RELEVANCE: 0.2,
            SignalType.RECENCY_BOOST: 0.1,
            SignalType.AUTHORITY_SCORE: 0.1
        }

        # Adaptive learning parameters
        self.click_data = defaultdict(list)
        self.user_feedback = defaultdict(list)
        self.performance_metrics = defaultdict(float)
        self.learning_rate = 0.1

    def update_weights(self, performance_metrics: Dict[str, float]):
        """Update signal weights based on performance feedback"""
        for signal_type, performance in performance_metrics.items():
            if signal_type in self.signal_weights:
                # Adjust weight based on performance
                current_weight = self.signal_weights[signal_type]
                adjustment = (performance - 0.5) * self.learning_rate
                new_weight = max(0.01, min(1.0, current_weight + adjustment))
                self.signal_weights[signal_type] = new_weight

        logger.info(f"Updated signal weights: {self.signal_weights}")

    def record_click(self, result_id: str, position: int, query_id: str, session_id: str):
        """Record user click for learning"""
        self.click_data[result_id].append({
            'position': position,
            'query_id': query_id,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        })

    def record_feedback(self, result_id: str, feedback: Dict[str, Any]):
        """Record user feedback for learning"""
        self.user_feedback[result_id].append(feedback)

    def calculate_click_through_rate(self, result_ids: List[str]) -> Dict[str, float]:
        """Calculate click-through rate for results"""
        ctr_scores = {}

        for result_id in result_ids:
            clicks = len(self.click_data.get(result_id, []))
            impressions = len(self.click_data.get(result_id, [])) + 1  # Avoid division by zero
            ctr_scores[result_id] = clicks / impressions

        return ctr_scores

    def rank_results(self, query: SearchQuery, results: List[SearchResult],
                     semantic_results: List[SearchResult],
                     keyword_results: List[SearchResult],
                     graph_results: List[SearchResult]) -> List[SearchResult]:
        """Rank results using multiple signals"""

        strategy = query.strategy

        if strategy == SearchStrategy.SEMANTIC_ONLY:
            return self._rank_semantic_only(results, query)
        elif strategy == SearchStrategy.KEYWORD_ONLY:
            return self._rank_keyword_only(results, query)
        elif strategy == SearchStrategy.GRAPH_ONLY:
            return self._rank_graph_only(results, query)
        elif strategy == SearchStrategy.HYBRID_BALANCED:
            return self._rank_hybrid_balanced(results, query, semantic_results, keyword_results, graph_results)
        elif strategy == SearchStrategy.HYBRID_SEMANTIC_WEIGHTED:
            return self._rank_hybrid_semantic_weighted(results, query, semantic_results, keyword_results, graph_results)
        elif strategy == SearchStrategy.HYBRID_GRAPH_WEIGHTED:
            return self._rank_hybrid_graph_weighted(results, query, semantic_results, keyword_results, graph_results)
        elif strategy == SearchStrategy.ADAPTIVE:
            return self._rank_adaptive(results, query, semantic_results, keyword_results, graph_results)
        else:
            return self._rank_hybrid_balanced(results, query, semantic_results, keyword_results, graph_results)

    def _rank_semantic_only(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Rank using semantic similarity only"""
        for result in results:
            result.score = result.semantic_score
            result.signals[SignalType.SEMANTIC_SIMILARITY] = result.score

        return sorted(results, key=lambda x: x.score, reverse=True)

    def _rank_keyword_only(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Rank using keyword BM25 only"""
        for result in results:
            result.score = result.keyword_score
            result.signals[SignalType.KEYWORD_BM25] = result.score

        return sorted(results, key=lambda x: x.score, reverse=True)

    def _rank_graph_only(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Rank using graph relevance only"""
        for result in results:
            result.score = result.graph_score
            result.signals[SignalType.GRAPH_RELEVANCE] = result.score

        return sorted(results, key=lambda x: x.score, reverse=True)

    def _rank_hybrid_balanced(self, results: List[SearchResult], query: SearchQuery,
                         semantic_results: List[SearchResult],
                         keyword_results: List[SearchResult],
                         graph_results: List[SearchResult]) -> List[SearchResult]:
        """Rank using balanced hybrid approach"""
        for result in results:
            # Combine scores with default weights
            semantic_score = result.semantic_score
            keyword_score = result.keyword_score
            graph_score = result.graph_score

            result.score = (
                semantic_score * self.signal_weights[SignalType.SEMANTIC_SIMILARITY] +
                keyword_score * self.signal_weights[SignalType.KEYWORD_BM25] +
                graph_score * self.signal_weights[SignalType.GRAPH_RELEVANCE]
            )

            result.signals[SignalType.SEMANTIC_SIMILARITY] = semantic_score
            result.signals[SignalType.KEYWORD_BM25] = keyword_score
            result.signals[SignalType.GRAPH_RELEVANCE] = graph_score

        # Apply recency boost
        self._apply_recency_boost(results, query)

        return sorted(results, key=lambda x: x.score, reverse=True)

    def _rank_hybrid_semantic_weighted(self, results: List[SearchResult], query: SearchQuery,
                                     semantic_results: List[SearchResult],
                                     keyword_results: List[SearchResult],
                                     graph_results: List[Result]) -> List[SearchResult]:
        """Rank with semantic emphasis"""
        for result in results:
            semantic_score = result.semantic_score
            keyword_score = result.keyword_score
            graph_score = result.graph_score

            # Heavily weight semantic search
            result.score = (
                semantic_score * 0.6 +
                keyword_score * 0.2 +
                graph_score * 0.2
            )

            result.signals[SignalType.SEMANTIC_SIMILARITY] = semantic_score * 1.5
            result.signals[SignalType.KEYWORD_BM25] = keyword_score
            result.signals[SignalType.GRAPH_RELEVANCE] = graph_score

        self._apply_recency_boost(results, query)
        return sorted(results, key=lambda x: x.score, reverse=True)

    def _rank_hybrid_graph_weighted(self, results: List[SearchResult], query: SearchQuery,
                                   semantic_results: List[SearchResult],
                                   keyword_results: List[SearchResult],
                                   graph_results: List[Result]) -> List[SearchResult]:
        """Rank with graph emphasis"""
        for result in results:
            semantic_score = result.semantic_score
            keyword_score = result.keyword_score
            graph_score = result.graph_score

            # Heavily weight graph search
            result.score = (
                semantic_score * 0.3 +
                keyword_score * 0.2 +
                graph_score * 0.5
            )

            result.signals[SignalType.SEMANTIC_SIMILARITY] = semantic_score
            result.signals[SignalType.KEYWORD_BM25] = keyword_score
            result.signals[SignalType.GRAPH_RELEVANCE] = graph_score * 1.5

        self._apply_recency_boost(results, query)
        return sorted(results, key=lambda x: x.score, reverse=True)

    def _rank_adaptive(self, results: List[SearchResult], query: SearchQuery,
                      semantic_results: List[SearchResult],
                      keyword_results: List[SearchResult],
                      graph_results: List[SearchResult]) -> List[SearchResult]:
        """Rank using adaptive learning"""
        # Calculate click-through rates if available
        result_ids = [r.id for r in results]
        ctr_scores = self.calculate_click_through_rate(result_ids)

        for result in results:
            base_score = result.score
            ctr_boost = ctr_scores.get(result.id, 0.0)

            # Apply CTR boost
            result.score = base_score * (1 + ctr_boost)

            # Store signals for explanation
            result.signals[SignalType.CLICK_THROUGH_RATE] = ctr_boost

        self._apply_recency_boost(results, query)
        return sorted(results, key=lambda x: x.score, reverse=True)

    def _apply_recency_boost(self, results: List[SearchResult], query: SearchQuery):
        """Apply recency boost to results"""
        now = datetime.now()
        time_range = query.time_range

        for result in results:
            if result.updated_at:
                time_diff = (now - result.updated_at).days
                recency_boost = max(0.1, 1.0 - time_diff / 365)

                # Apply recency boost
                result.score *= recency_boost
                result.signals[SignalType.RECENCY_BOOST] = recency_boost

    def get_ranking_explanation(self, result: SearchResult) -> Dict[str, Any]:
        """Get detailed explanation of ranking factors"""
        return {
            'final_score': result.score,
            'signal_weights': self.signal_weights,
            'signal_scores': dict(result.signals),
            'explanation': result.explanation,
            'factors': {
                'semantic_similarity': f"Score: {result.semantic_score:.3f}, Weight: {self.signal_weights.get(SignalType.SEMANTIC_SIMILARITY):.2f}",
                'keyword_relevance': f"Score: {result.keyword_score:.3f}, Weight: {self.signal_weights.get(SignalType.KEYWORD_BM25):.2f}",
                'graph_relevance': f"Score: {result.graph_score:.3f}, Weight: {self.signal_weights.get(SignalType.GRAPH_RELEVANCE):.2f}"
            }
        }

class HybridSearchEngine:
    """Main hybrid search engine orchestrating all search components"""

    def __init__(self):
        self.semantic_engine = SemanticSearchEngine()
        self.keyword_engine = KeywordSearchEngine()
        self.graph_engine = KnowledgeGraphEngine()
        self.ranking_engine = RankingEngine()
        self.search_sessions = {}
        self.search_history = []
        self.performance_metrics = defaultdict(list)

    async def initialize(self):
        """Initialize all search components"""
        logger.info("Initializing hybrid search engine...")
        await self.semantic_engine.initialize()
        logger.info("Hybrid search engine initialized")

    async def index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Index documents across all search engines"""
        logger.info(f"Indexing {len(documents)} documents in hybrid search engine")

        # Index in parallel for performance
        tasks = [
            self.semantic_engine.index(documents),
            self.keyword_engine.index(documents),
            self.graph_engine.index(documents)
        ]

        results = await asyncio.gather(*tasks)

        results_summary = {
            'semantic_search': results[0],
            'keyword_search': results[1],
            'graph_search': results[2],
            'total_documents': len(documents),
            'success': all(results)
        }

        logger.info(f"Indexing completed: {results_summary}")
        return results_summary

    async def search(self, query: SearchQuery) -> Dict[str, Any]:
        """Perform hybrid search"""
        search_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            logger.info(f"Performing hybrid search: {query.text[:100]}...")

            # Initialize search session if user context provided
            session_id = None
            if query.user_context and 'session_id' in query.user_context:
                session_id = query.user_context['session_id']
                if session_id not in self.search_sessions:
                    self.search_sessions[session_id] = SearchSession(
                        session_id=session_id,
                        user_id=query.user_context.get('user_id'),
                        created_at=datetime.now()
                    )

            # Perform searches in parallel
            tasks = [
                self.semantic_engine.search(query),
                self.keyword_engine.search(query),
                self.graph_engine.search(query)
            ]

            semantic_results, keyword_results, graph_results = await asyncio.gather(*tasks)

            # Combine all results
            all_results = []
            all_results.extend(semantic_results)
            all_results.extend(keyword_results)
            all_results.extend(graph_results)

            # Remove duplicates based on ID
            unique_results = {}
            for result in all_results:
                if result.id not in unique_results:
                    unique_results[result.id] = result

            combined_results = list(unique_results.values())

            # Rank results
            ranked_results = self.ranking_engine.rank_results(
                query, combined_results, semantic_results, keyword_results, graph_results
            )

            # Apply pagination
            paginated_results = ranked_results[query.offset:query.offset + query.limit]

            # Record search in history
            search_record = {
                'search_id': search_id,
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'result_count': len(paginated_results),
                'session_id': session_id,
                'total_found': len(ranked_results),
                'search_time': time.time() - start_time
            }
            self.search_history.append(search_record)

            # Update session
            if session_id:
                session = self.search_sessions[session_id]
                session.query_history.append(query)
                session.last_activity = datetime.now()
                session.search_metrics['total_searches'] += 1
                session.search_metrics['average_results'] = (
                    session.search_metrics.get('average_results', 0) * (session.search_metrics['total_searches'] - 1) +
                    len(paginated_results)
                ) / session.search_metrics['total_searches']

            # Record performance metrics
            self.performance_metrics['search_time'].append(time.time() - start_time)

            # Prepare response
            response = {
                'search_id': search_id,
                'query': {
                    'text': query.text,
                    'strategy': query.strategy.value,
                    'filters': query.filters,
                    'limit': query.limit,
                    'offset': query.offset
                },
                'results': [
                    {
                        'id': result.id,
                        'title': result.title,
                        'content': result.content[:200] + '...' if len(result.content) > 200 else result.content,
                        'url': result.url,
                        'score': result.score,
                        'metadata': result.metadata,
                        'explanation': result.explanation
                    }
                    for result in paginated_results
                ],
                'metrics': {
                    'total_found': len(ranked_results),
                    'returned': len(paginated_results),
                    'search_time': time.time() - start_time,
                    'components': {
                        'semantic_search': {
                            'results_count': len(semantic_results),
                            'success': len(semantic_results) > 0
                        },
                        'keyword_search': {
                            'results_count': len(keyword_results),
                            'success': len(keyword_results) > 0
                        },
                        'graph_search': {
                            'results_count': len(graph_results),
                            'success': len(graph_results) > 0
                        }
                    }
                },
                'explanation': self._generate_search_explanation(paginated_results, search_record)
            }

            logger.info(f"Hybrid search completed: {len(paginated_results)} results in {time.time() - start_time:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return {
                'search_id': search_id,
                'query': query.text,
                'error': str(e),
                'results': [],
                'metrics': {'error': True}
            }

    def _generate_search_explanation(self, results: List[SearchResult], search_record: Dict) -> Dict[str, Any]:
        """Generate explanation of search results"""
        if not results:
            return {'explanation': 'No results found'}

        # Get explanations from top results
        explanations = []
        for result in results[:3]:
            explanation = self.ranking_engine.get_ranking_explanation(result)
            explanations.append(explanation)

        return {
            'summary': f"Found {len(results)} results using {search_record['query']['strategy'].value} strategy",
            'top_results_explanations': explanations,
            'strategy_performance': {
                'components': search_record['metrics']['components'],
                'search_time': f"{search_record['search_time']:.2f}s"
            }
        }

    def get_session_history(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get search session history"""
        if session_id not in self.search_sessions:
            return None

        session = self.search_sessions[session_id]
        return {
            'session_id': session.session_id,
            'user_id': session.user_id,
            'query_count': len(session.query_history),
            'last_activity': session.last_activity.isoformat(),
            'metrics': session.search_metrics,
            'query_history': [
                {
                    'text': q.text,
                    'strategy': q.strategy.value,
                    'timestamp': q.created_at.isoformat()
                } for q in session.query_history
            ]
        }

    def get_search_analytics(self) -> Dict[str, Any]:
        """Get comprehensive search analytics"""
        return {
            'total_searches': len(self.search_history),
            'average_search_time': (
                sum(self.performance_metrics['search_time']) / len(self.performance_metrics['search_time'])
                if self.performance_metrics['search_time'] else 0
            ),
            'session_count': len(self.search_sessions),
            'indexed_documents': {
                'semantic_search': self.semantic_engine.indexed_count,
                'keyword_search': self.keyword_engine.indexed_count,
                'graph_search': self.graph_engine.indexed_count
            },
            'signal_weights': self.ranking_engine.signal_weights,
            'recent_searches': self.search_history[-10:],
            'popular_queries': self._get_popular_queries()
        }

    def _get_popular_queries(self) -> List[Dict[str, Any]]:
        """Get most popular search queries"""
        query_counts = Counter(
            record['query']['text'] for record in self.search_history
        )

        return [
            {'query': query, 'count': count}
            for query, count in query_counts.most_common(10)
        ]

    def update_learning_weights(self, performance_metrics: Dict[str, float]):
        """Update learning weights based on performance feedback"""
        self.ranking_engine.update_weights(performance_metrics)
        logger.info(f"Updated learning weights: {self.ranking_engine.signal_weights}")

# Example usage and testing
if __name__ == "__main__":
    async def test_hybrid_search():
        """Test the hybrid search engine"""

        # Initialize search engine
        search_engine = HybridSearchEngine()
        await search_engine.initialize()

        # Create sample documents
        sample_documents = [
            {
                'id': 'doc1',
                'title': 'Machine Learning Fundamentals',
                'content': 'Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.',
                'url': 'https://example.com/ml-fundamentals',
                'entity_type': 'concept',
                'relationships': [
                    {'target_id': 'doc2', 'relationship_type': 'prerequisite_of', 'weight': 0.8},
                    {'target_id': 'doc3', 'relationship_type': 'applies_to', 'weight': 0.6}
                ],
                'metadata': {'category': 'education', 'difficulty': 'beginner'}
            },
            {
                'id': 'doc2',
                'title': 'Deep Neural Networks',
                'content': 'Deep neural networks are artificial neural networks with multiple layers between input and output.',
                'url': 'https://example.com/deep-nn',
                'entity_type': 'concept',
                'relationships': [
                    {'target_id': 'doc4', 'relationship_type': 'evolved_from', 'weight': 0.9}
                ],
                'metadata': {'category': 'technology', 'difficulty': 'advanced'}
            },
            {
                'id': 'doc3',
                'title': 'Computer Vision Applications',
                'content': 'Computer vision enables machines to interpret and understand visual information from images and videos.',
                'url': 'https://example.com/computer-vision',
                'entity_type': 'application',
                'relationships': [],
                'metadata': {'category': 'technology', 'difficulty': 'intermediate'}
            },
            {
                'id': 'doc4',
                'title': 'Natural Language Processing',
                'content': 'NLP combines computational linguistics and machine learning to understand human language.',
                'url': 'https://example.com/nlp',
                'entity_type': 'application',
                'relationships': [
                    {'target_id': 'doc5', 'relationship_type': 'integrates_with', 'weight': 0.7}
                ],
                'metadata': {'category': 'technology', 'difficulty': 'intermediate'}
            },
            {
                'id': 'doc5',
                'title': 'Data Science Integration',
                'content': 'Data science combines statistics, mathematics, and domain expertise to extract insights from data.',
                'url': 'https://example.com/data-science',
                'entity_type': 'concept',
                'relationships': [],
                'metadata': {'category': 'business', 'difficulty': 'advanced'}
            }
        ]

        # Index documents
        index_result = await search_engine.index_documents(sample_documents)
        print(f"Indexing result: {index_result}")

        # Test different search strategies
        test_queries = [
            SearchQuery(text="machine learning neural networks", strategy=SearchStrategy.HYBRID_SEMANTIC_WEIGHTED),
            SearchQuery(text="computer vision applications", strategy=SearchStrategy.GRAPH_ONLY),
            SearchQuery(text="fundamentals deep learning", strategy=SearchStrategy.KEYWORD_ONLY),
            SearchQuery(text="artificial intelligence overview", strategy=SearchStrategy.HYBRID_BALANCED)
        ]

        for query in test_queries:
            print(f"\nSearching for: {query.text}")
            search_result = await search_engine.search(query)

            print(f"Found {len(search_result['results'])} results")
            print(f"Search time: {search_result['metrics']['search_time']:.2f}s")
            print(f"Strategy: {search_result['query']['strategy']}")
            print(f"Components: {search_result['metrics']['components']}")

            # Display top 3 results
            for i, result in enumerate(search_result['results'][:3]):
                print(f"\nResult {i+1}:")
                print(f"  Title: {result['title']}")
                print(f"  Score: {result['score']:.3f}")
                print(f"  Semantic: {result['signals']['semantic_similarity']:.3f}")
                print(f"  Keyword: {result['signals']['keyword_bm25']:.3f}")
                print(f"  Graph: {result['signals'].get('graph_relevance', 0):.3f}")
                print(f"  Explanation: {result['explanation']['method']}")

        # Get analytics
        analytics = search_engine.get_search_analytics()
        print(f"\nSearch Analytics:")
        print(f"  Total searches: {analytics['total_searches']}")
        print(f" Average search time: {analytics['average_search_time']:.2f}s")
        print(f" Active sessions: {analytics['session_count']}")
        print(f" Indexed documents: {analytics['indexed_documents']}")

        # Test session history
        if search_engine.search_sessions:
            session_id = list(search_engine.search_sessions.keys())[0]
            session_history = search_engine.get_session_history(session_id)
            print(f"\nSession History:")
            print(f"  Session ID: {session_history['session_id']}")
            print(f"  Queries: {session_history['query_count']}")
            print(f" Average results: {session_history['metrics']['average_results']:.1f}")

    # Run test
    asyncio.run(test_hybrid_search())