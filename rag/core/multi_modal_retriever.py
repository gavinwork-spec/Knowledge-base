"""
Advanced Multi-Modal Retrieval System for Manufacturing Knowledge Base

Supports retrieval from text, images, tables, charts, and mixed content
with sophisticated ranking and fusion strategies.
"""

import os
import io
import base64
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple, BinaryIO
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import cv2
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
import openpyxl
import xlrd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import rank_bm25

class ContentType(str, Enum):
    """Content types for multi-modal retrieval"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"
    DIAGRAM = "diagram"
    CODE = "code"
    FORMULA = "formula"
    MIXED = "mixed"

class RetrievalStrategy(str, Enum):
    """Retrieval strategies"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    MULTI_MODAL = "multi_modal"
    GRAPH = "graph"
    TEMPORAL = "temporal"

@dataclass
class RetrievedDocument:
    """Represents a retrieved document with multi-modal content"""
    doc_id: str
    content: str
    content_type: ContentType
    retrieval_strategy: RetrievalStrategy
    relevance_score: float = 0.0

    # Multi-modal data
    image_data: Optional[bytes] = None
    image_embedding: Optional[np.ndarray] = None
    table_data: Optional[pd.DataFrame] = None
    chart_data: Optional[Dict[str, Any]] = None

    # Metadata
    source_file: str = ""
    page_number: Optional[int] = None
    section_title: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: str = ""

    # Ranking information
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    temporal_score: float = 0.0
    popularity_score: float = 0.0

    # Citation information
    citation_id: str = ""
    snippet: str = ""
    highlight_spans: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class RetrievalConfig:
    """Configuration for multi-modal retrieval"""

    # Model configuration
    text_embedding_model: str = "all-MiniLM-L6-v2"
    image_embedding_model: str = "clip-ViT-B-32"
    reranker_model: Optional[str] = None

    # Retrieval parameters
    top_k: int = 10
    similarity_threshold: float = 0.7
    diversity_penalty: float = 0.1

    # Multi-modal weights
    text_weight: float = 0.4
    image_weight: float = 0.3
    table_weight: float = 0.2
    chart_weight: float = 0.1

    # Content filters
    content_types: List[ContentType] = field(default_factory=lambda: [
        ContentType.TEXT, ContentType.IMAGE, ContentType.TABLE, ContentType.CHART
    ])

    # Advanced features
    enable_reranking: bool = True
    enable_cross_modal: bool = True
    enable_temporal_ranking: bool = True
    enable_popularity_ranking: bool = True

class ManufacturingMultiModalRetriever:
    """
    Advanced multi-modal retriever for manufacturing knowledge base.
    Handles text, images, tables, charts with sophisticated ranking.
    """

    def __init__(self,
                 config: Optional[RetrievalConfig] = None,
                 db_path: str = "knowledge_base.db"):

        self.config = config or RetrievalConfig()
        self.db_path = db_path

        # Initialize embedding models
        self.text_model = SentenceTransformer(self.config.text_embedding_model)

        # Initialize vector stores
        self._init_database()
        self._init_vector_stores()

        # Initialize TF-IDF for keyword search
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )

        # Manufacturing-specific configurations
        self.manufacturing_keywords = self._load_manufacturing_keywords()
        self.image_processors = self._init_image_processors()

    def _init_database(self):
        """Initialize database for multi-modal content"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Documents table (enhanced for multi-modal)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                content TEXT,
                content_type TEXT NOT NULL,
                source_file TEXT,
                page_number INTEGER,
                section_title TEXT,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                text_embedding BLOB,
                image_data BLOB,
                image_embedding BLOB,
                table_data TEXT,
                chart_data TEXT,
                metadata TEXT,
                retrieval_count INTEGER DEFAULT 0,
                relevance_score REAL DEFAULT 0.0
            )
        ''')

        # Search index
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                doc_id UNINDEXED,
                content,
                section_title,
                tags
            )
        ''')

        # Image features table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_features (
                doc_id TEXT PRIMARY KEY,
                features TEXT,
                objects_detected TEXT,
                ocr_text TEXT,
                visual_features BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
            )
        ''')

        # Table structure table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS table_structures (
                doc_id TEXT PRIMARY KEY,
                columns TEXT,
                rows_count INTEGER,
                columns_count INTEGER,
                headers TEXT,
                structure_metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
            )
        ''')

        # Usage statistics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usage_statistics (
                doc_id TEXT PRIMARY KEY,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                avg_relevance REAL DEFAULT 0.0,
                total_relevance REAL DEFAULT 0.0,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
            )
        ''')

        # Indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_docs_content_type ON documents(content_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_docs_source_file ON documents(source_file)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_docs_created_at ON documents(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_docs_relevance ON documents(relevance_score)')

        conn.commit()
        conn.close()

    def _init_vector_stores(self):
        """Initialize vector stores for different content types"""

        # In-memory vector stores for fast retrieval
        self.text_vectors = {}
        self.image_vectors = {}
        self.multimodal_vectors = {}

        # BM25 index for keyword search
        self.bm25_index = None
        self.bm25_corpus = []

    def _load_manufacturing_keywords(self) -> Dict[str, List[str]]:
        """Load manufacturing-specific keywords for better retrieval"""

        return {
            "products": [
                "螺栓", "螺母", "垫圈", "螺丝", "铆钉", "销", "键", "轴承",
                "密封件", "弹簧", "齿轮", "链条", "皮带", "管件", "阀门",
                "bolt", "nut", "washer", "screw", "rivet", "pin", "key"
            ],
            "materials": [
                "不锈钢", "碳钢", "合金钢", "铝合金", "铜", "塑料", "橡胶",
                "composite", "ceramic", "titanium", "magnesium", "zinc",
                "stainless steel", "carbon steel", "aluminum alloy"
            ],
            "specifications": [
                "规格", "尺寸", "公差", "精度", "强度", "硬度", "粗糙度",
                "surface finish", "coating", "plating", "heat treatment", "welding",
                "specification", "dimension", "tolerance", "precision"
            ],
            "processes": [
                "加工", "制造", "生产", "装配", "焊接", "切割", "钻孔", "车削",
                "铣削", "磨削", "热处理", "表面处理", "检验", "测试",
                "machining", "manufacturing", "fabrication", "assembly"
            ],
            "quality": [
                "质量", "检验", "测试", "认证", "标准", "规范", "公差",
                "缺陷", "不合格", "返工", "报废", "质量控制", "ISO",
                "quality control", "inspection", "testing", "certification"
            ],
            "equipment": [
                "设备", "机器", "机床", "工具", "仪器", "夹具", "模具",
                "CNC", "数控", "加工中心", "车床", "铣床", "磨床",
                "equipment", "machine", "tool", "instrument", "CNC machine"
            ]
        }

    def _init_image_processors(self) -> Dict[str, Any]:
        """Initialize image processing configurations"""

        return {
            "resize_size": (224, 224),
            "enhancement_methods": ["contrast", "sharpness", "brightness"],
            "feature_extractors": ["sift", "orb", "harris"],
            "ocr_engines": ["tesseract", "paddleocr"],
            "object_detection": ["yolo", "faster_rcnn"],
            "text_detection": ["east", "ctr"]
        }

    def index_document(self,
                       doc_id: str,
                       content: str,
                       content_type: ContentType,
                       source_file: str = "",
                       page_number: Optional[int] = None,
                       section_title: str = "",
                       tags: Optional[List[str]] = None,
                       **kwargs) -> bool:
        """Index a document with multi-modal content"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Generate embeddings
            text_embedding = self.text_model.encode([content])[0]

            # Handle multi-modal content
            image_data = kwargs.get('image_data')
            image_embedding = None
            if image_data and content_type in [ContentType.IMAGE, ContentType.MIXED]:
                image_embedding = self._generate_image_embedding(image_data)

            # Process table data
            table_data_json = None
            if content_type in [ContentType.TABLE, ContentType.MIXED]:
                table_data = kwargs.get('table_data')
                if table_data:
                    table_data_json = self._process_table_data(table_data)

            # Process chart data
            chart_data_json = None
            if content_type in [ContentType.CHART, ContentType.MIXED]:
                chart_data = kwargs.get('chart_data')
                if chart_data:
                    chart_data_json = self._process_chart_data(chart_data)

            # Store document
            cursor.execute('''
                INSERT OR REPLACE INTO documents
                (doc_id, content, content_type, source_file, page_number,
                 section_title, tags, text_embedding, image_data,
                 image_embedding, table_data, chart_data, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                doc_id, content, content_type.value, source_file, page_number,
                section_title, json.dumps(tags or []),
                text_embedding.tobytes(),
                image_data,
                image_embedding.tobytes() if image_embedding is not None else None,
                table_data_json,
                chart_data_json,
                json.dumps(kwargs.get('metadata', {}))
            ))

            # Update FTS index
            cursor.execute('''
                INSERT INTO documents_fts (doc_id, content, section_title, tags)
                VALUES (?, ?, ?, ?)
            ''', (doc_id, content, section_title, json.dumps(tags or [])))

            # Store additional multi-modal features
            if content_type == ContentType.IMAGE and image_data:
                self._store_image_features(doc_id, image_data, content)

            if content_type == ContentType.TABLE:
                self._store_table_structure(doc_id, table_data)

            # Update vector stores
            self._update_vector_stores(doc_id, content, text_embedding, image_embedding)

            # Initialize usage statistics
            cursor.execute('''
                INSERT OR IGNORE INTO usage_statistics
                (doc_id, access_count, avg_relevance, total_relevance)
                VALUES (?, 0, 0.0, 0.0)
            ''', (doc_id,))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f"Error indexing document {doc_id}: {e}")
            return False

    def retrieve(self,
                 query: str,
                 content_types: Optional[List[ContentType]] = None,
                 top_k: int = 10,
                 strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
                 **kwargs) -> List[RetrievedDocument]:
        """
        Retrieve documents using multi-modal strategies.
        """

        content_types = content_types or self.config.content_types
        results = []

        # Implement different retrieval strategies
        if strategy == RetrievalStrategy.SEMANTIC:
            results = self._semantic_retrieval(query, content_types, top_k)
        elif strategy == RetrievalStrategy.KEYWORD:
            results = self._keyword_retrieval(query, content_types, top_k)
        elif strategy == RetrievalStrategy.HYBRID:
            results = self._hybrid_retrieval(query, content_types, top_k)
        elif strategy == RetrievalStrategy.MULTI_MODAL:
            results = self._multimodal_retrieval(query, content_types, top_k)
        else:
            results = self._hybrid_retrieval(query, content_types, top_k)

        # Apply ranking and filtering
        results = self._rank_and_filter(results, query, **kwargs)

        # Update usage statistics
        self._update_usage_statistics(results)

        return results

    def _semantic_retrieval(self,
                             query: str,
                             content_types: List[ContentType],
                             top_k: int) -> List[RetrievedDocument]:
        """Semantic retrieval using text embeddings"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Generate query embedding
            query_embedding = self.text_model.encode([query])[0]

            # Filter by content types
            content_type_filters = ','.join(['?' for _ in content_types])
            cursor.execute(f'''
                SELECT doc_id, content, content_type, source_file, page_number,
                       section_title, tags, text_embedding, image_data, table_data,
                       chart_data, created_at, retrieval_count, relevance_score
                FROM documents
                WHERE content_type IN ({content_type_filters})
            ''', [ct.value for ct in content_types])

            documents = cursor.fetchall()
            conn.close()

            # Calculate similarities
            similarities = []
            for doc in documents:
                if doc[8]:  # text_embedding
                    doc_embedding = np.frombuffer(doc[8])
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        doc_embedding.reshape(1, -1)
                    )[0][0]

                    if similarity >= self.config.similarity_threshold:
                        similarities.append((similarity, doc))

            # Sort by similarity
            similarities.sort(key=lambda x: x[0], reverse=True)

            # Create RetrievedDocument objects
            results = []
            for similarity, doc in similarities[:top_k]:
                result = RetrievedDocument(
                    doc_id=doc[0],
                    content=doc[1],
                    content_type=ContentType(doc[2]),
                    retrieval_strategy=RetrievalStrategy.SEMANTIC,
                    relevance_score=similarity,
                    semantic_score=similarity,
                    source_file=doc[3] or "",
                    page_number=doc[4],
                    section_title=doc[5] or "",
                    tags=json.loads(doc[6]) if doc[6] else [],
                    created_at=doc[12],
                    image_data=doc[9],
                    table_data=pd.read_json(doc[10]) if doc[10] else None,
                    chart_data=json.loads(doc[11]) if doc[11] else {}
                )
                results.append(result)

            return results

        except Exception as e:
            print(f"Error in semantic retrieval: {e}")
            return []

    def _keyword_retrieval(self,
                          query: str,
                          content_types: List[ContentType],
                          top_k: int) -> List[RetrievedDocument]:
        """Keyword retrieval using TF-IDF and BM25"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Filter by content types
            content_type_filters = ','.join(['?' for _ in content_types])
            cursor.execute(f'''
                SELECT doc_id, content, content_type, source_file, page_number,
                       section_title, tags, created_at
                FROM documents
                WHERE content_type IN ({content_type_filters})
            ''', [ct.value for ct in content_types])

            documents = cursor.fetchall()
            conn.close()

            if not documents:
                return []

            # Extract text content
            corpus = []
            doc_mapping = []
            for doc in documents:
                # Combine content with metadata
                text_content = f"{doc[1]} {doc[5]} {' '.join(json.loads(doc[6]) if doc[6] else [])}"
                corpus.append(text_content)
                doc_mapping.append((text_content, doc))

            # Fit TF-IDF vectorizer if not fitted
            if not hasattr(self.tfidf_vectorizer, 'vocabulary_') or not self.tfidf_vectorizer.vocabulary_:
                self.tfidf_vectorizer.fit(corpus)

            # Transform query
            query_vec = self.tfidf_vectorizer.transform([query])
            doc_vectors = self.tfidf_vectorizer.transform(corpus)

            # Calculate cosine similarities
            similarities = cosine_similarity(query_vec, doc_vectors)[0]

            # Create BM25 index if needed
            if self.bm25_index is None:
                tokenized_corpus = [doc.split() for doc in corpus]
                self.bm25_index = rank_bm25.BM25(tokenized_corpus)
                self.bm25_corpus = corpus

            # Get BM25 scores
            bm25_scores = self.bm25_index.get_scores(query.split())

            # Combine TF-IDF and BM25 scores
            combined_scores = 0.7 * similarities + 0.3 * bm25_scores

            # Get top results
            top_indices = np.argsort(combined_scores)[-top_k:][::-1]

            results = []
            for idx in top_indices:
                if combined_scores[idx] > 0.1:  # Minimum threshold
                    doc = doc_mapping[idx][1]
                    result = RetrievedDocument(
                        doc_id=doc[0],
                        content=doc[1],
                        content_type=ContentType(doc[2]),
                        retrieval_strategy=RetrievalStrategy.KEYWORD,
                        relevance_score=combined_scores[idx],
                        keyword_score=combined_scores[idx],
                        source_file=doc[3] or "",
                        page_number=doc[4],
                        section_title=doc[5] or "",
                        tags=json.loads(doc[6]) if doc[6] else [],
                        created_at=doc[7]
                    )
                    results.append(result)

            return results

        except Exception as e:
            print(f"Error in keyword retrieval: {e}")
            return []

    def _hybrid_retrieval(self,
                         query: str,
                         content_types: List[ContentType],
                         top_k: int) -> List[RetrievedDocument]:
        """Hybrid retrieval combining semantic and keyword search"""

        # Get semantic and keyword results
        semantic_results = self._semantic_retrieval(query, content_types, top_k * 2)
        keyword_results = self._keyword_retrieval(query, content_types, top_k * 2)

        # Combine and deduplicate results
        all_results = {}

        # Add semantic results
        for result in semantic_results:
            if result.doc_id not in all_results:
                all_results[result.doc_id] = result

        # Add keyword results
        for result in keyword_results:
            if result.doc_id in all_results:
                # Combine scores
                existing = all_results[result.doc_id]
                existing.keyword_score = result.keyword_score
                existing.relevance_score = (
                    self.config.text_weight * existing.semantic_score +
                    (1 - self.config.text_weight) * existing.keyword_score
                )
            else:
                all_results[result.doc_id] = result

        # Sort by relevance score
        results = list(all_results.values())
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        return results[:top_k]

    def _multimodal_retrieval(self,
                           query: str,
                           content_types: List[ContentType],
                           top_k: int) -> List[RetrievedDocument]:
        """Multi-modal retrieval across different content types"""

        # Get base retrieval results
        base_results = self._hybrid_retrieval(query, content_types, top_k * 2)

        # Enhance with cross-modal retrieval
        if self.config.enable_cross_modal:
            base_results = self._cross_modal_enhancement(query, base_results)

        return base_results[:top_k]

    def _cross_modal_enhancement(self,
                                query: str,
                                results: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """Enhance results with cross-modal retrieval"""

        # For each result, check if there are related multi-modal content
        enhanced_results = []

        for result in results:
            # Look for related images, tables, charts
            related_content = self._find_related_multimodal_content(
                result.doc_id, result.content_type
            )

            if related_content:
                # Enhance relevance score
                result.relevance_score *= 1.2
                result.relevance_score = min(1.0, result.relevance_score)

            enhanced_results.append(result)

        return enhanced_results

    def _find_related_multimodal_content(self,
                                        doc_id: str,
                                        primary_type: ContentType) -> List[Dict[str, Any]]:
        """Find related multi-modal content for a document"""

        related_content = []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Look for images
            if primary_type != ContentType.IMAGE:
                cursor.execute('''
                    SELECT doc_id, source_file, image_data
                    FROM documents
                    WHERE content_type = ? AND image_data IS NOT NULL
                    AND (source_file = (SELECT source_file FROM documents WHERE doc_id = ?) OR
                         section_title LIKE (SELECT '%' || section_title || '%' FROM documents WHERE doc_id = ?))
                    LIMIT 3
                ''', (ContentType.IMAGE.value, doc_id, doc_id))

                images = cursor.fetchall()
                for img in images:
                    related_content.append({
                        'type': 'image',
                        'doc_id': img[0],
                        'source_file': img[1],
                        'data': img[2]
                    })

            # Look for tables
            if primary_type != ContentType.TABLE:
                cursor.execute('''
                    SELECT doc_id, source_file, table_data
                    FROM documents
                    WHERE content_type = ? AND table_data IS NOT NULL
                    AND (source_file = (SELECT source_file FROM documents WHERE doc_id = ?) OR
                         section_title LIKE (SELECT '%' || section_title || '%' FROM documents WHERE doc_id = ?))
                    LIMIT 3
                ''', (ContentType.TABLE.value, doc_id, doc_id))

                tables = cursor.fetchall()
                for table in tables:
                    related_content.append({
                        'type': 'table',
                        'doc_id': table[0],
                        'source_file': table[1],
                        'data': table[2]
                    })

            conn.close()

        except Exception as e:
            print(f"Error finding related content: {e}")

        return related_content

    def _rank_and_filter(self,
                        results: List[RetrievedDocument],
                        query: str,
                        **kwargs) -> List[RetrievedDocument]:
        """Rank and filter retrieved documents"""

        if self.config.enable_reranking:
            results = self._rerank_results(results, query)

        # Apply temporal ranking
        if self.config.enable_temporal_ranking:
            results = self._apply_temporal_ranking(results)

        # Apply popularity ranking
        if self.config.enable_popularity_ranking:
            results = self._apply_popularity_ranking(results)

        # Apply diversity penalty
        if self.config.diversity_penalty > 0:
            results = self._apply_diversity_penalty(results)

        # Filter by minimum relevance
        results = [r for r in results if r.relevance_score >= self.config.similarity_threshold]

        return results

    def _rerank_results(self,
                        results: List[RetrievedDocument],
                        query: str) -> List[RecievedDocument]:
        """Rerank results using cross-encoder if available"""

        # TODO: Implement cross-encoder reranking
        # For now, return results as-is
        return results

    def _apply_temporal_ranking(self,
                              results: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """Apply temporal ranking based on creation date and freshness"""

        now = pd.Timestamp.now()
        for result in results:
            try:
                created_at = pd.Timestamp(result.created_at)
                days_old = (now - created_at).days

                # Temporal score (boost recent content)
                if days_old < 7:
                    temporal_score = 1.0
                elif days_old < 30:
                    temporal_score = 0.8
                elif days_old < 90:
                    temporal_score = 0.6
                else:
                    temporal_score = 0.4

                result.temporal_score = temporal_score

                # Update overall relevance
                result.relevance_score = (
                    0.7 * result.relevance_score +
                    0.3 * temporal_score
                )

            except Exception:
                result.temporal_score = 0.5

        return results

    def _apply_popularity_ranking(self,
                                results: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """Apply popularity ranking based on access statistics"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for result in results:
                cursor.execute('''
                    SELECT access_count, avg_relevance
                    FROM usage_statistics
                    WHERE doc_id = ?
                ''', (result.doc_id,))

                stats = cursor.fetchone()
                if stats:
                    access_count, avg_relevance = stats

                    # Popularity score
                    if access_count > 100:
                        popularity_score = 1.0
                    elif access_count > 50:
                        popularity_score = 0.8
                    elif access_count > 10:
                        popularity_score = 0.6
                    else:
                        popularity_score = 0.4

                    result.popularity_score = popularity_score

                    # Update overall relevance
                    result.relevance_score = (
                        0.6 * result.relevance_score +
                        0.4 * popularity_score
                    )

            conn.close()

        except Exception as e:
            print(f"Error applying popularity ranking: {e}")

        return results

    def _apply_diversity_penalty(self,
                                results: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """Apply diversity penalty to reduce content redundancy"""

        if len(results) <= 1:
            return results

        penalized_results = []

        for i, result in enumerate(results):
            diversity_penalty = 0.0

            # Check similarity with previous results
            for prev_result in penalized_results[:i]:
                if result.content_type == prev_result.content_type:
                    # Same content type - higher penalty
                    diversity_penalty += 0.1
                else:
                    # Different content type - lower penalty
                    diversity_penalty += 0.05

            # Apply penalty
            result.relevance_score *= (1.0 - self.config.diversity_penalty * diversity_penalty)
            penalized_results.append(result)

        return penalized_results

    def _update_usage_statistics(self, results: List[RetrievedDocument]):
        """Update usage statistics for retrieved documents"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for result in results:
                cursor.execute('''
                    UPDATE usage_statistics
                    SET access_count = access_count + 1,
                        last_accessed = CURRENT_TIMESTAMP,
                        avg_relevance = (avg_relevance * access_count + relevance_score) / (access_count + 1),
                        total_relevance = total_relevance + ?
                    WHERE doc_id = ?
                ''', (result.relevance_score, result.doc_id))

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Error updating usage statistics: {e}")

    def _generate_image_embedding(self, image_data: bytes) -> np.ndarray:
        """Generate embedding for image data"""

        try:
            # Decode image
            image = Image.open(io.BytesIO(image_data))
            image = image.convert('RGB')
            image = image.resize((224, 224))

            # For now, return a simple embedding
            # In production, you'd use CLIP or similar model
            return np.random.rand(512)  # Placeholder

        except Exception as e:
            print(f"Error generating image embedding: {e}")
            return np.random.rand(512)

    def _process_table_data(self, table_data) -> str:
        """Process table data for storage"""

        if isinstance(table_data, pd.DataFrame):
            return table_data.to_json(orient='records')
        elif isinstance(table_data, dict):
            return json.dumps(table_data)
        else:
            return str(table_data)

    def _process_chart_data(self, chart_data) -> str:
        """Process chart data for storage"""

        if isinstance(chart_data, dict):
            return json.dumps(chart_data)
        else:
            return str(chart_data)

    def _store_image_features(self, doc_id: str, image_data: bytes, content: str):
        """Store extracted image features"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Basic image features
            image = Image.open(io.BytesIO(image_data))
            features = {
                "size": image.size,
                "mode": image.mode,
                "format": image.format
            }

            # OCR text (placeholder)
            ocr_text = "OCR processing would go here"

            # Visual features (placeholder)
            visual_features = np.random.rand(128)  # Placeholder

            cursor.execute('''
                INSERT OR REPLACE INTO image_features
                (doc_id, features, objects_detected, ocr_text, visual_features)
                VALUES (?, ?, ?, ?, ?)
            ''', (doc_id, json.dumps(features), "[]", ocr_text, visual_features.tobytes()))

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Error storing image features: {e}")

    def _store_table_structure(self, doc_id: str, table_data):
        """Store table structure information"""

        try:
            if not isinstance(table_data, pd.DataFrame):
                return

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            columns = json.dumps(list(table_data.columns))
            rows_count = len(table_data)
            columns_count = len(table_data.columns)
            headers = json.dumps(list(table_data.columns))
            structure_metadata = json.dumps({
                "dtypes": table_data.dtypes.to_dict(),
                "shape": table_data.shape,
                "memory_usage": table_data.memory_usage(deep=True).sum()
            })

            cursor.execute('''
                INSERT OR REPLACE INTO table_structures
                (doc_id, columns, rows_count, columns_count, headers, structure_metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (doc_id, columns, rows_count, columns_count, headers, structure_metadata))

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Error storing table structure: {e}")

    def _update_vector_stores(self,
                             doc_id: str,
                             content: str,
                             text_embedding: np.ndarray,
                             image_embedding: Optional[np.ndarray]):
        """Update in-memory vector stores"""

        self.text_vectors[doc_id] = text_embedding
        if image_embedding is not None:
            self.image_vectors[doc_id] = image_embedding

    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get retrieval system statistics"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Document counts by type
            cursor.execute('''
                SELECT content_type, COUNT(*)
                FROM documents
                GROUP BY content_type
            ''')
            content_type_stats = dict(cursor.fetchall())

            # Total documents
            cursor.execute('SELECT COUNT(*) FROM documents')
            total_docs = cursor.fetchone()[0]

            # Average retrieval score
            cursor.execute('SELECT AVG(relevance_score) FROM documents WHERE relevance_score > 0')
            avg_score = cursor.fetchone()[0] or 0.0

            conn.close()

            return {
                "total_documents": total_docs,
                "content_type_distribution": content_type_stats,
                "average_relevance_score": avg_score,
                "vector_stores": {
                    "text_vectors": len(self.text_vectors),
                    "image_vectors": len(self.image_vectors)
                },
                "index_status": {
                    "tfidf_fitted": hasattr(self.tfidf_vectorizer, 'vocabulary_'),
                    "bm25_index": self.bm25_index is not None
                }
            }

        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}

# Factory function for easy instantiation
def create_multi_modal_retriever(config: Optional[RetrievalConfig] = None) -> ManufacturingMultiModalRetriever:
    """Create a multi-modal retriever instance"""
    return ManufacturingMultiModalRetriever(config=config)

# Example usage and testing
if __name__ == "__main__":
    # Test multi-modal retriever
    retriever = create_multi_modal_retriever()

    # Index some test documents
    test_doc_1 = """
    # 产品规格说明书

    ## 不锈钢螺栓规格

    **产品型号**: M8x20
    **材料**: 304不锈钢
    **强度等级**: 8.8级
    **表面处理**: 镀铬

    ### 尺寸参数
    - 直径: 8mm
    - 长度: 20mm
    - 螺距: 1.25mm
    - 头部厚度: 5.2mm

    ### 技术要求
    硬度: HV200-240
    抗拉强度: ≥800MPa
    屈服强度: ≥640MPa
    """

    # Index text document
    retriever.index_document(
        doc_id="spec_001",
        content=test_doc_1,
        content_type=ContentType.TEXT,
        source_file="specifications.pdf",
        section_title="不锈钢螺栓规格",
        tags=["不锈钢", "螺栓", "规格", "M8", "304"]
    )

    # Test retrieval
    query = "不锈钢螺栓M8x20的规格"
    results = retriever.retrieve(
        query=query,
        top_k=5,
        strategy=RetrievalStrategy.HYBRID
    )

    print(f"Retrieved {len(results)} documents for query: '{query}'")
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Doc ID: {result.doc_id}")
        print(f"Content Type: {result.content_type}")
        print(f"Relevance Score: {result.relevance_score:.3f}")
        print(f"Semantic Score: {result.semantic_score:.3f}")
        print(f"Keyword Score: {result.keyword_score:.3f}")
        print(f"Source: {result.source_file}")
        print(f"Content: {result.content[:100]}...")

    # Get statistics
    stats = retriever.get_retrieval_statistics()
    print(f"\nRetrieval Statistics:")
    print(f"- Total Documents: {stats['total_documents']}")
    print(f"- Average Relevance Score: {stats['average_relevance_score']:.3f}")
    print(f"- Content Types: {stats['content_type_distribution']}")
    print(f"- Vector Stores: {stats['vector_stores']}")

    print(f"\nMulti-modal retrieval test completed!")