#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Embedding System with LangChain Integration
增强型嵌入系统与LangChain集成

This module provides an enhanced embedding system that integrates with the advanced RAG system,
supporting LangChain retrievers, hierarchical document chunking, and multi-modal retrieval.
"""

import sqlite3
import json
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Union
import os
import hashlib
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum

# LangChain imports
try:
    from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma, FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
    from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever, ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    from langchain.schema import Document as LangChainDocument
    from langchain.embeddings import CacheBackedEmbeddings
    from langchain.storage import LocalFileStore
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available, using fallback implementations")

# Import advanced RAG components
try:
    from .core.document_chunker import DocumentChunker, ChunkingStrategy, ContentType
    from .core.multi_modal_retriever import MultiModalRetriever, RetrievalStrategy, RetrievedDocument
    from .core.citation_tracker import CitationTracker, CitationType
    from .core.database_integration import DatabaseIntegration, VectorDatabaseType
except ImportError:
    # Fallback imports if run as standalone
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from core.document_chunker import DocumentChunker, ChunkingStrategy, ContentType
        from core.multi_modal_retriever import MultiModalRetriever, RetrievalStrategy, RetrievedDocument
        from core.citation_tracker import CitationTracker, CitationType
        from core.database_integration import DatabaseIntegration, VectorDatabaseType
    except ImportError:
        logging.error("Advanced RAG components not available")
        DocumentChunker = None
        MultiModalRetriever = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/processed/enhanced_embeddings.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RetrievalMode(Enum):
    """检索模式枚举"""
    SIMPLE = "simple"
    VECTOR = "vector"
    MULTI_MODAL = "multi_modal"
    HIERARCHICAL = "hierarchical"
    ENSEMBLE = "ensemble"
    COMPRESSION = "compression"

@dataclass
class EnhancedSearchResult:
    """增强搜索结果"""
    entry_id: int
    content: str
    metadata: Dict[str, Any]
    score: float
    retrieval_mode: str
    chunk_info: Optional[Dict[str, Any]] = None
    citations: Optional[List[Dict[str, Any]]] = None
    context_relevance: float = 0.0

class EnhancedEmbeddingSystem:
    """增强型嵌入系统"""

    def __init__(self,
                 db_path: str = "knowledge_base.db",
                 vector_db_type: VectorDatabaseType = VectorDatabaseType.CHROMA,
                 cache_dir: str = "./cache",
                 enable_langchain: bool = True):
        """
        初始化增强嵌入系统

        Args:
            db_path: SQLite数据库路径
            vector_db_type: 向量数据库类型
            cache_dir: 缓存目录
            enable_langchain: 是否启用LangChain功能
        """
        self.db_path = db_path
        self.vector_db_type = vector_db_type
        self.cache_dir = cache_dir
        self.enable_langchain = enable_langchain and LANGCHAIN_AVAILABLE
        self.conn = None

        # 初始化组件
        self.embedding_model = None
        self.vector_store = None
        self.retriever = None
        self.document_chunker = None
        self.multi_modal_retriever = None
        self.citation_tracker = None
        self.database_integration = None

        # 配置参数
        self.embedding_dimension = 1536
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.top_k = 10

        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)

    async def initialize(self):
        """异步初始化所有组件"""
        try:
            logger.info("🚀 Initializing Enhanced Embedding System...")

            # 连接数据库
            self._connect_database()

            # 初始化RAG组件
            if DocumentChunker and MultiModalRetriever:
                await self._initialize_rag_components()

            # 初始化LangChain组件
            if self.enable_langchain:
                await self._initialize_langchain_components()

            # 初始化向量存储
            await self._initialize_vector_store()

            logger.info("✅ Enhanced Embedding System initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize enhanced embedding system: {e}")
            raise

    def _connect_database(self):
        """连接数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.execute("PRAGMA foreign_keys = ON")
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    async def _initialize_rag_components(self):
        """初始化RAG组件"""
        try:
            # 初始化文档分块器
            self.document_chunker = DocumentChunker()
            await self.document_chunker.initialize()
            logger.info("✅ Document chunker initialized")

            # 初始化多模态检索器
            self.multi_modal_retriever = MultiModalRetriever(
                db_path=self.db_path,
                vector_db_type=self.vector_db_type
            )
            await self.multi_modal_retriever.initialize()
            logger.info("✅ Multi-modal retriever initialized")

            # 初始化引用跟踪器
            self.citation_tracker = CitationTracker(self.db_path)
            await self.citation_tracker.initialize()
            logger.info("✅ Citation tracker initialized")

            # 初始化数据库集成
            self.database_integration = DatabaseIntegration(self.db_path)
            await self.database_integration.initialize()
            logger.info("✅ Database integration initialized")

        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}")
            raise

    async def _initialize_langchain_components(self):
        """初始化LangChain组件"""
        try:
            # 初始化嵌入模型
            await self._initialize_embeddings()

            # 初始化文档分割器
            self._initialize_text_splitters()

            logger.info("✅ LangChain components initialized")

        except Exception as e:
            logger.error(f"Failed to initialize LangChain components: {e}")
            raise

    async def _initialize_embeddings(self):
        """初始化嵌入模型"""
        try:
            # 尝试OpenAI嵌入
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.embedding_model = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    openai_api_key=api_key
                )
                self.embedding_dimension = 1536
                logger.info("✅ Using OpenAI embeddings: text-embedding-3-small")
                return

            # 尝试HuggingFace嵌入
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                cache_folder=self.cache_dir
            )
            self.embedding_dimension = 384
            logger.info("✅ Using HuggingFace embeddings: all-MiniLM-L6-v2")

        except Exception as e:
            logger.warning(f"Failed to initialize advanced embeddings: {e}")
            # 使用简化版嵌入
            self.embedding_model = None
            logger.warning("⚠️ Using fallback embedding system")

    def _initialize_text_splitters(self):
        """初始化文本分割器"""
        try:
            # 递归字符分割器（用于分层文档）
            self.recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )

            # 字符分割器（用于简单文档）
            self.character_splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

            logger.info("✅ Text splitters initialized")

        except Exception as e:
            logger.error(f"Failed to initialize text splitters: {e}")

    async def _initialize_vector_store(self):
        """初始化向量存储"""
        try:
            if not self.embedding_model:
                logger.warning("No embedding model available, skipping vector store initialization")
                return

            # 创建本地文件存储
            store = LocalFileStore(self.cache_dir)

            # 创建缓存支持的嵌入
            cached_embeddings = CacheBackedEmbeddings(
                underlying_embeddings=self.embedding_model,
                document_embedding_cache=store
            )

            # 根据类型初始化向量存储
            vector_store_path = os.path.join(self.cache_dir, "vector_store")
            os.makedirs(vector_store_path, exist_ok=True)

            if self.vector_db_type == VectorDatabaseType.CHROMA:
                self.vector_store = Chroma(
                    embedding_function=cached_embeddings,
                    persist_directory=vector_store_path
                )
                logger.info("✅ Chroma vector store initialized")

            elif self.vector_db_type == VectorDatabaseType.FAISS:
                index_path = os.path.join(vector_store_path, "faiss.index")
                if os.path.exists(index_path):
                    self.vector_store = FAISS.load_local(
                        index_path,
                        cached_embeddings,
                        "faiss.index"
                    )
                else:
                    # 创建新的FAISS索引
                    self.vector_store = FAISS.from_texts(
                        [""],  # 空文档用于初始化
                        cached_embeddings
                    )
                logger.info("✅ FAISS vector store initialized")

            # 初始化检索器
            self._initialize_retrievers()

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def _initialize_retrievers(self):
        """初始化检索器"""
        try:
            if not self.vector_store:
                logger.warning("No vector store available, skipping retriever initialization")
                return

            # 基础向量检索器
            base_retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k}
            )

            # 如果有分层文档支持，创建父文档检索器
            if self.recursive_splitter and self.character_splitter:
                try:
                    self.parent_retriever = ParentDocumentRetriever(
                        vectorstore=self.vector_store,
                        child_splitter=self.character_splitter,
                        parent_splitter=self.recursive_splitter
                    )
                    logger.info("✅ Parent document retriever initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize parent retriever: {e}")

            # 基础检索器
            self.retriever = base_retriever
            logger.info("✅ Base retriever initialized")

        except Exception as e:
            logger.error(f"Failed to initialize retrievers: {e}")

    async def process_and_index_documents(self,
                                        chunking_strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL,
                                        force_reprocess: bool = False) -> Dict[str, Any]:
        """
        处理并索引文档

        Args:
            chunking_strategy: 文档分块策略
            force_reprocess: 是否强制重新处理

        Returns:
            处理结果统计
        """
        try:
            logger.info(f"📚 Processing documents with strategy: {chunking_strategy.value}")

            if not self.document_chunker:
                logger.error("Document chunker not initialized")
                return {"success": False, "error": "Document chunker not initialized"}

            # 获取所有知识条目
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, entity_type, name, description, attributes_json, created_at
                FROM knowledge_entries
                ORDER BY created_at DESC
            """)

            entries = []
            for row in cursor.fetchall():
                entry = {
                    'id': row[0],
                    'entity_type': row[1],
                    'name': row[2],
                    'description': row[3],
                    'attributes_json': row[4],
                    'created_at': row[5]
                }
                entries.append(entry)

            logger.info(f"Found {len(entries)} entries to process")

            # 处理每个文档
            processed_count = 0
            chunk_count = 0

            for entry in entries:
                # 检查是否需要重新处理
                if not force_reprocess:
                    cursor.execute(
                        "SELECT 1 FROM document_chunks WHERE source_id = ? LIMIT 1",
                        (entry['id'],)
                    )
                    if cursor.fetchone():
                        continue

                # 生成文档内容
                content = self._generate_document_content(entry)
                if not content.strip():
                    continue

                # 创建元数据
                metadata = {
                    'source_id': entry['id'],
                    'entity_type': entry['entity_type'],
                    'title': entry['name'],
                    'created_at': entry['created_at']
                }

                # 处理文档分块
                chunks = await self.document_chunker.chunk_document(
                    doc_id=str(entry['id']),
                    content=content,
                    content_type=ContentType.TEXT,
                    metadata=metadata,
                    strategy=chunking_strategy
                )

                # 使用LangChain处理分块（如果可用）
                if self.enable_langchain and chunks:
                    await self._process_chunks_with_langchain(chunks, entry)

                processed_count += 1
                chunk_count += len(chunks)

                if processed_count % 10 == 0:
                    logger.info(f"Processed {processed_count} entries, {chunk_count} chunks")

            # 保存向量存储
            if self.vector_store and self.vector_db_type == VectorDatabaseType.CHROMA:
                self.vector_store.persist()
            elif self.vector_store and self.vector_db_type == VectorDatabaseType.FAISS:
                vector_store_path = os.path.join(self.cache_dir, "vector_store")
                self.vector_store.save_local(vector_store_path, "faiss.index")

            logger.info(f"✅ Document processing completed")
            logger.info(f"📊 Processed: {processed_count} entries")
            logger.info(f"🧩 Generated: {chunk_count} chunks")

            return {
                "success": True,
                "processed_entries": processed_count,
                "generated_chunks": chunk_count,
                "strategy_used": chunking_strategy.value
            }

        except Exception as e:
            logger.error(f"Failed to process documents: {e}")
            return {"success": False, "error": str(e)}

    async def _process_chunks_with_langchain(self, chunks: List[Any], entry: Dict):
        """使用LangChain处理文档分块"""
        try:
            if not self.embedding_model:
                return

            # 转换为LangChain文档格式
            langchain_docs = []
            for chunk in chunks:
                metadata = {
                    'source_id': chunk.source_id,
                    'chunk_id': chunk.chunk_id,
                    'entity_type': entry['entity_type'],
                    'title': entry['name'],
                    'chunk_index': chunk.chunk_index,
                    'content_type': chunk.content_type.value,
                    **chunk.metadata
                }

                doc = LangChainDocument(
                    page_content=chunk.content,
                    metadata=metadata
                )
                langchain_docs.append(doc)

            # 批量添加到向量存储
            if langchain_docs:
                if self.vector_db_type == VectorDatabaseType.FAISS:
                    self.vector_store.add_documents(langchain_docs)
                else:
                    # Chroma支持批量添加
                    self.vector_store.add_documents(langchain_docs)

        except Exception as e:
            logger.error(f"Failed to process chunks with LangChain: {e}")

    def _generate_document_content(self, entry: Dict) -> str:
        """生成文档内容"""
        content_parts = []

        if entry.get('name'):
            content_parts.append(f"标题: {entry['name']}")

        if entry.get('description'):
            content_parts.append(f"描述: {entry['description']}")

        if entry.get('attributes_json'):
            try:
                attributes = json.loads(entry['attributes_json'])
                for key, value in attributes.items():
                    if value:
                        content_parts.append(f"{key}: {value}")
            except json.JSONDecodeError:
                pass

        if entry.get('entity_type'):
            content_parts.append(f"类型: {entry['entity_type']}")

        return "\n".join(content_parts)

    async def search(self,
                    query: str,
                    mode: RetrievalMode = RetrievalMode.MULTI_MODAL,
                    top_k: int = None,
                    content_types: Optional[List[str]] = None,
                    filters: Optional[Dict[str, Any]] = None) -> List[EnhancedSearchResult]:
        """
        增强搜索功能

        Args:
            query: 搜索查询
            mode: 检索模式
            top_k: 返回结果数量
            content_types: 内容类型过滤
            filters: 过滤条件

        Returns:
            增强搜索结果列表
        """
        try:
            top_k = top_k or self.top_k

            if mode == RetrievalMode.MULTI_MODAL and self.multi_modal_retriever:
                return await self._multi_modal_search(query, top_k, content_types, filters)

            elif mode == RetrievalMode.HIERARCHICAL and hasattr(self, 'parent_retriever'):
                return await self._hierarchical_search(query, top_k, filters)

            elif mode == RetrievalMode.ENSEMBLE and self.enable_langchain:
                return await self._ensemble_search(query, top_k, filters)

            elif mode == RetrievalMode.COMPRESSION and self.enable_langchain:
                return await self._compression_search(query, top_k, filters)

            else:
                # 默认向量搜索
                return await self._vector_search(query, top_k, filters)

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def _multi_modal_search(self, query: str, top_k: int, content_types: Optional[List[str]], filters: Optional[Dict[str, Any]]) -> List[EnhancedSearchResult]:
        """多模态搜索"""
        try:
            if not self.multi_modal_retriever:
                return []

            # 转换内容类型
            content_type_enums = []
            if content_types:
                for ct in content_types:
                    try:
                        content_type_enums.append(ContentType(ct))
                    except ValueError:
                        continue

            # 执行检索
            results = await self.multi_modal_retriever.retrieve(
                query=query,
                content_types=content_type_enums or None,
                top_k=top_k,
                strategy=RetrievalStrategy.MULTI_MODAL,
                filters=filters or {}
            )

            # 转换结果格式
            enhanced_results = []
            for result in results:
                enhanced_result = EnhancedSearchResult(
                    entry_id=int(result.document.chunk_id.split('_')[0]) if '_' in result.document.chunk_id else 0,
                    content=result.document.content,
                    metadata=result.document.metadata,
                    score=result.relevance_score,
                    retrieval_mode="multi_modal",
                    chunk_info={
                        'chunk_id': result.document.chunk_id,
                        'content_type': result.document.content_type.value,
                        'section': result.document.metadata.get('section', ''),
                        'keywords': result.document.metadata.get('keywords', [])
                    }
                )
                enhanced_results.append(enhanced_result)

            return enhanced_results

        except Exception as e:
            logger.error(f"Multi-modal search failed: {e}")
            return []

    async def _vector_search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]]) -> List[EnhancedSearchResult]:
        """向量搜索"""
        try:
            if not self.retriever:
                return []

            # 检索文档
            docs = self.retriever.get_relevant_documents(query)

            # 转换结果
            enhanced_results = []
            for i, doc in enumerate(docs[:top_k]):
                enhanced_result = EnhancedSearchResult(
                    entry_id=doc.metadata.get('source_id', 0),
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=1.0 - (i * 0.1),  # 简单的递减分数
                    retrieval_mode="vector"
                )
                enhanced_results.append(enhanced_result)

            return enhanced_results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def _hierarchical_search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]]) -> List[EnhancedSearchResult]:
        """分层搜索"""
        try:
            if not hasattr(self, 'parent_retriever'):
                return await self._vector_search(query, top_k, filters)

            docs = self.parent_retriever.get_relevant_documents(query)

            enhanced_results = []
            for i, doc in enumerate(docs[:top_k]):
                enhanced_result = EnhancedSearchResult(
                    entry_id=doc.metadata.get('source_id', 0),
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=1.0 - (i * 0.1),
                    retrieval_mode="hierarchical"
                )
                enhanced_results.append(enhanced_result)

            return enhanced_results

        except Exception as e:
            logger.error(f"Hierarchical search failed: {e}")
            return []

    async def _ensemble_search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]]) -> List[EnhancedSearchResult]:
        """集成搜索"""
        try:
            # 结合多种检索策略
            vector_results = await self._vector_search(query, top_k, filters)

            # 如果有多模态检索器，也使用它
            if self.multi_modal_retriever:
                multi_modal_results = await self._multi_modal_search(query, top_k, None, filters)
                # 合并和重新排序结果
                all_results = vector_results + multi_modal_results
                # 简单的去重和重排序
                unique_results = {}
                for result in all_results:
                    key = f"{result.entry_id}_{result.content[:100]}"
                    if key not in unique_results or result.score > unique_results[key].score:
                        unique_results[key] = result

                results = list(unique_results.values())
                results.sort(key=lambda x: x.score, reverse=True)
                return results[:top_k]

            return vector_results

        except Exception as e:
            logger.error(f"Ensemble search failed: {e}")
            return []

    async def _compression_search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]]) -> List[EnhancedSearchResult]:
        """压缩搜索（压缩检索结果）"""
        try:
            # 先进行基础检索
            base_results = await self._vector_search(query, top_k * 2, filters)

            # 这里可以添加内容压缩逻辑
            # 简化版本：直接返回基础结果
            return base_results[:top_k]

        except Exception as e:
            logger.error(f"Compression search failed: {e}")
            return []

    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        try:
            stats = {
                'enable_langchain': self.enable_langchain,
                'vector_db_type': self.vector_db_type.value,
                'embedding_dimension': self.embedding_dimension,
                'chunk_size': self.chunk_size,
                'top_k': self.top_k,
                'components_initialized': {
                    'document_chunker': self.document_chunker is not None,
                    'multi_modal_retriever': self.multi_modal_retriever is not None,
                    'citation_tracker': self.citation_tracker is not None,
                    'database_integration': self.database_integration is not None,
                    'embedding_model': self.embedding_model is not None,
                    'vector_store': self.vector_store is not None,
                    'retriever': self.retriever is not None
                }
            }

            # 获取向量存储统计
            if self.vector_store:
                try:
                    if hasattr(self.vector_store, '_collection'):
                        # Chroma
                        stats['vector_store_stats'] = {
                            'type': 'Chroma',
                            'document_count': self.vector_store._collection.count()
                        }
                    elif hasattr(self.vector_store, 'index'):
                        # FAISS
                        stats['vector_store_stats'] = {
                            'type': 'FAISS',
                            'dimension': self.vector_store.index.d if hasattr(self.vector_store.index, 'd') else 'unknown'
                        }
                except Exception as e:
                    logger.warning(f"Failed to get vector store stats: {e}")

            return stats

        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {'error': str(e)}

    def close(self):
        """关闭系统"""
        try:
            if self.conn:
                self.conn.close()
                logger.info("Database connection closed")

            # 保存向量存储
            if self.vector_store:
                if self.vector_db_type == VectorDatabaseType.CHROMA:
                    self.vector_store.persist()
                elif self.vector_db_type == VectorDatabaseType.FAISS:
                    vector_store_path = os.path.join(self.cache_dir, "vector_store")
                    self.vector_store.save_local(vector_store_path, "faiss.index")

                logger.info("Vector store saved")

        except Exception as e:
            logger.error(f"Error closing system: {e}")

async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Embedding System')
    parser.add_argument('--init', action='store_true', help='Initialize system')
    parser.add_argument('--process', action='store_true', help='Process and index documents')
    parser.add_argument('--search', type=str, help='Search query')
    parser.add_argument('--mode', type=str, default='multi_modal',
                       choices=['simple', 'vector', 'multi_modal', 'hierarchical', 'ensemble', 'compression'],
                       help='Search mode')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    parser.add_argument('--stats', action='store_true', help='Show system statistics')
    parser.add_argument('--force', action='store_true', help='Force reprocess')

    args = parser.parse_args()

    # 创建系统实例
    system = EnhancedEmbeddingSystem()

    try:
        if args.init or args.process or args.search or args.stats:
            await system.initialize()

        if args.process:
            result = await system.process_and_index_documents(force_reprocess=args.force)
            print(f"Processing result: {result}")

        elif args.search:
            mode = RetrievalMode(args.mode)
            results = await system.search(args.search, mode=mode, top_k=args.top_k)

            print(f"\n🔍 Search Results for: '{args.search}' (Mode: {args.mode})")
            print("=" * 80)

            if results:
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. [Score: {result.score:.3f}] {result.retrieval_mode.upper()}")
                    print(f"   Entry ID: {result.entry_id}")
                    print(f"   Content: {result.content[:200]}...")
                    if result.chunk_info:
                        print(f"   Chunk: {result.chunk_info}")
            else:
                print("No results found.")

        elif args.stats:
            stats = system.get_system_stats()
            print("\n📊 Enhanced Embedding System Statistics")
            print("=" * 50)
            print(json.dumps(stats, indent=2, ensure_ascii=False))

        else:
            parser.print_help()

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        system.close()

if __name__ == "__main__":
    asyncio.run(main())