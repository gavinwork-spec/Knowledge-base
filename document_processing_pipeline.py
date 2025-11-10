#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document Processing Pipeline for RAG System
RAG系统的文档处理管道

This script processes various document formats and creates embeddings
for the enhanced knowledge base with LangChain RAG capabilities.
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import sqlite3

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import (
    TextLoader,
    PDFMinerLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader
)
from langchain.schema import Document
from langchain.document_transformers import Html2TextTransformer

# LangFuse imports
from langfuse import Langfuse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler('data/processed/document_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Enhanced document processor for RAG system"""

    def __init__(self, embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings_model = embeddings_model
        self.embeddings = None
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        # Initialize LangFuse for observability
        try:
            self.langfuse = Langfuse(
                secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-..."),
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-..."),
                host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            )
            logger.info("✅ LangFuse observability enabled for document processing")
        except Exception as e:
            logger.warning(f"⚠️ LangFuse initialization failed: {e}")
            self.langfuse = None

    def initialize_embeddings(self) -> bool:
        """Initialize embeddings model"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embeddings_model,
                cache_folder="data/processed/embeddings_cache"
            )
            logger.info(f"✅ Initialized embeddings: {self.embeddings_model}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize embeddings: {e}")
            return False

    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """Load documents from a directory with various formats"""
        documents = []
        directory = Path(directory_path)

        if not directory.exists():
            logger.error(f"❌ Directory not found: {directory_path}")
            return documents

        # Create trace for document loading
        if self.langfuse:
            trace = self.langfuse.trace(
                name="document_loading",
                input={"directory": directory_path}
            )

        supported_extensions = {
            '.txt': self._load_text,
            '.md': self._load_markdown,
            '.pdf': self._load_pdf,
            '.csv': self._load_csv,
            '.docx': self._load_word,
            '.doc': self._load_word,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
            '.html': self._load_html,
            '.htm': self._load_html
        }

        total_files = 0
        processed_files = 0

        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                total_files += 1
                try:
                    loader_func = supported_extensions[file_path.suffix.lower()]
                    file_docs = loader_func(str(file_path))
                    if file_docs:
                        documents.extend(file_docs)
                        processed_files += 1
                        logger.info(f"✅ Loaded: {file_path.name}")
                    else:
                        logger.warning(f"⚠️ No content loaded from: {file_path.name}")
                except Exception as e:
                    logger.error(f"❌ Failed to load {file_path.name}: {e}")

        if self.langfuse:
            trace.update(
                output={
                    "total_files": total_files,
                    "processed_files": processed_files,
                    "total_documents": len(documents)
                }
            )

        logger.info(f"📄 Loaded {len(documents)} documents from {processed_files}/{total_files} files")
        return documents

    def _load_text(self, file_path: str) -> List[Document]:
        """Load text documents"""
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            return loader.load()
        except Exception as e:
            logger.error(f"❌ Failed to load text file {file_path}: {e}")
            return []

    def _load_markdown(self, file_path: str) -> List[Document]:
        """Load markdown documents"""
        try:
            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()
            # Add metadata for markdown
            for doc in docs:
                doc.metadata['document_type'] = 'markdown'
                doc.metadata['file_type'] = 'md'
            return docs
        except Exception as e:
            logger.error(f"❌ Failed to load markdown file {file_path}: {e}")
            return []

    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF documents"""
        try:
            loader = PDFMinerLoader(file_path)
            docs = loader.load()
            # Add metadata for PDF
            for doc in docs:
                doc.metadata['document_type'] = 'pdf'
                doc.metadata['file_type'] = 'pdf'
            return docs
        except Exception as e:
            logger.error(f"❌ Failed to load PDF file {file_path}: {e}")
            return []

    def _load_csv(self, file_path: str) -> List[Document]:
        """Load CSV documents"""
        try:
            loader = CSVLoader(file_path, encoding='utf-8')
            docs = loader.load()
            # Add metadata for CSV
            for doc in docs:
                doc.metadata['document_type'] = 'csv'
                doc.metadata['file_type'] = 'csv'
            return docs
        except Exception as e:
            logger.error(f"❌ Failed to load CSV file {file_path}: {e}")
            return []

    def _load_word(self, file_path: str) -> List[Document]:
        """Load Word documents"""
        try:
            loader = UnstructuredWordDocumentLoader(file_path)
            docs = loader.load()
            # Add metadata for Word
            for doc in docs:
                doc.metadata['document_type'] = 'word'
                doc.metadata['file_type'] = 'docx'
            return docs
        except Exception as e:
            logger.error(f"❌ Failed to load Word file {file_path}: {e}")
            return []

    def _load_excel(self, file_path: str) -> List[Document]:
        """Load Excel documents"""
        try:
            loader = UnstructuredExcelLoader(file_path)
            docs = loader.load()
            # Add metadata for Excel
            for doc in docs:
                doc.metadata['document_type'] = 'excel'
                doc.metadata['file_type'] = 'xlsx'
            return docs
        except Exception as e:
            logger.error(f"❌ Failed to load Excel file {file_path}: {e}")
            return []

    def _load_html(self, file_path: str) -> List[Document]:
        """Load HTML documents"""
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()

            # Transform HTML to text
            html_transformer = Html2TextTransformer()
            transformed_docs = html_transformer.transform_documents(docs)

            # Add metadata for HTML
            for doc in transformed_docs:
                doc.metadata['document_type'] = 'html'
                doc.metadata['file_type'] = 'html'
            return transformed_docs
        except Exception as e:
            logger.error(f"❌ Failed to load HTML file {file_path}: {e}")
            return []

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process and split documents for RAG"""
        if not documents:
            logger.warning("⚠️ No documents to process")
            return []

        # Create trace for document processing
        if self.langfuse:
            trace = self.langfuse.trace(
                name="document_processing",
                input={"documents_count": len(documents)}
            )

        processed_docs = []

        for doc in documents:
            # Add enhanced metadata
            doc.metadata.update({
                'processed_at': datetime.now().isoformat(),
                'chunk_id': f"doc_{len(processed_docs)}",
                'source_type': doc.metadata.get('document_type', 'unknown')
            })

            # Choose appropriate splitter based on document type
            if doc.metadata.get('document_type') == 'markdown':
                chunks = self.markdown_splitter.split_documents([doc])
            else:
                chunks = self.text_splitter.split_documents([doc])

            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'parent_document': doc.metadata.get('source', 'unknown')
                })

            processed_docs.extend(chunks)

        if self.langfuse:
            trace.update(
                output={
                    "processed_documents": len(processed_docs),
                    "avg_chunk_size": sum(len(doc.page_content) for doc in processed_docs) // len(processed_docs)
                }
            )

        logger.info(f"📝 Processed {len(processed_docs)} document chunks from {len(documents)} original documents")
        return processed_docs

    def create_vector_store(self, documents: List[Document], save_path: str = "data/processed/vector_store") -> bool:
        """Create FAISS vector store from documents"""
        if not documents:
            logger.error("❌ No documents to create vector store")
            return False

        try:
            # Initialize embeddings if not already done
            if not self.embeddings:
                self.initialize_embeddings()

            # Create trace for vector store creation
            if self.langfuse:
                trace = self.langfuse.trace(
                    name="vector_store_creation",
                    input={"documents_count": len(documents), "model": self.embeddings_model}
                )

            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(documents, self.embeddings)

            # Save vector store
            os.makedirs(save_path, exist_ok=True)
            self.vector_store.save_local(save_path)

            if self.langfuse:
                trace.update(
                    output={
                        "vector_store_path": save_path,
                        "embedding_dimension": self.embeddings.client.get_sentence_embedding_dimension(),
                        "documents_indexed": len(documents)
                    }
                )

            logger.info(f"✅ Created vector store with {len(documents)} documents at {save_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to create vector store: {e}")
            if self.langfuse:
                trace.update(output={"error": str(e)}, level="ERROR")
            return False

    def load_existing_vector_store(self, load_path: str = "data/processed/vector_store") -> bool:
        """Load existing vector store"""
        try:
            if not self.embeddings:
                self.initialize_embeddings()

            if os.path.exists(load_path):
                self.vector_store = FAISS.load_local(
                    load_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"✅ Loaded existing vector store from {load_path}")
                return True
            else:
                logger.warning(f"⚠️ No existing vector store found at {load_path}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to load vector store: {e}")
            return False

    def add_documents_to_vector_store(self, documents: List[Document]) -> bool:
        """Add documents to existing vector store"""
        try:
            if not self.vector_store:
                logger.error("❌ No existing vector store. Create one first.")
                return False

            self.vector_store.add_documents(documents)

            # Save updated vector store
            self.vector_store.save_local("data/processed/vector_store")

            logger.info(f"✅ Added {len(documents)} documents to vector store")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to add documents to vector store: {e}")
            return False

    def process_knowledge_base_database(self, db_path: str = "knowledge_base.db") -> bool:
        """Process documents from knowledge base database"""
        try:
            # Create trace for database processing
            if self.langfuse:
                trace = self.langfuse.trace(
                    name="database_processing",
                    input={"database_path": db_path}
                )

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get all knowledge entries
            cursor.execute("SELECT id, title, content, metadata FROM knowledge_entries")
            rows = cursor.fetchall()

            documents = []
            for row in rows:
                entry_id, title, content, metadata_json = row

                # Parse metadata
                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                except:
                    metadata = {}

                # Create document
                doc = Document(
                    page_content=content,
                    metadata={
                        'id': entry_id,
                        'title': title,
                        'source': 'knowledge_base',
                        'document_type': 'knowledge_entry',
                        'processed_at': datetime.now().isoformat(),
                        **metadata
                    }
                )
                documents.append(doc)

            conn.close()

            if self.langfuse:
                trace.update(
                    output={"database_entries": len(documents)}
                )

            logger.info(f"📚 Loaded {len(documents)} entries from knowledge base database")

            # Process the documents
            processed_docs = self.process_documents(documents)

            # Create vector store
            return self.create_vector_store(processed_docs)

        except Exception as e:
            logger.error(f"❌ Failed to process knowledge base database: {e}")
            if self.langfuse:
                trace.update(output={"error": str(e)}, level="ERROR")
            return False

    def search_similar_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        try:
            if not self.vector_store:
                logger.error("❌ Vector store not initialized")
                return []

            docs = self.vector_store.similarity_search(query, k=k)

            results = []
            for i, doc in enumerate(docs):
                results.append({
                    'rank': i + 1,
                    'content': doc.page_content,
                    'metadata': doc.metadata
                })

            return results
        except Exception as e:
            logger.error(f"❌ Document search failed: {e}")
            return []

def main():
    """Main processing function"""
    logger.info("🚀 Starting Document Processing Pipeline for RAG System")

    # Initialize processor
    processor = DocumentProcessor()

    # Create directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/documents", exist_ok=True)

    # Process knowledge base database
    logger.info("📚 Processing knowledge base database...")
    success = processor.process_knowledge_base_database()

    if success:
        logger.info("✅ Knowledge base processing completed successfully")
    else:
        logger.error("❌ Knowledge base processing failed")

    # Process additional documents if they exist
    documents_dir = "data/documents"
    if os.path.exists(documents_dir):
        logger.info(f"📄 Processing documents from {documents_dir}...")
        documents = processor.load_documents_from_directory(documents_dir)

        if documents:
            processed_docs = processor.process_documents(documents)

            # Add to existing vector store or create new one
            if processor.load_existing_vector_store():
                processor.add_documents_to_vector_store(processed_docs)
            else:
                processor.create_vector_store(processed_docs)

    # Test search functionality
    if processor.vector_store:
        test_queries = [
            "safety procedures",
            "quality control",
            "manufacturing process",
            "equipment maintenance"
        ]

        logger.info("🔍 Testing document search...")
        for query in test_queries:
            results = processor.search_similar_documents(query, k=3)
            logger.info(f"Query: '{query}' - Found {len(results)} results")

    logger.info("🎉 Document processing pipeline completed")

if __name__ == "__main__":
    main()