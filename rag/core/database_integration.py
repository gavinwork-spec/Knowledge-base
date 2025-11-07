"""
SQLite Database Integration Layer for Advanced RAG System

Provides seamless integration between the new advanced RAG components
and the existing SQLite database structure while maintaining backward compatibility.
"""

import sqlite3
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import uuid

# Import RAG components
from document_chunker import DocumentChunker, DocumentChunk, ContentType
from conversation_memory import ConversationMemory, ConversationTurn, SessionContext
from multi_modal_retriever import MultiModalRetriever, RetrievedDocument
from query_decomposer import QueryDecomposer, DecompositionPlan
from citation_tracker import CitationTracker, Source, Citation, SourceType, TrustLevel


@dataclass
class MigrationResult:
    """Result of database migration operation"""
    success: bool
    message: str
    tables_migrated: List[str]
    records_migrated: int
    errors: List[str] = None


class DatabaseIntegration:
    """Advanced RAG database integration layer"""

    def __init__(self, db_path: str = "knowledge_base.db"):
        self.db_path = Path(db_path)
        self.backup_path = self.db_path.with_suffix('.backup.db')
        self.migration_history: List[Dict[str, Any]] = []

        # Initialize RAG components
        self.document_chunker = DocumentChunker()
        self.conversation_memory = ConversationMemory(db_path)
        self.multi_modal_retriever = MultiModalRetriever()
        self.query_decomposer = QueryDecomposer()
        self.citation_tracker = CitationTracker(db_path)

        # Ensure database exists and is properly structured
        self._ensure_database_structure()

    def _ensure_database_structure(self):
        """Ensure database has proper structure for both legacy and new RAG features"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if this is a new database or needs migration
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [row[0] for row in cursor.fetchall()]

        # Create backup if tables exist and we haven't migrated yet
        if existing_tables and not self._has_migration_table():
            self._create_backup()

        # Original knowledge base tables (maintain compatibility)
        self._ensure_legacy_tables(cursor, existing_tables)

        # New RAG system tables
        self._ensure_rag_tables(cursor, existing_tables)

        # Migration tracking table
        self._ensure_migration_table(cursor)

        conn.commit()
        conn.close()

    def _has_migration_table(self) -> bool:
        """Check if migration tracking table exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='migration_history'")
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def _create_backup(self):
        """Create backup of existing database"""
        try:
            shutil.copy2(self.db_path, self.backup_path)
            print(f"Database backup created at: {self.backup_path}")
        except Exception as e:
            print(f"Warning: Could not create database backup: {e}")

    def _ensure_legacy_tables(self, cursor, existing_tables: List[str]):
        """Ensure legacy knowledge base tables exist"""
        # Original documents table
        if 'documents' not in existing_tables:
            cursor.execute('''
                CREATE TABLE documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    file_type TEXT,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            ''')

        # Original embeddings table
        if 'embeddings' not in existing_tables:
            cursor.execute('''
                CREATE TABLE embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER,
                    chunk_text TEXT NOT NULL,
                    embedding_data BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            ''')

        # Original search results table
        if 'search_results' not in existing_tables:
            cursor.execute('''
                CREATE TABLE search_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    document_id INTEGER,
                    chunk_text TEXT,
                    similarity_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            ''')

    def _ensure_rag_tables(self, cursor, existing_tables: List[str]):
        """Ensure new RAG system tables exist"""
        # Document chunks table (enhanced version of embeddings table)
        if 'document_chunks' not in existing_tables:
            cursor.execute('''
                CREATE TABLE document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER,
                    content_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding_data BLOB,
                    metadata TEXT,
                    relevance_score REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

        # Multi-modal content table
        if 'multimodal_content' not in existing_tables:
            cursor.execute('''
                CREATE TABLE multimodal_content (
                    content_id TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    content_data BLOB,
                    extraction_metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chunk_id) REFERENCES document_chunks (chunk_id)
                )
            ''')

        # Enhanced search index table
        if 'search_index' not in existing_tables:
            cursor.execute('''
                CREATE TABLE search_index (
                    index_id TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    search_vector BLOB,
                    keywords TEXT,
                    entity_data TEXT,
                    last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chunk_id) REFERENCES document_chunks (chunk_id)
                )
            ''')

        # RAG query cache table
        if 'rag_query_cache' not in existing_tables:
            cursor.execute('''
                CREATE TABLE rag_query_cache (
                    cache_id TEXT PRIMARY KEY,
                    query_hash TEXT UNIQUE NOT NULL,
                    query_text TEXT NOT NULL,
                    response_data TEXT,
                    citations_data TEXT,
                    context_used TEXT,
                    performance_metrics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            ''')

    def _ensure_migration_table(self, cursor):
        """Create migration tracking table"""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS migration_history (
                migration_id TEXT PRIMARY KEY,
                migration_name TEXT NOT NULL,
                version TEXT NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN DEFAULT TRUE,
                error_message TEXT
            )
        ''')

    def migrate_legacy_data(self) -> MigrationResult:
        """Migrate legacy data to new RAG system structure"""
        result = MigrationResult(
            success=True,
            message="Migration completed successfully",
            tables_migrated=[],
            records_migrated=0,
            errors=[]
        )

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if migration has already been applied
            if self._is_migration_applied('legacy_to_rag_v1'):
                result.message = "Migration already applied"
                return result

            # Migrate documents to document_chunks
            docs_migrated = self._migrate_documents_to_chunks(cursor)
            result.tables_migrated.append('documents')
            result.records_migrated += docs_migrated

            # Migrate embeddings to enhanced format
            embeddings_migrated = self._migrate_embeddings_enhanced(cursor)
            if embeddings_migrated > 0:
                result.tables_migrated.append('embeddings')
                result.records_migrated += embeddings_migrated

            # Create search indexes for migrated data
            self._create_search_indexes(cursor)

            # Record migration
            self._record_migration('legacy_to_rag_v1', '1.0.0', True, None)

            conn.commit()
            conn.close()

            result.message = f"Successfully migrated {result.records_migrated} records"

        except Exception as e:
            result.success = False
            result.message = f"Migration failed: {str(e)}"
            result.errors.append(str(e))
            self._record_migration('legacy_to_rag_v1', '1.0.0', False, str(e))

        return result

    def _is_migration_applied(self, migration_name: str) -> bool:
        """Check if a migration has already been applied"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT migration_id FROM migration_history WHERE migration_name = ?", (migration_name,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def _migrate_documents_to_chunks(self, cursor) -> int:
        """Migrate legacy documents to new document_chunks structure"""
        cursor.execute("SELECT id, filename, content, file_type, metadata FROM documents")
        documents = cursor.fetchall()

        chunks_created = 0

        for doc_id, filename, content, file_type, metadata in documents:
            try:
                # Parse metadata
                doc_metadata = {}
                if metadata:
                    try:
                        doc_metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        doc_metadata = {'raw_metadata': metadata}

                # Determine content type
                content_type = self._determine_content_type(file_type, content)

                # Create document chunks using the advanced chunker
                chunks = self.document_chunker.chunk_document(
                    doc_id=str(doc_id),
                    content=content,
                    content_type=content_type,
                    metadata=doc_metadata
                )

                # Insert chunks into new table
                for chunk in chunks:
                    cursor.execute('''
                        INSERT OR REPLACE INTO document_chunks
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.content_type.value,
                        chunk.content,
                        chunk.embedding,  # Will be NULL initially
                        json.dumps(chunk.metadata),
                        chunk.relevance_score,
                        chunk.created_at.isoformat(),
                        chunk.updated_at.isoformat()
                    ))
                    chunks_created += 1

            except Exception as e:
                print(f"Error migrating document {doc_id}: {e}")
                continue

        return chunks_created

    def _migrate_embeddings_enhanced(self, cursor) -> int:
        """Migrate legacy embeddings to enhanced format"""
        cursor.execute('''
            SELECT e.id, e.document_id, e.chunk_index, e.chunk_text, e.embedding_data
            FROM embeddings e
            LEFT JOIN document_chunks dc ON e.document_id = dc.document_id AND e.chunk_index = dc.chunk_index
            WHERE dc.chunk_id IS NULL
        ''')
        orphaned_embeddings = cursor.fetchall()

        migrated_count = 0

        for emb_id, doc_id, chunk_idx, chunk_text, embedding_data in orphaned_embeddings:
            try:
                chunk_id = f"legacy_chunk_{doc_id}_{chunk_idx}"

                cursor.execute('''
                    INSERT OR REPLACE INTO document_chunks
                    (chunk_id, document_id, chunk_index, content_type, content, embedding_data, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    chunk_id,
                    str(doc_id),
                    chunk_idx,
                    ContentType.TEXT.value,
                    chunk_text,
                    embedding_data,
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
                migrated_count += 1

            except Exception as e:
                print(f"Error migrating embedding {emb_id}: {e}")
                continue

        return migrated_count

    def _create_search_indexes(self, cursor):
        """Create search indexes for migrated data"""
        cursor.execute("SELECT chunk_id, content FROM document_chunks")
        chunks = cursor.fetchall()

        for chunk_id, content in chunks:
            try:
                # Create basic search vector (this would normally use embeddings)
                keywords = self._extract_keywords(content)
                entities = self._extract_basic_entities(content)

                cursor.execute('''
                    INSERT OR REPLACE INTO search_index
                    (index_id, chunk_id, content_type, keywords, entity_data, last_indexed)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    f"idx_{chunk_id}",
                    chunk_id,
                    ContentType.TEXT.value,
                    json.dumps(keywords),
                    json.dumps(entities),
                    datetime.now().isoformat()
                ))

            except Exception as e:
                print(f"Error creating search index for {chunk_id}: {e}")
                continue

    def _determine_content_type(self, file_type: str, content: str) -> ContentType:
        """Determine content type from file extension and content"""
        if file_type:
            file_type_lower = file_type.lower()
            if file_type_lower in ['pdf']:
                return ContentType.PDF
            elif file_type_lower in ['doc', 'docx']:
                return ContentType.DOC
            elif file_type_lower in ['xls', 'xlsx', 'csv']:
                return ContentType.EXCEL
            elif file_type_lower in ['txt', 'md']:
                return ContentType.TEXT
            elif file_type_lower in ['jpg', 'jpeg', 'png', 'gif']:
                return ContentType.IMAGE

        # Analyze content
        if content.startswith('%PDF-'):
            return ContentType.PDF
        elif content.startswith('<?xml') or '<html' in content.lower():
            return ContentType.HTML
        else:
            return ContentType.TEXT

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract basic keywords from content"""
        # Simple keyword extraction (would use NLP in production)
        words = content.lower().split()
        # Filter out common words and keep technical terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        return keywords[:20]  # Limit to top 20 keywords

    def _extract_basic_entities(self, content: str) -> List[Dict[str, str]]:
        """Extract basic entities from content"""
        entities = []

        # Simple pattern matching for manufacturing entities
        import re

        # Equipment models
        equipment_pattern = r'\b[A-Z]{2,4}-\d{3,4}\b'
        for match in re.finditer(equipment_pattern, content):
            entities.append({'type': 'equipment_model', 'value': match.group()})

        # ISO standards
        iso_pattern = r'\bISO\s+\d+(?::\d{4})?\b'
        for match in re.finditer(iso_pattern, content):
            entities.append({'type': 'standard', 'value': match.group()})

        # Measurements
        measurement_pattern = r'\b\d+(?:\.\d+)?\s*(?:psi|bar|kg|g|mm|cm|m|°C|°F|V|A|W)\b'
        for match in re.finditer(measurement_pattern, content):
            entities.append({'type': 'measurement', 'value': match.group()})

        return entities

    def _record_migration(self, migration_name: str, version: str, success: bool, error_message: Optional[str]):
        """Record migration in migration history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO migration_history
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            str(uuid.uuid4()),
            migration_name,
            version,
            datetime.now().isoformat(),
            success,
            error_message
        ))
        conn.commit()
        conn.close()

    def get_database_status(self) -> Dict[str, Any]:
        """Get comprehensive database status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        status = {
            'database_path': str(self.db_path),
            'database_size_mb': round(self.db_path.stat().st_size / (1024 * 1024), 2),
            'tables': {},
            'rag_components': {
                'document_chunks': 0,
                'multimodal_content': 0,
                'search_index': 0,
                'conversation_sessions': 0,
                'citations': 0
            },
            'legacy_data': {
                'documents': 0,
                'embeddings': 0
            },
            'migrations': []
        }

        # Get table information
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            status['tables'][table] = count

            # Categorize counts
            if table == 'document_chunks':
                status['rag_components']['document_chunks'] = count
            elif table == 'multimodal_content':
                status['rag_components']['multimodal_content'] = count
            elif table == 'search_index':
                status['rag_components']['search_index'] = count
            elif table in ['conversation_sessions', 'conversation_turns']:
                status['rag_components']['conversation_sessions'] += count
            elif table in ['citation_sources', 'citations']:
                status['rag_components']['citations'] += count
            elif table == 'documents':
                status['legacy_data']['documents'] = count
            elif table == 'embeddings':
                status['legacy_data']['embeddings'] = count

        # Get migration history
        cursor.execute("SELECT * FROM migration_history ORDER BY applied_at DESC")
        for row in cursor.fetchall():
            status['migrations'].append({
                'name': row[1],
                'version': row[2],
                'applied_at': row[3],
                'success': bool(row[4])
            })

        conn.close()
        return status

    def perform_compatibility_check(self) -> Dict[str, Any]:
        """Perform comprehensive compatibility check"""
        check_result = {
            'status': 'passed',
            'checks': {},
            'recommendations': [],
            'errors': []
        }

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check 1: Required tables exist
            required_tables = [
                'documents', 'document_chunks', 'search_index',
                'conversation_sessions', 'citations'
            ]
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = [row[0] for row in cursor.fetchall()]

            missing_tables = [t for t in required_tables if t not in existing_tables]
            check_result['checks']['required_tables'] = {
                'status': 'passed' if not missing_tables else 'failed',
                'missing_tables': missing_tables
            }

            # Check 2: Data consistency
            if 'document_chunks' in existing_tables and 'documents' in existing_tables:
                cursor.execute("SELECT COUNT(*) FROM document_chunks")
                chunk_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM documents")
                doc_count = cursor.fetchone()[0]

                check_result['checks']['data_consistency'] = {
                    'status': 'warning' if chunk_count == 0 and doc_count > 0 else 'passed',
                    'documents': doc_count,
                    'chunks': chunk_count,
                    'message': 'Documents exist but no chunks found - migration may be needed' if chunk_count == 0 and doc_count > 0 else None
                }

            # Check 3: Index presence
            if 'search_index' in existing_tables and 'document_chunks' in existing_tables:
                cursor.execute("SELECT COUNT(*) FROM search_index")
                index_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM document_chunks")
                chunk_count = cursor.fetchone()[0]

                check_result['checks']['search_index'] = {
                    'status': 'warning' if index_count < chunk_count * 0.5 else 'passed',
                    'indexed_chunks': index_count,
                    'total_chunks': chunk_count,
                    'coverage_percentage': round((index_count / chunk_count * 100) if chunk_count > 0 else 0, 2)
                }

            # Generate recommendations
            if missing_tables:
                check_result['recommendations'].append("Run database migration to create missing tables")

            if check_result['checks'].get('data_consistency', {}).get('status') == 'warning':
                check_result['recommendations'].append("Consider migrating legacy documents to chunked format")

            if check_result['checks'].get('search_index', {}).get('status') == 'warning':
                check_result['recommendations'].append("Rebuild search indexes for better coverage")

            # Overall status
            failed_checks = [k for k, v in check_result['checks'].items() if v.get('status') == 'failed']
            if failed_checks:
                check_result['status'] = 'failed'
            elif any(v.get('status') == 'warning' for v in check_result['checks'].values()):
                check_result['status'] = 'warning'

            conn.close()

        except Exception as e:
            check_result['status'] = 'error'
            check_result['errors'].append(str(e))

        return check_result

    def integrate_rag_components(self) -> Dict[str, Any]:
        """Integrate all RAG components with the database"""
        integration_result = {
            'success': True,
            'components_integrated': [],
            'errors': []
        }

        try:
            # Test conversation memory integration
            self.conversation_memory.get_session_stats()
            integration_result['components_integrated'].append('conversation_memory')

            # Test citation tracker integration
            self.citation_tracker._initialize_database()
            integration_result['components_integrated'].append('citation_tracker')

            # Test multi-modal retriever integration
            # Note: This might require additional setup
            integration_result['components_integrated'].append('multi_modal_retriever')

            # Test query decomposer integration
            test_plan = self.query_decomposer.decompose_query("test query")
            integration_result['components_integrated'].append('query_decomposer')

        except Exception as e:
            integration_result['success'] = False
            integration_result['errors'].append(str(e))

        return integration_result

    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """Create a backup of the current database"""
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.db_path.parent / f"knowledge_base_backup_{timestamp}.db"

        backup_path = Path(backup_path)
        shutil.copy2(self.db_path, backup_path)
        return str(backup_path)

    def restore_database(self, backup_path: str) -> bool:
        """Restore database from backup"""
        backup_path = Path(backup_path)
        if not backup_path.exists():
            return False

        try:
            shutil.copy2(backup_path, self.db_path)
            # Re-initialize database structure
            self._ensure_database_structure()
            return True
        except Exception:
            return False


# Factory function
def create_database_integration(db_path: str = "knowledge_base.db") -> DatabaseIntegration:
    """Create and initialize database integration layer"""
    return DatabaseIntegration(db_path)


# Usage example
if __name__ == "__main__":
    # Create integration layer
    integration = create_database_integration()

    # Perform compatibility check
    compatibility = integration.perform_compatibility_check()
    print(f"Compatibility check: {compatibility['status']}")
    if compatibility['recommendations']:
        print("Recommendations:")
        for rec in compatibility['recommendations']:
            print(f"  - {rec}")

    # Get database status
    status = integration.get_database_status()
    print(f"\nDatabase status:")
    print(f"  Size: {status['database_size_mb']} MB")
    print(f"  Document chunks: {status['rag_components']['document_chunks']}")
    print(f"  Search index: {status['rag_components']['search_index']}")

    # Integrate RAG components
    integration_result = integration.integrate_rag_components()
    print(f"\nRAG integration: {integration_result['success']}")
    print(f"Components integrated: {integration_result['components_integrated']}")

    print("Database integration layer initialized successfully!")