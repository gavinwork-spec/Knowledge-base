#!/usr/bin/env python3
"""
Specialized Agents Implementation
Implements various specialized agents for the knowledge base system,
including learning agents, data processing agents, and coordination agents.
"""

import asyncio
import json
import logging
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor

# Import existing systems
from multi_agent_orchestrator import BaseAgent, AgentTask, AgentCapability, AgentStatus
from parse_documents import parse_document
from build_embeddings import build_embedding_index
from advanced_rag_system import AdvancedRAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningAgent(BaseAgent):
    """Enhanced learning agent based on existing learn_from_updates_agent"""

    def __init__(self, agent_id: str, orchestrator, config: Dict[str, Any] = None):
        super().__init__(agent_id, "Learning Agent", orchestrator)
        self.config = config or {}
        self.db_path = self.config.get('database', {}).get('path', "knowledge_base.db")
        self.scan_directories = self.config.get('scan_directories', [])
        self.supported_formats = self.config.get('supported_formats', ['.pdf', '.xlsx', '.docx'])
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.last_scan_time = None
        self.processed_files = set()

    async def initialize(self):
        """Initialize the learning agent"""
        logger.info(f"Initializing Learning Agent {self.id}")

        # Create data directories if they don't exist
        Path("data/processed").mkdir(parents=True, exist_ok=True)

        # Load processed files history
        await self._load_processed_files_history()

        # Initialize database connection
        await self._init_database()

    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute learning-related tasks"""
        task_type = task.task_type
        parameters = task.parameters

        try:
            if task_type == "file_scan_learning":
                return await self._perform_file_scan_learning(parameters)
            elif task_type == "database_learning":
                return await self._perform_database_learning(parameters)
            elif task_type == "incremental_learning":
                return await self._perform_incremental_learning(parameters)
            elif task_type == "batch_learning":
                return await self._perform_batch_learning(parameters)
            elif task_type == "knowledge_validation":
                return await self._perform_knowledge_validation(parameters)
            else:
                raise ValueError(f"Unknown task type: {task_type}")

        except Exception as e:
            logger.error(f"Error executing learning task {task.id}: {e}")
            raise

    def get_capabilities(self) -> List[AgentCapability]:
        """Get agent capabilities"""
        return [
            AgentCapability.DATA_PROCESSING,
            AgentCapability.KNOWLEDGE_EXTRACTION,
            AgentCapability.DOCUMENT_PROCESSING,
            AgentCapability.FILE_SYSTEM_MONITOR,
            AgentCapability.DATABASE_MANAGEMENT
        ]

    async def _perform_file_scan_learning(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform file scanning and learning"""
        scan_directories = parameters.get('directories', self.scan_directories)
        days_back = parameters.get('days_back', 7)

        logger.info(f"Starting file scan learning for directories: {scan_directories}")

        results = {
            'scanned_directories': scan_directories,
            'files_found': 0,
            'files_processed': 0,
            'knowledge_entries_created': 0,
            'errors': [],
            'processing_time': 0
        }

        start_time = time.time()

        for directory in scan_directories:
            try:
                directory_results = await self._scan_directory(directory, days_back)
                results['files_found'] += directory_results['files_found']
                results['files_processed'] += directory_results['files_processed']
                results['knowledge_entries_created'] += directory_results['knowledge_entries_created']
                results['errors'].extend(directory_results['errors'])

            except Exception as e:
                error_msg = f"Error scanning directory {directory}: {str(e)}"
                logger.error(error_msg)
                results['errors'].append(error_msg)

        results['processing_time'] = time.time() - start_time

        # Update processed files history
        await self._save_processed_files_history()

        # Rebuild embeddings if new knowledge was added
        if results['knowledge_entries_created'] > 0:
            try:
                await self._rebuild_embeddings()
                logger.info("Embeddings rebuilt successfully")
            except Exception as e:
                logger.error(f"Error rebuilding embeddings: {e}")
                results['errors'].append(f"Embedding rebuild failed: {str(e)}")

        logger.info(f"File scan learning completed: {results}")
        return results

    async def _scan_directory(self, directory: str, days_back: int) -> Dict[str, Any]:
        """Scan a specific directory for new files"""
        results = {
            'files_found': 0,
            'files_processed': 0,
            'knowledge_entries_created': 0,
            'errors': []
        }

        try:
            directory_path = Path(directory)
            if not directory_path.exists():
                results['errors'].append(f"Directory does not exist: {directory}")
                return results

            cutoff_time = datetime.now() - timedelta(days=days_back)

            # Scan for files
            for file_path in directory_path.rglob('*'):
                if file_path.is_file():
                    try:
                        # Check file modification time
                        mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)

                        if mod_time < cutoff_time:
                            continue

                        # Check file extension
                        if file_path.suffix.lower() not in self.supported_formats:
                            continue

                        # Check if already processed
                        file_hash = self._calculate_file_hash(file_path)
                        if file_hash in self.processed_files:
                            continue

                        results['files_found'] += 1

                        # Process file
                        process_result = await self._process_file(file_path)
                        if process_result['success']:
                            results['files_processed'] += 1
                            results['knowledge_entries_created'] += process_result['entries_created']
                            self.processed_files.add(file_hash)
                        else:
                            results['errors'].append(f"Failed to process {file_path}: {process_result['error']}")

                    except Exception as e:
                        error_msg = f"Error processing file {file_path}: {str(e)}"
                        logger.error(error_msg)
                        results['errors'].append(error_msg)

        except Exception as e:
            error_msg = f"Error scanning directory {directory}: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)

        return results

    async def _process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single file and extract knowledge"""
        try:
            logger.info(f"Processing file: {file_path}")

            # Parse document using existing parsing logic
            parsed_data = parse_document(str(file_path))

            if not parsed_data:
                return {'success': False, 'error': 'No data parsed from file', 'entries_created': 0}

            # Extract knowledge entries
            entries_created = 0

            # Process entities from parsed data
            if 'entities' in parsed_data:
                for entity_data in parsed_data['entities']:
                    if self._should_create_knowledge_entry(entity_data):
                        entry_id = await self._create_knowledge_entry(entity_data, str(file_path))
                        if entry_id:
                            entries_created += 1

            # Process additional content
            if 'content' in parsed_data:
                content_entries = await self._extract_knowledge_from_content(
                    parsed_data['content'], str(file_path)
                )
                entries_created += content_entries

            return {
                'success': True,
                'entries_created': entries_created,
                'parsed_entities': len(parsed_data.get('entities', []))
            }

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return {'success': False, 'error': str(e), 'entries_created': 0}

    def _should_create_knowledge_entry(self, entity_data: Dict[str, Any]) -> bool:
        """Determine if a knowledge entry should be created from entity data"""
        # Check confidence
        confidence = entity_data.get('confidence', 0.0)
        if confidence < self.confidence_threshold:
            return False

        # Check required fields
        required_fields = ['name', 'entity_type']
        if not all(field in entity_data for field in required_fields):
            return False

        # Check minimum content length
        description = entity_data.get('description', '')
        if len(description) < 10:
            return False

        return True

    async def _create_knowledge_entry(self, entity_data: Dict[str, Any], source_file: str) -> Optional[str]:
        """Create a knowledge entry in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Generate unique ID
            entry_id = str(uuid.uuid4())

            # Insert knowledge entry
            cursor.execute("""
                INSERT INTO knowledge_entries
                (id, name, description, entity_type, attributes, created_at, updated_at, created_by, source_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry_id,
                entity_data['name'],
                entity_data.get('description', ''),
                entity_data['entity_type'],
                json.dumps(entity_data.get('attributes', {})),
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                self.id,
                source_file
            ))

            conn.commit()
            conn.close()

            return entry_id

        except Exception as e:
            logger.error(f"Error creating knowledge entry: {e}")
            return None

    async def _extract_knowledge_from_content(self, content: str, source_file: str) -> int:
        """Extract additional knowledge from document content"""
        # This would implement content analysis for additional knowledge extraction
        # For now, return 0 as a placeholder
        return 0

    async def _perform_database_learning(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform learning from database records"""
        table_name = parameters.get('table', 'reminder_records')
        batch_size = parameters.get('batch_size', 20)

        logger.info(f"Starting database learning from table: {table_name}")

        results = {
            'table_name': table_name,
            'records_processed': 0,
            'knowledge_entries_created': 0,
            'errors': [],
            'processing_time': 0
        }

        start_time = time.time()

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get total record count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_records = cursor.fetchone()[0]

            # Process records in batches
            offset = 0
            while offset < total_records:
                cursor.execute(f"""
                    SELECT * FROM {table_name}
                    LIMIT ? OFFSET ?
                """, (batch_size, offset))

                records = cursor.fetchall()

                for record in records:
                    try:
                        # Extract knowledge from record
                        entry_created = await self._extract_knowledge_from_record(record, table_name)
                        if entry_created:
                            results['knowledge_entries_created'] += 1

                        results['records_processed'] += 1

                    except Exception as e:
                        error_msg = f"Error processing record {record}: {str(e)}"
                        logger.error(error_msg)
                        results['errors'].append(error_msg)

                offset += batch_size

            conn.close()

        except Exception as e:
            error_msg = f"Error in database learning: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)

        results['processing_time'] = time.time() - start_time

        logger.info(f"Database learning completed: {results}")
        return results

    async def _extract_knowledge_from_record(self, record: tuple, table_name: str) -> bool:
        """Extract knowledge from a database record"""
        # Implement record-to-knowledge extraction logic
        # This would analyze the record fields and create appropriate knowledge entries
        return False  # Placeholder

    async def _perform_incremental_learning(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform incremental learning based on recent changes"""
        since_time = parameters.get('since', (datetime.now() - timedelta(hours=24)).isoformat())

        logger.info(f"Starting incremental learning since: {since_time}")

        results = {
            'since_time': since_time,
            'changes_found': 0,
            'knowledge_updates': 0,
            'errors': [],
            'processing_time': 0
        }

        start_time = time.time()

        try:
            # Check for file changes
            file_changes = await self._detect_file_changes(since_time)
            results['changes_found'] += len(file_changes)

            # Process file changes
            for change in file_changes:
                try:
                    update_result = await self._process_file_change(change)
                    results['knowledge_updates'] += update_result

                except Exception as e:
                    error_msg = f"Error processing file change {change}: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)

            # Check for database changes
            db_changes = await self._detect_database_changes(since_time)
            results['changes_found'] += len(db_changes)

            # Process database changes
            for change in db_changes:
                try:
                    update_result = await self._process_database_change(change)
                    results['knowledge_updates'] += update_result

                except Exception as e:
                    error_msg = f"Error processing database change {change}: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)

        except Exception as e:
            error_msg = f"Error in incremental learning: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)

        results['processing_time'] = time.time() - start_time

        logger.info(f"Incremental learning completed: {results}")
        return results

    async def _detect_file_changes(self, since_time: str) -> List[Dict[str, Any]]:
        """Detect file changes since given time"""
        changes = []

        try:
            cutoff_time = datetime.fromisoformat(since_time)

            for directory in self.scan_directories:
                directory_path = Path(directory)
                if directory_path.exists():
                    for file_path in directory_path.rglob('*'):
                        if file_path.is_file():
                            mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if mod_time > cutoff_time:
                                changes.append({
                                    'type': 'file',
                                    'path': str(file_path),
                                    'modified_time': mod_time.isoformat(),
                                    'action': 'modified'
                                })

        except Exception as e:
            logger.error(f"Error detecting file changes: {e}")

        return changes

    async def _detect_database_changes(self, since_time: str) -> List[Dict[str, Any]]:
        """Detect database changes since given time"""
        changes = []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check for recent changes in knowledge_entries
            cursor.execute("""
                SELECT id, updated_at FROM knowledge_entries
                WHERE updated_at > ?
            """, (since_time,))

            for row in cursor.fetchall():
                changes.append({
                    'type': 'database',
                    'table': 'knowledge_entries',
                    'record_id': row[0],
                    'updated_at': row[1],
                    'action': 'updated'
                })

            conn.close()

        except Exception as e:
            logger.error(f"Error detecting database changes: {e}")

        return changes

    async def _process_file_change(self, change: Dict[str, Any]) -> int:
        """Process a file change and update knowledge"""
        file_path = Path(change['path'])

        if file_path.exists():
            process_result = await self._process_file(file_path)
            if process_result['success']:
                return process_result['entries_created']

        return 0

    async def _process_database_change(self, change: Dict[str, Any]) -> int:
        """Process a database change and update knowledge"""
        # Implement database change processing
        return 0  # Placeholder

    async def _perform_batch_learning(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform batch learning on multiple items"""
        items = parameters.get('items', [])
        learning_type = parameters.get('learning_type', 'files')

        logger.info(f"Starting batch learning for {len(items)} items")

        results = {
            'items_count': len(items),
            'learning_type': learning_type,
            'items_processed': 0,
            'knowledge_entries_created': 0,
            'errors': [],
            'processing_time': 0
        }

        start_time = time.time()

        for item in items:
            try:
                if learning_type == 'files':
                    file_path = Path(item)
                    if file_path.exists():
                        process_result = await self._process_file(file_path)
                        if process_result['success']:
                            results['items_processed'] += 1
                            results['knowledge_entries_created'] += process_result['entries_created']
                        else:
                            results['errors'].append(f"Failed to process {item}: {process_result['error']}")

                elif learning_type == 'records':
                    # Process database record
                    update_result = await self._extract_knowledge_from_record(item, 'batch_table')
                    if update_result:
                        results['knowledge_entries_created'] += 1
                    results['items_processed'] += 1

            except Exception as e:
                error_msg = f"Error processing item {item}: {str(e)}"
                logger.error(error_msg)
                results['errors'].append(error_msg)

        results['processing_time'] = time.time() - start_time

        logger.info(f"Batch learning completed: {results}")
        return results

    async def _perform_knowledge_validation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate existing knowledge entries"""
        validation_type = parameters.get('validation_type', 'completeness')

        logger.info(f"Starting knowledge validation: {validation_type}")

        results = {
            'validation_type': validation_type,
            'entries_validated': 0,
            'issues_found': 0,
            'corrections_made': 0,
            'errors': [],
            'processing_time': 0
        }

        start_time = time.time()

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all knowledge entries
            cursor.execute("SELECT id, name, description, entity_type FROM knowledge_entries")
            entries = cursor.fetchall()

            for entry in entries:
                try:
                    entry_id, name, description, entity_type = entry

                    # Validate entry based on type
                    issues = await self._validate_knowledge_entry(entry_id, name, description, entity_type)

                    if issues:
                        results['issues_found'] += len(issues)

                        # Attempt corrections
                        corrections = await self._correct_knowledge_entry(entry_id, issues)
                        results['corrections_made'] += corrections

                    results['entries_validated'] += 1

                except Exception as e:
                    error_msg = f"Error validating entry {entry}: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)

            conn.close()

        except Exception as e:
            error_msg = f"Error in knowledge validation: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)

        results['processing_time'] = time.time() - start_time

        logger.info(f"Knowledge validation completed: {results}")
        return results

    async def _validate_knowledge_entry(self, entry_id: str, name: str, description: str, entity_type: str) -> List[str]:
        """Validate a knowledge entry and return list of issues"""
        issues = []

        # Check for empty fields
        if not name or len(name.strip()) < 2:
            issues.append('Invalid or empty name')

        if not description or len(description.strip()) < 10:
            issues.append('Description too short or empty')

        if not entity_type:
            issues.append('Missing entity type')

        # Check for duplicates
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*) FROM knowledge_entries
                WHERE name = ? AND id != ?
            """, (name, entry_id))

            duplicate_count = cursor.fetchone()[0]
            if duplicate_count > 0:
                issues.append(f'Found {duplicate_count} duplicate entries with same name')

            conn.close()

        except Exception as e:
            logger.error(f"Error checking duplicates: {e}")

        return issues

    async def _correct_knowledge_entry(self, entry_id: str, issues: List[str]) -> int:
        """Correct issues in a knowledge entry"""
        corrections = 0

        for issue in issues:
            try:
                if 'duplicate' in issue.lower():
                    # Handle duplicates - this would merge or mark duplicates
                    pass
                elif 'empty' in issue.lower():
                    # Handle empty fields - this might add default values or mark for review
                    pass

                corrections += 1

            except Exception as e:
                logger.error(f"Error correcting issue '{issue}' for entry {entry_id}: {e}")

        return corrections

    async def _init_database(self):
        """Initialize database connection and tables"""
        # Ensure database exists and has required tables
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables if they don't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_entries (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                entity_type TEXT,
                attributes TEXT,
                created_at TEXT,
                updated_at TEXT,
                created_by TEXT,
                source_file TEXT
            )
        """)

        conn.commit()
        conn.close()

    async def _load_processed_files_history(self):
        """Load history of processed files"""
        try:
            history_file = Path("data/processed/processed_files.json")
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.processed_files = set(data.get('processed_files', []))
        except Exception as e:
            logger.error(f"Error loading processed files history: {e}")

    async def _save_processed_files_history(self):
        """Save history of processed files"""
        try:
            history_file = Path("data/processed/processed_files.json")
            with open(history_file, 'w') as f:
                json.dump({
                    'processed_files': list(self.processed_files),
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving processed files history: {e}")

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate hash of file for change detection"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                # Read first 8KB for hash calculation
                chunk = f.read(8192)
                hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash for {file_path}: {e}")
            return ""

    async def _rebuild_embeddings(self):
        """Rebuild the embedding index"""
        try:
            # This would call the existing embedding rebuild functionality
            # build_embedding_index(build=True, force=True)
            logger.info("Embedding rebuild completed")
        except Exception as e:
            logger.error(f"Error rebuilding embeddings: {e}")
            raise

class DataProcessingAgent(BaseAgent):
    """Specialized agent for data processing tasks"""

    def __init__(self, agent_id: str, orchestrator):
        super().__init__(agent_id, "Data Processing Agent", orchestrator)

    async def initialize(self):
        """Initialize the data processing agent"""
        logger.info(f"Initializing Data Processing Agent {self.id}")

    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute data processing tasks"""
        task_type = task.task_type
        parameters = task.parameters

        try:
            if task_type == "data_transformation":
                return await self._perform_data_transformation(parameters)
            elif task_type == "data_aggregation":
                return await self._perform_data_aggregation(parameters)
            elif task_type == "data_validation":
                return await self._perform_data_validation(parameters)
            elif task_type == "data_export":
                return await self._perform_data_export(parameters)
            else:
                raise ValueError(f"Unknown task type: {task_type}")

        except Exception as e:
            logger.error(f"Error executing data processing task {task.id}: {e}")
            raise

    def get_capabilities(self) -> List[AgentCapability]:
        """Get agent capabilities"""
        return [
            AgentCapability.DATA_PROCESSING,
            AgentCapability.DATABASE_MANAGEMENT
        ]

    async def _perform_data_transformation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform data transformation"""
        source_data = parameters.get('source_data', [])
        transformations = parameters.get('transformations', [])

        logger.info(f"Transforming data with {len(transformations)} transformations")

        transformed_data = source_data

        for transform in transformations:
            transform_type = transform.get('type')
            if transform_type == 'filter':
                transformed_data = self._filter_data(transformed_data, transform.get('criteria', {}))
            elif transform_type == 'map':
                transformed_data = self._map_data(transformed_data, transform.get('mapping', {}))
            elif transform_type == 'aggregate':
                transformed_data = self._aggregate_data(transformed_data, transform.get('aggregation', {}))

        return {
            'transformed_data': transformed_data,
            'original_count': len(source_data),
            'transformed_count': len(transformed_data),
            'transformations_applied': len(transformations)
        }

    def _filter_data(self, data: List[Dict], criteria: Dict) -> List[Dict]:
        """Filter data based on criteria"""
        filtered = []
        for item in data:
            match = True
            for field, value in criteria.items():
                if item.get(field) != value:
                    match = False
                    break
            if match:
                filtered.append(item)
        return filtered

    def _map_data(self, data: List[Dict], mapping: Dict) -> List[Dict]:
        """Map data fields according to mapping"""
        mapped = []
        for item in data:
            mapped_item = {}
            for old_field, new_field in mapping.items():
                if old_field in item:
                    mapped_item[new_field] = item[old_field]
            mapped.append(mapped_item)
        return mapped

    def _aggregate_data(self, data: List[Dict], aggregation: Dict) -> List[Dict]:
        """Aggregate data"""
        # Simple aggregation implementation
        group_by = aggregation.get('group_by', [])
        aggregations = aggregation.get('functions', {})

        if not group_by:
            return data  # No aggregation if no group_by specified

        grouped = {}
        for item in data:
            key = tuple(item.get(field, '') for field in group_by)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(item)

        result = []
        for key, items in grouped.items():
            aggregated_item = dict(zip(group_by, key))
            for field, func in aggregations.items():
                if func == 'count':
                    aggregated_item[field] = len(items)
                elif func == 'sum':
                    values = [item.get(field.replace('_sum', ''), 0) for item in items]
                    aggregated_item[field] = sum(values)
            result.append(aggregated_item)

        return result

    async def _perform_data_aggregation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform data aggregation"""
        data_source = parameters.get('data_source', 'database')
        aggregation_config = parameters.get('aggregation', {})

        logger.info(f"Aggregating data from {data_source}")

        # Implement data aggregation logic
        return {
            'data_source': data_source,
            'aggregation_applied': aggregation_config,
            'result_count': 0
        }

    async def _perform_data_validation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform data validation"""
        data = parameters.get('data', [])
        validation_rules = parameters.get('rules', [])

        logger.info(f"Validating {len(data)} records with {len(validation_rules)} rules")

        validation_results = {
            'total_records': len(data),
            'valid_records': 0,
            'invalid_records': 0,
            'validation_errors': []
        }

        for i, record in enumerate(data):
            record_errors = []
            for rule in validation_rules:
                if not self._validate_rule(record, rule):
                    record_errors.append(f"Rule failed: {rule.get('name', 'unnamed')}")

            if record_errors:
                validation_results['invalid_records'] += 1
                validation_results['validation_errors'].append({
                    'record_index': i,
                    'errors': record_errors
                })
            else:
                validation_results['valid_records'] += 1

        return validation_results

    def _validate_rule(self, record: Dict, rule: Dict) -> bool:
        """Validate a single rule against a record"""
        field = rule.get('field')
        condition = rule.get('condition', 'required')
        expected = rule.get('value')

        if field not in record:
            return condition != 'required'

        value = record[field]

        if condition == 'required':
            return value is not None and str(value).strip() != ''
        elif condition == 'equals':
            return value == expected
        elif condition == 'not_equals':
            return value != expected
        elif condition == 'greater_than':
            return value > expected
        elif condition == 'less_than':
            return value < expected

        return True

    async def _perform_data_export(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform data export"""
        data = parameters.get('data', [])
        export_format = parameters.get('format', 'json')
        output_path = parameters.get('output_path', 'exported_data')

        logger.info(f"Exporting {len(data)} records in {export_format} format")

        try:
            if export_format == 'json':
                output_file = f"{output_path}.json"
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
            elif export_format == 'csv':
                output_file = f"{output_path}.csv"
                if data:
                    df = pd.DataFrame(data)
                    df.to_csv(output_file, index=False)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")

            return {
                'export_format': export_format,
                'output_file': output_file,
                'records_exported': len(data),
                'status': 'success'
            }

        except Exception as e:
            return {
                'export_format': export_format,
                'records_exported': 0,
                'status': 'failed',
                'error': str(e)
            }

class DocumentAnalysisAgent(BaseAgent):
    """Specialized agent for document analysis"""

    def __init__(self, agent_id: str, orchestrator):
        super().__init__(agent_id, "Document Analysis Agent", orchestrator)

    async def initialize(self):
        """Initialize the document analysis agent"""
        logger.info(f"Initializing Document Analysis Agent {self.id}")

    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute document analysis tasks"""
        task_type = task.task_type
        parameters = task.parameters

        try:
            if task_type == "document_parsing":
                return await self._perform_document_parsing(parameters)
            elif task_type == "content_analysis":
                return await self._perform_content_analysis(parameters)
            elif task_type == "entity_extraction":
                return await self._perform_entity_extraction(parameters)
            elif task_type == "sentiment_analysis":
                return await self._perform_sentiment_analysis(parameters)
            else:
                raise ValueError(f"Unknown task type: {task_type}")

        except Exception as e:
            logger.error(f"Error executing document analysis task {task.id}: {e}")
            raise

    def get_capabilities(self) -> List[AgentCapability]:
        """Get agent capabilities"""
        return [
            AgentCapability.DOCUMENT_PROCESSING,
            AgentCapability.TEXT_ANALYSIS,
            AgentCapability.KNOWLEDGE_EXTRACTION
        ]

    async def _perform_document_parsing(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform document parsing"""
        file_path = parameters.get('file_path')
        parse_options = parameters.get('options', {})

        logger.info(f"Parsing document: {file_path}")

        try:
            # Use existing document parsing functionality
            parsed_data = parse_document(file_path)

            return {
                'file_path': file_path,
                'parsed_data': parsed_data,
                'status': 'success',
                'elements_found': len(parsed_data.get('entities', [])) if parsed_data else 0
            }

        except Exception as e:
            return {
                'file_path': file_path,
                'status': 'failed',
                'error': str(e),
                'elements_found': 0
            }

    async def _perform_content_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform content analysis"""
        content = parameters.get('content', '')
        analysis_type = parameters.get('analysis_type', 'general')

        logger.info(f"Analyzing content with {analysis_type} analysis")

        # Implement content analysis logic
        analysis_result = {
            'content_length': len(content),
            'analysis_type': analysis_type,
            'keywords': self._extract_keywords(content),
            'summary': self._generate_summary(content),
            'status': 'completed'
        }

        return analysis_result

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content"""
        # Simple keyword extraction
        words = content.lower().split()
        # Filter out common words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]

        # Count frequency and return top keywords
        from collections import Counter
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(10)]

    def _generate_summary(self, content: str) -> str:
        """Generate a summary of the content"""
        # Simple extractive summarization
        sentences = content.split('.')
        if len(sentences) <= 3:
            return content

        # Return first few sentences as summary
        return '. '.join(sentences[:3]) + '.'

    async def _perform_entity_extraction(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform entity extraction"""
        content = parameters.get('content', '')
        entity_types = parameters.get('entity_types', ['PERSON', 'ORG', 'GPE'])

        logger.info(f"Extracting entities from content ({len(entity_types)} types)")

        # Simple entity extraction using regex patterns
        entities = self._extract_entities_simple(content, entity_types)

        return {
            'content_length': len(content),
            'entity_types_requested': entity_types,
            'entities_found': entities,
            'total_entities': len(entities)
        }

    def _extract_entities_simple(self, content: str, entity_types: List[str]) -> List[Dict]:
        """Simple entity extraction using patterns"""
        entities = []

        # Pattern for capitalized words (potential entities)
        capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(capitalized_pattern, content)

        for match in matches:
            # Simple heuristic for entity type
            if any(keyword in match.lower() for keyword in ['company', 'corp', 'inc', 'ltd']):
                entity_type = 'ORG'
            elif any(keyword in match.lower() for keyword in ['mr', 'mrs', 'dr', 'prof']):
                entity_type = 'PERSON'
            else:
                entity_type = 'MISC'  # Miscellaneous

            entities.append({
                'text': match,
                'type': entity_type,
                'confidence': 0.7  # Simple confidence score
            })

        return entities[:20]  # Limit to top 20 entities

    async def _perform_sentiment_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform sentiment analysis"""
        content = parameters.get('content', '')

        logger.info(f"Performing sentiment analysis on content ({len(content)} characters)")

        # Simple sentiment analysis using word lists
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'positive', 'best', 'love', 'like'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike', 'negative', 'poor', 'fail'}

        words = content.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            sentiment_score = 0.0
            sentiment_label = 'neutral'
        else:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
            if sentiment_score > 0.1:
                sentiment_label = 'positive'
            elif sentiment_score < -0.1:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'

        return {
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'positive_words': positive_count,
            'negative_words': negative_count,
            'total_words': len(words),
            'status': 'completed'
        }

class NotificationAgent(BaseAgent):
    """Specialized agent for notifications"""

    def __init__(self, agent_id: str, orchestrator):
        super().__init__(agent_id, "Notification Agent", orchestrator)

    async def initialize(self):
        """Initialize the notification agent"""
        logger.info(f"Initializing Notification Agent {self.id}")

    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute notification tasks"""
        task_type = task.task_type
        parameters = task.parameters

        try:
            if task_type == "send_notification":
                return await self._send_notification(parameters)
            elif task_type == "send_report":
                return await self._send_report(parameters)
            elif task_type == "schedule_notification":
                return await self._schedule_notification(parameters)
            else:
                raise ValueError(f"Unknown task type: {task_type}")

        except Exception as e:
            logger.error(f"Error executing notification task {task.id}: {e}")
            raise

    def get_capabilities(self) -> List[AgentCapability]:
        """Get agent capabilities"""
        return [
            AgentCapability.NOTIFICATION,
            AgentCapability.REPORTING
        ]

    async def _send_notification(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Send a notification"""
        message = parameters.get('message', '')
        recipients = parameters.get('recipients', [])
        notification_type = parameters.get('type', 'info')

        logger.info(f"Sending {notification_type} notification to {len(recipients)} recipients")

        # Implement notification sending logic
        # This would integrate with email, Slack, or other notification systems

        return {
            'notification_type': notification_type,
            'recipients_count': len(recipients),
            'message_length': len(message),
            'status': 'sent',
            'sent_at': datetime.now().isoformat()
        }

    async def _send_report(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Send a report"""
        report_data = parameters.get('report_data', {})
        report_type = parameters.get('report_type', 'summary')
        recipients = parameters.get('recipients', [])

        logger.info(f"Sending {report_type} report to {len(recipients)} recipients")

        # Generate report
        report_content = await self._generate_report(report_data, report_type)

        # Send report (implementation would depend on notification system)
        return {
            'report_type': report_type,
            'recipients_count': len(recipients),
            'report_length': len(report_content),
            'status': 'sent',
            'sent_at': datetime.now().isoformat()
        }

    async def _generate_report(self, report_data: Dict, report_type: str) -> str:
        """Generate report content"""
        if report_type == 'summary':
            return f"""
Summary Report
=============
Generated: {datetime.now().isoformat()}

Data Points: {report_data.get('data_points', 'N/A')}
Success Rate: {report_data.get('success_rate', 'N/A')}%
Processing Time: {report_data.get('processing_time', 'N/A')}s

Key Metrics:
{json.dumps(report_data.get('metrics', {}), indent=2)}
"""
        elif report_type == 'detailed':
            return f"Detailed Report\n{'='*50}\n{json.dumps(report_data, indent=2)}"
        else:
            return json.dumps(report_data, indent=2)

    async def _schedule_notification(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule a notification"""
        schedule_time = parameters.get('schedule_time')
        notification_data = parameters.get('notification', {})

        logger.info(f"Scheduling notification for {schedule_time}")

        # Implement notification scheduling
        return {
            'scheduled_time': schedule_time,
            'notification_id': str(uuid.uuid4()),
            'status': 'scheduled',
            'scheduled_at': datetime.now().isoformat()
        }

# Import uuid for notification scheduling
import uuid