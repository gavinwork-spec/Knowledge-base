#!/usr/bin/env python3
"""
Knowledge Base CLI Tool - A comprehensive command-line interface for managing knowledge bases
Inspired by AI Shell with natural language processing capabilities
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import subprocess
import shutil
import hashlib
import sqlite3
from dataclasses import dataclass
from enum import Enum

# Import user management
try:
    from user_manager import UserManager, User, Permission, UserRole, create_user, delete_user, list_users, get_user_stats
except ImportError:
    print("Warning: User management features not available")
    UserManager = None
    User = None
    Permission = None
    UserRole = None

try:
    import click
    import rich
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.tree import Tree
    from rich.syntax import Syntax
    from rich.prompt import Prompt, Confirm
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.markdown import Markdown
except ImportError:
    print("Installing required dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "click", "rich"])
    import click
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

# Initialize rich console
console = Console()

class CommandCategory(Enum):
    """Command categories for organization"""
    DOCUMENTS = "documents"
    SYSTEM = "system"
    MONITORING = "monitoring"
    BACKUP = "backup"
    USERS = "users"
    SEARCH = "search"
    ANALYTICS = "analytics"

@dataclass
class CommandResult:
    """Result of command execution"""
    success: bool
    message: str
    data: Optional[Any] = None
    execution_time: float = 0.0
    details: Optional[Dict[str, Any]] = None

class NLCommandParser:
    """Natural Language Command Parser"""

    def __init__(self):
        self.patterns = {
            # Document operations
            r'(?:show|list|display|get)\s+(?:all\s+)?documents?\s+(?:about|regarding|related to)\s+(.+?)\s*(?:added|created|updated|modified)\s+(?:last|past|in the)\s+(.+)$':
                ('search_documents', {'query': 1, 'timeframe': 2}),

            r'(?:show|list|display)\s+(?:all\s+)?documents?\s+added\s+(?:last|past|in the)\s+(.+)$':
                ('list_recent_documents', {'timeframe': 1}),

            r'(?:ingest|import|add|process)\s+(?:bulk\s+)?documents?\s+(?:from|in)\s+(.+)$':
                ('bulk_ingest', {'path': 1}),

            r'(?:delete|remove)\s+(?:document|documents?)\s+(?:with|containing)\s+(.+)$':
                ('delete_documents', {'query': 1}),

            # System operations
            r'(?:check|run|perform)\s+(?:system\s+)?health\s+check':
                ('health_check', {}),

            r'(?:show|display|get)\s+(?:system\s+)?status':
                ('system_status', {}),

            r'(?:monitor|track)\s+(?:performance|system\s+performance)':
                ('performance_monitor', {}),

            # Backup operations
            r'(?:create|make|generate)\s+(?:a\s+)?backup\s+(?:of\s+)?(?:the\s+)?(knowledge\s+base|kb)?':
                ('create_backup', {}),

            r'(?:restore|recover)\s+(?:knowledge\s+base|kb)\s+from\s+(.+)$':
                ('restore_backup', {'backup_path': 1}),

            r'(?:list|show)\s+(?:all\s+)?backups':
                ('list_backups', {}),

            # User operations
            r'(?:create|add)\s+(?:new\s+)?user\s+(.+?)(?:\s+with\s+role\s+(.+))?$':
                ('create_user', {'username': 1, 'role': 2}),

            r'(?:delete|remove)\s+user\s+(.+)$':
                ('delete_user', {'username': 1}),

            r'(?:list|show)\s+(?:all\s+)?users':
                ('list_users', {}),

            # Analytics
            r'(?:show|display|get)\s+(?:analytics|statistics|stats)(?:\s+for)?(.+)?$':
                ('show_analytics', {'scope': 1}),

            r'(?:analyze|analysis)\s+(?:document|documents?)(?:\s+for)?(.+)?$':
                ('analyze_documents', {'scope': 1}),
        }

        self.timeframe_patterns = {
            r'(\d+)\s+(?:day|days)': lambda x: timedelta(days=int(x.group(1))),
            r'(\d+)\s+(?:week|weeks)': lambda x: timedelta(weeks=int(x.group(1))),
            r'(\d+)\s+(?:month|months)': lambda x: timedelta(days=int(x.group(1))*30),
            r'(\d+)\s+(?:year|years)': lambda x: timedelta(days=int(x.group(1))*365),
            r'yesterday': lambda x: timedelta(days=1),
            r'today': lambda x: timedelta(0),
            r'this\s+week': lambda x: timedelta(weeks=1),
            r'last\s+week': lambda x: timedelta(weeks=1),
            r'this\s+month': lambda x: timedelta(days=30),
            r'last\s+month': lambda x: timedelta(days=30),
        }

    def parse(self, command: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Parse natural language command into structured command"""
        command = command.strip().lower()

        for pattern, (cmd_name, groups) in self.patterns.items():
            match = re.match(pattern, command, re.IGNORECASE)
            if match:
                params = {}
                for group_name, group_idx in groups.items():
                    if group_idx < len(match.groups()) and match.group(group_idx + 1):
                        value = match.group(group_idx + 1).strip()

                        # Special handling for timeframe
                        if group_name == 'timeframe':
                            params[group_name] = self._parse_timeframe(value)
                        else:
                            params[group_name] = value

                return cmd_name, params

        return None

    def _parse_timeframe(self, timeframe_str: str) -> datetime:
        """Parse timeframe string into datetime"""
        for pattern, handler in self.timeframe_patterns.items():
            match = re.search(pattern, timeframe_str, re.IGNORECASE)
            if match:
                return datetime.now() - handler(match)

        # Default to last 7 days if parsing fails
        return datetime.now() - timedelta(days=7)

class KnowledgeBaseManager:
    """Main knowledge base management system"""

    def __init__(self, config_path: str = "kb_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.db_path = self.config.get("database_path", "knowledge_base.db")
        self.backup_dir = Path(self.config.get("backup_dir", "backups"))
        self.backup_dir.mkdir(exist_ok=True)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return self._create_default_config()

    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        default_config = {
            "database_path": "knowledge_base.db",
            "backup_dir": "backups",
            "max_backups": 10,
            "auto_backup_interval": 24,  # hours
            "supported_formats": [".txt", ".md", ".pdf", ".docx", ".html", ".json"],
            "indexing_enabled": True,
            "performance_monitoring": True,
            "log_level": "INFO"
        }

        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)

        return default_config

    def _execute_with_timing(self, func, *args, **kwargs) -> CommandResult:
        """Execute function with timing"""
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            if isinstance(result, CommandResult):
                result.execution_time = execution_time
                return result
            else:
                return CommandResult(True, "Operation completed successfully", result, execution_time)
        except Exception as e:
            execution_time = time.time() - start_time
            return CommandResult(False, str(e), None, execution_time)

class DocumentManager(KnowledgeBaseManager):
    """Document management operations"""

    def __init__(self, config_path: str = "kb_config.json"):
        super().__init__(config_path)
        self._init_database()

    def _init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT,
                file_path TEXT,
                file_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tags TEXT,
                category TEXT,
                metadata TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_tags (
                document_id INTEGER,
                tag TEXT,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        ''')

        conn.commit()
        conn.close()

    def search_documents(self, query: str, timeframe: Optional[datetime] = None) -> CommandResult:
        """Search documents by query and optional timeframe"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        base_query = """
            SELECT id, title, content, created_at, updated_at, tags, category
            FROM documents
            WHERE (title LIKE ? OR content LIKE ?)
        """
        params = [f"%{query}%", f"%{query}%"]

        if timeframe:
            base_query += " AND created_at >= ?"
            params.append(timeframe.isoformat())

        base_query += " ORDER BY created_at DESC"

        cursor.execute(base_query, params)
        results = cursor.fetchall()

        documents = []
        for row in results:
            documents.append({
                'id': row[0],
                'title': row[1],
                'content': row[2][:200] + "..." if row[2] and len(row[2]) > 200 else row[2],
                'created_at': row[3],
                'updated_at': row[4],
                'tags': row[5],
                'category': row[6]
            })

        conn.close()

        return CommandResult(True, f"Found {len(documents)} documents", documents)

    def list_recent_documents(self, timeframe: datetime) -> CommandResult:
        """List documents added in the specified timeframe"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, title, created_at, category, tags
            FROM documents
            WHERE created_at >= ?
            ORDER BY created_at DESC
        """, (timeframe.isoformat(),))

        results = cursor.fetchall()

        documents = []
        for row in results:
            documents.append({
                'id': row[0],
                'title': row[1],
                'created_at': row[2],
                'category': row[3],
                'tags': row[4]
            })

        conn.close()

        return CommandResult(True, f"Found {len(documents)} recent documents", documents)

    def bulk_ingest(self, path: str) -> CommandResult:
        """Bulk ingest documents from a directory"""
        import_path = Path(path)
        if not import_path.exists():
            return CommandResult(False, f"Path does not exist: {path}")

        supported_formats = self.config.get("supported_formats", [".txt", ".md"])
        files_to_process = []

        if import_path.is_file():
            if import_path.suffix.lower() in supported_formats:
                files_to_process.append(import_path)
        else:
            for ext in supported_formats:
                files_to_process.extend(import_path.glob(f"**/*{ext}"))

        if not files_to_process:
            return CommandResult(False, "No supported files found")

        ingested_count = 0
        errors = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Ingesting {len(files_to_process)} documents...", total=len(files_to_process))

            for file_path in files_to_process:
                try:
                    # Calculate file hash
                    file_hash = self._calculate_file_hash(file_path)

                    # Check if file already exists
                    if self._file_exists(file_hash):
                        progress.advance(task)
                        continue

                    # Read file content
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    # Insert into database
                    self._insert_document(
                        title=file_path.stem,
                        content=content,
                        file_path=str(file_path),
                        file_hash=file_hash
                    )

                    ingested_count += 1
                    progress.advance(task)

                except Exception as e:
                    errors.append(f"Error processing {file_path}: {str(e)}")
                    progress.advance(task)

        message = f"Successfully ingested {ingested_count} documents"
        if errors:
            message += f" with {len(errors)} errors"

        return CommandResult(True, message, {"ingested": ingested_count, "errors": errors})

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _file_exists(self, file_hash: str) -> bool:
        """Check if file with given hash already exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM documents WHERE file_hash = ?", (file_hash,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def _insert_document(self, title: str, content: str, file_path: str, file_hash: str):
        """Insert document into database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO documents (title, content, file_path, file_hash, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (title, content, file_path, file_hash, datetime.now().isoformat(), datetime.now().isoformat()))
        conn.commit()
        conn.close()

class SystemManager(KnowledgeBaseManager):
    """System monitoring and health checks"""

    def health_check(self) -> CommandResult:
        """Perform comprehensive health check"""
        health_status = {
            "database": self._check_database(),
            "disk_space": self._check_disk_space(),
            "memory": self._check_memory_usage(),
            "performance": self._check_performance_metrics(),
            "backups": self._check_backup_status()
        }

        overall_health = all(status["healthy"] for status in health_status.values())

        return CommandResult(
            overall_health,
            "System healthy" if overall_health else "System has issues",
            health_status
        )

    def _check_database(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            conn.close()

            return {
                "healthy": True,
                "document_count": doc_count,
                "table_count": len(tables),
                "message": "Database operational"
            }
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Database error: {str(e)}"
            }

    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        try:
            stat = shutil.disk_usage('.')
            free_gb = stat.free / (1024**3)
            total_gb = stat.total / (1024**3)
            used_percent = ((stat.total - stat.free) / stat.total) * 100

            return {
                "healthy": free_gb > 1,  # At least 1GB free
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
                "used_percent": round(used_percent, 2),
                "message": f"{free_gb:.1f}GB free ({used_percent:.1f}% used)"
            }
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Disk space check failed: {str(e)}"
            }

    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                "healthy": memory.percent < 90,
                "used_gb": round(memory.used / (1024**3), 2),
                "total_gb": round(memory.total / (1024**3), 2),
                "percent": memory.percent,
                "message": f"{memory.percent:.1f}% memory used"
            }
        except ImportError:
            return {
                "healthy": True,
                "message": "psutil not available for memory monitoring"
            }
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Memory check failed: {str(e)}"
            }

    def _check_performance_metrics(self) -> Dict[str, Any]:
        """Check performance metrics"""
        try:
            # Test database response time
            start_time = time.time()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            cursor.fetchone()
            conn.close()
            response_time = time.time() - start_time

            return {
                "healthy": response_time < 1.0,
                "db_response_time": round(response_time, 3),
                "message": f"Database response time: {response_time:.3f}s"
            }
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Performance check failed: {str(e)}"
            }

    def _check_backup_status(self) -> Dict[str, Any]:
        """Check backup status"""
        try:
            backup_files = list(self.backup_dir.glob("*.db"))
            latest_backup = max(backup_files, key=os.path.getctime) if backup_files else None

            if latest_backup:
                backup_age = time.time() - os.path.getctime(latest_backup)
                backup_age_hours = backup_age / 3600

                return {
                    "healthy": backup_age_hours < 72,  # Backup less than 3 days old
                    "backup_count": len(backup_files),
                    "latest_backup_age_hours": round(backup_age_hours, 1),
                    "message": f"Latest backup: {backup_age_hours:.1f} hours ago"
                }
            else:
                return {
                    "healthy": False,
                    "backup_count": 0,
                    "message": "No backups found"
                }
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Backup check failed: {str(e)}"
            }

    def system_status(self) -> CommandResult:
        """Get overall system status"""
        status = {
            "uptime": self._get_uptime(),
            "version": self._get_version(),
            "config": self.config,
            "statistics": self._get_statistics()
        }

        return CommandResult(True, "System status retrieved", status)

    def _get_uptime(self) -> str:
        """Get system uptime"""
        try:
            import psutil
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            uptime_days = int(uptime_seconds // 86400)
            uptime_hours = int((uptime_seconds % 86400) // 3600)
            uptime_minutes = int((uptime_seconds % 3600) // 60)

            return f"{uptime_days}d {uptime_hours}h {uptime_minutes}m"
        except ImportError:
            return "Unknown (psutil not available)"

    def _get_version(self) -> str:
        """Get CLI version"""
        return "1.0.0"

    def _get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Document statistics
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_docs = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT category) FROM documents")
            categories = cursor.fetchone()[0]

            cursor.execute("SELECT created_at FROM documents ORDER BY created_at DESC LIMIT 1")
            latest_doc = cursor.fetchone()

            conn.close()

            return {
                "total_documents": total_docs,
                "categories": categories,
                "latest_document": latest_doc[0] if latest_doc else None
            }
        except Exception as e:
            return {"error": str(e)}

class BackupManager(KnowledgeBaseManager):
    """Backup and restore operations"""

    def create_backup(self) -> CommandResult:
        """Create a backup of the knowledge base"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"kb_backup_{timestamp}.db"
        backup_path = self.backup_dir / backup_filename

        try:
            # Copy database file
            shutil.copy2(self.db_path, backup_path)

            # Also backup config
            config_backup = self.backup_dir / f"kb_config_{timestamp}.json"
            shutil.copy2(self.config_path, config_backup)

            # Clean old backups (keep only the most recent N)
            self._cleanup_old_backups()

            return CommandResult(True, f"Backup created: {backup_filename}", {"backup_path": str(backup_path)})

        except Exception as e:
            return CommandResult(False, f"Backup failed: {str(e)}")

    def restore_backup(self, backup_path: str) -> CommandResult:
        """Restore from a backup"""
        backup_file = Path(backup_path)

        if not backup_file.exists():
            return CommandResult(False, f"Backup file not found: {backup_path}")

        try:
            # Create a backup of current state before restoring
            current_backup = f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            shutil.copy2(self.db_path, self.backup_dir / current_backup)

            # Restore from backup
            shutil.copy2(backup_file, self.db_path)

            return CommandResult(True, f"Successfully restored from {backup_path}", {"current_backup": current_backup})

        except Exception as e:
            return CommandResult(False, f"Restore failed: {str(e)}")

    def list_backups(self) -> CommandResult:
        """List all available backups"""
        try:
            backup_files = []
            for backup_file in self.backup_dir.glob("kb_backup_*.db"):
                stat = backup_file.stat()
                backup_files.append({
                    "filename": backup_file.name,
                    "size_mb": round(stat.st_size / (1024*1024), 2),
                    "created_at": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S")
                })

            # Sort by creation time (newest first)
            backup_files.sort(key=lambda x: x["created_at"], reverse=True)

            return CommandResult(True, f"Found {len(backup_files)} backups", backup_files)

        except Exception as e:
            return CommandResult(False, f"Failed to list backups: {str(e)}")

    def _cleanup_old_backups(self):
        """Clean up old backups, keeping only the most recent N"""
        max_backups = self.config.get("max_backups", 10)
        backup_files = list(self.backup_dir.glob("kb_backup_*.db"))

        if len(backup_files) > max_backups:
            # Sort by creation time (oldest first)
            backup_files.sort(key=os.path.getctime)

            # Remove oldest backups
            for old_backup in backup_files[:-max_backups]:
                old_backup.unlink()

            # Also clean up corresponding config backups
            config_backups = list(self.backup_dir.glob("kb_config_*.json"))
            config_backups.sort(key=os.path.getctime)

            for old_config in config_backups[:-max_backups]:
                old_config.unlink()

class CLIInterface:
    """Main CLI interface with natural language processing"""

    def __init__(self):
        self.nl_parser = NLCommandParser()
        self.doc_manager = DocumentManager()
        self.system_manager = SystemManager()
        self.backup_manager = BackupManager()
        self.user_manager = UserManager() if UserManager else None

    def execute_command(self, command: str) -> CommandResult:
        """Execute a command (natural language or structured)"""
        # Try to parse as natural language first
        parsed = self.nl_parser.parse(command)

        if parsed:
            cmd_name, params = parsed
            return self._execute_structured_command(cmd_name, params)
        else:
            # Try to execute as structured command
            return self._execute_direct_command(command)

    def _execute_structured_command(self, cmd_name: str, params: Dict[str, Any]) -> CommandResult:
        """Execute a structured command"""
        command_map = {
            'search_documents': lambda: self.doc_manager.search_documents(
                params.get('query', ''), params.get('timeframe')
            ),
            'list_recent_documents': lambda: self.doc_manager.list_recent_documents(
                params.get('timeframe', datetime.now() - timedelta(days=7))
            ),
            'bulk_ingest': lambda: self.doc_manager.bulk_ingest(
                params.get('path', '.')
            ),
            'delete_documents': lambda: CommandResult(False, "Delete functionality not implemented yet"),
            'health_check': lambda: self.system_manager.health_check(),
            'system_status': lambda: self.system_manager.system_status(),
            'performance_monitor': lambda: self._performance_monitor(),
            'create_backup': lambda: self.backup_manager.create_backup(),
            'restore_backup': lambda: self.backup_manager.restore_backup(
                params.get('backup_path', '')
            ),
            'list_backups': lambda: self.backup_manager.list_backups(),
        }

        # Add user management commands if available
        if self.user_manager:
            command_map.update({
                'create_user': lambda: self._create_user(params.get('username', ''), params.get('email', ''), params.get('role', 'viewer')),
                'delete_user': lambda: self._delete_user(params.get('username', '')),
                'list_users': lambda: self._list_users(),
            })

        command_map.update({
            'show_analytics': lambda: self._show_analytics(params.get('scope', 'overall')),
            'analyze_documents': lambda: self._analyze_documents(params.get('scope', 'all')),
        })

        if cmd_name in command_map:
            return command_map[cmd_name]()
        else:
            return CommandResult(False, f"Unknown command: {cmd_name}")

    def _execute_direct_command(self, command: str) -> CommandResult:
        """Execute command directly (fallback)"""
        # This could be expanded to support traditional CLI syntax
        return CommandResult(False, f"Command not recognized: {command}")

    def _performance_monitor(self) -> CommandResult:
        """Performance monitoring command"""
        console.print("[yellow]Starting performance monitoring... Press Ctrl+C to stop[/yellow]")

        try:
            while True:
                health = self.system_manager.health_check()

                # Create a simple status display
                status_text = f"""
                System Health: {'✅' if health.success else '❌'}
                Database Response: {health.data.get('performance', {}).get('db_response_time', 'N/A')}s
                Memory Usage: {health.data.get('memory', {}).get('percent', 'N/A')}%
                Disk Free: {health.data.get('disk_space', {}).get('free_gb', 'N/A')}GB
                """

                console.clear()
                console.print(Panel(status_text, title="Performance Monitor"))
                time.sleep(5)

        except KeyboardInterrupt:
            console.print("\n[yellow]Performance monitoring stopped[/yellow]")
            return CommandResult(True, "Performance monitoring completed")

    def _show_analytics(self, scope: str) -> CommandResult:
        """Show analytics for different scopes"""
        conn = sqlite3.connect(self.doc_manager.db_path)
        cursor = conn.cursor()

        # Get basic statistics
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_docs = cursor.fetchone()[0]

        cursor.execute("SELECT category, COUNT(*) FROM documents GROUP BY category")
        categories = cursor.fetchall()

        cursor.execute("SELECT DATE(created_at), COUNT(*) FROM documents GROUP BY DATE(created_at) ORDER BY DATE(created_at) DESC LIMIT 7")
        daily_stats = cursor.fetchall()

        conn.close()

        analytics = {
            "total_documents": total_docs,
            "categories": dict(categories),
            "daily_additions": dict(daily_stats),
            "scope": scope
        }

        return CommandResult(True, f"Analytics for {scope}", analytics)

    def _analyze_documents(self, scope: str) -> CommandResult:
        """Analyze documents"""
        # This is a placeholder for document analysis
        return CommandResult(True, f"Document analysis for {scope}", {"analysis": "Analysis results would go here"})

    def _create_user(self, username: str, email: str, role: str = "viewer") -> CommandResult:
        """Create a new user"""
        if not self.user_manager:
            return CommandResult(False, "User management not available")

        if not username:
            return CommandResult(False, "Username is required")

        if not email:
            return CommandResult(False, "Email is required")

        # For CLI, we'll need to prompt for password or generate a temporary one
        import getpass
        try:
            password = getpass.getpass("Enter password for new user: ")
            confirm_password = getpass.getpass("Confirm password: ")

            if password != confirm_password:
                return CommandResult(False, "Passwords do not match")

            success, message, user = self.user_manager.create_user(username, email, password, role)
            return CommandResult(success, message, user.to_dict() if user else None)
        except KeyboardInterrupt:
            return CommandResult(False, "User creation cancelled")

    def _delete_user(self, username: str) -> CommandResult:
        """Delete a user"""
        if not self.user_manager:
            return CommandResult(False, "User management not available")

        if not username:
            return CommandResult(False, "Username is required")

        user = self.user_manager.get_user_by_username(username)
        if not user:
            return CommandResult(False, f"User '{username}' not found")

        success, message = self.user_manager.delete_user(user.id)
        return CommandResult(success, message)

    def _list_users(self) -> CommandResult:
        """List all users"""
        if not self.user_manager:
            return CommandResult(False, "User management not available")

        users = self.user_manager.list_users()
        user_data = [user.to_dict() for user in users]
        return CommandResult(True, f"Found {len(users)} users", user_data)

    def interactive_mode(self):
        """Start interactive shell mode"""
        console.print(Panel.fit(
            "[bold green]Knowledge Base CLI[/bold green]\n"
            "Type 'help' for commands or 'exit' to quit\n"
            "Try natural language like: 'show me documents about AI added last week'",
            title="Welcome"
        ))

        while True:
            try:
                command = Prompt.ask("\n[bold cyan]kb-cli[/bold cyan]", default="")

                if command.lower() in ['exit', 'quit', 'q']:
                    console.print("[yellow]Goodbye![/yellow]")
                    break

                if command.lower() in ['help', '?']:
                    self._show_help()
                    continue

                if not command.strip():
                    continue

                # Execute command
                with console.status("[bold green]Executing command..."):
                    result = self.execute_command(command)

                # Display result
                self._display_result(result)

            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit[/yellow]")
            except EOFError:
                break

    def _show_help(self):
        """Show help information"""
        help_text = """
        [bold]Natural Language Commands:[/bold]

        📄 [cyan]Documents:[/cyan]
        • show me all documents about AI added last week
        • list documents added in the past 7 days
        • ingest documents from /path/to/folder
        • delete documents containing 'old project'

        🔧 [cyan]System:[/cyan]
        • run health check
        • show system status
        • monitor performance

        💾 [cyan]Backup:[/cyan]
        • create backup
        • restore from backup_20240101_120000.db
        • list all backups

        👥 [cyan]Users:[/cyan]
        • create user john with role admin
        • delete user jane
        • list all users

        📊 [cyan]Analytics:[/cyan]
        • show analytics for last month
        • analyze documents for performance

        [dim]Type 'exit' to quit[/dim]
        """

        console.print(Panel(help_text, title="Help", border_style="blue"))

    def _display_result(self, result: CommandResult):
        """Display command result in a formatted way"""
        if result.success:
            console.print(f"✅ [green]{result.message}[/green]")

            if result.execution_time > 0:
                console.print(f"[dim]Completed in {result.execution_time:.3f}s[/dim]")

            if result.data:
                self._display_data(result.data)
        else:
            console.print(f"❌ [red]{result.message}[/red]")

    def _display_data(self, data: Any):
        """Display data in appropriate format"""
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                # Display as table
                table = Table(show_header=True, header_style="bold magenta")

                # Add columns from first item
                for key in data[0].keys():
                    table.add_column(key.replace('_', ' ').title())

                # Add rows
                for item in data[:10]:  # Limit to 10 items
                    table.add_row(*[str(item.get(key, '')) for key in data[0].keys()])

                console.print(table)

                if len(data) > 10:
                    console.print(f"[dim]... and {len(data) - 10} more items[/dim]")
            else:
                # Display as simple list
                for item in data:
                    console.print(f"• {item}")

        elif isinstance(data, dict):
            # Display as key-value pairs
            for key, value in data.items():
                if isinstance(value, dict):
                    console.print(f"[bold]{key.replace('_', ' ').title()}:[/bold]")
                    for sub_key, sub_value in value.items():
                        console.print(f"  {sub_key}: {sub_value}")
                else:
                    console.print(f"[bold]{key.replace('_', ' ').title()}:[/bold] {value}")

        else:
            console.print(data)

# CLI Entry Points
@click.group()
@click.pass_context
def cli(ctx):
    """Knowledge Base CLI - Comprehensive knowledge base management tool"""
    ctx.ensure_object(dict)
    ctx.obj['cli_interface'] = CLIInterface()

@cli.command()
@click.argument('command', required=True)
@click.pass_context
def exec(ctx, command):
    """Execute a single command"""
    cli_interface = ctx.obj['cli_interface']
    result = cli_interface.execute_command(command)
    cli_interface._display_result(result)
    sys.exit(0 if result.success else 1)

@cli.command()
def interactive():
    """Start interactive shell mode"""
    cli_interface = CLIInterface()
    cli_interface.interactive_mode()

@cli.command()
@click.option('--query', '-q', help='Search query')
@click.option('--timeframe', '-t', help='Timeframe (e.g., "7 days", "last week")')
@click.option('--category', '-c', help='Document category')
def search(query, timeframe, category):
    """Search documents"""
    cli_interface = CLIInterface()

    # Build search command
    search_cmd = f"show documents about {query}" if query else "show recent documents"
    if timeframe:
        search_cmd += f" added {timeframe}"
    if category:
        search_cmd += f" in category {category}"

    result = cli_interface.execute_command(search_cmd)
    cli_interface._display_result(result)
    sys.exit(0 if result.success else 1)

@cli.command()
@click.argument('path', type=click.Path(exists=True))
def ingest(path):
    """Bulk ingest documents from path"""
    cli_interface = CLIInterface()
    result = cli_interface.execute_command(f"ingest documents from {path}")
    cli_interface._display_result(result)
    sys.exit(0 if result.success else 1)

@cli.command()
def health():
    """Perform system health check"""
    cli_interface = CLIInterface()
    result = cli_interface.execute_command("run health check")
    cli_interface._display_result(result)
    sys.exit(0 if result.success else 1)

@cli.command()
def status():
    """Show system status"""
    cli_interface = CLIInterface()
    result = cli_interface.execute_command("show system status")
    cli_interface._display_result(result)
    sys.exit(0 if result.success else 1)

@cli.command()
def backup():
    """Create a backup"""
    cli_interface = CLIInterface()
    result = cli_interface.backup_manager.create_backup()
    cli_interface._display_result(result)
    sys.exit(0 if result.success else 1)

@cli.command()
@click.argument('backup_path', type=click.Path(exists=True))
def restore(backup_path):
    """Restore from backup"""
    cli_interface = CLIInterface()
    result = cli_interface.execute_command(f"restore from {backup_path}")
    cli_interface._display_result(result)
    sys.exit(0 if result.success else 1)

@cli.command()
def backups():
    """List all backups"""
    cli_interface = CLIInterface()
    result = cli_interface.execute_command("list backups")
    cli_interface._display_result(result)
    sys.exit(0 if result.success else 1)

@cli.command()
def monitor():
    """Start performance monitoring"""
    cli_interface = CLIInterface()
    result = cli_interface.execute_command("monitor performance")
    # Note: monitor runs until interrupted

@cli.command()
@click.option('--scope', default='overall', help='Analytics scope')
def analytics(scope):
    """Show analytics"""
    cli_interface = CLIInterface()
    result = cli_interface.execute_command(f"show analytics for {scope}")
    cli_interface._display_result(result)
    sys.exit(0 if result.success else 1)

@cli.command()
@click.argument('username')
@click.argument('email')
@click.option('--role', default='viewer', help='User role (admin, editor, viewer, guest)')
def user_create(username, email, role):
    """Create a new user"""
    cli_interface = CLIInterface()
    result = cli_interface.execute_command(f"create user {username} {email} with role {role}")
    cli_interface._display_result(result)
    sys.exit(0 if result.success else 1)

@cli.command()
@click.argument('username')
@click.confirmation_option(prompt='Are you sure you want to delete this user?')
def user_delete(username):
    """Delete a user"""
    cli_interface = CLIInterface()
    result = cli_interface.execute_command(f"delete user {username}")
    cli_interface._display_result(result)
    sys.exit(0 if result.success else 1)

@cli.command()
def user_list():
    """List all users"""
    cli_interface = CLIInterface()
    result = cli_interface.execute_command("list all users")
    cli_interface._display_result(result)
    sys.exit(0 if result.success else 1)

@cli.command()
def user_stats():
    """Show user statistics"""
    if not UserManager:
        console.print("[red]User management not available[/red]")
        sys.exit(1)

    stats = get_user_stats()
    console.print(Panel.fit(
        "\n".join([f"[bold]{key.replace('_', ' ').title()}:[/bold] {value}" for key, value in stats.items()]),
        title="User Statistics"
    ))

if __name__ == '__main__':
    # Handle direct execution without click
    if len(sys.argv) == 1:
        # Start interactive mode
        cli_interface = CLIInterface()
        cli_interface.interactive_mode()
    else:
        # Use click for command parsing
        cli()