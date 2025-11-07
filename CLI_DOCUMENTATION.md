# Knowledge Base CLI Tool - Comprehensive Documentation

A powerful command-line interface for managing knowledge bases with natural language processing capabilities, inspired by AI Shell.

## 🚀 Overview

The Knowledge Base CLI (kb-cli) is a comprehensive tool that provides both traditional CLI commands and natural language processing capabilities. Users can interact with their knowledge base using either structured commands or conversational language.

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Required dependencies (automatically installed):
  - `click` - Command-line interface creation kit
  - `rich` - Rich text and beautiful formatting in the terminal

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd Knowledge-base

# Make the CLI executable
chmod +x kb_cli.py

# (Optional) Install globally
sudo cp kb_cli.py /usr/local/bin/kb-cli
```

## 🎯 Features

### 🔥 Natural Language Processing
Type commands in plain English:
- "show me all documents about AI added last week"
- "create a backup of the knowledge base"
- "monitor system performance"
- "list users with admin role"

### 📄 Document Management
- **Bulk Ingestion**: Process multiple documents at once
- **Smart Search**: Natural language document search
- **Time-based Filtering**: Search documents by date ranges
- **Format Support**: .txt, .md, .pdf, .docx, .html, .json

### 🔧 System Administration
- **Health Checks**: Comprehensive system monitoring
- **Performance Tracking**: Real-time performance metrics
- **Backup/Restore**: Automated backup and recovery
- **Analytics**: Detailed usage and content analytics

### 👥 User Management
- **Role-based Access**: Admin, Editor, Viewer, Guest roles
- **Authentication**: Secure login system with session management
- **Audit Logging**: Complete audit trail of all actions
- **Security Features**: Account lockout, password policies

## 🛠️ Usage

### Interactive Mode
Start the interactive shell for conversational interaction:

```bash
python kb_cli.py
# or
./kb_cli.py
```

### One-shot Commands
Execute single commands directly:

```bash
python kb_cli.py exec "show me documents about AI"
python kb_cli.py exec "run health check"
python kb_cli.py exec "create backup"
```

### Traditional CLI Commands
Use traditional CLI syntax:

```bash
# Document operations
python kb_cli.py search --query "machine learning" --timeframe "7 days"
python kb_cli.py ingest /path/to/documents

# System operations
python kb_cli.py health
python kb_cli.py status
python kb_cli.py monitor

# Backup operations
python kb_cli.py backup
python kb_cli.py restore backup_20240101_120000.db
python kb_cli.py backups

# User management
python kb_cli.py user-create john john@example.com --role editor
python kb_cli.py user-list
python kb-cli user-delete jane

# Analytics
python kb_cli.py analytics --scope "last month"
```

## 📚 Command Reference

### Natural Language Commands

#### 📄 Document Operations
```bash
# Search documents
"show me all documents about AI added last week"
"list documents about machine learning"
"display documents containing 'Python' added yesterday"
"find documents about neural networks created this month"

# Document ingestion
"ingest documents from /path/to/folder"
"import all markdown files from ./docs"
"process documents in /home/user/Documents"

# Document management
"delete documents about old project"
"remove documents containing 'deprecated'"
```

#### 🔧 System Operations
```bash
# Health monitoring
"run health check"
"perform system health check"
"check system status"
"monitor performance"

# System information
"show system status"
"display system information"
"get system statistics"
```

#### 💾 Backup Operations
```bash
# Create backups
"create backup"
"make a backup of the knowledge base"
"backup the system"

# Restore backups
"restore from backup_20240101_120000.db"
"recover from backup file"

# List backups
"show all backups"
"list available backups"
"display backup history"
```

#### 👥 User Management
```bash
# Create users
"create user john with role admin"
"add user jane@example.com as editor"
"make new user bob with viewer role"

# Delete users
"delete user john"
"remove user jane"
"delete account bob"

# List users
"show all users"
"display user list"
"list active users"
```

#### 📊 Analytics
```bash
# View analytics
"show analytics"
"display system statistics"
"get usage analytics for last month"
"show document analytics"
```

### Traditional CLI Commands

#### Document Commands
```bash
# Search documents
python kb_cli.py search --query "machine learning" --timeframe "7 days" --category "AI"

# Ingest documents
python kb_cli.py ingest /path/to/folder
python kb_cli.py ingest ./docs --format md
```

#### System Commands
```bash
# Health check
python kb_cli.py health

# System status
python kb_cli.py status

# Performance monitoring
python kb_cli.py monitor
```

#### Backup Commands
```bash
# Create backup
python kb_cli.py backup

# Restore backup
python kb_cli.py restore backup_20240101_120000.db

# List backups
python kb_cli.py backups
```

#### User Management Commands
```bash
# Create user
python kb_cli.py user-create username email@example.com --role admin

# List users
python kb_cli.py user-list

# Delete user
python kb_cli.py user-delete username

# User statistics
python kb_cli.py user-stats
```

#### Analytics Commands
```bash
# Show analytics
python kb_cli.py analytics --scope "overall"
python kb_cli.py analytics --scope "last month"
```

## 🔐 User Roles and Permissions

### Role Hierarchy
1. **Admin** - Full system access
2. **Editor** - Can read, write, and bulk ingest
3. **Viewer** - Read-only access with analytics
4. **Guest** - Limited read access

### Permissions Matrix

| Permission | Guest | Viewer | Editor | Admin |
|------------|-------|--------|--------|-------|
| Read Documents | ✅ | ✅ | ✅ | ✅ |
| Write Documents | ❌ | ❌ | ✅ | ✅ |
| Delete Documents | ❌ | ❌ | ✅ | ✅ |
| Bulk Ingest | ❌ | ❌ | ✅ | ✅ |
| View Analytics | ❌ | ✅ | ✅ | ✅ |
| Manage Users | ❌ | ❌ | ❌ | ✅ |
| System Admin | ❌ | ❌ | ⚠️ | ✅ |
| Backup/Restore | ❌ | ❌ | ❌ | ✅ |

## ⚙️ Configuration

### Default Configuration File (`kb_config.json`)
```json
{
  "database_path": "knowledge_base.db",
  "backup_dir": "backups",
  "max_backups": 10,
  "auto_backup_interval": 24,
  "supported_formats": [".txt", ".md", ".pdf", ".docx", ".html", ".json"],
  "indexing_enabled": true,
  "performance_monitoring": true,
  "log_level": "INFO"
}
```

### Environment Variables
```bash
# Database path
export KB_DB_PATH="/custom/path/to/database.db"

# Backup directory
export KB_BACKUP_DIR="/custom/backup/directory"

# Log level
export KB_LOG_LEVEL="DEBUG"
```

## 🔍 Advanced Usage

### Scripting and Automation
```bash
#!/bin/bash
# Example automation script

# Health check
python kb_cli.py health

# Ingest new documents
python kb_cli.py ingest ./new-docs

# Create backup
python kb_cli.py backup

# Send notification if backup successful
if [ $? -eq 0 ]; then
    echo "Backup completed successfully"
fi
```

### Integration with Other Tools
```bash
# Integration with cron for automated tasks
# Add to crontab:
# 0 2 * * * cd /path/to/knowledge-base && python kb_cli.py backup
# 0 */6 * * * cd /path/to/knowledge-base && python kb_cli.py health

# Integration with monitoring systems
python kb_cli.py health | grep -q "healthy" && echo "OK" || echo "ALERT"
```

### Custom Natural Language Patterns
Extend the NLP parser by modifying the `patterns` dictionary in `NLCommandParser`:

```python
# Example: Add custom pattern
self.patterns[r'custom pattern here'] = ('custom_command', {'param1': 1})
```

## 🚨 Troubleshooting

### Common Issues

#### Installation Problems
```bash
# If click/rich not found
pip install click rich

# If permission denied
chmod +x kb_cli.py

# If module not found
python -m pip install -r requirements.txt
```

#### Database Issues
```bash
# If database locked
python kb_cli.py health  # Will show database status

# Reset database (caution: deletes all data)
rm knowledge_base.db users.db
```

#### Authentication Issues
```bash
# Default admin credentials
Username: admin
Password: admin123

# Reset admin password
python user_manager.py create-user admin admin@localhost admin123 admin
```

### Debug Mode
Enable debug logging:
```bash
export KB_LOG_LEVEL="DEBUG"
python kb_cli.py health
```

### Performance Issues
```bash
# Monitor system performance
python kb_cli.py monitor

# Check database size
ls -lh knowledge_base.db

# Clean old backups
python kb_cli.py backups  # Shows backup count
```

## 📊 Monitoring and Analytics

### Health Check Components
- **Database**: Connectivity and performance
- **Disk Space**: Available storage monitoring
- **Memory Usage**: System memory consumption
- **Performance**: Response time metrics
- **Backup Status**: Backup age and availability

### Analytics Data
- **Document Statistics**: Total documents, categories, growth trends
- **User Activity**: Login patterns, session duration
- **System Performance**: Response times, resource usage
- **Content Analytics**: Document categories, tagging patterns

## 🔒 Security Features

### Authentication Security
- **Password Hashing**: PBKDF2 with salt
- **Session Management**: Token-based sessions with expiration
- **Account Lockout**: Automatic lockout after failed attempts
- **Audit Logging**: Complete action audit trail

### Data Protection
- **Input Validation**: All user inputs validated
- **SQL Injection Protection**: Parameterized queries
- **File Access Control**: Restricted file system access
- **Backup Encryption**: Encrypted backup files

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd Knowledge-base

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
python -m flake8 kb_cli.py
```

### Adding New Commands
1. Add command pattern to `NLCommandParser`
2. Implement command method in appropriate manager
3. Add CLI command using Click decorator
4. Update documentation

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add docstrings for all functions
- Include error handling

## 📖 API Reference

### Core Classes

#### `CLIInterface`
Main interface for command execution.

```python
cli = CLIInterface()
result = cli.execute_command("show documents about AI")
```

#### `DocumentManager`
Handles document operations.

```python
doc_manager = DocumentManager()
result = doc_manager.search_documents("AI", timeframe)
```

#### `SystemManager`
System monitoring and health checks.

```python
system_manager = SystemManager()
health = system_manager.health_check()
```

#### `BackupManager`
Backup and restore operations.

```python
backup_manager = BackupManager()
result = backup_manager.create_backup()
```

#### `UserManager`
User authentication and authorization.

```python
user_manager = UserManager()
success, token, user = user_manager.authenticate("admin", "password")
```

### Data Structures

#### `CommandResult`
Result object for command execution.

```python
@dataclass
class CommandResult:
    success: bool
    message: str
    data: Optional[Any] = None
    execution_time: float = 0.0
```

#### `User`
User data model.

```python
@dataclass
class User:
    id: Optional[int] = None
    username: str = ""
    email: str = ""
    role: str = "viewer"
    # ... other fields
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
1. Check this documentation
2. Review troubleshooting section
3. Check GitHub issues
4. Contact development team

## 🔄 Version History

### Version 1.0.0
- Initial release
- Natural language processing
- Document management
- System monitoring
- User management
- Backup/restore functionality
- Interactive shell mode

---

*Last updated: January 2024*