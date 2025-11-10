# Knowledge Base CLI Tool 🚀

A comprehensive command-line interface for managing knowledge bases with natural language processing capabilities, inspired by AI Shell.

## ✨ Features

### 🔥 Natural Language Processing
- **Conversational Commands**: Type commands in plain English
- **Smart Parsing**: Understands context and intent
- **Flexible Syntax**: Multiple ways to express the same command

### 📄 Document Management
- **Bulk Ingestion**: Process hundreds of documents at once
- **Smart Search**: Natural language and keyword search
- **Time-based Filtering**: Search by date ranges and timeframes
- **Multi-format Support**: .txt, .md, .pdf, .docx, .html, .json

### 🔧 System Administration
- **Health Monitoring**: Comprehensive system health checks
- **Performance Tracking**: Real-time metrics and monitoring
- **Backup/Restore**: Automated backup with one-click restore
- **Analytics**: Detailed usage and content analytics

### 👥 User Management
- **Role-based Access**: Admin, Editor, Viewer, Guest roles
- **Security Features**: Account lockout, audit logging
- **Session Management**: Secure authentication system
- **User Analytics**: Track user activity and patterns

## 🚀 Quick Start

### Installation
```bash
# Clone and setup
git clone <repository-url>
cd Knowledge-base

# Install dependencies
pip install -r requirements.txt

# Make executable
chmod +x kb_cli.py
```

### First Run
```bash
# Start interactive mode
python kb_cli.py

# Or run single commands
python kb_cli.py health
python kb_cli.py backup
python kb_cli.py status
```

### Default Admin User
The CLI automatically creates a default admin user:
- **Username**: `admin`
- **Password**: `admin123`
⚠️ Change this password after first login!

## 💡 Usage Examples

### Natural Language Commands
```bash
# Document search
python kb_cli.py exec "show me documents about AI added last week"
python kb_cli.py exec "find documents about machine learning"

# System operations
python kb_cli.py exec "run health check"
python kb_cli.py exec "monitor performance"

# Backup operations
python kb_cli.py exec "create backup"
python kb_cli.py exec "show all backups"

# User management
python kb_cli.py exec "create user john with role editor"
python kb_cli.py exec "list all users"
```

### Traditional CLI Commands
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
python kb_cli.py user-delete jane

# Analytics
python kb_cli.py analytics --scope "last month"
```

### Interactive Mode
```bash
python kb_cli.py
# Then type commands like:
> show me documents about AI
> run health check
> create backup
> list users
> exit
```

## 📊 Command Reference

### Document Commands
| Command | Description | Example |
|---------|-------------|---------|
| `search` | Search documents | `python kb_cli.py search --query "AI"` |
| `ingest` | Bulk import documents | `python kb_cli.py ingest ./docs` |

### System Commands
| Command | Description | Example |
|---------|-------------|---------|
| `health` | Health check | `python kb_cli.py health` |
| `status` | System status | `python kb_cli.py status` |
| `monitor` | Performance monitoring | `python kb_cli.py monitor` |

### Backup Commands
| Command | Description | Example |
|---------|-------------|---------|
| `backup` | Create backup | `python kb_cli.py backup` |
| `restore` | Restore backup | `python kb_cli.py restore backup.db` |
| `backups` | List backups | `python kb_cli.py backups` |

### User Management
| Command | Description | Example |
|---------|-------------|---------|
| `user-create` | Create user | `python kb_cli.py user-create john@email.com --role admin` |
| `user-list` | List users | `python kb_cli.py user-list` |
| `user-delete` | Delete user | `python kb_cli.py user-delete john` |
| `user-stats` | User statistics | `python kb_cli.py user-stats` |

## 🔐 Security

### User Roles
- **Admin**: Full system access
- **Editor**: Read, write, bulk ingest
- **Viewer**: Read-only + analytics
- **Guest**: Limited read access

### Security Features
- **Password Hashing**: PBKDF2 with salt
- **Session Management**: Token-based authentication
- **Account Lockout**: Automatic lockout after failed attempts
- **Audit Logging**: Complete action audit trail

## ⚙️ Configuration

### Environment Variables
```bash
export KB_DB_PATH="/custom/path/database.db"
export KB_BACKUP_DIR="/custom/backup/dir"
export KB_LOG_LEVEL="DEBUG"
```

### Configuration File
Edit `kb_config.json` to customize:
- Database paths
- Backup settings
- Supported file formats
- Performance monitoring
- Logging levels

## 📈 Monitoring

### Health Check Components
- Database connectivity and performance
- Disk space monitoring
- Memory usage tracking
- System response times
- Backup status and age

### Analytics
- Document statistics and growth
- User activity patterns
- System performance metrics
- Content analysis and categorization

## 🛠️ Advanced Usage

### Automation Scripts
```bash
#!/bin/bash
# Daily backup and health check
python kb_cli.py health
python kb_cli.py backup

# Document ingestion automation
python kb_cli.py ingest ./new-docs
python kb_cli.py analytics --scope "today"
```

### Integration with Cron
```bash
# Add to crontab:
0 2 * * * cd /path/to/kb && python kb_cli.py backup
0 */6 * * * cd /path/to/kb && python kb_cli.py health
```

## 🐛 Troubleshooting

### Common Issues

**Installation Problems**
```bash
# Install dependencies
pip install click rich psutil

# Set permissions
chmod +x kb_cli.py
```

**Database Issues**
```bash
# Check database status
python kb_cli.py health

# Reset database (caution!)
rm knowledge_base.db users.db
```

**Authentication Issues**
```bash
# Default credentials
Username: admin
Password: admin123

# Create new admin user
python user_manager.py create-user admin admin@localhost admin123 admin
```

## 📚 Documentation

- **Full Documentation**: [CLI_DOCUMENTATION.md](CLI_DOCUMENTATION.md)
- **API Reference**: Included in full documentation
- **Troubleshooting Guide**: See troubleshooting section
- **Security Guide**: See security features section

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For support:
1. Check the documentation
2. Review troubleshooting section
3. Check GitHub issues
4. Contact development team

---

**Knowledge Base CLI v1.0.0** - Your intelligent knowledge management companion 🤖✨