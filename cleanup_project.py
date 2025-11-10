#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Cleanup Script
项目清理脚本

Clean up unnecessary files, organize the project structure, and prepare for GitHub upload.
"""

import os
import shutil
import glob
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProjectCleanup:
    """项目清理器"""

    def __init__(self, project_path="/Users/gavin/Knowledge base"):
        self.project_path = project_path
        self.cleanup_log = []
        self.backup_created = False

    def run_cleanup(self):
        """执行完整的项目清理"""
        logger.info("🧹 Starting project cleanup...")

        # 1. 创建备份
        self.create_backup()

        # 2. 清理临时文件和缓存
        self.cleanup_temp_files()

        # 3. 清理重复文件
        self.cleanup_duplicate_files()

        # 4. 整理项目结构
        self.organize_project_structure()

        # 5. 清理日志文件
        self.cleanup_log_files()

        # 6. 清理JSON报告文件
        self.cleanup_json_reports()

        # 7. 整理配置文件
        self.organize_config_files()

        # 8. 创建更新的README
        self.create_updated_readme()

        # 9. 生成项目统计
        self.generate_project_stats()

        # 10. 创建部署清单
        self.create_deployment_checklist()

        self.save_cleanup_log()
        logger.info("✅ Project cleanup completed successfully!")

    def create_backup(self):
        """创建项目备份"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"backup_{timestamp}"

            # 创建备份目录
            backup_path = os.path.join(self.project_path, backup_dir)
            os.makedirs(backup_path, exist_ok=True)

            # 备份重要配置文件
            important_files = [
                ".env",
                ".gitignore",
                "requirements.txt",
                "docker-compose.yml",
                "openapi_spec.yaml"
            ]

            for file in important_files:
                src = os.path.join(self.project_path, file)
                if os.path.exists(src):
                    dst = os.path.join(backup_path, file)
                    shutil.copy2(src, dst)
                    logger.info(f"  Backed up: {file}")

            self.backup_created = True
            logger.info(f"✅ Created backup in {backup_dir}")

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")

    def cleanup_temp_files(self):
        """清理临时文件"""
        temp_patterns = [
            "*.tmp",
            "*.temp",
            "*.cache",
            "__pycache__",
            "*.pyc",
            "*.pyo",
            ".DS_Store",
            "*.swp",
            "*.swo",
            "*~"
        ]

        cleaned_files = []
        for pattern in temp_patterns:
            for file_path in glob.glob(os.path.join(self.project_path, "**", pattern), recursive=True):
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        cleaned_files.append(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        cleaned_files.append(file_path)
                except Exception as e:
                    logger.warning(f"Could not remove {file_path}: {e}")

        if cleaned_files:
            logger.info(f"🗑️ Removed {len(cleaned_files)} temporary files")
            self.cleanup_log.append(f"Temporary files removed: {len(cleaned_files)}")

    def cleanup_duplicate_files(self):
        """清理重复文件"""
        duplicate_patterns = [
            "*_old*",
            "*_backup*",
            "*_copy*",
            "*_duplicate*"
        ]

        # 保留的重要文件模式
        keep_patterns = [
            "README*",
            "*.md",
            "requirements*.txt",
            "setup.py",
            "main.py"
        ]

        removed_count = 0
        for pattern in duplicate_patterns:
            for file_path in glob.glob(os.path.join(self.project_path, "**", pattern), recursive=True):
                # 检查是否是重要文件
                should_keep = any(file_path.endswith(keep_pat) for keep_pat in keep_patterns)

                if not should_keep and os.path.exists(file_path):
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            removed_count += 1
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                            removed_count += 1
                    except Exception as e:
                        logger.warning(f"Could not remove {file_path}: {e}")

        if removed_count > 0:
            logger.info(f"🗑️ Removed {removed_count} duplicate files")
            self.cleanup_log.append(f"Duplicate files removed: {removed_count}")

    def organize_project_structure(self):
        """整理项目结构"""
        # 确保必要的目录存在
        required_dirs = [
            "docs",
            "tests",
            "scripts",
            "config",
            "data",
            "logs"
        ]

        for dir_name in required_dirs:
            dir_path = os.path.join(self.project_path, dir_name)
            os.makedirs(dir_path, exist_ok=True)

        # 移动文档文件到docs目录
        doc_patterns = ["*.md", "DOCUMENTATION*"]
        for pattern in doc_patterns:
            for file_path in glob.glob(os.path.join(self.project_path, pattern)):
                if os.path.isfile(file_path) and not file_path.endswith("README.md"):
                    try:
                        filename = os.path.basename(file_path)
                        dst = os.path.join(self.project_path, "docs", filename)
                        shutil.move(file_path, dst)
                        logger.info(f"  Moved to docs/: {filename}")
                    except Exception as e:
                        logger.warning(f"Could not move {file_path}: {e}")

        # 移动测试文件到tests目录
        test_patterns = ["test_*.py", "*_test.py", "verify_*.py"]
        for pattern in test_patterns:
            for file_path in glob.glob(os.path.join(self.project_path, pattern)):
                try:
                    filename = os.path.basename(file_path)
                    dst = os.path.join(self.project_path, "tests", filename)
                    shutil.move(file_path, dst)
                    logger.info(f"  Moved to tests/: {filename}")
                except Exception as e:
                    logger.warning(f"Could not move {file_path}: {e}")

    def cleanup_log_files(self):
        """清理日志文件"""
        log_files = glob.glob(os.path.join(self.project_path, "**", "*.log"), recursive=True)

        # 只保留最近的日志文件
        log_files.sort(key=os.path.getmtime, reverse=True)
        recent_logs = log_files[:5]  # 保留最近5个日志文件

        for log_file in log_files[5:]:
            try:
                os.remove(log_file)
                logger.info(f"  Removed old log: {os.path.basename(log_file)}")
            except Exception as e:
                logger.warning(f"Could not remove {log_file}: {e}")

    def cleanup_json_reports(self):
        """清理JSON报告文件"""
        json_files = glob.glob(os.path.join(self.project_path, "**", "*.json"), recursive=True)

        # 移动报告文件到专门的目录
        reports_dir = os.path.join(self.project_path, "data", "reports")
        os.makedirs(reports_dir, exist_ok=True)

        moved_count = 0
        for json_file in json_files:
            if "verification_report" in json_file or "metrics_report" in json_file:
                try:
                    filename = os.path.basename(json_file)
                    dst = os.path.join(reports_dir, filename)
                    shutil.move(json_file, dst)
                    moved_count += 1
                    logger.info(f"  Moved to reports/: {filename}")
                except Exception as e:
                    logger.warning(f"Could not move {json_file}: {e}")

        if moved_count > 0:
            logger.info(f"📊 Moved {moved_count} JSON report files to data/reports/")

    def organize_config_files(self):
        """整理配置文件"""
        config_dir = os.path.join(self.project_path, "config")
        os.makedirs(config_dir, exist_ok=True)

        # 移动YAML配置文件
        yaml_files = glob.glob(os.path.join(self.project_path, "*.yaml"))
        for yaml_file in yaml_files:
            try:
                filename = os.path.basename(yaml_file)
                dst = os.path.join(config_dir, filename)
                shutil.move(yaml_file, dst)
                logger.info(f"  Moved to config/: {filename}")
            except Exception as e:
                logger.warning(f"Could not move {yaml_file}: {e}")

    def create_updated_readme(self):
        """创建更新的README"""
        readme_content = """# Manufacturing Knowledge Base System

A comprehensive AI-powered knowledge base system specifically designed for manufacturing operations, featuring advanced RAG capabilities, multi-agent orchestration, and real-time observability.

## 🚀 Key Features

### 🤖 Advanced AI Capabilities
- **Advanced RAG System**: State-of-the-art retrieval with LangChain and LlamaIndex integration
- **Multi-Agent Orchestration**: Intelligent agent coordination for complex tasks
- **Multi-Modal Processing**: Handle text, images, tables, and technical drawings
- **Query Decomposition**: Break down complex manufacturing queries
- **Conversation Memory**: Context-aware dialogue management

### 🏭 Manufacturing-Specific Features
- **Quote Management**: Automated quote generation and analysis
- **Quality Control**: Integrated quality assurance workflows
- **Compliance Tracking**: ISO and industry standard compliance
- **Document Processing**: Technical drawing and specification analysis
- **Safety Management**: Safety procedure enforcement and monitoring

### 📊 Comprehensive Observability
- **Real-time Monitoring**: WebSocket-based dashboard with live metrics
- **AI Interaction Tracking**: Detailed logging with LangFuse patterns
- **Cost Analysis**: API call cost breakdown and forecasting
- **User Analytics**: Behavior pattern recognition and insights
- **Intelligent Alerting**: Proactive anomaly detection and notification

### 🔍 Advanced Search & Retrieval
- **Hybrid Search Engine**: Multiple search strategies combined
- **Personalized Search**: User-adaptive search results
- **Semantic Search**: Concept-based understanding and matching
- **Cross-Modal Retrieval**: Search across different content types
- **Citation Tracking**: Source verification and trust scoring

## 📁 Project Structure

```
├── rag/                          # Advanced RAG System
├── multi_agent_system/           # Multi-Agent Architecture
├── observability/                # Comprehensive Monitoring System
├── github-frontend/              # Modern React Frontend
├── python_sdk/                   # Python Client SDK
├── microservices/                # Microservices Architecture
├── frontend/                     # Legacy Frontend
├── docs/                         # Documentation
├── tests/                        # Test Files
├── config/                       # Configuration Files
├── data/                         # Data and Reports
└── scripts/                      # Utility Scripts
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Node.js 16+ (for frontend)
- SQLite 3
- Docker (optional)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd manufacturing-knowledge-base
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**
   ```bash
   cd github-frontend
   npm install
   ```

4. **Initialize the database**
   ```bash
   python setup_models.py
   ```

5. **Start the system**
   ```bash
   # Start the main API server
   python api_server_knowledge.py --port 8001

   # Start the chat interface
   python api_chat_interface.py --port 8002

   # Start the frontend (optional)
   cd github-frontend && npm start
   ```

## 📖 Usage Examples

### Basic RAG Query
```python
from rag.advanced_rag_system import create_advanced_rag_system

# Initialize RAG system
rag_system = await create_advanced_rag_system()
await rag_system.initialize()

# Query the system
response = await rag_system.query(
    "What are the safety procedures for HAAS VF-2 CNC machines?"
)

print(response.answer)
```

### Multi-Agent Orchestration
```python
from multi_agent_system import create_multi_agent_orchestrator

# Initialize agent system
orchestrator = await create_multi_agent_orchestrator()

# Process complex manufacturing query
result = await orchestrator.process_query(
    "Analyze quote trends for titanium aerospace parts"
)
```

### Observability Integration
```python
from observability import create_observability_orchestrator

# Initialize observability
observability = await create_observability_orchestrator()

# Track AI interactions
await observability.log_ai_interaction(
    session_id="session_001",
    user_id="user_123",
    query="Manufacturing safety procedures",
    response="Detailed safety guidelines...",
    performance_data={"response_time_ms": 1200}
)
```

## 🔧 Configuration

### Environment Variables
```bash
# Database
DATABASE_PATH=knowledge_base.db

# AI Services (Optional)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# LangFuse (Optional)
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
```

### Advanced Configuration
See `config/` directory for detailed configuration options.

## 📊 Dashboard & Monitoring

### Real-time Dashboard
- **WebSocket Connection**: `ws://localhost:8765`
- **System Health**: CPU, memory, API performance
- **Manufacturing KPIs**: Quote accuracy, quality metrics, customer satisfaction
- **User Analytics**: Behavior patterns and knowledge gaps

### Monitoring Features
- **AI Interaction Tracking**: Complete audit trail
- **Cost Analysis**: Per-operation cost breakdown
- **Performance Metrics**: Real-time system performance
- **Alert Management**: Intelligent anomaly detection
- **User Insights**: Behavior analytics and recommendations

## 🚀 Deployment

### Docker Deployment
```bash
# Build and start all services
docker-compose up -d
```

### Production Setup
1. Configure environment variables
2. Set up monitoring and alerting
3. Configure database backups
4. Set up SSL/TLS certificates
5. Configure load balancing

## 📚 Documentation

- [API Documentation](docs/API_DESIGN.md)
- [Multi-Agent System](docs/MULTI_AGENT_SYSTEM_DOCUMENTATION.md)
- [Advanced RAG System](docs/ADVANCED_RAG_SYSTEM_DOCUMENTATION.md)
- [Observability Guide](docs/OBSERVABILITY_GUIDE.md)
- [Microservices Architecture](docs/MICROSERVICES_README.md)

## 🔍 Manufacturing Use Cases

### Quote Management
- Automated quote generation with cost analysis
- Accuracy tracking and improvement
- Customer preference learning
- Competitive analysis integration

### Quality Control
- Document classification and processing
- Quality procedure enforcement
- Compliance tracking and reporting
- Defect analysis and prevention

### Document Processing
- Technical drawing analysis
- Specification extraction
- Cross-reference linking
- Version control management

### Customer Service
- Intelligent query routing
- Personalized response generation
- Feedback integration and analysis
- Satisfaction tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the documentation
- Review the examples in the `examples/` directory

---

Built with ❤️ for Advanced Manufacturing Knowledge Management

This system combines state-of-the-art AI technology with manufacturing domain expertise to create a comprehensive knowledge management solution.
"""

        try:
            readme_path = os.path.join(self.project_path, "README.md")
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            logger.info("✅ Created updated README.md")
            self.cleanup_log.append("Updated README.md with comprehensive project overview")
        except Exception as e:
            logger.error(f"Failed to create updated README: {e}")

    def generate_project_stats(self):
        """生成项目统计"""
        try:
            stats = {
                "python_files": 0,
                "javascript_files": 0,
                "yaml_files": 0,
                "markdown_files": 0,
                "json_files": 0,
                "total_files": 0,
                "total_size_mb": 0,
                "directories": 0
            }

            total_size = 0

            for root, dirs, files in os.walk(self.project_path):
                # 跳过某些目录
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

                stats["directories"] += len(dirs)

                for file in files:
                    file_path = os.path.join(root, file)

                    try:
                        if file.endswith('.py'):
                            stats["python_files"] += 1
                        elif file.endswith(('.js', '.jsx', '.ts', '.tsx')):
                            stats["javascript_files"] += 1
                        elif file.endswith(('.yaml', '.yml')):
                            stats["yaml_files"] += 1
                        elif file.endswith('.md'):
                            stats["markdown_files"] += 1
                        elif file.endswith('.json'):
                            stats["json_files"] += 1

                        stats["total_files"] += 1

                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                    except:
                        pass

            stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)

            # 保存统计信息
            stats_path = os.path.join(self.project_path, "data", "project_stats.json")
            os.makedirs(os.path.dirname(stats_path), exist_ok=True)

            import json
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

            logger.info(f"📊 Project Statistics:")
            logger.info(f"  Python files: {stats['python_files']}")
            logger.info(f"  JavaScript files: {stats['javascript_files']}")
            logger.info(f"  Markdown files: {stats['markdown_files']}")
            logger.info(f"  Total files: {stats['total_files']}")
            logger.info(f"  Total size: {stats['total_size_mb']} MB")

            self.cleanup_log.append(f"Project stats: {stats}")

        except Exception as e:
            logger.error(f"Failed to generate project stats: {e}")

    def create_deployment_checklist(self):
        """创建部署清单"""
        checklists = [
            "✅ Create project backup",
            "✅ Clean temporary files",
            "✅ Organize project structure",
            "✅ Update README.md",
            "✅ Generate project statistics",
            "⏳ Add all new files to Git",
            "⏳ Create comprehensive commit",
            "⏳ Push to GitHub repository",
            "⏳ Verify deployment status"
        ]

        checklist_path = os.path.join(self.project_path, "DEPLOYMENT_CHECKLIST.md")
        with open(checklist_path, 'w', encoding='utf-8') as f:
            f.write("# Deployment Checklist\n\n")
            for item in checklists:
                f.write(f"{item}\n")

        logger.info("✅ Created deployment checklist")
        self.cleanup_log.append("Created deployment checklist")

    def save_cleanup_log(self):
        """保存清理日志"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(self.project_path, "cleanup_log.json")

            import json
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "backup_created": self.backup_created,
                "cleanup_actions": self.cleanup_log
            }

            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 Cleanup log saved to cleanup_log.json")

        except Exception as e:
            logger.error(f"Failed to save cleanup log: {e}")

def main():
    """主函数"""
    print("🧹 Manufacturing Knowledge Base - Project Cleanup")
    print("=" * 50)

    try:
        cleaner = ProjectCleanup()
        cleaner.run_cleanup()
        print("\n🎉 Project cleanup completed successfully!")
        print("\nNext steps:")
        print("1. Review the cleanup_log.json for details")
        print("2. Check the updated project structure")
        print("3. Run: git add .")
        print("4. Run: git commit -m 'Project cleanup and organization update'")
        print("5. Run: git push origin main")

    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()