# Multi-Agent Orchestration System

## Overview

A comprehensive XAgent-inspired multi-agent orchestration system designed for advanced manufacturing knowledge base management. This system provides intelligent agent coordination, specialized task execution, and seamless integration with existing infrastructure.

## 🚀 Key Features

### Advanced Agent Communication
- **Message Routing**: Priority-based message queues with Redis pub/sub and compression
- **Task Delegation**: Intelligent agent selection with performance-based scoring
- **Real-time Monitoring**: Heartbeat system with automatic status updates
- **Cross-Agent Collaboration**: Seamless cooperation between specialized agents

### Specialized Agents
- **DocumentProcessor**: Multi-modal processing (PDF, Excel, images, OCR)
- **PriceAnalyzer**: ML-powered price analysis, market comparison, risk assessment
- **TrendPredictor**: Time series forecasting, anomaly detection, market signal identification
- **CustomerInsights**: Behavioral segmentation, churn prediction, LTV analysis

### Coordination & Orchestration
- **Task Planning**: Dependency graph analysis with critical path optimization
- **Execution Strategies**: Sequential, parallel, collaborative, and adaptive execution
- **Load Balancing**: Dynamic agent selection based on performance and availability
- **Result Synthesis**: Intelligent aggregation and conflict resolution

### Agent Marketplace
- **Dynamic Registration**: Self-service agent registration with capability indexing
- **Intelligent Matching**: ML-based agent matching for optimal task assignment
- **Performance Tracking**: Real-time performance metrics and agent ranking
- **Health Monitoring**: Comprehensive marketplace health checks and alerts

### Performance Monitoring
- **Real-time Metrics**: CPU, memory, response time, throughput monitoring
- **Anomaly Detection**: ML-based anomaly detection for performance patterns
- **Resource Optimization**: Automated resource utilization analysis and optimization
- **Prometheus Integration**: Comprehensive metrics collection and visualization

### Configuration Compatibility
- **YAML Migration**: Seamless migration from existing YAML configurations
- **Backward Compatibility**: Full compatibility with legacy agent configurations
- **Enhanced Features**: Automatic enhancement suggestions and migration paths
- **Validation Tools**: Configuration validation and compliance checking

## 📁 Architecture

```
multi_agent_system/
├── __init__.py                         # Main module entry point
├── README.md                            # This documentation
├── core/
│   └── yaml_compatibility.py          # YAML configuration compatibility
├── protocols/
│   └── agent_communication.py        # Advanced communication protocols
├── agents/
│   ├── specialized_agents.py          # Enhanced base agents
│   ├── coordinator_agent.py           # Task coordination and planning
│   ├── trend_predictor_agent.py       # Trend forecasting
│   └── customer_insights_agent.py     # Customer behavior analysis
├── marketplace/
│   └── agent_marketplace.py           # Agent registration and discovery
└── monitoring/
    └── performance_monitor.py        # Performance monitoring and optimization
```

## 🛠️ Installation

### Prerequisites

```bash
# Required Python packages
pip install asyncio aiohttp aiofiles psutil prometheus-client
pip install scikit-learn scipy networkx pandas numpy
pip install cryptography pydantic redis
pip install pyyaml joblib pdfplumber openpyxl pytesseract
pip install statsmodels
```

### Basic Setup

```python
# Import the multi-agent system
from multi_agent_system import XAgentOrchestrator

# Create and initialize the orchestrator
orchestrator = XAgentOrchestrator()
await orchestrator.initialize()

# The orchestrator is now ready to coordinate agents
print("Multi-agent system initialized successfully!")
```

## 📖 Usage Examples

### Basic Agent Creation

```python
from multi_agent_system.agents.specialized_agents import DocumentProcessorAgent
from multi_agent_system.agents.trend_predictor_agent import TrendPredictorAgent

# Create specialized agents
doc_processor = DocumentProcessorAgent(orchestrator)
trend_predictor = TrendPredictorAgent(orchestrator)

# Register agents
await orchestrator.register_agent(doc_processor)
await orchestrator.register_agent(trend_predictor)
```

### Task Coordination

```python
from multi_agent_system.protocols.agent_communication import TaskRequest, Priority

# Create a complex task
complex_task = TaskRequest(
    task_id="comprehensive_analysis_001",
    task_type="comprehensive_analysis",
    description="Analyze manufacturing trends and customer insights",
    parameters={
        'data_sources': ['market_data', 'customer_feedback'],
        'analysis_depth': 'comprehensive',
        'output_format': 'detailed_report'
    },
    priority=Priority.HIGH
)

# Submit task for coordinated execution
result = await orchestrator.submit_task(complex_task)
print(f"Task completed: {result.success}")
```

### Agent Marketplace

```python
from multi_agent_system.marketplace.agent_marketplace import create_agent_marketplace

# Create marketplace
marketplace = create_agent_marketplace()

# Start marketplace
await marketplace.start()

# Discover agents for specific tasks
suitable_agents = marketplace.discover_agents(
    capabilities=['document_processing', 'text_extraction'],
    min_success_rate=0.8
)
print(f"Found {len(suitable_agents)} suitable agents")
```

### Performance Monitoring

```python
from multi_agent_system.monitoring.performance_monitor import create_performance_monitor

# Create performance monitor
monitor = create_performance_monitor({
    'monitoring_interval': 30,
    'prometheus_enabled': True,
    'retention_days': 30
})

# Start monitoring
await monitor.start_monitoring()

# Get performance summary
summary = monitor.get_performance_summary()
print(f"Performance status: {summary}")
```

### YAML Configuration Migration

```python
from multi_agent_system.core.yaml_compatibility import create_yaml_compatibility_layer

# Create compatibility layer
compatibility = create_yaml_compatibility_layer("agent_configs")

# Load legacy configurations
legacy_configs = await compatibility.load_legacy_configurations()

# Migrate to enhanced format
from multi_agent_system.marketplace.agent_marketplace import create_agent_marketplace
marketplace = create_agent_marketplace()
migrated = await compatibility.migrate_all_configs(marketplace)

# Get migration summary
summary = compatibility.get_migration_summary()
print(f"Migration completed: {summary['migration_status']}")
```

## 🔧 Configuration

### Environment Configuration

```python
# Example configuration for orchestrator
config = {
    'message_router': {
        'redis_url': 'redis://localhost:6379',
        'timeout_seconds': 300
    },
    'task_delegator': {
        'max_retries': 3,
        'retry_delay': 60,
        'load_balancing': True
    },
    'performance_monitor': {
        'monitoring_interval': 30,
        'prometheus_enabled': True,
        'retention_days': 30
    },
    'marketplace': {
        'db_path': 'marketplace.db',
        'monitoring_enabled': True,
        'auto_scaling_enabled': True,
        'security_enabled': True
    }
}

orchestrator = XAgentOrchestrator(config=config)
```

### YAML Configuration Format

```yaml
# Legacy agent configuration (automatically compatible)
name: document_processor_agent
description: "Automated document processing agent"
version: "2.0.0"
author: "System Administrator"

triggers:
  scheduled:
    - name: daily_processing
      cron: "0 9 * * *"
      enabled: true
  file_system:
    - name: document_monitor
      directories: ["/path/to/documents"]
      patterns: ["*.pdf", "*.docx", "*.xlsx"]
      enabled: true

actions:
  primary:
    - name: process_new_documents
      command: "python3 process_documents.py --mode automated"
      working_directory: "/app/agents/document_processor"

config:
  database:
    path: "knowledge_base.db"
    connection_timeout: 30
  processing:
    batch_size: 20
    timeout_seconds: 600

tools_allowed:
  FileSystem:
    permissions: ["Read", "Write"]
    allowed_paths: ["/app/data/documents"]
  PythonExecution:
    allowed_scripts: ["process_documents.py"]
    max_execution_time: 600

resources:
  max_memory: "2GB"
  max_cpu_time: 600
  max_concurrent_actions: 2
```

## 🔍 API Reference

### Main Orchestration API

#### MultiAgentOrchestrator

```python
class MultiAgentOrchestrator:
    async def initialize() -> bool
    async def register_agent(self, agent: BaseAgent) -> bool
    async def unregister_agent(self, agent_id: str) -> bool
    async def submit_task(self, task: TaskRequest) -> Dict[str, Any]
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]
    async def get_system_status(self) -> Dict[str, Any]
    async def shutdown(self) -> None
```

#### TaskRequest

```python
@dataclass
class TaskRequest:
    task_id: str
    task_type: str
    description: str
    parameters: Dict[str, Any]
    requirements: Dict[str, Any]
    priority: Priority
    deadline: Optional[datetime] = None
    collaboration_needed: bool = False
```

### Communication Protocol API

#### MessageRouter

```python
class MessageRouter:
    async def start() -> None
    async def stop() -> None
    def register_handler(self, message_type: MessageType, handler: Callable) -> None
    async def send_message(self, message: AgentMessage) -> bool
    def subscribe_to_channel(self, channel: str) -> None
```

#### TaskDelegator

```python
class TaskDelegator:
    async def delegate_task(self, task: TaskRequest, preferred_agents: List[str] = None) -> str
    def update_agent_capabilities(self, agent_id: str, capabilities: Set[str]) -> None
    def update_agent_performance(self, agent_id: str, performance_metrics: Dict[str, float]) -> None
    def get_delegation_stats(self) -> Dict[str, Any]
```

### Marketplace API

#### AgentMarketplace

```python
class AgentMarketplace:
    async def start() -> None
    async def stop() -> None
    async def register_external_agent(self, agent_info: Dict[str, Any]) -> str
    async def discover_and_match(self, task_request: TaskRequest) -> Optional[str]
    async def get_marketplace_status(self) -> Dict[str, Any]
```

### Monitoring API

#### PerformanceMonitor

```python
class PerformanceMonitor:
    async def start_monitoring() -> None
    async def stop_monitoring() -> None
    def get_performance_summary(self) -> Dict[str, Any]
    def calculate_system_health(self) -> Dict[str, Any]
    def get_resource_utilization_summary(self) -> Dict[str, Any]
```

## 🔧 Agent Specializations

### DocumentProcessorAgent

**Capabilities:**
- Multi-format document processing (PDF, Excel, Word, images)
- OCR and text extraction
- Metadata extraction and classification
- Batch processing workflows
- Quality validation and error handling

**Usage:**
```python
processor = DocumentProcessorAgent(orchestrator)
await processor.initialize()

task = TaskRequest(
    task_id="doc_process_001",
    task_type="process_document",
    parameters={'file_path': '/path/to/document.pdf'}
)
result = await processor.execute_delegated_task(task)
```

### PriceAnalyzerAgent

**Capabilities:**
- Market price comparison and analysis
- ML-powered price prediction
- Competitor analysis and tracking
- Risk assessment and optimization
- Trend analysis and forecasting

**Usage:**
```python
analyzer = PriceAnalyzerAgent(orchestrator)
await analyzer.initialize()

task = TaskRequest(
    task_id="price_analysis_001",
    task_type="analyze_quote",
    parameters={'quote_data': {'price': 1000, 'product_category': 'fasteners'}}
)
result = await analyzer.execute_delegated_task(task)
```

### TrendPredictorAgent

**Capabilities:**
- Time series forecasting
- Seasonal decomposition
- Anomaly detection
- Market signal identification
- Predictive modeling

**Usage:**
```python
predictor = TrendPredictorAgent(orchestrator)
await predictor.initialize()

task = TaskRequest(
    task_id="trend_pred_001",
    task_type="predict_price_trend",
    parameters={
        'product_category': 'bolts',
        'time_horizon': '30d',
        'confidence_level': 0.95
    }
)
result = await predictor.execute_delegated_task(task)
```

### CustomerInsightsAgent

**Capabilities:**
- Customer segmentation
- Behavioral analysis
- Churn prediction
- Lifetime value calculation
- Personalized recommendations

**Usage:**
```python
insights = CustomerInsightsAgent(orchestrator)
await insights.initialize()

task = TaskRequest(
    task_id="insights_001",
    task_type="segment_customers",
    parameters={'segmentation_method': 'kmeans', 'num_segments': 5}
)
result = await insights.execute_delegated_task(task)
```

### CoordinatorAgent

**Capabilities:**
- Complex task decomposition
- Dependency resolution
- Load balancing
- Collaborative execution
- Result synthesis

**Usage:**
```python
coordinator = CoordinatorAgent(orchestrator)
await coordinator.initialize()

task = TaskRequest(
    task_id="coordination_001",
    task_type="coordinate_complex_task",
    parameters={
        'root_task': {
            'task_type': 'comprehensive_analysis',
            'parameters': {...}
        }
    }
)
result = await coordinator.execute_delegated_task(task)
```

## 🚨 Advanced Features

### 1. Intelligent Task Decomposition

The coordinator agent automatically decomposes complex tasks into manageable sub-tasks:

- **Complexity Analysis**: Automatic assessment of task complexity
- **Dependency Resolution**: Creation of execution dependency graphs
- **Resource Allocation**: Optimal agent selection and load balancing
- **Execution Planning**: Multiple execution strategies (sequential, parallel, collaborative)

### 2. Cross-Agent Collaboration

Agents can collaborate on complex tasks through:

- **Task Sharing**: Automatic delegation of sub-tasks to specialized agents
- **Result Aggregation**: Intelligent combination of results from multiple agents
- **Conflict Resolution**: Handling conflicting results and recommendations
- **Dynamic Replanning**: Adaptation to changing conditions and agent availability

### 3. Performance Optimization

The system includes comprehensive performance optimization:

- **Load Balancing**: Dynamic distribution of tasks based on agent capabilities and load
- **Resource Monitoring**: Real-time tracking of CPU, memory, and network usage
- **Anomaly Detection**: ML-based identification of performance issues
- **Auto-scaling**: Automatic scaling of agent resources based on demand

### 4. Configuration Migration

Seamless migration from existing YAML configurations:

- **Automatic Detection**: Recognition of legacy agent configurations
- **Intelligent Mapping**: Mapping of legacy capabilities to enhanced features
- **Gradual Enhancement**: Step-by-step migration with compatibility preservation
- **Validation Tools**: Comprehensive configuration validation and error checking

## 📊 Monitoring & Analytics

### System Metrics

The system provides comprehensive monitoring of:

- **Agent Performance**: Response time, success rate, throughput
- **Resource Utilization**: CPU, memory, disk, network usage
- **Task Execution**: Completion rates, execution times, error rates
- **Marketplace Health**: Agent availability, registration trends, capability coverage

### Prometheus Integration

Prometheus metrics are available for:

- Agent-level metrics (CPU, memory, response time)
- System-level metrics (load average, resource usage)
- Task execution metrics (throughput, success rate)
- Marketplace metrics (agent count, registration rate)

### Alerting System

Automatic alerts for:

- Performance anomalies and degradation
- Agent failures and recovery
- Resource utilization thresholds
- System health issues

## 🔄 Integration with Existing Systems

### Database Compatibility

The system maintains full compatibility with the existing SQLite database structure:

- **Automatic Migration**: Seamless migration of legacy data
- **Schema Extension**: Enhanced schema while preserving existing tables
- **Rollback Support**: Ability to rollback to legacy configurations if needed

### API Integration

Compatible with existing REST APIs:

- **Knowledge Base API** (Port 8001): Extended with multi-agent capabilities
- **Chat Interface API** (Port 8002): Enhanced with agent coordination
- **File System Integration**: Direct access to monitored directories
- **External Service Integration**: Support for webhooks and external APIs

### Legacy Agent Integration

Legacy agents can be:

- **Automatically Migrated**: Automatic conversion to enhanced format
- **Hybrid Operation**: Legacy and enhanced agents working together
- **Gradual Enhancement**: Step-by-step migration path
- **Configuration Preservation**: Full preservation of existing YAML configurations

## 🚀 Deployment

### Production Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up Redis (for message routing)
redis-server

# 3. Start the multi-agent system
python -m multi_agent_system.main

# 4. Verify deployment
curl http://localhost:8000/api/v1/status
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY multi_agent_system/ /app/
WORKDIR /app

# Expose ports
EXPOSE 8000 8001 8002

# Start the system
CMD ["python", "-m", "multi_agent_system.main"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: multi-agent-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: multi-agent-system
  template:
    metadata:
      labels:
        app: multi-agent-system
    spec:
      containers:
      - name: multi-agent-system
        image: multi-agent-system:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis:6379"
        - name: DB_PATH
          value: "knowledge_base.db"
```

## 🔒 Customization

### Creating Custom Agents

```python
from multi_agent_system.agents.specialized_agents import EnhancedBaseAgent

class CustomAgent(EnhancedBaseAgent):
    def __init__(self, orchestrator):
        super().__init__("custom_agent", "Custom Agent", orchestrator)

    def get_capabilities(self):
        return [
            AgentCapability.CUSTOM_CAPABILITY
        ]

    async def _execute_task_logic(self, task: TaskRequest) -> Dict[str, Any]:
        # Custom task execution logic
        return {"result": "Custom task completed"}

    async def initialize(self):
        await super().initialize()
        # Custom initialization logic
```

### Custom Communication Protocols

```python
from multi_agent_system.protocols.agent_communication import MessageType

# Register custom message handler
agent.message_router.register_handler(
    MessageType.CUSTOM_MESSAGE,
    self.handle_custom_message
)

async def handle_custom_message(self, message: AgentMessage):
    # Handle custom message
    pass
```

### Custom Performance Metrics

```python
from prometheus_client import Gauge, Counter

# Custom metrics
custom_metric = Gauge('custom_operation_count', 'Total custom operations', ['agent_id'])

# Update metrics
custom_metric.labels['agent_id'].inc()
```

## 📚 Best Practices

### 1. Agent Design

- **Single Responsibility**: Each agent should have a focused purpose
- **Stateless Operation**: Design agents to be as stateless as possible
- **Error Handling**: Implement comprehensive error handling and recovery
- **Resource Management**: Monitor and limit resource usage

### 2. Task Design

- **Atomic Operations**: Design tasks to be atomic and idempotent
- **Timeout Handling**: Set appropriate timeouts for task execution
- **Retry Logic**: Implement intelligent retry mechanisms
- **Result Validation**: Validate and sanitize task results

### 3. Communication

- **Message Size**: Keep messages compact and efficient
- **Error Handling**: Handle communication failures gracefully
- **Security**: Validate and sanitize all message content
- **Logging**: Log communication for debugging and auditing

### 4. Performance Optimization

- **Caching**: Implement intelligent caching for frequently accessed data
- **Batching**: Use batch processing for similar operations
- **Async Operations**: Use async/await for I/O operations
- **Resource Monitoring**: Monitor and optimize resource usage

## 🐛 Troubleshooting

### Common Issues

#### Agent Registration Fails
```
Error: Agent registration failed
Solution: Check agent capabilities and marketplace requirements
```

#### Task Execution Timeout
```
Error: Task execution timeout
Solution: Check resource limits and task complexity
```

#### Communication Failure
```
Error: Agent communication failed
Solution: Verify Redis connection and message routing configuration
```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug mode in configuration
config['debug_mode'] = True
orchestrator = XAgentOrchestrator(config=config)
```

### Health Checks

```python
# Check system health
status = await orchestrator.get_system_status()
print(f"System status: {status}")

# Check specific agent health
agent_status = await orchestrator.get_agent_status("agent_id")
print(f"Agent status: {agent_status}")
```

## 📚 Roadmap

### Version 2.1 (Planned)
- Enhanced ML models for better task prediction
- Multi-cloud deployment support
- Advanced visualization dashboards
- Natural language task specification
- Automated agent optimization

### Version 2.2 (Future)
- Swarm intelligence for emergent behavior
- Federated learning across deployments
- Advanced security features
- Edge computing support
- Real-time collaboration features

## 📞 Support

For support, documentation, or contributions:

1. **Documentation**: Check the comprehensive documentation
2. **Issues**: Report issues on the project repository
3. **Discussions**: Join discussions in the community forum
4. **Examples**: Review the usage examples and code samples

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Built with ❤️ for Advanced Manufacturing Knowledge Management using XAgent-inspired principles