# Quick Start Guide
# Manufacturing Knowledge Base - Open-Source Integration Framework

This guide will help you get started with the complete open-source integration framework for the Manufacturing Knowledge Base System.

## 🚀 Prerequisites

### Required Dependencies

```bash
# Core dependencies
pip install langchain openai chromadb redis sqlalchemy

# Open-source integrations
pip install langfuse  # For observability
pip install websockets  # For LobeChat WebSocket support
pip install asyncio    # For XAgent coordination

# Manufacturing-specific dependencies
pip install pandas numpy matplotlib  # For analytics and reporting
```

### Environment Variables

```bash
# OpenAI API (for LangChain)
export OPENAI_API_KEY="your-openai-api-key"

# LangFuse (for observability)
export LANGFUSE_PUBLIC_KEY="your-langfuse-public-key"
export LANGFUSE_SECRET_KEY="your-langfuse-secret-key"

# Database and Cache
export REDIS_PASSWORD="your-redis-password"
export DATABASE_URL="sqlite:///knowledge_base.db"

# Security
export ENCRYPTION_KEY="your-encryption-key"
export JWT_SECRET="your-jwt-secret"
```

## 📦 Installation

### 1. Clone and Setup

```bash
cd /Users/gavin/Knowledge\ base/integrations
```

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt  # Create this file with the dependencies above

# Or install individually
pip install langchain openai chromadb langfuse websockets redis
```

### 3. Configure Integrations

```bash
# Copy and edit the configuration
cp config/integrations.yaml config/integrations.yaml.local

# Edit the configuration with your settings
nano config/integrations.yaml.local
```

### 4. Initialize the Integration Manager

```bash
# Run the setup script
python setup_integration.py --config config/integrations.yaml.local --env production

# Or use with options
python setup_integration.py \
  --config config/integrations.yaml.local \
  --env production \
  --health-check \
  --verbose
```

## 🧪 Testing

### Run Health Checks

```bash
# Check all integrations
python run_tests.py --health-check --config config/integrations.yaml.local

# Check specific integration
python run_tests.py --health-check --config config/integrations.yaml.local --tests langchain
```

### Run Integration Tests

```bash
# Run all tests
python run_tests.py --config config/integrations.yaml.local

# Run specific test suites
python run_tests.py --config config/integrations.yaml.local --tests langchain langfuse

# Run with detailed output
python run_tests.py --config config/integrations.yaml.local --verbose --report test_results.json
```

## 💻 Usage Examples

### Basic Integration Manager Usage

```python
import asyncio
from integrations import IntegrationManager
from integrations.shared import ManufacturingContext

async def main():
    # Initialize the integration manager
    manager = IntegrationManager()
    await manager.initialize("config/integrations.yaml.local")

    # Get LangChain integration
    langchain = manager.get_integration("langchain")

    # Create manufacturing context
    context = ManufacturingContext(
        equipment_type="CNC_Milling",
        user_role="operator",
        facility_id="Plant_A"
    )

    # Process a manufacturing query
    response = await langchain.process_request({
        "query": "What are the safety procedures for HAAS VF-2?",
        "type": "safety_procedure",
        "context": context
    })

    print(response["response"])

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Usage with Multiple Integrations

```python
import asyncio
from integrations import IntegrationManager
from integrations.shared import ManufacturingContext

async def manufacturing_workflow():
    manager = IntegrationManager()
    await manager.initialize("config/integrations.yaml.local")

    # Get all integrations
    langchain = manager.get_integration("langchain")
    lobechat = manager.get_integration("lobechat")
    xagent = manager.get_integration("xagent")
    langfuse = manager.get_integration("langfuse")

    context = ManufacturingContext(
        equipment_type="CNC_Milling",
        user_role="quality_inspector",
        facility_id="Plant_B"
    )

    # Create trace for workflow
    trace_id = await langfuse.create_trace(
        name="quality_inspection_workflow",
        inputs={"operation": "first_article_inspection"},
        manufacturing_context=context
    )

    try:
        # Step 1: Generate quality inspection plan using LangChain
        plan_response = await langchain.process_request({
            "type": "quality_inspection",
            "query": "Generate first article inspection plan for aerospace component",
            "product_spec": "Aerospace_Bracket_001",
            "inspection_type": "first_article"
        })

        # Step 2: Coordinate with XAgent for task decomposition
        task_result = await xagent.decompose_task({
            "task": "Execute quality inspection workflow",
            "complexity": "medium",
            "steps": [
                "Review technical specifications",
                "Prepare measurement equipment",
                "Execute dimensional inspection",
                "Document results",
                "Generate inspection report"
            ]
        })

        # Step 3: Track quality metrics with LangFuse
        await langfuse.track_quality_event(
            inspection_type="first_article",
            result="pass",
            measurements={"length": 100.001, "width": 50.000, "height": 25.001},
            specifications={"length": {"min": 99.999, "max": 100.001}, "width": {"target": 50.000}},
            manufacturing_context=context
        )

        # Step 4: Create chat session in LobeChat for collaboration
        session_id = await lobechat.create_session()
        await lobechat.send_message(
            session_id=session_id,
            message="Quality inspection completed successfully. All dimensions within tolerance.",
            metadata={"trace_id": trace_id, "operation": "quality_control"}
        )

        return {
            "trace_id": trace_id,
            "inspection_plan": plan_response,
            "task_decomposition": task_result,
            "session_id": session_id,
            "status": "completed"
        }

    except Exception as e:
        # Track error in LangFuse
        await langfuse.create_span(
            trace_id=trace_id,
            name="workflow_error",
            span_type="error",
            inputs={"error": str(e)}
        )
        raise

if __name__ == "__main__":
    asyncio.run(manufacturing_workflow())
```

### Safety Event Monitoring

```python
import asyncio
from integrations import IntegrationManager
from integrations.shared import ManufacturingContext

async def monitor_safety_events():
    manager = IntegrationManager()
    await manager.initialize("config/integrations.yaml.local")

    langfuse = manager.get_integration("langfuse")
    xagent = manager.get_integration("xagent")

    context = ManufacturingContext(
        equipment_type="CNC_Turning",
        user_role="operator",
        facility_id="Plant_A"
    )

    # Track a safety event
    await langfuse.track_safety_event(
        event_type="emergency_stop",
        severity="high",
        description="Emergency stop activated during machining operation",
        equipment_type="CNC_Turning",
        action_taken="Equipment inspected and safety procedures reviewed",
        manufacturing_context=context
    )

    # Trigger safety agent to analyze the event
    safety_analysis = await xagent.execute_agent_task({
        "agent_type": "safety_agent",
        "task": "Analyze emergency stop event and provide recommendations",
        "event_data": {
            "equipment_type": "CNC_Turning",
            "severity": "high",
            "description": "Emergency stop during machining"
        }
    })

    return safety_analysis

if __name__ == "__main__":
    asyncio.run(monitor_safety_events())
```

## 📊 Monitoring and Analytics

### Get Manufacturing Dashboard

```python
import asyncio
from integrations import IntegrationManager

async def get_dashboard():
    manager = IntegrationManager()
    await manager.initialize("config/integrations.yaml.local")

    langfuse = manager.get_integration("langfuse")

    # Get dashboard for last 24 hours
    dashboard = await langfuse.get_manufacturing_dashboard(
        time_range="24h",
        equipment_type="CNC_Milling",
        facility_id="Plant_A"
    )

    print("Manufacturing Dashboard:")
    print(f"AI Performance Metrics: {dashboard['metrics']['ai_performance']}")
    print(f"Safety Compliance: {dashboard['metrics']['safety_compliance']}")
    print(f"Quality Control: {dashboard['metrics']['quality_control']}")
    print(f"Active Alerts: {len(dashboard['alerts'])}")

    return dashboard

if __name__ == "__main__":
    asyncio.run(get_dashboard())
```

### Generate Compliance Report

```python
import asyncio
from datetime import datetime, timedelta
from integrations import IntegrationManager

async def generate_compliance_report():
    manager = IntegrationManager()
    await manager.initialize("config/integrations.yaml.local")

    langfuse = manager.get_integration("langfuse")

    # Generate ISO 9001 compliance report for last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    report = await langfuse.export_compliance_report(
        standard="ISO_9001",
        start_date=start_date,
        end_date=end_date,
        format="json"
    )

    print(f"Compliance Report Generated:")
    print(f"Standard: {report['standard']}")
    print(f"Period: {report['period']['start_date']} to {report['period']['end_date']}")
    print(f"Overall Score: {report['overall_score']:.1f}%")
    print(f"Requirements Checked: {len(report['compliance_metrics'])}")

    return report

if __name__ == "__main__":
    asyncio.run(generate_compliance_report())
```

## 🔧 Configuration

### Environment-Specific Configurations

```yaml
# config/integrations.yaml.local - Development
development:
  debug_mode: true
  log_level: "DEBUG"
  mock_external_apis: true

# config/integrations.yaml.local - Production
production:
  debug_mode: false
  log_level: "INFO"
  require_auth: true
  rate_limiting: true
```

### Manufacturing-Specific Settings

```yaml
manufacturing:
  domain:
    industry: "manufacturing"
    sub_industry: "precision_machining"
    compliance_standards: ["ISO_9001", "AS9100", "OSHA"]

  user_roles: ["operator", "engineer", "quality_inspector", "safety_officer"]
  equipment_types: ["cnc_milling", "cnc_turning", "grinding", "measurement"]
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install langchain openai chromadb langfuse websockets
   ```

2. **Connection Errors**: Check environment variables
   ```bash
   echo $OPENAI_API_KEY
   echo $LANGFUSE_PUBLIC_KEY
   ```

3. **Configuration Errors**: Validate YAML syntax
   ```bash
   python -c "import yaml; yaml.safe_load(open('config/integrations.yaml'))"
   ```

### Health Check

```bash
# Run comprehensive health check
python run_tests.py --health-check --verbose
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Next Steps

1. **Review the Documentation**: Read `INTEGRATION_SUMMARY.md` for detailed features
2. **Run the Tests**: Execute `python run_tests.py` to verify all integrations work
3. **Explore Examples**: Check the `examples/` directory for more use cases
4. **Customize Configuration**: Modify `config/integrations.yaml` for your specific needs
5. **Monitor Performance**: Use the LangFuse dashboard to track AI performance and costs

## 🆘 Support

For issues and questions:
1. Check the health check output for specific errors
2. Review the integration logs in `tests/integration_tests.log`
3. Consult the configuration examples in `config/integrations.yaml`
4. Run individual integration tests to isolate issues

---

**Happy Manufacturing! 🏭**

This integration framework provides a solid foundation for building advanced manufacturing knowledge systems with modern AI capabilities.