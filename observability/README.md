# Knowledge Base Observability System

A comprehensive observability system inspired by Langfuse, providing detailed logging, metrics tracking, cost analysis, user behavior analytics, system health monitoring, and intelligent alerting for knowledge base applications.

## Features

### 🔍 **Comprehensive Logging**
- Structured logging with trace context
- AI interaction tracking with detailed metadata
- Performance metrics logging
- User behavior logging
- System event logging
- Thread-local context management

### 📊 **Metrics Collection**
- Real-time metrics collection
- Counter, Gauge, Histogram, and Summary metrics
- Thread-safe operations
- Background aggregation and cleanup
- Prometheus integration
- Custom metric definitions

### 💰 **Cost Analysis**
- AI/ML operation cost tracking
- Model pricing configuration
- Budget management and monitoring
- Cost optimization recommendations
- Multi-cost-center support
- Token usage analysis

### 👥 **User Analytics**
- User behavior tracking
- Session management
- Engagement scoring
- User segmentation
- Performance metrics
- Satisfaction tracking

### 🏥 **Health Monitoring**
- Component health checking
- System health aggregation
- Health trend analysis
- Automated recommendations
- Concurrent health checks
- Custom health check support

### 🚨 **Intelligent Alerting**
- Rule-based alerting
- Multiple notification channels (Email, Slack, Webhook)
- Alert escalation and suppression
- Rate limiting and cooldowns
- Alert acknowledgment
- Historical alert tracking

### 📈 **Visualization & Monitoring**
- Prometheus metrics export
- Grafana dashboard configuration
- Real-time monitoring
- Historical data analysis
- Custom dashboards
- Alert management UI

## Architecture

```
observability/
├── core/                   # Core observability components
│   ├── logging.py         # Structured logging system
│   └── metrics.py         # Metrics collection
├── analytics/             # Analytics and analysis
│   ├── cost_analyzer.py   # Cost analysis
│   └── user_analytics.py  # User behavior analytics
├── monitoring/            # Monitoring and health checks
│   ├── health_checker.py  # System health monitoring
│   └── alerting.py        # Alerting system
├── dashboard/             # Dashboards and visualization
│   ├── metrics_exporter.py # Prometheus export
│   └── grafana_dashboard.json # Grafana configuration
├── prometheus/           # Prometheus configuration
│   ├── prometheus.yml    # Prometheus config
│   └── alert_rules.yml   # Alert rules
├── integration.py        # System integration
└── README.md            # This file
```

## Quick Start

### Installation

```bash
# Install required dependencies
pip install prometheus-client pydantic psutil

# Navigate to the observability directory
cd observability
```

### Basic Usage

```python
from observability import get_observability_system

# Initialize the observability system
obs = get_observability_system()
obs.initialize()

# Log an AI interaction
obs.log_ai_interaction(
    interaction_type="search",
    query="What is machine learning?",
    response_length=250,
    model_used="gpt-4",
    tokens_used={"input": 15, "output": 40},
    cost_estimate=0.025,
    confidence_score=0.92,
    results_count=5,
    duration_ms=350
)

# Track custom metrics
obs.track_metric("custom_metric", 42.5, label="demo")

# Get system status
status = obs.get_system_status()
print(f"System health: {status['system']['status']}")
```

### Running the Demo

```bash
python integration.py
```

This will:
- Initialize all observability components
- Simulate AI interactions
- Track metrics and user behavior
- Run health checks
- Start the Prometheus metrics server

Access metrics at: http://localhost:9090/metrics

## Configuration

### System Configuration

```python
from observability import configure_observability

config = {
    "logging": {
        "level": "INFO",
        "format": "structured"
    },
    "metrics": {
        "retention_hours": 24,
        "aggregation_interval": 60
    },
    "cost_analyzer": {
        "budget_warning_threshold": 0.8,
        "budget_critical_threshold": 0.95
    },
    "health_checker": {
        "check_interval": 30.0,
        "max_concurrent_checks": 10
    },
    "metrics_port": 9090
}

obs = configure_observability(config)
```

### Prometheus Setup

1. **Configure Prometheus** (prometheus/prometheus.yml):
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'knowledge-base'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: /metrics
```

2. **Start Prometheus**:
```bash
prometheus --config.file=prometheus/prometheus.yml
```

3. **Access Prometheus**: http://localhost:9090

### Grafana Setup

1. **Import Dashboard**:
   - Go to Grafana UI
   - Import `dashboard/grafana_dashboard.json`
   - Select Prometheus data source

2. **View Dashboards**:
   - System Overview
   - Performance Metrics
   - User Analytics
   - Cost Analysis
   - Health Monitoring

## API Reference

### Logging

```python
from observability.core.logging import log_ai_interaction

# Log AI interactions
log_ai_interaction(
    interaction_type="search",  # search, chat, completion
    query="User query",
    response_length=150,
    model_used="gpt-4",
    tokens_used={"input": 20, "output": 30},
    cost_estimate=0.01,
    confidence_score=0.85,
    results_count=5,
    duration_ms=200
)
```

### Metrics

```python
from observability.core.metrics import get_metrics_collector

metrics = get_metrics_collector()

# Track different metric types
metrics.increment_counter("requests_total", labels={"endpoint": "/search"})
metrics.set_gauge("active_users", 42)
metrics.record_histogram("response_time", 0.25, labels={"method": "GET"})
metrics.record_summary("processing_time", 150.5)
```

### Cost Analysis

```python
from observability.analytics.cost_analyzer import get_cost_analyzer

cost_analyzer = get_cost_analyzer()

# Calculate costs
cost = cost_analyzer.calculate_cost(
    token_usage={"input": 100, "output": 200},
    interaction_type="search",
    cost_center="ai_operations",
    user_id="user123"
)

# Get budget status
budget_status = cost_analyzer.get_budget_status()
```

### User Analytics

```python
from observability.analytics.user_analytics import get_user_analytics

user_analytics = get_user_analytics()

# Track user behavior
user_analytics.track_behavior(
    user_id="user123",
    session_id="session456",
    action_type="search",
    resource_id="doc789"
)

# Get user profile
profile = user_analytics.get_user_profile("user123")
```

### Health Monitoring

```python
from observability.monitoring.health_checker import get_health_checker

health_checker = get_health_checker()

# Run health checks
system_health = health_checker.run_all_health_checks()

# Get health summary
summary = health_checker.get_health_summary()
```

### Alerting

```python
from observability.monitoring.alerting import get_alert_manager

alert_manager = get_alert_manager()

# Create alert rule
rule = alert_manager.create_alert_rule(
    name="High Error Rate",
    description="Error rate is above threshold",
    metric_name="error_rate",
    threshold=0.1,
    severity="warning"
)

# Trigger manual alert
alert_manager.trigger_alert(
    rule_name="High Error Rate",
    title="High Error Rate Detected",
    description="Error rate is 15%"
)
```

## Metrics Reference

### System Metrics
- `kb_system_uptime_seconds` - System uptime
- `kb_system_health_score` - Overall health score (0-100)
- `kb_system_component_status` - Component health status

### Performance Metrics
- `kb_request_duration_seconds` - Request duration histogram
- `kb_requests_per_second` - Request rate gauge
- `kb_error_rate` - Error rate gauge
- `kb_active_connections` - Active connections gauge

### AI/ML Metrics
- `kb_ai_requests_total` - Total AI requests counter
- `kb_ai_response_time_seconds` - AI response time histogram
- `kb_ai_tokens_total` - Total tokens processed counter
- `kb_ai_cost_total_usd` - Total AI cost gauge

### User Metrics
- `kb_active_users_total` - Active users gauge
- `kb_user_sessions_total` - User sessions counter
- `kb_user_engagement_score` - Engagement score histogram
- `kb_user_satisfaction_score` - Satisfaction score histogram

### Cost Metrics
- `kb_daily_cost_budget_utilization` - Budget utilization gauge
- `kb_cost_per_query_usd` - Cost per query histogram

## Alert Rules

The system includes comprehensive alert rules for:

- **System Health**: Low health scores, component failures
- **Performance**: High response times, error rates
- **Resource Usage**: CPU, memory, disk usage
- **AI Services**: Response times, costs
- **Search Metrics**: Response times, relevance scores
- **User Metrics**: Satisfaction, engagement
- **Budget Monitoring**: Cost thresholds
- **Availability**: Service downtime, request rates

## Integration with Applications

### FastAPI Integration

```python
from fastapi import FastAPI
from observability import get_observability_system

app = FastAPI()
obs = get_observability_system()
obs.initialize()

@app.middleware("http")
async def observability_middleware(request, call_next):
    start_time = time.time()

    response = await call_next(request)

    # Log request
    obs.log_ai_interaction(
        interaction_type="api_request",
        query=str(request.url),
        response_length=len(response.body),
        duration_ms=(time.time() - start_time) * 1000,
        metadata={
            "method": request.method,
            "status_code": response.status_code
        }
    )

    return response
```

### Flask Integration

```python
from flask import Flask
from observability import get_observability_system

app = Flask(__name__)
obs = get_observability_system()
obs.initialize()

@app.before_request
def before_request():
    g.start_time = time.time()

@app.after_request
def after_request(response):
    duration = (time.time() - g.start_time) * 1000

    obs.track_metric(
        "request_duration",
        duration,
        labels={"endpoint": request.endpoint}
    )

    return response
```

## Troubleshooting

### Common Issues

1. **Metrics server not starting**:
   - Check if port 9090 is available
   - Verify dependencies are installed

2. **Health checks failing**:
   - Check component configurations
   - Verify network connectivity

3. **Alerts not sending**:
   - Verify notification channel configurations
   - Check alert rule conditions

### Debug Mode

Enable debug logging:

```python
config = {
    "logging": {
        "level": "DEBUG",
        "format": "structured"
    }
}

obs = configure_observability(config)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## License

This observability system is part of the Knowledge Base project and follows the same licensing terms.

## Support

For questions and support:
- Check the documentation
- Review the demo code
- Examine the configuration examples
- Check system logs for errors