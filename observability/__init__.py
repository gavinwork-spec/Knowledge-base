"""
Knowledge Base Observability System

A comprehensive observability system inspired by Langfuse for monitoring AI interactions,
performance metrics, cost analysis, user behavior analytics, and system health.

Features:
- 📊 Structured logging for all AI interactions
- 🚀 Performance metrics tracking with Prometheus
- 💰 Cost analysis for API calls and token usage
- 👥 User behavior analytics and session tracking
- 🏥 System health monitoring and alerting
- 📈 Real-time dashboards with Grafana
- 🔔 Intelligent alerting for system anomalies
"""

__version__ = "1.0.0"
__author__ = "Knowledge Base Team"

from .core.logging import (
    ObservabilityLogger,
    LogEvent,
    LogLevel,
    log_ai_interaction,
    log_performance_metric,
    log_user_action,
    log_system_event
)

from .core.metrics import (
    MetricsCollector,
    MetricType,
    PerformanceMetrics,
    CostMetrics,
    UserMetrics,
    SystemMetrics
)

from .core.tracing import (
    TracingManager,
    TraceContext,
    Span,
    trace_function,
    trace_async_function
)

from .analytics.user_analytics import (
    UserAnalyticsManager,
    UserSession,
    UserBehavior,
    UserProfile
)

from .analytics.cost_analyzer import (
    CostAnalyzer,
    CostCalculation,
    TokenUsage,
    APIUsageCost
)

from .monitoring.health_checker import (
    HealthChecker,
    HealthStatus,
    ComponentHealth,
    SystemHealth
)

from .monitoring.alerting import (
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertChannel,
    Alert
)

from .dashboard.metrics_exporter import (
    PrometheusExporter,
    GrafanaDashboardManager,
    DashboardConfig
)

__all__ = [
    # Core logging
    "ObservabilityLogger",
    "LogEvent",
    "LogLevel",
    "log_ai_interaction",
    "log_performance_metric",
    "log_user_action",
    "log_system_event",

    # Metrics
    "MetricsCollector",
    "MetricType",
    "PerformanceMetrics",
    "CostMetrics",
    "UserMetrics",
    "SystemMetrics",

    # Tracing
    "TracingManager",
    "TraceContext",
    "Span",
    "trace_function",
    "trace_async_function",

    # Analytics
    "UserAnalyticsManager",
    "UserSession",
    "UserBehavior",
    "UserProfile",
    "CostAnalyzer",
    "CostCalculation",
    "TokenUsage",
    "APIUsageCost",

    # Monitoring
    "HealthChecker",
    "HealthStatus",
    "ComponentHealth",
    "SystemHealth",
    "AlertManager",
    "AlertRule",
    "AlertSeverity",
    "AlertChannel",
    "Alert",

    # Dashboards
    "PrometheusExporter",
    "GrafanaDashboardManager",
    "DashboardConfig",
]