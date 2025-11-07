"""
Prometheus metrics exporter for observability.

Exports collected metrics to Prometheus format with proper labels,
metric types, and help text for comprehensive monitoring.
"""

import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from prometheus_client import (
    CollectorRegistry,
    Gauge,
    Counter,
    Histogram,
    Summary,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from ..core.metrics import get_metrics_collector, MetricType
from ..core.logging import get_logger
from ..analytics.cost_analyzer import get_cost_analyzer
from ..analytics.user_analytics import get_user_analytics


class PrometheusExporter:
    """
    Exports metrics to Prometheus format with comprehensive coverage
    of system performance, costs, user behavior, and business metrics.
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.logger = get_logger()
        self.metrics_collector = get_metrics_collector()
        self.cost_analyzer = get_cost_analyzer()
        self.user_analytics = get_user_analytics()

        # Use custom registry or default
        self.registry = registry or CollectorRegistry()

        # Initialize Prometheus metrics
        self._initialize_metrics()

        # Background update thread
        self._update_thread = threading.Thread(target=self._background_updates, daemon=True)
        self._update_thread.start()

    def _initialize_metrics(self):
        """Initialize all Prometheus metrics"""
        # System Metrics
        self.system_uptime = Gauge(
            'kb_system_uptime_seconds',
            'System uptime in seconds',
            ['service', 'version', 'environment'],
            registry=self.registry
        )

        self.system_health_score = Gauge(
            'kb_system_health_score',
            'Overall system health score (0-100)',
            ['service'],
            registry=self.registry
        )

        self.system_component_status = Gauge(
            'kb_system_component_status',
            'Component status (1=healthy, 0=unhealthy)',
            ['component', 'service'],
            registry=self.registry
        )

        # Performance Metrics
        self.response_time = Histogram(
            'kb_request_duration_seconds',
            'Request duration in seconds',
            ['endpoint', 'method', 'service'],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )

        self.request_rate = Gauge(
            'kb_requests_per_second',
            'Current request rate',
            ['endpoint', 'service'],
            registry=self.registry
        )

        self.error_rate = Gauge(
            'kb_error_rate',
            'Current error rate (0-1)',
            ['endpoint', 'error_type', 'service'],
            registry=self.registry
        )

        self.active_connections = Gauge(
            'kb_active_connections',
            'Number of active connections',
            ['connection_type', 'service'],
            registry=self.registry
        )

        # Resource Usage Metrics
        self.cpu_usage = Gauge(
            'kb_cpu_usage_percent',
            'CPU usage percentage',
            ['service', 'instance'],
            registry=self.registry
        )

        self.memory_usage = Gauge(
            'kb_memory_usage_bytes',
            'Memory usage in bytes',
            ['service', 'instance', 'type'],
            registry=self.registry
        )

        self.disk_usage = Gauge(
            'kb_disk_usage_bytes',
            'Disk usage in bytes',
            ['service', 'instance', 'mount_point'],
            registry=self.registry
        )

        # AI/ML Metrics
        self.ai_requests_total = Counter(
            'kb_ai_requests_total',
            'Total AI requests',
            ['model', 'provider', 'interaction_type', 'success'],
            registry=self.registry
        )

        self.ai_response_time = Histogram(
            'kb_ai_response_time_seconds',
            'AI model response time in seconds',
            ['model', 'provider', 'interaction_type'],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
            registry=self.registry
        )

        self.ai_tokens_total = Counter(
            'kb_ai_tokens_total',
            'Total AI tokens processed',
            ['model', 'provider', 'token_type', 'direction'],
            registry=self.registry
        )

        self.ai_cost_total = Gauge(
            'kb_ai_cost_total_usd',
            'Total AI cost in USD',
            ['model', 'provider', 'cost_center', 'period'],
            registry=self.registry
        )

        # Search Metrics
        self.search_requests_total = Counter(
            'kb_search_requests_total',
            'Total search requests',
            ['search_type', 'strategy', 'success'],
            registry=self.registry
        )

        self.search_results_count = Histogram(
            'kb_search_results_count',
            'Number of search results returned',
            ['search_type', 'strategy'],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
            registry=self.registry
        )

        self.search_relevance_score = Histogram(
            'kb_search_relevance_score',
            'Search result relevance scores',
            ['search_type', 'strategy'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )

        # User Metrics
        self.active_users_total = Gauge(
            'kb_active_users_total',
            'Number of active users',
            ['period', 'segment'],
            registry=self.registry
        )

        self.user_sessions_total = Counter(
            'kb_user_sessions_total',
            'Total user sessions',
            ['status', 'segment', 'is_first_session'],
            registry=self.registry
        )

        self.user_engagement_score = Histogram(
            'kb_user_engagement_score',
            'User engagement scores',
            ['segment'],
            buckets=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            registry=self.registry
        )

        self.user_satisfaction_score = Histogram(
            'kb_user_satisfaction_score',
            'User satisfaction scores',
            ['segment'],
            buckets=[1, 2, 3, 4, 5],
            registry=self.registry
        )

        # Document Metrics
        self.documents_total = Gauge(
            'kb_documents_total',
            'Total number of documents',
            ['status', 'type', 'source'],
            registry=self.registry
        )

        self.document_processing_time = Histogram(
            'kb_document_processing_time_seconds',
            'Document processing time in seconds',
            ['operation', 'document_type'],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0],
            registry=self.registry
        )

        self.document_size_bytes = Histogram(
            'kb_document_size_bytes',
            'Document sizes in bytes',
            ['document_type'],
            buckets=[1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216],
            registry=self.registry
        )

        # Business Metrics
        self.conversion_events_total = Counter(
            'kb_conversion_events_total',
            'Total conversion events',
            ['event_type', 'segment'],
            registry=self.registry
        )

        self.daily_active_users = Gauge(
            'kb_daily_active_users',
            'Daily active users',
            registry=self.registry
        )

        self.monthly_active_users = Gauge(
            'kb_monthly_active_users',
            'Monthly active users',
            registry=self.registry
        )

        # Cost and Budget Metrics
        self.daily_cost_budget_utilization = Gauge(
            'kb_daily_cost_budget_utilization',
            'Daily cost budget utilization (0-1)',
            ['cost_center'],
            registry=self.registry
        )

        self.cost_per_query = Histogram(
            'kb_cost_per_query_usd',
            'Cost per query in USD',
            ['cost_center', 'model'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry
        )

    def update_system_metrics(self, metrics: Dict[str, Any]):
        """Update system-related metrics"""
        if 'uptime_seconds' in metrics:
            self.system_uptime.labels(
                service='knowledge-base',
                version='1.0.0',
                environment='production'
            ).set(metrics['uptime_seconds'])

        if 'health_score' in metrics:
            self.system_health_score.labels(
                service='knowledge-base'
            ).set(metrics['health_score'])

        if 'component_status' in metrics:
            for component, status in metrics['component_status'].items():
                self.system_component_status.labels(
                    component=component,
                    service='knowledge-base'
                ).set(1.0 if status else 0.0)

        if 'cpu_usage' in metrics:
            self.cpu_usage.labels(
                service='knowledge-base',
                instance='main'
            ).set(metrics['cpu_usage'])

        if 'memory_usage' in metrics:
            self.memory_usage.labels(
                service='knowledge-base',
                instance='main',
                type='used'
            ).set(metrics['memory_usage'])

    def update_performance_metrics(self, metrics: Dict[str, Any]):
        """Update performance-related metrics"""
        if 'throughput_rps' in metrics:
            self.request_rate.labels(
                endpoint='all',
                service='knowledge-base'
            ).set(metrics['throughput_rps'])

        if 'error_rate' in metrics:
            self.error_rate.labels(
                endpoint='all',
                error_type='all',
                service='knowledge-base'
            ).set(metrics['error_rate'])

    def update_ai_metrics(self, metrics: Dict[str, Any]):
        """Update AI/ML related metrics"""
        # These would be updated by the AI interaction tracking system
        pass

    def update_user_metrics(self, metrics: Dict[str, Any]):
        """Update user-related metrics"""
        if 'active_users_24h' in metrics:
            self.active_users_total.labels(
                period='24h',
                segment='all'
            ).set(metrics['active_users_24h'])

        if 'active_users_7d' in metrics:
            self.active_users_total.labels(
                period='7d',
                segment='all'
            ).set(metrics['active_users_7d'])

        if 'active_users_30d' in metrics:
            self.active_users_total.labels(
                period='30d',
                segment='all'
            ).set(metrics['active_users_30d'])

        if 'user_satisfaction_score' in metrics:
            # This would come from user feedback
            pass

    def update_cost_metrics(self, metrics: Dict[str, Any]):
        """Update cost-related metrics"""
        budget_status = self.cost_analyzer.get_budget_status()

        for cost_center, status in budget_status.items():
            self.daily_cost_budget_utilization.labels(
                cost_center=cost_center
            ).set(status['daily_utilization'])

    def _background_updates(self):
        """Background thread to update metrics from collectors"""
        while True:
            try:
                time.sleep(30)  # Update every 30 seconds

                # Update system metrics
                system_metrics = self.metrics_collector.system_metrics
                if system_metrics:
                    self.update_system_metrics(system_metrics)

                # Update performance metrics
                performance_metrics = self.metrics_collector.performance_metrics
                if performance_metrics:
                    self.update_performance_metrics(performance_metrics)

                # Update user metrics
                user_metrics = self.metrics_collector.user_metrics
                if user_metrics:
                    self.update_user_metrics(user_metrics)

                # Update cost metrics
                cost_metrics = self.metrics_collector.cost_metrics
                if cost_metrics:
                    self.update_cost_metrics(cost_metrics)

            except Exception as e:
                self.logger.error(f"Error updating Prometheus metrics: {e}")

    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')

    def get_content_type(self) -> str:
        """Get the content type for the metrics export"""
        return CONTENT_TYPE_LATEST


# Global Prometheus exporter instance
_global_exporter: Optional[PrometheusExporter] = None


def get_prometheus_exporter() -> PrometheusExporter:
    """Get global Prometheus exporter instance"""
    global _global_exporter
    if _global_exporter is None:
        _global_exporter = PrometheusExporter()
    return _global_exporter


def configure_prometheus_exporter(**kwargs) -> PrometheusExporter:
    """Configure global Prometheus exporter"""
    global _global_exporter
    _global_exporter = PrometheusExporter(**kwargs)
    return _global_exporter


class MetricsHTTPServer:
    """
    Simple HTTP server to expose Prometheus metrics endpoint.
    """

    def __init__(self, exporter: PrometheusExporter, port: int = 9090):
        self.exporter = exporter
        self.port = port
        self.logger = get_logger()

    def start(self):
        """Start the HTTP server"""
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler

            class MetricsHandler(BaseHTTPRequestHandler):
                def __init__(self, exporter):
                    self.exporter = exporter
                    super().__init__()

                def do_GET(self):
                    if self.path == '/metrics':
                        self.send_response(200)
                        self.send_header('Content-Type', self.exporter.get_content_type())
                        self.end_headers()
                        self.wfile.write(self.exporter.export_metrics().encode('utf-8'))
                    else:
                        self.send_response(404)
                        self.end_headers()

                def log_message(self, format, *args):
                    # Suppress default logging
                    pass

            # Create handler with exporter
            handler = type('MetricsHandler', (MetricsHandler,), {'__init__': lambda self: MetricsHandler.__init__(self, exporter)})

            server = HTTPServer(('0.0.0.0', self.port), handler)
            self.logger.info(f"Prometheus metrics server started on port {self.port}")
            server.serve_forever()

        except ImportError:
            self.logger.error("http.server not available - cannot start metrics server")
            raise
        except Exception as e:
            self.logger.error(f"Failed to start metrics server: {e}")
            raise


def start_metrics_server(port: int = 9090) -> None:
    """Start the Prometheus metrics HTTP server"""
    exporter = get_prometheus_exporter()
    server = MetricsHTTPServer(exporter, port)
    server.start()