"""
Observability system integration script.

This script provides a complete integration point for all observability components
including logging, metrics, analytics, health monitoring, and alerting.
"""

import os
import sys
import time
import asyncio
import threading
from datetime import datetime
from typing import Dict, Any, Optional

# Add the observability module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from core.logging import get_logger, configure_logging, log_ai_interaction
from core.metrics import get_metrics_collector, configure_metrics_collector
from analytics.cost_analyzer import get_cost_analyzer, configure_cost_analyzer
from analytics.user_analytics import get_user_analytics, configure_user_analytics
from monitoring.health_checker import get_health_checker, configure_health_checker
from monitoring.alerting import get_alert_manager, configure_alert_manager
from dashboard.metrics_exporter import get_prometheus_exporter, start_metrics_server


class ObservabilitySystem:
    """
    Main observability system coordinator.

    Integrates all observability components and provides a unified interface
    for monitoring the knowledge base application.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = None
        self.metrics_collector = None
        self.cost_analyzer = None
        self.user_analytics = None
        self.health_checker = None
        self.alert_manager = None
        self.prometheus_exporter = None

        self._initialized = False
        self._start_time = datetime.now()

    def initialize(self):
        """Initialize all observability components"""
        if self._initialized:
            return

        print("🚀 Initializing Knowledge Base Observability System...")

        # Configure logging
        print("📋 Setting up structured logging...")
        configure_logging(**self.config.get("logging", {}))
        self.logger = get_logger()

        # Configure metrics collector
        print("📊 Initializing metrics collector...")
        configure_metrics_collector(**self.config.get("metrics", {}))
        self.metrics_collector = get_metrics_collector()

        # Configure cost analyzer
        print("💰 Setting up cost analysis...")
        configure_cost_analyzer(**self.config.get("cost_analyzer", {}))
        self.cost_analyzer = get_cost_analyzer()

        # Configure user analytics
        print("👥 Initializing user analytics...")
        configure_user_analytics(**self.config.get("user_analytics", {}))
        self.user_analytics = get_user_analytics()

        # Configure health checker
        print("🏥 Setting up health monitoring...")
        configure_health_checker(**self.config.get("health_checker", {}))
        self.health_checker = get_health_checker()

        # Configure alert manager
        print("🚨 Initializing alerting system...")
        configure_alert_manager(**self.config.get("alerting", {}))
        self.alert_manager = get_alert_manager()

        # Configure Prometheus exporter
        print("📈 Setting up Prometheus metrics export...")
        self.prometheus_exporter = get_prometheus_exporter()

        # Start metrics server
        metrics_port = self.config.get("metrics_port", 9090)
        print(f"🌐 Starting metrics server on port {metrics_port}...")

        def start_server():
            try:
                start_metrics_server(metrics_port)
            except Exception as e:
                self.logger.error(f"Failed to start metrics server: {e}")

        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()

        self._initialized = True
        print("✅ Observability system initialized successfully!")

        # Log system event
        self.log_system_event(
            event_type="observability_initialized",
            message="Observability system successfully initialized",
            metadata={
                "components": [
                    "logging", "metrics", "cost_analysis",
                    "user_analytics", "health_monitoring",
                    "alerting", "prometheus_export"
                ],
                "start_time": self._start_time.isoformat()
            }
        )

    def log_ai_interaction(self, **kwargs):
        """Log AI interaction"""
        if not self._initialized:
            self.initialize()
        log_ai_interaction(**kwargs)

    def track_metric(self, metric_name: str, value: float, **labels):
        """Track a metric"""
        if not self._initialized:
            self.initialize()
        self.metrics_collector.set_gauge(metric_name, value, labels=labels)

    def track_cost(self, **kwargs):
        """Track cost data"""
        if not self._initialized:
            self.initialize()
        return self.cost_analyzer.calculate_cost(**kwargs)

    def track_user_behavior(self, **kwargs):
        """Track user behavior"""
        if not self._initialized:
            self.initialize()
        self.user_analytics.track_behavior(**kwargs)

    def check_health(self):
        """Run health checks"""
        if not self._initialized:
            self.initialize()
        return self.health_checker.run_all_health_checks()

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        if not self._initialized:
            self.initialize()

        # Get health status
        health_status = self.health_checker.get_system_health()

        # Get metrics summary
        metrics_summary = self.metrics_collector.get_metrics_summary()

        # Get cost summary
        cost_summary = self.cost_analyzer.get_cost_summary()

        # Get user analytics summary
        user_summary = self.user_analytics.get_analytics_summary()

        # Get alert summary
        alert_summary = self.alert_manager.get_alert_summary()

        return {
            "system": {
                "status": health_status.status.value if health_status else "unknown",
                "uptime_seconds": (datetime.now() - self._start_time).total_seconds(),
                "health_score": health_status.overall_score if health_status else 0,
                "components": len(health_status.components) if health_status else 0
            },
            "metrics": metrics_summary,
            "costs": cost_summary,
            "users": user_summary,
            "alerts": alert_summary,
            "observability": {
                "initialized": self._initialized,
                "start_time": self._start_time.isoformat(),
                "components": [
                    "logging", "metrics", "cost_analysis",
                    "user_analytics", "health_monitoring",
                    "alerting", "prometheus_export"
                ]
            }
        }

    def log_system_event(self, event_type: str, message: str, **metadata):
        """Log a system event"""
        if not self._initialized:
            self.initialize()

        self.logger.info(f"System Event: {message}", extra={
            "event_type": event_type,
            "metadata": metadata
        })

    def create_custom_alert_rule(self, **kwargs):
        """Create a custom alert rule"""
        if not self._initialized:
            self.initialize()
        return self.alert_manager.create_alert_rule(**kwargs)

    def get_metrics_export(self) -> str:
        """Get Prometheus metrics export"""
        if not self._initialized:
            self.initialize()
        return self.prometheus_exporter.export_metrics()

    def shutdown(self):
        """Shutdown observability system"""
        if not self._initialized:
            return

        print("🛑 Shutting down observability system...")

        # Log shutdown event
        self.log_system_event(
            event_type="observability_shutdown",
            message="Observability system shutting down",
            metadata={
                "uptime_seconds": (datetime.now() - self._start_time).total_seconds()
            }
        )

        # Cleanup components
        if self.metrics_collector:
            self.metrics_collector.cleanup()

        print("✅ Observability system shutdown complete")


# Global observability system instance
_global_observability: Optional[ObservabilitySystem] = None


def get_observability_system(config: Optional[Dict[str, Any]] = None) -> ObservabilitySystem:
    """Get global observability system instance"""
    global _global_observability
    if _global_observability is None:
        _global_observability = ObservabilitySystem(config)
    return _global_observability


def configure_observability(config: Dict[str, Any]) -> ObservabilitySystem:
    """Configure global observability system"""
    global _global_observability
    _global_observability = ObservabilitySystem(config)
    _global_observability.initialize()
    return _global_observability


# Example configuration
DEFAULT_CONFIG = {
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
    "user_analytics": {
        "session_timeout_minutes": 30,
        "retention_days": 30
    },
    "health_checker": {
        "check_interval": 30.0,
        "max_concurrent_checks": 10
    },
    "alerting": {
        "max_active_alerts": 1000,
        "alert_retention_days": 30
    },
    "metrics_port": 9090
}


def demo_observability():
    """Demonstrate observability system functionality"""
    print("🎯 Running observability system demo...")

    # Initialize system
    obs = get_observability_system(DEFAULT_CONFIG)
    obs.initialize()

    # Simulate some AI interactions
    print("🤖 Simulating AI interactions...")
    for i in range(5):
        obs.log_ai_interaction(
            interaction_type="search",
            query=f"Sample query {i}",
            response_length=150 + i * 10,
            model_used="gpt-4",
            tokens_used={"input": 20, "output": 30 + i * 5},
            cost_estimate=0.01 + i * 0.002,
            confidence_score=0.85 + i * 0.02,
            results_count=5 + i,
            duration_ms=200 + i * 50
        )
        time.sleep(0.5)

    # Track some metrics
    print("📊 Tracking custom metrics...")
    obs.track_metric("demo_metric", 42.5, label="demo")
    obs.track_metric("demo_metric", 38.7, label="test")

    # Track user behavior
    print("👥 Tracking user behavior...")
    obs.track_user_behavior(
        user_id="demo_user",
        session_id="demo_session",
        action_type="search",
        resource_id="demo_resource"
    )

    # Run health check
    print("🏥 Running health checks...")
    health = obs.check_health()
    print(f"   System health: {health.status.value} (score: {health.overall_score:.1f})")

    # Get system status
    print("📋 Getting system status...")
    status = obs.get_system_status()
    print(f"   System status: {status['system']['status']}")
    print(f"   Uptime: {status['system']['uptime_seconds']:.1f}s")
    print(f"   Components: {status['system']['components']}")

    # Get metrics export
    print("📈 Getting Prometheus metrics...")
    metrics = obs.get_metrics_export()
    print(f"   Metrics export size: {len(metrics)} characters")

    print("✅ Demo completed!")
    print(f"🌐 Access metrics at: http://localhost:9090/metrics")


if __name__ == "__main__":
    # Run demo if script is executed directly
    demo_observability()