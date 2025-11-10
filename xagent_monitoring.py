#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAgent Monitoring and Observability System
XAgent监控和可观测性系统

This module provides comprehensive monitoring, metrics collection, and observability
for XAgent agents, including performance tracking, health monitoring, and analytics.
"""

import asyncio
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import statistics
import uuid
from pathlib import Path

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler('data/processed/xagent_monitoring.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent status enumeration"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    STARTING = "starting"
    STOPPING = "stopping"

class MetricType(Enum):
    """Metric types for monitoring"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    SUMMARY = "summary"

class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class AgentMetric:
    """Single agent metric data point"""
    agent_id: str
    metric_name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""

@dataclass
class AgentEvent:
    """Agent event for monitoring"""
    event_id: str
    agent_id: str
    event_type: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None

@dataclass
class PerformanceSnapshot:
    """Performance snapshot for an agent"""
    agent_id: str
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    task_count: int
    queue_size: int
    response_time_avg: float
    error_rate: float
    uptime: float

class MetricsCollector:
    """Collects and manages metrics for agents"""

    def __init__(self, retention_hours: int = 24, max_metrics_per_type: int = 10000):
        self.retention_hours = retention_hours
        self.max_metrics_per_type = max_metrics_per_type
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics_per_type))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()

    def record_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Record a counter metric"""
        with self.lock:
            key = self._make_key(name, tags)
            self.counters[key] += value

            metric = AgentMetric(
                agent_id=tags.get("agent_id", "system"),
                metric_name=name,
                metric_type=MetricType.COUNTER,
                value=self.counters[key],
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.metrics[key].append(metric)

    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric"""
        with self.lock:
            key = self._make_key(name, tags)
            self.gauges[key] = value

            metric = AgentMetric(
                agent_id=tags.get("agent_id", "system"),
                metric_name=name,
                metric_type=MetricType.GAUGE,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.metrics[key].append(metric)

    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram metric"""
        with self.lock:
            key = self._make_key(name, tags)
            self.histograms[key].append(value)

            # Keep only recent values
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]

            metric = AgentMetric(
                agent_id=tags.get("agent_id", "system"),
                metric_name=name,
                metric_type=MetricType.HISTOGRAM,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.metrics[key].append(metric)

    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timer metric"""
        with self.lock:
            key = self._make_key(name, tags)
            self.timers[key].append(duration)

            # Keep only recent values
            if len(self.timers[key]) > 1000:
                self.timers[key] = self.timers[key][-1000:]

            metric = AgentMetric(
                agent_id=tags.get("agent_id", "system"),
                metric_name=name,
                metric_type=MetricType.TIMER,
                value=duration,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.metrics[key].append(metric)

    def get_metric_summary(self, name: str, tags: Dict[str, str] = None,
                          time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        key = self._make_key(name, tags)

        if key not in self.metrics:
            return {}

        metrics_list = list(self.metrics[key])

        # Apply time window filter
        if time_window:
            cutoff_time = datetime.now() - time_window
            metrics_list = [m for m in metrics_list if m.timestamp > cutoff_time]

        if not metrics_list:
            return {}

        values = [m.value for m in metrics_list]

        summary = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": statistics.mean(values),
            "sum": sum(values),
            "latest": values[-1] if values else 0,
            "time_range": {
                "start": min(m.timestamp for m in metrics_list).isoformat(),
                "end": max(m.timestamp for m in metrics_list).isoformat()
            }
        }

        # Add statistics for numeric metrics
        if len(values) > 1:
            summary["median"] = statistics.median(values)
            summary["stdev"] = statistics.stdev(values) if len(values) > 2 else 0
            summary["p95"] = self._percentile(values, 95)
            summary["p99"] = self._percentile(values, 99)

        return summary

    def _make_key(self, name: str, tags: Dict[str, str] = None) -> str:
        """Create a unique key for metric with tags"""
        if not tags:
            return name

        tag_parts = [f"{k}={v}" for k, v in sorted(tags.items())]
        return f"{name}({','.join(tag_parts)})"

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)

        with self.lock:
            for key, metric_deque in self.metrics.items():
                # Filter old metrics
                filtered_metrics = deque(
                    (m for m in metric_deque if m.timestamp > cutoff_time),
                    maxlen=self.max_metrics_per_type
                )
                self.metrics[key] = filtered_metrics

class AlertManager:
    """Manages alerts and notifications"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_rules: Dict[str, Dict] = {}
        self.active_alerts: Dict[str, AgentEvent] = {}
        self.alert_history: List[AgentEvent] = []
        self.notification_handlers: List[Callable] = []
        self.max_history_size = config.get("max_history_size", 10000)

    def add_alert_rule(self, rule_id: str, condition: Dict[str, Any],
                      severity: AlertSeverity, message_template: str):
        """Add an alert rule"""
        self.alert_rules[rule_id] = {
            "condition": condition,
            "severity": severity,
            "message_template": message_template,
            "enabled": True,
            "created_at": datetime.now()
        }
        logger.info(f"✅ Added alert rule: {rule_id}")

    def evaluate_alerts(self, metrics_collector: MetricsCollector):
        """Evaluate all alert rules against current metrics"""
        for rule_id, rule in self.alert_rules.items():
            if not rule["enabled"]:
                continue

            try:
                if self._evaluate_rule(rule["condition"], metrics_collector):
                    self._trigger_alert(rule_id, rule)
                else:
                    self._resolve_alert(rule_id)

            except Exception as e:
                logger.error(f"❌ Error evaluating alert rule {rule_id}: {e}")

    def _evaluate_rule(self, condition: Dict[str, Any],
                      metrics_collector: MetricsCollector) -> bool:
        """Evaluate a single alert condition"""
        metric_name = condition["metric"]
        operator = condition["operator"]
        threshold = condition["threshold"]
        tags = condition.get("tags", {})
        time_window = condition.get("time_window")

        summary = metrics_collector.get_metric_summary(metric_name, tags, time_window)

        if not summary:
            return False

        value = summary.get("avg", summary.get("latest", 0))

        if operator == "gt":
            return value > threshold
        elif operator == "lt":
            return value < threshold
        elif operator == "eq":
            return abs(value - threshold) < 0.001
        elif operator == "gte":
            return value >= threshold
        elif operator == "lte":
            return value <= threshold
        else:
            logger.warning(f"Unknown alert operator: {operator}")
            return False

    def _trigger_alert(self, rule_id: str, rule: Dict[str, Any]):
        """Trigger an alert"""
        if rule_id in self.active_alerts:
            return  # Already active

        alert = AgentEvent(
            event_id=str(uuid.uuid4()),
            agent_id=rule["condition"].get("tags", {}).get("agent_id", "system"),
            event_type="alert_triggered",
            severity=rule["severity"],
            message=rule["message_template"].format(**rule["condition"]),
            timestamp=datetime.now(),
            metadata={"rule_id": rule_id, "condition": rule["condition"]}
        )

        self.active_alerts[rule_id] = alert
        self.alert_history.append(alert)

        # Trim history if too large
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size:]

        logger.warning(f"🚨 Alert triggered: {rule_id} - {alert.message}")

        # Send notifications
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"❌ Error in notification handler: {e}")

    def _resolve_alert(self, rule_id: str):
        """Resolve an alert"""
        if rule_id not in self.active_alerts:
            return

        alert = self.active_alerts[rule_id]
        resolution = AgentEvent(
            event_id=str(uuid.uuid4()),
            agent_id=alert.agent_id,
            event_type="alert_resolved",
            severity=AlertSeverity.INFO,
            message=f"Alert resolved: {alert.message}",
            timestamp=datetime.now(),
            metadata={"original_alert_id": alert.event_id, "rule_id": rule_id}
        )

        del self.active_alerts[rule_id]
        self.alert_history.append(resolution)

        logger.info(f"✅ Alert resolved: {rule_id}")

    def get_active_alerts(self) -> List[AgentEvent]:
        """Get all active alerts"""
        return list(self.active_alerts.values())

    def get_alert_history(self, time_window: Optional[timedelta] = None) -> List[AgentEvent]:
        """Get alert history with optional time filtering"""
        if not time_window:
            return self.alert_history

        cutoff_time = datetime.now() - time_window
        return [alert for alert in self.alert_history if alert.timestamp > cutoff_time]

class HealthMonitor:
    """Monitors agent health and availability"""

    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.agent_health: Dict[str, Dict] = {}
        self.health_checks: Dict[str, Callable] = {}
        self.running = False
        self.monitor_thread = None

    def register_agent(self, agent_id: str, health_check: Callable = None):
        """Register an agent for health monitoring"""
        self.agent_health[agent_id] = {
            "status": AgentStatus.STARTING,
            "last_seen": datetime.now(),
            "last_check": datetime.now(),
            "consecutive_failures": 0,
            "uptime": 0.0,
            "total_checks": 0,
            "successful_checks": 0
        }

        if health_check:
            self.health_checks[agent_id] = health_check

        logger.info(f"✅ Registered health monitoring for agent: {agent_id}")

    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agent_health:
            self.agent_health[agent_id]["status"] = AgentStatus.OFFLINE
            logger.info(f"📤 Unregistered health monitoring for agent: {agent_id}")

    def update_agent_status(self, agent_id: str, status: AgentStatus,
                           metadata: Dict[str, Any] = None):
        """Update agent status (called by agents themselves)"""
        if agent_id not in self.agent_health:
            self.register_agent(agent_id)

        health_data = self.agent_health[agent_id]
        health_data["status"] = status
        health_data["last_seen"] = datetime.now()
        health_data["consecutive_failures"] = 0

        if metadata:
            health_data.update(metadata)

    async def start_monitoring(self):
        """Start health monitoring loop"""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🚀 Health monitoring started")

    async def stop_monitoring(self):
        """Stop health monitoring loop"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("🛑 Health monitoring stopped")

    def _monitor_loop(self):
        """Health monitoring loop"""
        while self.running:
            try:
                self._check_agent_health()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"❌ Error in health monitoring loop: {e}")
                time.sleep(self.check_interval)

    def _check_agent_health(self):
        """Check health of all registered agents"""
        for agent_id, health_data in self.agent_health.items():
            try:
                if health_data["status"] == AgentStatus.OFFLINE:
                    continue

                # Perform health check
                if agent_id in self.health_checks:
                    is_healthy = self.health_checks[agent_id]()
                else:
                    # Default health check: check if agent reported recently
                    time_since_last_seen = datetime.now() - health_data["last_seen"]
                    is_healthy = time_since_last_seen < timedelta(minutes=5)

                health_data["last_check"] = datetime.now()
                health_data["total_checks"] += 1

                if is_healthy:
                    health_data["successful_checks"] += 1
                    health_data["consecutive_failures"] = 0

                    if health_data["status"] in [AgentStatus.ERROR, AgentStatus.OFFLINE]:
                        health_data["status"] = AgentStatus.IDLE
                else:
                    health_data["consecutive_failures"] += 1

                    if health_data["consecutive_failures"] >= 3:
                        health_data["status"] = AgentStatus.ERROR
                        logger.warning(f"⚠️ Agent {agent_id} health check failed: {health_data['consecutive_failures']} consecutive failures")

                # Update uptime calculation
                if health_data["total_checks"] > 0:
                    health_data["uptime"] = health_data["successful_checks"] / health_data["total_checks"]

            except Exception as e:
                logger.error(f"❌ Error checking health for agent {agent_id}: {e}")

    def get_agent_health(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get health status for a specific agent"""
        return self.agent_health.get(agent_id)

    def get_all_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all agents"""
        return {
            agent_id: {
                "status": health["status"].value,
                "last_seen": health["last_seen"].isoformat(),
                "uptime": health["uptime"],
                "consecutive_failures": health["consecutive_failures"],
                "total_checks": health["total_checks"],
                "successful_checks": health["successful_checks"]
            }
            for agent_id, health in self.agent_health.items()
        }

class XAgentObservability:
    """Main observability system for XAgent"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector(
            retention_hours=config.get("metrics_retention_hours", 24),
            max_metrics_per_type=config.get("max_metrics_per_type", 10000)
        )
        self.alert_manager = AlertManager(config.get("alerts", {}))
        self.health_monitor = HealthMonitor(
            check_interval=config.get("health_check_interval", 30.0)
        )
        self.performance_snapshots: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.running = False
        self.tasks: List[asyncio.Task] = []

        # Add default alert rules
        self._setup_default_alert_rules()

    def _setup_default_alert_rules(self):
        """Setup default alert rules for common scenarios"""
        self.alert_manager.add_alert_rule(
            "high_error_rate",
            {
                "metric": "error_rate",
                "operator": "gt",
                "threshold": 0.1,  # 10%
                "time_window": timedelta(minutes=5)
            },
            AlertSeverity.HIGH,
            "Agent error rate is {threshold:.1%} (threshold: {operator} {threshold:.1%})"
        )

        self.alert_manager.add_alert_rule(
            "high_response_time",
            {
                "metric": "response_time",
                "operator": "gt",
                "threshold": 5000,  # 5 seconds
                "tags": {"percentile": "p95"}
            },
            AlertSeverity.MEDIUM,
            "Agent response time P95 is {threshold}ms (threshold: {operator} {threshold}ms)"
        )

        self.alert_manager.add_alert_rule(
            "agent_offline",
            {
                "metric": "uptime",
                "operator": "lt",
                "threshold": 0.5,  # 50%
                "time_window": timedelta(minutes=10)
            },
            AlertSeverity.CRITICAL,
            "Agent uptime is {threshold:.1%} (threshold: {operator} {threshold:.1%})"
        )

    async def start(self):
        """Start the observability system"""
        if self.running:
            logger.warning("⚠️ Observability system already running")
            return

        self.running = True

        # Start health monitoring
        await self.health_monitor.start_monitoring()

        # Start periodic tasks
        self.tasks = [
            asyncio.create_task(self._periodic_metrics_cleanup()),
            asyncio.create_task(self._periodic_alert_evaluation()),
            asyncio.create_task(self._periodic_performance_collection())
        ]

        logger.info("🚀 XAgent Observability System started")

    async def stop(self):
        """Stop the observability system"""
        self.running = False

        # Stop health monitoring
        await self.health_monitor.stop_monitoring()

        # Cancel periodic tasks
        for task in self.tasks:
            task.cancel()

        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        logger.info("🛑 XAgent Observability System stopped")

    async def _periodic_metrics_cleanup(self):
        """Periodically clean up old metrics"""
        while self.running:
            try:
                self.metrics_collector.cleanup_old_metrics()
                await asyncio.sleep(3600)  # Every hour
            except Exception as e:
                logger.error(f"❌ Error in metrics cleanup: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes

    async def _periodic_alert_evaluation(self):
        """Periodically evaluate alert conditions"""
        while self.running:
            try:
                self.alert_manager.evaluate_alerts(self.metrics_collector)
                await asyncio.sleep(60)  # Every minute
            except Exception as e:
                logger.error(f"❌ Error in alert evaluation: {e}")
                await asyncio.sleep(30)  # Retry in 30 seconds

    async def _periodic_performance_collection(self):
        """Periodically collect performance snapshots"""
        while self.running:
            try:
                await self._collect_performance_snapshots()
                await asyncio.sleep(30)  # Every 30 seconds
            except Exception as e:
                logger.error(f"❌ Error in performance collection: {e}")
                await asyncio.sleep(60)  # Retry in 1 minute

    async def _collect_performance_snapshots(self):
        """Collect performance snapshots for all agents"""
        # This would integrate with system monitoring tools
        # For now, we'll create mock data
        for agent_id in self.health_monitor.agent_health.keys():
            snapshot = PerformanceSnapshot(
                agent_id=agent_id,
                timestamp=datetime.now(),
                cpu_usage=0.3 + (hash(agent_id) % 100) / 200,  # Mock CPU usage
                memory_usage=0.4 + (hash(agent_id) % 100) / 250,  # Mock memory usage
                disk_usage=0.2 + (hash(agent_id) % 100) / 500,  # Mock disk usage
                network_io={"bytes_in": 1000, "bytes_out": 800},
                task_count=self.metrics_collector.counters.get(f"tasks_completed_{agent_id}", 0),
                queue_size=5,
                response_time_avg=100 + (hash(agent_id) % 200),
                error_rate=0.01 + (hash(agent_id) % 100) / 10000,
                uptime=self.health_monitor.agent_health[agent_id]["uptime"]
            )

            self.performance_snapshots[agent_id].append(snapshot)

            # Record some metrics
            self.metrics_collector.set_gauge(
                "cpu_usage", snapshot.cpu_usage, {"agent_id": agent_id}
            )
            self.metrics_collector.set_gauge(
                "memory_usage", snapshot.memory_usage, {"agent_id": agent_id}
            )
            self.metrics_collector.set_gauge(
                "response_time", snapshot.response_time_avg, {"agent_id": agent_id}
            )
            self.metrics_collector.set_gauge(
                "error_rate", snapshot.error_rate, {"agent_id": agent_id}
            )

    def record_agent_event(self, event_type: str, agent_id: str,
                          severity: AlertSeverity, message: str,
                          metadata: Dict[str, Any] = None):
        """Record an agent event"""
        event = AgentEvent(
            event_id=str(uuid.uuid4()),
            agent_id=agent_id,
            event_type=event_type,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

        self.alert_manager.alert_history.append(event)
        logger.info(f"📝 Event recorded: {event_type} - {agent_id} - {message}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            "system_overview": {
                "total_agents": len(self.health_monitor.agent_health),
                "active_agents": len([
                    h for h in self.health_monitor.agent_health.values()
                    if h["status"] not in [AgentStatus.OFFLINE, AgentStatus.ERROR]
                ]),
                "error_agents": len([
                    h for h in self.health_monitor.agent_health.values()
                    if h["status"] == AgentStatus.ERROR
                ]),
                "active_alerts": len(self.alert_manager.get_active_alerts()),
                "total_metrics": sum(len(deque) for deque in self.metrics_collector.metrics.values())
            },
            "agent_health": self.health_monitor.get_all_health_status(),
            "active_alerts": [
                {
                    "event_id": alert.event_id,
                    "agent_id": alert.agent_id,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in self.alert_manager.get_active_alerts()
            ],
            "recent_events": [
                {
                    "event_id": event.event_id,
                    "agent_id": event.agent_id,
                    "event_type": event.event_type,
                    "severity": event.severity.value,
                    "message": event.message,
                    "timestamp": event.timestamp.isoformat()
                }
                for event in self.alert_manager.get_alert_history(timedelta(hours=1))[-20:]
            ]
        }

# Example usage and testing
async def test_xagent_monitoring():
    """Test XAgent monitoring system"""
    logger.info("🧪 Testing XAgent Monitoring System")

    # Create observability system
    config = {
        "metrics_retention_hours": 24,
        "health_check_interval": 10.0,
        "alerts": {
            "max_history_size": 1000
        }
    }

    obs = XAgentObservability(config)

    # Register some test agents
    obs.health_monitor.register_agent("test_agent_1")
    obs.health_monitor.register_agent("test_agent_2")

    # Start observability system
    await obs.start()

    try:
        # Simulate some activity
        for i in range(10):
            # Record some metrics
            obs.metrics_collector.record_counter(
                "tasks_completed", 1, {"agent_id": "test_agent_1"}
            )
            obs.metrics_collector.set_gauge(
                "cpu_usage", 0.3 + (i * 0.05), {"agent_id": "test_agent_1"}
            )
            obs.metrics_collector.record_timer(
                "response_time", 100 + (i * 10), {"agent_id": "test_agent_1"}
            )

            # Update agent status
            if i % 5 == 0:
                obs.health_monitor.update_agent_status(
                    "test_agent_1", AgentStatus.BUSY
                )
            else:
                obs.health_monitor.update_agent_status(
                    "test_agent_1", AgentStatus.IDLE
                )

            await asyncio.sleep(1)

        # Record an event
        obs.record_agent_event(
            "task_completed", "test_agent_1", AlertSeverity.INFO,
            "Test task completed successfully"
        )

        # Get dashboard data
        dashboard = obs.get_dashboard_data()
        logger.info(f"📊 Dashboard data: {json.dumps(dashboard, indent=2)}")

        # Get specific metrics
        cpu_summary = obs.metrics_collector.get_metric_summary(
            "cpu_usage", {"agent_id": "test_agent_1"}
        )
        logger.info(f"📈 CPU usage summary: {cpu_summary}")

    finally:
        await obs.stop()

if __name__ == "__main__":
    asyncio.run(test_xagent_monitoring())