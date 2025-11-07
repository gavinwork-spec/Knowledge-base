"""
Metrics collection system for observability.

Collects, aggregates, and exports performance metrics, cost metrics,
user metrics, and system metrics with Prometheus integration.
"""

import time
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
import statistics
import uuid

from pydantic import BaseModel, Field


class MetricType(str, Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AggregationType(str, Enum):
    """Types of metric aggregations"""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    P50 = "p50"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"
    RATE = "rate"


@dataclass
class MetricValue:
    """Single metric value with timestamp and labels"""
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


class Metric(BaseModel):
    """Base metric structure"""
    name: str
    description: str
    metric_type: MetricType
    labels: Dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class Counter(Metric):
    """Counter metric that only increases"""
    value: float = 0.0

    def increment(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment counter by amount"""
        self.value += amount
        if labels:
            self.labels.update(labels)

    def reset(self):
        """Reset counter to zero"""
        self.value = 0.0


class Gauge(Metric):
    """Gauge metric that can go up or down"""
    value: float = 0.0

    def set(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Set gauge value"""
        self.value = value
        if labels:
            self.labels.update(labels)

    def increment(self, amount: float = 1.0):
        """Increment gauge"""
        self.value += amount

    def decrement(self, amount: float = 1.0):
        """Decrement gauge"""
        self.value -= amount


class Histogram(Metric):
    """Histogram metric with configurable buckets"""
    buckets: List[float] = Field(default_factory=lambda: [0.1, 0.5, 1.0, 2.5, 5.0, 10.0])
    bucket_counts: Dict[float, int] = Field(default_factory=dict)
    count: int = 0
    sum: float = 0.0

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize bucket counts
        for bucket in self.buckets:
            self.bucket_counts[bucket] = 0

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value"""
        self.count += 1
        self.sum += value

        # Update bucket counts
        for bucket in self.buckets:
            if value <= bucket:
                self.bucket_counts[bucket] += 1

        if labels:
            self.labels.update(labels)

    def get_quantile(self, quantile: float) -> float:
        """Calculate approximate quantile from histogram data"""
        if self.count == 0:
            return 0.0

        # This is a simplified calculation
        # In practice, you'd want more sophisticated quantile estimation
        return self.sum / self.count


class Summary(Metric):
    """Summary metric with calculated quantiles"""
    count: int = 0
    sum: float = 0.0
    values: deque = Field(default_factory=lambda: deque(maxlen=1000))
    quantiles: List[float] = Field(default_factory=lambda: [0.5, 0.9, 0.95, 0.99])

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value"""
        self.count += 1
        self.sum += value
        self.values.append(value)

        if labels:
            self.labels.update(labels)

    def get_quantile(self, quantile: float) -> float:
        """Calculate quantile from observed values"""
        if not self.values:
            return 0.0

        sorted_values = sorted(self.values)
        index = int(quantile * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]


@dataclass
class PerformanceMetrics:
    """Performance-related metrics"""
    response_time_p50: float = 0.0
    response_time_p90: float = 0.0
    response_time_p95: float = 0.0
    response_time_p99: float = 0.0
    throughput_rps: float = 0.0
    error_rate: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_io: float = 0.0
    network_io: float = 0.0


@dataclass
class CostMetrics:
    """Cost-related metrics"""
    total_cost_usd: float = 0.0
    cost_per_query_usd: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    api_calls_count: int = 0
    model_usage_costs: Dict[str, float] = field(default_factory=dict)
    daily_cost_limit_usage: float = 0.0


@dataclass
class UserMetrics:
    """User-related metrics"""
    active_users_24h: int = 0
    active_users_7d: int = 0
    active_users_30d: int = 0
    total_sessions: int = 0
    avg_session_duration: float = 0.0
    user_retention_rate: float = 0.0
    conversion_rate: float = 0.0
    user_satisfaction_score: float = 0.0


@dataclass
class SystemMetrics:
    """System-level metrics"""
    uptime_seconds: float = 0.0
    health_score: float = 100.0
    component_status: Dict[str, bool] = field(default_factory=dict)
    alert_count: int = 0
    last_restart: Optional[datetime] = None
    version: str = "unknown"


class MetricsCollector:
    """
    Main metrics collector that manages different types of metrics,
    aggregations, and exports.
    """

    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, Union[Counter, Gauge, Histogram, Summary]] = {}
        self.time_series: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.aggregations: Dict[str, Dict[AggregationType, float]] = {}
        self.lock = threading.RLock()

        # Predefined metric groups
        self.performance_metrics: Dict[str, float] = {}
        self.cost_metrics: Dict[str, float] = {}
        self.user_metrics: Dict[str, float] = {}
        self.system_metrics: Dict[str, float] = {}

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_old_data, daemon=True)
        self._cleanup_thread.start()

    def create_counter(
        self,
        name: str,
        description: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Counter:
        """Create a counter metric"""
        with self.lock:
            counter = Counter(
                name=name,
                description=description,
                metric_type=MetricType.COUNTER,
                labels=labels or {}
            )
            self.metrics[name] = counter
            return counter

    def create_gauge(
        self,
        name: str,
        description: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Gauge:
        """Create a gauge metric"""
        with self.lock:
            gauge = Gauge(
                name=name,
                description=description,
                metric_type=MetricType.GAUGE,
                labels=labels or {}
            )
            self.metrics[name] = gauge
            return gauge

    def create_histogram(
        self,
        name: str,
        description: str,
        buckets: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> Histogram:
        """Create a histogram metric"""
        with self.lock:
            histogram = Histogram(
                name=name,
                description=description,
                metric_type=MetricType.HISTOGRAM,
                buckets=buckets or [0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
                labels=labels or {}
            )
            self.metrics[name] = histogram
            return histogram

    def create_summary(
        self,
        name: str,
        description: str,
        quantiles: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> Summary:
        """Create a summary metric"""
        with self.lock:
            summary = Summary(
                name=name,
                description=description,
                metric_type=MetricType.SUMMARY,
                quantiles=quantiles or [0.5, 0.9, 0.95, 0.99],
                labels=labels or {}
            )
            self.metrics[name] = summary
            return summary

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ):
        """Record a metric value"""
        timestamp = timestamp or datetime.now()

        with self.lock:
            if name not in self.metrics:
                # Auto-create metric if it doesn't exist
                if metric_type == MetricType.COUNTER:
                    self.create_counter(name, f"Auto-created counter {name}", labels)
                elif metric_type == MetricType.GAUGE:
                    self.create_gauge(name, f"Auto-created gauge {name}", labels)
                elif metric_type == MetricType.HISTOGRAM:
                    self.create_histogram(name, f"Auto-created histogram {name}", labels=labels)
                elif metric_type == MetricType.SUMMARY:
                    self.create_summary(name, f"Auto-created summary {name}", labels=labels)

            metric = self.metrics[name]

            # Record based on metric type
            if isinstance(metric, Counter):
                metric.increment(value, labels)
            elif isinstance(metric, Gauge):
                metric.set(value, labels)
            elif isinstance(metric, (Histogram, Summary)):
                metric.observe(value, labels)

            # Store time series data
            metric_value = MetricValue(
                value=value,
                timestamp=timestamp,
                labels=labels or {}
            )
            self.time_series[name].append(metric_value)

    def increment_counter(self, name: str, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        with self.lock:
            if name not in self.metrics or not isinstance(self.metrics[name], Counter):
                self.create_counter(name, f"Counter {name}")
            self.metrics[name].increment(amount, labels)

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value"""
        with self.lock:
            if name not in self.metrics or not isinstance(self.metrics[name], Gauge):
                self.create_gauge(name, f"Gauge {name}")
            self.metrics[name].set(value, labels)

    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a histogram value"""
        with self.lock:
            if name not in self.metrics or not isinstance(self.metrics[name], Histogram):
                self.create_histogram(name, f"Histogram {name}")
            self.metrics[name].observe(value, labels)

    def observe_summary(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a summary value"""
        with self.lock:
            if name not in self.metrics or not isinstance(self.metrics[name], Summary):
                self.create_summary(name, f"Summary {name}")
            self.metrics[name].observe(value, labels)

    def get_metric(self, name: str) -> Optional[Union[Counter, Gauge, Histogram, Summary]]:
        """Get a metric by name"""
        with self.lock:
            return self.metrics.get(name)

    def get_all_metrics(self) -> Dict[str, Union[Counter, Gauge, Histogram, Summary]]:
        """Get all registered metrics"""
        with self.lock:
            return self.metrics.copy()

    def get_time_series(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MetricValue]:
        """Get time series data for a metric"""
        end_time = end_time or datetime.now()
        start_time = start_time or end_time - timedelta(hours=1)

        with self.lock:
            if name not in self.time_series:
                return []

            return [
                mv for mv in self.time_series[name]
                if start_time <= mv.timestamp <= end_time
            ]

    def calculate_aggregations(
        self,
        name: str,
        aggregation_types: List[AggregationType],
        time_window: timedelta = timedelta(hours=1)
    ) -> Dict[AggregationType, float]:
        """Calculate aggregations for a metric over a time window"""
        end_time = datetime.now()
        start_time = end_time - time_window

        time_series = self.get_time_series(name, start_time, end_time)
        if not time_series:
            return {}

        values = [mv.value for mv in time_series]
        aggregations = {}

        for agg_type in aggregation_types:
            if agg_type == AggregationType.SUM:
                aggregations[agg_type] = sum(values)
            elif agg_type == AggregationType.AVG:
                aggregations[agg_type] = statistics.mean(values) if values else 0.0
            elif agg_type == AggregationType.MIN:
                aggregations[agg_type] = min(values) if values else 0.0
            elif agg_type == AggregationType.MAX:
                aggregations[agg_type] = max(values) if values else 0.0
            elif agg_type == AggregationType.P50:
                aggregations[agg_type] = statistics.median(values) if values else 0.0
            elif agg_type in [AggregationType.P90, AggregationType.P95, AggregationType.P99]:
                percentile = int(agg_type[1:]) / 100
                sorted_values = sorted(values)
                index = int(percentile * len(sorted_values))
                aggregations[agg_type] = sorted_values[min(index, len(sorted_values) - 1)] if sorted_values else 0.0
            elif agg_type == AggregationType.RATE:
                # Calculate rate per second
                time_diff = (time_window.total_seconds())
                aggregations[agg_type] = len(values) / time_diff if time_diff > 0 else 0.0

        return aggregations

    def update_performance_metrics(self, metrics: PerformanceMetrics):
        """Update performance metrics"""
        self.performance_metrics = {
            "response_time_p50": metrics.response_time_p50,
            "response_time_p90": metrics.response_time_p90,
            "response_time_p95": metrics.response_time_p95,
            "response_time_p99": metrics.response_time_p99,
            "throughput_rps": metrics.throughput_rps,
            "error_rate": metrics.error_rate,
            "cpu_usage": metrics.cpu_usage,
            "memory_usage": metrics.memory_usage,
            "disk_io": metrics.disk_io,
            "network_io": metrics.network_io,
        }

        # Update corresponding gauges
        for key, value in self.performance_metrics.items():
            self.set_gauge(f"performance_{key}", value)

    def update_cost_metrics(self, metrics: CostMetrics):
        """Update cost metrics"""
        self.cost_metrics = {
            "total_cost_usd": metrics.total_cost_usd,
            "cost_per_query_usd": metrics.cost_per_query_usd,
            "api_calls_count": metrics.api_calls_count,
            "daily_cost_limit_usage": metrics.daily_cost_limit_usage,
        }

        # Update corresponding gauges
        for key, value in self.cost_metrics.items():
            self.set_gauge(f"cost_{key}", value)

        # Update token usage counters
        for token_type, count in metrics.token_usage.items():
            self.increment_counter(f"tokens_{token_type}", count)

        # Update model usage costs
        for model, cost in metrics.model_usage_costs.items():
            self.increment_counter(f"cost_model_{model}", cost)

    def update_user_metrics(self, metrics: UserMetrics):
        """Update user metrics"""
        self.user_metrics = {
            "active_users_24h": metrics.active_users_24h,
            "active_users_7d": metrics.active_users_7d,
            "active_users_30d": metrics.active_users_30d,
            "total_sessions": metrics.total_sessions,
            "avg_session_duration": metrics.avg_session_duration,
            "user_retention_rate": metrics.user_retention_rate,
            "conversion_rate": metrics.conversion_rate,
            "user_satisfaction_score": metrics.user_satisfaction_score,
        }

        # Update corresponding gauges
        for key, value in self.user_metrics.items():
            self.set_gauge(f"user_{key}", value)

    def update_system_metrics(self, metrics: SystemMetrics):
        """Update system metrics"""
        self.system_metrics = {
            "uptime_seconds": metrics.uptime_seconds,
            "health_score": metrics.health_score,
            "alert_count": metrics.alert_count,
        }

        # Update corresponding gauges
        for key, value in self.system_metrics.items():
            self.set_gauge(f"system_{key}", value)

        # Update component status
        for component, status in metrics.component_status.items():
            self.set_gauge(f"system_component_{component}_status", 1.0 if status else 0.0)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics"""
        with self.lock:
            return {
                "registered_metrics": len(self.metrics),
                "performance_metrics": self.performance_metrics,
                "cost_metrics": self.cost_metrics,
                "user_metrics": self.user_metrics,
                "system_metrics": self.system_metrics,
                "total_time_series": sum(len(ts) for ts in self.time_series.values()),
            }

    def _cleanup_old_data(self):
        """Background thread to clean up old data"""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)

                with self.lock:
                    for name, ts in self.time_series.items():
                        # Remove old data points
                        while ts and ts[0].timestamp < cutoff_time:
                            ts.popleft()

            except Exception as e:
                # Log error but continue cleanup
                print(f"Error in metrics cleanup: {e}")


# Global metrics collector instance
_global_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector


def configure_metrics_collector(**kwargs):
    """Configure global metrics collector"""
    global _global_metrics_collector
    _global_metrics_collector = MetricsCollector(**kwargs)