"""
System health checker for observability.

Monitors component health, performs comprehensive health checks,
and provides health status for the entire system.
"""

import time
import asyncio
import threading
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import statistics

from pydantic import BaseModel, Field

from ..core.metrics import get_metrics_collector
from ..core.logging import get_logger, log_system_event


class HealthStatus(str, Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(str, Enum):
    """Types of system components"""
    DATABASE = "database"
    SEARCH_ENGINE = "search_engine"
    AI_SERVICE = "ai_service"
    CACHE = "cache"
    WEB_SERVER = "web_server"
    LOAD_BALANCER = "load_balancer"
    EXTERNAL_API = "external_api"
    STORAGE = "storage"
    NETWORK = "network"
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"


class CheckSeverity(str, Enum):
    """Severity levels for health checks"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    ERROR = "error"


@dataclass
class HealthCheck:
    """Individual health check configuration"""
    name: str
    component: ComponentType
    check_function: Callable[[], Dict[str, Any]]
    timeout: float = 10.0
    severity: CheckSeverity = CheckSeverity.INFO
    enabled: bool = True
    interval: float = 60.0
    last_run: Optional[datetime] = None
    last_result: Optional[Dict[str, Any]] = None


class ComponentHealth(BaseModel):
    """Health status of a component"""
    component: ComponentType
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    response_time_ms: float = 0.0
    last_check: datetime = Field(default_factory=datetime.now)
    uptime_percentage: float = 100.0
    total_checks: int = 0
    successful_checks: int = 0
    consecutive_failures: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SystemHealth(BaseModel):
    """Overall system health status"""
    status: HealthStatus
    overall_score: float = 0.0
    component_count: int = 0
    healthy_components: int = 0
    warning_components: int = 0
    unhealthy_components: int = 0
    unknown_components: int = 0
    components: Dict[str, ComponentHealth] = Field(default_factory=dict)
    system_uptime_seconds: float = 0.0
    last_check: datetime = Field(default_factory=datetime.now)
    health_trend: str = "stable"  # improving, declining, stable
    recommendations: List[str] = Field(default_factory=list)
    alerts: List[str] = Field(default_factory=list)


class HealthChecker:
    """
    Comprehensive health checker for system components.

    Monitors individual components, aggregates health status,
    and provides recommendations for system improvements.
    """

    def __init__(
        self,
        check_interval: float = 30.0,
        max_concurrent_checks: int = 10,
        failure_threshold: int = 3,
        warning_threshold: float = 0.8
    ):
        self.logger = get_logger()
        self.metrics = get_metrics_collector()

        self.check_interval = check_interval
        self.max_concurrent_checks = max_concurrent_checks
        self.failure_threshold = failure_threshold
        self.warning_threshold = warning_threshold

        # Health checks registry
        self.health_checks: List[HealthCheck] = []
        self.component_history: Dict[str, List[ComponentHealth]] = defaultdict(list)

        # System tracking
        self.start_time = datetime.now()
        self.last_system_health: Optional[SystemHealth] = None
        self.health_history: List[SystemHealth] = []

        # Thread safety
        self.lock = threading.RLock()

        # Initialize default health checks
        self._initialize_default_checks()

        # Start health monitoring
        self._start_health_monitoring()

    def _initialize_default_checks(self):
        """Initialize default health checks for common components"""

        # Database health check
        self.add_health_check(
            name="database_connection",
            component=ComponentType.DATABASE,
            check_function=self._check_database_health,
            timeout=5.0,
            severity=CheckSeverity.CRITICAL
        )

        # Search engine health check
        self.add_health_check(
            name="search_engine",
            component=ComponentType.SEARCH_ENGINE,
            check_function=self._check_search_engine_health,
            timeout=10.0,
            severity=CheckSeverity.CRITICAL
        )

        # AI service health check
        self.add_health_check(
            name="ai_service",
            component=ComponentType.AI_SERVICE,
            check_function=self._check_ai_service_health,
            timeout=15.0,
            severity=CheckSeverity.WARNING
        )

        # Cache health check
        self.add_health_check(
            name="cache",
            component=ComponentType.CACHE,
            check_function=self._check_cache_health,
            timeout=3.0,
            severity=CheckSeverity.WARNING
        )

        # Memory health check
        self.add_health_check(
            name="memory",
            component=ComponentType.MEMORY,
            check_function=self._check_memory_health,
            timeout=5.0,
            severity=CheckSeverity.WARNING
        )

        # Disk health check
        self.add_health_check(
            name="disk",
            component=ComponentType.DISK,
            check_function=self._check_disk_health,
            timeout=5.0,
            severity=CheckSeverity.WARNING
        )

        # Network health check
        self.add_health_check(
            name="network",
            component=ComponentType.NETWORK,
            check_function=self._check_network_health,
            timeout=5.0,
            severity=CheckSeverity.INFO
        )

    def add_health_check(
        self,
        name: str,
        component: ComponentType,
        check_function: Callable[[], Dict[str, Any]],
        timeout: float = 10.0,
        severity: CheckSeverity = CheckSeverity.INFO,
        enabled: bool = True,
        interval: Optional[float] = None
    ):
        """Add a custom health check"""
        with self.lock:
            health_check = HealthCheck(
                name=name,
                component=component,
                check_function=check_function,
                timeout=timeout,
                severity=severity,
                enabled=enabled,
                interval=interval or self.check_interval
            )
            self.health_checks.append(health_check)
            self.logger.info(f"Added health check: {name} for {component.value}")

    def remove_health_check(self, name: str):
        """Remove a health check"""
        with self.lock:
            self.health_checks = [hc for hc in self.health_checks if hc.name != name]
            self.logger.info(f"Removed health check: {name}")

    def run_health_check(self, check: HealthCheck) -> ComponentHealth:
        """Run a single health check"""
        start_time = time.time()

        try:
            # Execute health check with timeout
            result = self._execute_with_timeout(check.check_function, check.timeout)

            # Determine health status based on result
            status = self._determine_health_status(result, check.severity)
            message = result.get("message", "Health check completed")
            details = result.get("details", {})

            response_time_ms = (time.time() - start_time) * 1000

            return ComponentHealth(
                component=check.component,
                name=check.name,
                status=status,
                message=message,
                details=details,
                response_time_ms=response_time_ms,
                metadata={
                    "severity": check.severity.value,
                    "timeout": check.timeout,
                    "interval": check.interval
                }
            )

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000

            self.logger.error(f"Health check failed: {check.name} - {str(e)}")

            return ComponentHealth(
                component=check.component,
                name=check.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                response_time_ms=response_time_ms,
                metadata={
                    "error_type": type(e).__name__,
                    "severity": check.severity.value,
                    "timeout": check.timeout
                }
            )

    def _execute_with_timeout(self, func, timeout: float) -> Dict[str, Any]:
        """Execute function with timeout"""
        # In a real implementation, this would use proper timeout handling
        # For now, we'll simulate it with threading
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            try:
                return future.result(timeout=timeout)
            except TimeoutError:
                raise Exception(f"Health check timed out after {timeout}s")

    def _determine_health_status(self, result: Dict[str, Any], severity: CheckSeverity) -> HealthStatus:
        """Determine health status from check result"""
        if result.get("healthy", True):
            return HealthStatus.HEALTHY
        elif result.get("warning", False):
            return HealthStatus.WARNING
        else:
            return HealthStatus.UNHEALTHY

    def run_all_health_checks(self) -> SystemHealth:
        """Run all enabled health checks and return system health"""
        with self.lock:
            enabled_checks = [check for check in self.health_checks if check.enabled]

            if not enabled_checks:
                return SystemHealth(
                    status=HealthStatus.UNKNOWN,
                    message="No health checks configured"
                )

            components = {}
            executor = ThreadPoolExecutor(max_workers=self.max_concurrent_checks)

            # Run health checks concurrently
            future_to_check = {
                executor.submit(self.run_health_check, check): check
                for check in enabled_checks
            }

            # Collect results
            for future in as_completed(future_to_check):
                check = future_to_check[future]
                try:
                    component_health = future.result()
                    component_key = f"{check.component.value}:{check.name}"
                    components[component_key] = component_health

                    # Update component history
                    self._update_component_history(component_key, component_health)

                except Exception as e:
                    self.logger.error(f"Health check execution failed: {check.name} - {str(e)}")
                    component_key = f"{check.component.value}:{check.name}"
                    components[component_key] = ComponentHealth(
                        component=check.component,
                        name=check.name,
                        status=HealthStatus.UNKNOWN,
                        message=f"Execution failed: {str(e)}"
                    )

        # Calculate overall system health
        system_health = self._calculate_system_health(components)

        # Update metrics
        self._update_health_metrics(system_health)

        # Store result
        self.last_system_health = system_health
        self.health_history.append(system_health)

        # Keep history manageable
        if len(self.health_history) > 1000:
            self.health_history = self.health_history[-500:]

        return system_health

    def _update_component_history(self, component_key: str, health: ComponentHealth):
        """Update component health history"""
        self.component_history[component_key].append(health)

        # Keep history manageable (last 100 results)
        if len(self.component_history[component_key]) > 100:
            self.component_history[component_key] = self.component_history[component_key][-50]

    def _calculate_system_health(self, components: Dict[str, ComponentHealth]) -> SystemHealth:
        """Calculate overall system health from component health"""
        if not components:
            return SystemHealth(
                status=HealthStatus.UNKNOWN,
                message="No components to evaluate"
            )

        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 0,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.UNKNOWN: 0
        }

        total_response_time = 0
        component_scores = []

        for health in components.values():
            status_counts[health.status] += 1
            total_response_time += health.response_time_ms

            # Calculate component score
            if health.status == HealthStatus.HEALTHY:
                score = 100
            elif health.status == HealthStatus.WARNING:
                score = 70
            elif health.status == HealthStatus.UNHEALTHY:
                score = 30
            else:
                score = 50

            component_scores.append(score)

        # Determine overall status
        total_components = len(components)
        healthy_count = status_counts[HealthStatus.HEALTHY]

        if healthy_count == total_components:
            overall_status = HealthStatus.HEALTHY
            overall_score = 100
        elif healthy_count >= total_components * 0.8:
            overall_status = HealthStatus.WARNING
            overall_score = 70
        elif healthy_count > 0:
            overall_status = HealthStatus.WARNING
            overall_score = 40
        else:
            overall_status = HealthStatus.UNHEALTHY
            overall_score = 20

        # Calculate health trend
        health_trend = self._calculate_health_trend()

        # Generate recommendations
        recommendations = self._generate_recommendations(components)

        # Generate alerts
        alerts = self._generate_alerts(components)

        return SystemHealth(
            status=overall_status,
            overall_score=overall_score,
            component_count=total_components,
            healthy_components=status_counts[HealthStatus.HEALTHY],
            warning_components=status_counts[HealthStatus.WARNING],
            unhealthy_components=status_counts[HealthStatus.UNHEALTHY],
            unknown_components=status_counts[HealthStatus.UNKNOWN],
            components=components,
            system_uptime_seconds=(datetime.now() - self.start_time).total_seconds(),
            last_check=datetime.now(),
            health_trend=health_trend,
            recommendations=recommendations,
            alerts=alerts
        )

    def _calculate_health_trend(self) -> str:
        """Calculate health trend based on history"""
        if len(self.health_history) < 2:
            return "insufficient_data"

        recent_health = self.health_history[-10:]
        older_health = self.health_history[-20:-10] if len(self.health_history) >= 20 else self.health_history[:-10]

        if not older_health:
            return "insufficient_data"

        recent_avg_score = sum(h.overall_score for h in recent_health) / len(recent_health)
        older_avg_score = sum(h.overall_score for h in older_health) / len(older_health)

        if recent_avg_score > older_avg_score * 1.1:
            return "improving"
        elif recent_avg_score < older_avg_score * 0.9:
            return "declining"
        else:
            return "stable"

    def _generate_recommendations(self, components: Dict[str, ComponentHealth]) -> List[str]:
        """Generate health recommendations based on component status"""
        recommendations = []

        for health in components.values():
            if health.status == HealthStatus.UNHEALTHY:
                recommendations.append(f"Critical: {health.name} is unhealthy - {health.message}")
            elif health.status == HealthStatus.WARNING:
                recommendations.append(f"Warning: {health.name} needs attention - {health.message}")
            elif health.response_time_ms > 5000:
                recommendations.append(f"Performance: {health.name} has high response time ({health.response_time_ms:.0f}ms)")

        # General recommendations
        unhealthy_count = sum(1 for h in components.values() if h.status == HealthStatus.UNHEALTHY)
        if unhealthy_count > 0:
            recommendations.append(f"Action required: {unhealthy_count} components are unhealthy")

        return recommendations[:10]  # Limit to top 10 recommendations

    def _generate_alerts(self, components: Dict[str, ComponentHealth]) -> List[str]:
        """Generate alerts based on component status"""
        alerts = []

        for health in components.values():
            if health.status == HealthStatus.UNHEALTHY and health.consecutive_failures >= 3:
                alerts.append(f"ALERT: {health.name} has failed {health.consecutive_failures} times consecutively")

        # System-wide alerts
        unhealthy_count = sum(1 for h in components.values() if h.status == HealthStatus.UNHEALTHY)
        total_count = len(components)

        if unhealthy_count > 0:
            alert_level = "CRITICAL" if unhealthy_count >= total_count * 0.5 else "WARNING"
            alerts.append(f"{alert_level}: System health degraded - {unhealthy_count}/{total_count} components unhealthy")

        return alerts

    def _update_health_metrics(self, health: SystemHealth):
        """Update metrics based on health check results"""
        # Update overall health score
        self.metrics.set_gauge("system_health_score", health.overall_score)

        # Update component counts
        self.metrics.set_gauge("system_components_total", health.component_count)
        self.metrics.set_gauge("system_components_healthy", health.healthy_components)
        self.metrics.set_gauge("system_components_warning", health.warning_components)
        self.metrics.set_gauge("system_components_unhealthy", health.unhealthy_components)

        # Update component status
        for component_key, component_health in health.components.items():
            self.metrics.set_gauge(
                "component_health_score",
                100 if component_health.status == HealthStatus.HEALTHY else 0,
                labels={
                    "component": component_health.component.value,
                    "name": component_health.name
                }
            )

        # Log system event
        log_system_event(
            message=f"Health check completed - Status: {health.status.value}, Score: {health.overall_score:.1f}",
            metadata={
                "healthy_components": health.healthy_components,
                "unhealthy_components": health.unhealthy_components,
                "recommendations_count": len(health.recommendations),
                "alerts_count": len(health.alerts)
            }
        )

    def get_component_health(self, component: ComponentType, name: str) -> Optional[ComponentHealth]:
        """Get health status for a specific component"""
        component_key = f"{component.value}:{name}"

        if self.last_system_health and component_key in self.last_system_health.components:
            return self.last_system_health.components[component_key]
        return None

    def get_system_health(self) -> Optional[SystemHealth]:
        """Get current system health status"""
        return self.last_system_health

    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for dashboard"""
        if not self.last_system_health:
            return {
                "status": "unknown",
                "message": "No health checks have been run yet"
            }

        health = self.last_system_health

        return {
            "status": health.status.value,
            "overall_score": health.overall_score,
            "uptime_hours": health.system_uptime_seconds / 3600,
            "component_counts": {
                "total": health.component_count,
                "healthy": health.healthy_components,
                "warning": health.warning_components,
                "unhealthy": health.unhealthy_components,
                "unknown": health.unknown_components
            },
            "last_check": health.last_check.isoformat(),
            "health_trend": health.health_trend,
            "recommendations_count": len(health.recommendations),
            "alerts_count": len(health.alerts),
            "top_recommendations": health.recommendations[:3]
        }

    def _start_health_monitoring(self):
        """Start background health monitoring"""
        def monitor_health():
            while True:
                try:
                    self.run_all_health_checks()
                    time.sleep(self.check_interval)
                except Exception as e:
                    self.logger.error(f"Error in health monitoring: {e}")
                    time.sleep(60)  # Wait longer before retrying

        thread = threading.Thread(target=monitor_health, daemon=True)
        thread.start()
        self.logger.info("Health monitoring started")

    # Default health check implementations
    def _check_database_health(self) -> Dict[str, Any]:
        """Check database health"""
        # Simulate database health check
        # In a real implementation, this would connect to the database and run queries
        import random

        # Simulate random health status
        is_healthy = random.random() > 0.1  # 90% chance of being healthy

        return {
            "healthy": is_healthy,
            "message": "Database connection successful" if is_healthy else "Database connection failed",
            "details": {
                "connection_pool_size": random.randint(5, 20),
                "active_connections": random.randint(1, 10),
                "query_latency_ms": random.uniform(1, 50)
            }
        }

    def _check_search_engine_health(self) -> Dict[str, Any]:
        """Check search engine health"""
        import random

        is_healthy = random.random() > 0.05  # 95% chance of being healthy

        return {
            "healthy": is_healthy,
            "message": "Search engine operational" if is_healthy else "Search engine degraded",
            "details": {
                "index_size": random.randint(1000, 10000),
                "query_latency_ms": random.uniform(10, 200),
                "indexing_queue_size": random.randint(0, 100)
            }
        }

    def _check_ai_service_health(self) -> Dict[str, Any]:
        """Check AI service health"""
        import random

        is_healthy = random.random() > 0.1  # 90% chance of being healthy

        return {
            "healthy": is_healthy,
            "message": "AI service responding" if is_healthy else "AI service unavailable",
            "details": {
                "model_load": random.uniform(0.1, 0.9),
                "queue_length": random.randint(0, 50),
                "response_time_ms": random.uniform(100, 5000)
            }
        }

    def _check_cache_health(self) -> Dict[str, Any]:
        """Check cache health"""
        import random

        is_healthy = random.random() > 0.05  # 95% chance of being healthy

        return {
            "healthy": is_healthy,
            "message": "Cache operational" if is_healthy else "Cache issues detected",
            "details": {
                "hit_rate": random.uniform(0.7, 0.95),
                "memory_usage_percent": random.uniform(10, 80),
                "eviction_rate": random.uniform(0.01, 0.1)
            }
        }

    def _check_memory_health(self) -> Dict[str, Any]:
        """Check memory usage"""
        import psutil

        try:
            memory = psutil.virtual_memory()

            # Calculate memory usage percentage
            usage_percent = (memory.used / memory.total) * 100

            # Determine health based on usage
            if usage_percent < 80:
                status = "healthy"
                message = "Memory usage normal"
            elif usage_percent < 90:
                status = "warning"
                message = "Memory usage high"
            else:
                status = "unhealthy"
                message = "Memory usage critical"

            return {
                "healthy": status == "healthy",
                "message": message,
                "details": {
                    "total_gb": memory.total / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "usage_percent": usage_percent
                }
            }

        except Exception as e:
            return {
                "healthy": False,
                "message": f"Failed to check memory: {str(e)}",
                "details": {"error": str(e)}
            }

    def _check_disk_health(self) -> Dict[str, Any]:
        """Check disk usage"""
        import psutil

        try:
            disk = psutil.disk_usage('/')

            usage_percent = (disk.used / disk.total) * 100

            if usage_percent < 70:
                status = "healthy"
                message = "Disk usage normal"
            elif usage_percent < 85:
                status = "warning"
                message = "Disk usage high"
            else:
                status = "unhealthy"
                message = "Disk usage critical"

            return {
                "healthy": status == "healthy",
                "message": message,
                "details": {
                    "total_gb": disk.total / (1024**3),
                    "used_gb": disk.used / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "usage_percent": usage_percent
                }
            }

        except Exception as e:
            return {
                "healthy": False,
                "message": f"Failed to check disk: {str(e)}",
                "details": {"error": str(e)}
            }

    def _check_network_health(self) -> Dict[str, Any]:
        """Check network connectivity"""
        import socket
        import random

        # Test connectivity to common services
        test_hosts = ["8.8.8.8", "1.1.1.1"]
        successful_tests = 0

        for host in test_hosts:
            try:
                socket.create_connection((host, 53), timeout=2).close()
                successful_tests += 1
            except:
                pass

        is_healthy = successful_tests >= len(test_hosts) * 0.8
        message = f"Network connectivity: {successful_tests}/{len(test_hosts)} hosts reachable"

        return {
            "healthy": is_healthy,
            "message": message,
            "details": {
                "tested_hosts": len(test_hosts),
                "successful_tests": successful_tests,
                "success_rate": successful_tests / len(test_hosts)
            }
        }


# Global health checker instance
_global_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get global health checker instance"""
    global _global_health_checker
    if _global_health_checker is None:
        _global_health_checker = HealthChecker()
    return _global_health_checker


def configure_health_checker(**kwargs):
    """Configure global health checker"""
    global _global_health_checker
    _global_health_checker = HealthChecker(**kwargs)
    return _global_health_checker