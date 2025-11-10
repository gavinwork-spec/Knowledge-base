"""
Performance Monitoring and Optimization System
XAgent-inspired comprehensive monitoring, analytics, and optimization system
for multi-agent orchestration with real-time performance tracking.
"""

import asyncio
import json
import logging
import sqlite3
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import statistics
import uuid
import pickle
import psutil
import numpy as np

# Analytics and monitoring
import networkx as nx
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Performance monitoring
import prometheus_client as prometheus
from prometheus_client import Gauge, Counter, Histogram, start_http_server

# Import agent framework
from multi_agent_orchestrator import BaseAgent, AgentTask, AgentCapability, AgentStatus
from multi_agent_system.protocols.agent_communication import (
    MessageRouter, TaskDelegator, AgentMessage, TaskRequest, TaskResponse,
    MessageType, Priority
)
from multi_agent_system.agents.specialized_agents import EnhancedBaseAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for agents"""
    agent_id: str
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    response_time: float
    throughput: float
    success_rate: float
    error_rate: float
    active_tasks: int
    queued_tasks: int
    resource_efficiency: float
    network_io: float
    disk_io: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class AgentHealth:
    """Agent health status and metrics"""
    agent_id: str
    status: str
    health_score: float
    last_check: datetime
    uptime: float
    crash_count: int
    recovery_time: float
    performance_trend: str
    resource_alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class SystemOptimization:
    """System optimization recommendations and actions"""
    optimization_id: str
    optimization_type: str
    priority: str
    description: str
    affected_agents: List[str]
    expected_improvement: Dict[str, float]
    implementation_steps: List[str]
    estimated_time: float
    risk_level: str
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"


class PerformanceMonitor:
    """Advanced performance monitoring system for multi-agent orchestration"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.db_path = Path(self.config.get('db_path', 'performance_monitor.db'))
        self.monitoring_interval = self.config.get('monitoring_interval', 30)
        self.retention_days = self.config.get('retention_days', 30)

        # Performance data storage
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.agent_health: Dict[str, AgentHealth] = {}
        self.alerts: deque = deque(maxlen=1000)
        self.optimizations: List[SystemOptimization] = []

        # Monitoring components
        self.anomaly_detector = None
        self.performance_predictor = None
        self.resource_analyzer = None

        # Prometheus metrics
        self.prometheus_enabled = self.config.get('prometheus_enabled', False)
        if self.prometheus_enabled:
            self._setup_prometheus_metrics()

        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread = None

        # Initialize components
        self._initialize_database()
        self._initialize_analytics()

    def _initialize_database(self):
        """Initialize performance monitoring database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                cpu_usage REAL,
                memory_usage REAL,
                response_time REAL,
                throughput REAL,
                success_rate REAL,
                error_rate REAL,
                active_tasks INTEGER,
                queued_tasks INTEGER,
                resource_efficiency REAL,
                network_io REAL,
                disk_io REAL,
                custom_metrics TEXT
            )
        ''')

        # Health status table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_health (
                agent_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                health_score REAL,
                last_check TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                uptime REAL,
                crash_count INTEGER DEFAULT 0,
                recovery_time REAL,
                performance_trend TEXT,
                resource_alerts TEXT,
                recommendations TEXT
            )
        ''')

        # Optimizations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_optimizations (
                optimization_id TEXT PRIMARY KEY,
                optimization_type TEXT NOT NULL,
                priority TEXT NOT NULL,
                description TEXT,
                affected_agents TEXT,
                expected_improvement TEXT,
                implementation_steps TEXT,
                estimated_time REAL,
                risk_level TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending'
            )
        ''')

        # Alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_alerts (
                alert_id TEXT PRIMARY KEY,
                agent_id TEXT,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT,
                metric_name TEXT,
                threshold_value REAL,
                actual_value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                acknowledged BOOLEAN DEFAULT FALSE,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')

        conn.commit()
        conn.close()

    def _initialize_analytics(self):
        """Initialize analytics components"""
        try:
            # Anomaly detection for performance patterns
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )

            # Performance trend analyzer
            self.performance_predictor = None  # Will be initialized with data

            # Resource utilization analyzer
            self.resource_analyzer = ResourceAnalyzer()

        except Exception as e:
            logger.error(f"Error initializing analytics components: {e}")

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics for monitoring"""
        self.prometheus_metrics = {
            'cpu_usage': Gauge('agent_cpu_usage_percent', 'Agent CPU usage percentage', ['agent_id']),
            'memory_usage': Gauge('agent_memory_usage_percent', 'Agent memory usage percentage', ['agent_id']),
            'response_time': Histogram('agent_response_time_seconds', 'Agent response time in seconds', ['agent_id']),
            'throughput': Counter('agent_throughput_total', 'Total agent throughput', ['agent_id']),
            'success_rate': Gauge('agent_success_rate_percent', 'Agent success rate percentage', ['agent_id']),
            'active_tasks': Gauge('agent_active_tasks', 'Number of active tasks', ['agent_id']),
            'system_load': Gauge('system_load_average', 'System load average'),
            'health_score': Gauge('agent_health_score', 'Agent health score', ['agent_id'])
        }

    async def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring_active = True
        logger.info("Performance monitoring started")

        # Start background monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        # Start Prometheus HTTP server if enabled
        if self.prometheus_enabled:
            start_http_server(8001)

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                await self._collect_system_metrics()

                # Collect agent metrics
                await self._collect_agent_metrics()

                # Analyze performance patterns
                await self._analyze_performance_patterns()

                # Check for anomalies
                await self._detect_anomalies()

                # Generate optimizations
                await self._generate_optimizations()

                # Cleanup old data
                await self._cleanup_old_data()

                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

    async def _collect_system_metrics(self):
        """Collect system-level performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
            disk_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0

            # Network I/O
            net_io = psutil.net_io_counters()
            net_sent_mb = net_io.bytes_sent / (1024 * 1024) if net_io else 0
            net_recv_mb = net_io.bytes_recv / (1024 * 1024) if net_io else 0

            # Store system metrics
            system_metrics = {
                'timestamp': datetime.now(),
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent,
                'disk_read_mb': disk_read_mb,
                'disk_write_mb': disk_write_mb,
                'net_sent_mb': net_sent_mb,
                'net_recv_mb': net_recv_mb,
                'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
            }

            # Update Prometheus metrics
            if self.prometheus_enabled:
                self.prometheus_metrics['system_load'].set(system_metrics['load_average'])

            logger.debug(f"System metrics collected: CPU={cpu_percent:.1f}%, Memory={memory_percent:.1f}%")

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    async def _collect_agent_metrics(self):
        """Collect metrics from individual agents"""
        # This would be implemented to collect metrics from actual agents
        # For now, we'll simulate some basic metrics

        # In a real implementation, this would:
        # 1. Query each agent for its current metrics
        # 2. Collect performance data via message protocols
        # 3. Aggregate and store the metrics

        pass

    async def _analyze_performance_patterns(self):
        """Analyze performance patterns and trends"""
        try:
            for agent_id, metrics_history in self.metrics_history.items():
                if len(metrics_history) < 10:  # Need sufficient data for analysis
                    continue

                # Extract metric values for analysis
                response_times = [m.response_time for m in metrics_history if m.response_time > 0]
                success_rates = [m.success_rate for m in metrics_history if m.success_rate > 0]
                cpu_usages = [m.cpu_usage for m in metrics_history if m.cpu_usage > 0]

                # Analyze trends
                if len(response_times) >= 10:
                    response_trend = self._calculate_trend(response_times)
                    await self._update_performance_trend(agent_id, 'response_time', response_trend)

                if len(success_rates) >= 10:
                    success_trend = self._calculate_trend(success_rates)
                    await self._update_performance_trend(agent_id, 'success_rate', success_trend)

                if len(cpu_usages) >= 10:
                    cpu_trend = self._calculate_trend(cpu_usages)
                    await self._update_performance_trend(agent_id, 'cpu_usage', cpu_trend)

        except Exception as e:
            logger.error(f"Error analyzing performance patterns: {e}")

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return 'stable'

        # Simple linear regression to determine trend
        x = list(range(len(values)))
        slope, _, _, _, _ = stats.linregress(x, values)

        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'

    async def _update_performance_trend(self, agent_id: str, metric_type: str, trend: str):
        """Update performance trend for an agent"""
        if agent_id not in self.agent_health:
            self.agent_health[agent_id] = AgentHealth(
                agent_id=agent_id,
                status='active',
                health_score=0.0,
                last_check=datetime.now(),
                uptime=0.0,
                crash_count=0,
                recovery_time=0.0,
                performance_trend='stable'
            )

        # Update the overall performance trend based on individual metric trends
        # This is a simplified implementation
        self.agent_health[agent_id].performance_trend = trend

    async def _detect_anomalies(self):
        """Detect performance anomalies"""
        try:
            # Collect current metrics across all agents
            current_metrics = []
            agent_ids = []

            for agent_id, metrics_history in self.metrics_history.items():
                if metrics_history:
                    latest_metrics = metrics_history[-1]
                    current_metrics.append([
                        latest_metrics.cpu_usage,
                        latest_metrics.memory_usage,
                        latest_metrics.response_time,
                        latest_metrics.success_rate,
                        latest_metrics.throughput
                    ])
                    agent_ids.append(agent_id)

            if len(current_metrics) < 5:  # Need sufficient data for anomaly detection
                return

            # Standardize metrics
            scaler = StandardScaler()
            normalized_metrics = scaler.fit_transform(current_metrics)

            # Detect anomalies
            anomalies = self.anomaly_detector.fit_predict(normalized_metrics)

            # Process anomalies
            for i, is_anomaly in enumerate(anomalies):
                if is_anomaly == -1:  # Anomaly detected
                    agent_id = agent_ids[i]
                    await self._handle_anomaly(agent_id, current_metrics[i])

        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")

    async def _handle_anomaly(self, agent_id: str, metrics: List[float]):
        """Handle detected performance anomaly"""
        try:
            # Create alert
            alert_id = str(uuid.uuid4())
            alert = {
                'alert_id': alert_id,
                'agent_id': agent_id,
                'alert_type': 'performance_anomaly',
                'severity': 'medium',
                'message': f'Performance anomaly detected for agent {agent_id}',
                'metric_names': ['cpu_usage', 'memory_usage', 'response_time', 'success_rate', 'throughput'],
                'threshold_values': [0.8, 0.8, 5.0, 0.9, 10.0],  # Example thresholds
                'actual_values': metrics,
                'created_at': datetime.now().isoformat(),
                'acknowledged': False,
                'resolved': False
            }

            # Store alert
            self.alerts.append(alert)

            # Log alert
            logger.warning(f"Performance anomaly alert for agent {agent_id}: {alert}")

            # Generate recommendations
            recommendations = await self._generate_anomaly_recommendations(agent_id, metrics)

            # Update agent health
            if agent_id in self.agent_health:
                self.agent_health[agent_id].resource_alerts.extend(recommendations)

        except Exception as e:
            logger.error(f"Error handling anomaly for agent {agent_id}: {e}")

    async def _generate_anomaly_recommendations(self, agent_id: str, metrics: List[float]) -> List[str]:
        """Generate recommendations for anomaly resolution"""
        recommendations = []

        metric_names = ['cpu_usage', 'memory_usage', 'response_time', 'success_rate', 'throughput']
        thresholds = [0.8, 0.8, 5.0, 0.9, 10.0]

        for i, (metric_name, metric_value, threshold) in enumerate(zip(metric_names, metrics, thresholds)):
            if metric_value > threshold:
                if metric_name == 'cpu_usage':
                    recommendations.append(f"High CPU usage ({metric_value:.1f}%). Consider optimizing code or adding CPU resources.")
                elif metric_name == 'memory_usage':
                    recommendations.append(f"High memory usage ({metric_value:.1f}%). Check for memory leaks or increase memory allocation.")
                elif metric_name == 'response_time':
                    recommendations.append(f"High response time ({metric_value:.1f}s). Optimize algorithms or reduce computational complexity.")
                elif metric_name == 'success_rate':
                    recommendations.append(f"Low success rate ({metric_value:.1%}). Review error handling and input validation.")
                elif metric_name == 'throughput':
                    recommendations.append(f"Low throughput ({metric_value:.1f}). Optimize processing pipeline or add parallelism.")

        return recommendations

    async def _generate_optimizations(self):
        """Generate system optimization recommendations"""
        try:
            optimizations = []

            # Analyze resource utilization
            resource_optimizations = await self._analyze_resource_utilization()
            optimizations.extend(resource_optimizations)

            # Analyze performance bottlenecks
            bottleneck_optimizations = await self._identify_bottlenecks()
            optimizations.extend(bottleneck_optimizations)

            # Analyze agent performance distribution
            distribution_optimizations = await self._optimize_agent_distribution()
            optimizations.extend(distribution_optimizations)

            # Store optimizations
            self.optimizations.extend(optimizations)

            # Keep only recent optimizations
            if len(self.optimizations) > 100:
                self.optimizations = self.optimizations[-100:]

        except Exception as e:
            logger.error(f"Error generating optimizations: {e}")

    async def _analyze_resource_utilization(self) -> List[SystemOptimization]:
        """Analyze resource utilization patterns"""
        optimizations = []

        # Collect resource usage data
        cpu_usage = []
        memory_usage = []

        for metrics_history in self.metrics_history.values():
            if metrics_history:
                latest = metrics_history[-1]
                cpu_usage.append(latest.cpu_usage)
                memory_usage.append(latest.memory_usage)

        if cpu_usage:
            avg_cpu = statistics.mean(cpu_usage)
            if avg_cpu > 0.8:
                optimizations.append(SystemOptimization(
                    optimization_id=str(uuid.uuid4()),
                    optimization_type="resource_optimization",
                    priority="high",
                    description="High average CPU utilization detected",
                    affected_agents=["all"],
                    expected_improvement={"cpu_usage": -0.2},
                    implementation_steps=[
                        "Identify CPU-intensive tasks",
                        "Implement code optimizations",
                        "Consider load balancing",
                        "Add CPU resources if needed"
                    ],
                    estimated_time=120.0,
                    risk_level="low"
                ))

        if memory_usage:
            avg_memory = statistics.mean(memory_usage)
            if avg_memory > 0.8:
                optimizations.append(SystemOptimization(
                    optimization_id=str(uuid.uuid4()),
                    optimization_type="resource_optimization",
                    priority="high",
                    description="High average memory utilization detected",
                    affected_agents=["all"],
                    expected_improvement={"memory_usage": -0.2},
                    implementation_steps=[
                        "Check for memory leaks",
                        "Optimize data structures",
                        "Implement memory pooling",
                        "Add memory resources if needed"
                    ],
                    estimated_time=90.0,
                    risk_level="low"
                ))

        return optimizations

    async def _identify_bottlenecks(self) -> List[SystemOptimization]:
        """Identify performance bottlenecks"""
        optimizations = []

        # Analyze response times across agents
        response_times = []
        for metrics_history in self.metrics_history.values():
            if metrics_history:
                latest = metrics_history[-1]
                if latest.response_time > 0:
                    response_times.append((latest.agent_id, latest.response_time))

        if response_times:
            # Find slowest agents
            response_times.sort(key=lambda x: x[1], reverse=True)
            slowest_agents = response_times[:3]  # Top 3 slowest

            for agent_id, response_time in slowest_agents:
                if response_time > 10.0:  # Threshold for slow performance
                    optimizations.append(SystemOptimization(
                        optimization_id=str(uuid.uuid4()),
                        optimization_type="performance_optimization",
                        priority="medium",
                        description=f"Agent {agent_id} has slow response time",
                        affected_agents=[agent_id],
                        expected_improvement={"response_time": -response_time * 0.3},
                        implementation_steps=[
                            "Profile agent execution",
                            "Identify performance bottlenecks",
                            "Optimize critical path",
                            "Consider agent specialization"
                        ],
                        estimated_time=60.0,
                        risk_level="medium"
                    ))

        return optimizations

    async def _optimize_agent_distribution(self) -> List[SystemOptimization]:
        """Optimize agent distribution and load balancing"""
        optimizations = []

        # Analyze load distribution
        agent_loads = {}
        for agent_id, metrics_history in self.metrics_history.items():
            if metrics_history:
                latest = metrics_history[-1]
                agent_loads[agent_id] = latest.active_tasks

        if agent_loads:
            max_load = max(agent_loads.values())
            min_load = min(agent_loads.values())

            # Check for load imbalance
            if max_load > 0 and min_load / max_load < 0.5:
                overloaded_agents = [agent_id for agent_id, load in agent_loads.items() if load > max_load * 0.8]

                if overloaded_agents:
                    optimizations.append(SystemOptimization(
                        optimization_id=str(uuid.uuid4()),
                        optimization_type="load_balancing",
                        priority="medium",
                        description="Load imbalance detected between agents",
                        affected_agents=overloaded_agents,
                        expected_improvement={"load_balance": 0.3},
                        implementation_steps=[
                            "Implement dynamic load balancing",
                            "Scale underutilized agents",
                            "Optimize task assignment algorithm",
                            "Monitor load distribution"
                        ],
                        estimated_time=45.0,
                        risk_level="low"
                    ))

        return optimizations

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'agents_monitored': len(self.metrics_history),
                'total_alerts': len(self.alerts),
                'active_optimizations': len([opt for opt in self.optimizations if opt.status == 'pending']),
                'system_health': self._calculate_system_health(),
                'resource_utilization': self._get_resource_utilization_summary(),
                'performance_trends': self._get_performance_trends_summary()
            }

            return summary

        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return {'error': str(e)}

    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health score"""
        try:
            health_scores = []

            # Include agent health scores
            for health in self.agent_health.values():
                health_scores.append(health.health_score)

            # Include system resource health
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent

            # System health based on resource usage
            resource_health = max(0, 100 - (cpu_usage + memory_usage) / 2)
            health_scores.append(resource_health)

            # Overall health score
            if health_scores:
                overall_health = statistics.mean(health_scores)
            else:
                overall_health = 50.0

            return {
                'overall_score': overall_health,
                'status': 'healthy' if overall_health > 70 else 'warning' if overall_health > 50 else 'critical',
                'agents_health': len([h for h in self.agent_health.values() if h.health_score > 70]),
                'resource_health': resource_health
            }

        except Exception as e:
            logger.error(f"Error calculating system health: {e}")
            return {'overall_score': 0, 'status': 'error'}

    def _get_resource_utilization_summary(self) -> Dict[str, Any]:
        """Get resource utilization summary"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'cpu': {
                    'usage_percent': cpu_percent,
                    'status': 'normal' if cpu_percent < 80 else 'high' if cpu_percent < 95 else 'critical'
                },
                'memory': {
                    'usage_percent': memory.percent,
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'status': 'normal' if memory.percent < 80 else 'high' if memory.percent < 95 else 'critical'
                },
                'disk': {
                    'usage_percent': (disk.used / disk.total) * 100,
                    'total_gb': disk.total / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'status': 'normal' if (disk.used / disk.total) * 100 < 85 else 'high'
                }
            }

        except Exception as e:
            logger.error(f"Error getting resource utilization: {e}")
            return {'error': str(e)}

    def _get_performance_trends_summary(self) -> Dict[str, Any]:
        """Get performance trends summary"""
        trends = {
            'improving': [],
            'declining': [],
            'stable': []
        }

        try:
            for agent_id, health in self.agent_health.items():
                if health.performance_trend:
                    trends[health.performance_trend].append(agent_id)

        except Exception as e:
            logger.error(f"Error getting performance trends: {e}")

        return trends

    async def _cleanup_old_data(self):
        """Clean up old performance data"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)

            # Clean up database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM performance_metrics WHERE timestamp < ?", (cutoff_date,))
            cursor.execute("DELETE FROM performance_alerts WHERE created_at < ?", (cutoff_date,))

            deleted_rows = cursor.rowcount
            conn.commit()
            conn.close()

            if deleted_rows > 0:
                logger.info(f"Cleaned up {deleted_rows} old performance records")

        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")


class ResourceAnalyzer:
    """Resource utilization analyzer for performance optimization"""

    def __init__(self):
        self.resource_history = deque(maxlen=100)
        self.utilization_patterns = {}
        self.optimization_thresholds = {
            'cpu_warning': 70,
            'cpu_critical': 90,
            'memory_warning': 75,
            'memory_critical': 90,
            'disk_warning': 85,
            'disk_critical': 95
        }

    def analyze_utilization_pattern(self, resource_type: str, current_usage: float) -> Dict[str, Any]:
        """Analyze resource utilization pattern"""
        pattern = {
            'current_usage': current_usage,
            'status': self._get_status(resource_type, current_usage),
            'trend': 'stable',
            'recommendation': self._get_recommendation(resource_type, current_usage)
        }

        # Add to history
        self.resource_history.append({
            'type': resource_type,
            'usage': current_usage,
            'timestamp': datetime.now()
        })

        # Analyze trend
        if len(self.resource_history) >= 10:
            recent_usage = [h['usage'] for h in list(self.resource_history)[-10:] if h['type'] == resource_type]
            if len(recent_usage) >= 5:
                slope = self._calculate_trend_slope(recent_usage)
                if slope > 0.5:
                    pattern['trend'] = 'increasing'
                elif slope < -0.5:
                    pattern['trend'] = 'decreasing'

        return pattern

    def _get_status(self, resource_type: str, usage: float) -> str:
        """Get resource status based on usage"""
        if resource_type == 'cpu':
            if usage >= self.optimization_thresholds['cpu_critical']:
                return 'critical'
            elif usage >= self.optimization_thresholds['cpu_warning']:
                return 'warning'
            else:
                return 'normal'
        elif resource_type == 'memory':
            if usage >= self.optimization_thresholds['memory_critical']:
                return 'critical'
            elif usage >= self.optimization_thresholds['memory_warning']:
                return 'warning'
            else:
                return 'normal'
        else:
            return 'normal'

    def _get_recommendation(self, resource_type: str, usage: float) -> str:
        """Get optimization recommendation based on resource usage"""
        if usage >= 90:
            return f"Critical {resource_type} usage. Immediate action required."
        elif usage >= 75:
            return f"High {resource_type} usage. Consider optimization."
        elif usage >= 50:
            return f"Moderate {resource_type} usage. Monitor closely."
        else:
            return f"{resource_type.capitalize()} usage within normal range."

    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate slope for trend analysis"""
        if len(values) < 2:
            return 0.0

        x = list(range(len(values)))
        slope, _, _, _, _ = stats.linregress(x, values)
        return slope


# Factory function
def create_performance_monitor(config: Dict[str, Any] = None) -> PerformanceMonitor:
    """Create a performance monitor"""
    return PerformanceMonitor(config)


# Usage example
if __name__ == "__main__":
    async def test_performance_monitor():
        # Create performance monitor
        monitor = create_performance_monitor({
            'monitoring_interval': 10,
            'prometheus_enabled': True,
            'retention_days': 7
        })

        # Start monitoring
        await monitor.start_monitoring()

        # Run for a while to collect data
        await asyncio.sleep(30)

        # Get performance summary
        summary = monitor.get_performance_summary()
        print(f"Performance Summary: {summary}")

        # Stop monitoring
        monitor.stop_monitoring()

    asyncio.run(test_performance_monitor())