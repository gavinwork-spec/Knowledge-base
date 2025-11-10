#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Metrics Tracker
性能指标跟踪器

Comprehensive performance tracking system for API response times, accuracy metrics,
resource usage, and manufacturing-specific performance indicators.
"""

import sqlite3
import json
import logging
import asyncio
import time
import psutil
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics
import numpy as np

from .langfuse_integration import (
    LangFuseIntegration,
    InteractionType,
    MetricType,
    get_langfuse_integration
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricAggregation(Enum):
    """指标聚合方式"""
    MEAN = "mean"
    MEDIAN = "median"
    P95 = "p95"
    P99 = "p99"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    COUNT = "count"

class PerformanceThreshold(Enum):
    """性能阈值类型"""
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    ACCURACY = "accuracy"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"

@dataclass
class MetricValue:
    """指标值"""
    metric_type: MetricType
    value: float
    unit: str
    timestamp: datetime
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AggregatedMetrics:
    """聚合指标"""
    metric_type: MetricType
    aggregation_type: MetricAggregation
    value: float
    count: int
    period_start: datetime
    period_end: datetime
    tags: List[str] = field(default_factory=list)

@dataclass
class PerformanceThresholdRule:
    """性能阈值规则"""
    rule_id: str
    metric_type: MetricType
    threshold_value: float
    comparison_operator: str  # ">", "<", ">=", "<=", "=="
    aggregation_window: timedelta
    alert_cooldown: timedelta
    enabled: bool = True
    notification_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemResourceMetrics:
    """系统资源指标"""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    network_io: Dict[str, float]
    timestamp: datetime

@dataclass
class APIPerformanceMetrics:
    """API性能指标"""
    endpoint: str
    method: str
    response_time_ms: float
    status_code: int
    request_size_bytes: int
    response_size_bytes: int
    timestamp: datetime
    user_agent: str
    session_id: str

@dataclass
class AccuracyMetrics:
    """准确性指标"""
    task_type: str
    model_name: str
    ground_truth: str
    prediction: str
    accuracy_score: float
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class PerformanceTracker:
    """性能跟踪器"""

    def __init__(self,
                 db_path: str = "knowledge_base.db",
                 langfuse_integration: Optional[LangFuseIntegration] = None,
                 buffer_size: int = 10000,
                 aggregation_intervals: List[timedelta] = None):
        """
        初始化性能跟踪器

        Args:
            db_path: SQLite数据库路径
            langfuse_integration: LangFuse集成实例
            buffer_size: 内存缓冲区大小
            aggregation_intervals: 聚合时间间隔列表
        """
        self.db_path = db_path
        self.langfuse_integration = langfuse_integration or get_langfuse_integration()
        self.buffer_size = buffer_size
        self.aggregation_intervals = aggregation_intervals or [
            timedelta(minutes=5),
            timedelta(hours=1),
            timedelta(days=1)
        ]

        # 内存缓冲区
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.api_metrics_buffer = deque(maxlen=buffer_size)
        self.accuracy_metrics_buffer = deque(maxlen=buffer_size)

        # 实时统计
        self.real_time_stats = defaultdict(lambda: deque(maxlen=1000))
        self.threshold_rules = {}
        self.alert_cooldowns = {}

        # 系统监控线程
        self.system_monitoring = False
        self.system_monitor_thread = None

        # 初始化数据库
        self._init_database()

        # 启动后台任务
        self._start_background_tasks()

    def _init_database(self):
        """初始化数据库表"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute("PRAGMA foreign_keys = ON")

            # 创建指标表
            self.conn.executescript("""
                -- 原始指标数据表
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    tags TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- 聚合指标表
                CREATE TABLE IF NOT EXISTS aggregated_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_type TEXT NOT NULL,
                    aggregation_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    count INTEGER NOT NULL,
                    period_start DATETIME NOT NULL,
                    period_end DATETIME NOT NULL,
                    tags TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- API性能表
                CREATE TABLE IF NOT EXISTS api_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL,
                    response_time_ms REAL NOT NULL,
                    status_code INTEGER NOT NULL,
                    request_size_bytes INTEGER,
                    response_size_bytes INTEGER,
                    user_agent TEXT,
                    session_id TEXT,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- 准确性指标表
                CREATE TABLE IF NOT EXISTS accuracy_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_type TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    ground_truth TEXT,
                    prediction TEXT,
                    accuracy_score REAL NOT NULL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- 系统资源指标表
                CREATE TABLE IF NOT EXISTS system_resources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cpu_percent REAL NOT NULL,
                    memory_percent REAL NOT NULL,
                    memory_available_gb REAL NOT NULL,
                    disk_usage_percent REAL NOT NULL,
                    network_io TEXT,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- 性能阈值规则表
                CREATE TABLE IF NOT EXISTS performance_thresholds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id TEXT UNIQUE NOT NULL,
                    metric_type TEXT NOT NULL,
                    threshold_value REAL NOT NULL,
                    comparison_operator TEXT NOT NULL,
                    aggregation_window_minutes INTEGER NOT NULL,
                    alert_cooldown_minutes INTEGER NOT NULL,
                    enabled BOOLEAN DEFAULT 1,
                    notification_config TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- 警报历史表
                CREATE TABLE IF NOT EXISTS performance_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    threshold_value REAL NOT NULL,
                    triggered_at DATETIME NOT NULL,
                    resolved_at DATETIME,
                    context_data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- 创建索引
                CREATE INDEX IF NOT EXISTS idx_metrics_type_timestamp ON performance_metrics(metric_type, timestamp);
                CREATE INDEX IF NOT EXISTS idx_api_performance_timestamp ON api_performance(timestamp);
                CREATE INDEX IF NOT EXISTS idx_api_performance_endpoint ON api_performance(endpoint, method);
                CREATE INDEX IF NOT EXISTS idx_accuracy_metrics_timestamp ON accuracy_metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_system_resources_timestamp ON system_resources(timestamp);
            """)

            logger.info("✅ Performance tracking database initialized")

        except sqlite3.Error as e:
            logger.error(f"Failed to initialize performance database: {e}")
            raise

    def _start_background_tasks(self):
        """启动后台任务"""
        # 启动系统监控
        self.start_system_monitoring()

        # 启动指标聚合任务
        asyncio.create_task(self._periodic_aggregation())

        # 启动阈值检查任务
        asyncio.create_task(self._periodic_threshold_check())

    def record_metric(self,
                     metric_type: MetricType,
                     value: float,
                     unit: str,
                     tags: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """记录指标"""
        try:
            timestamp = datetime.now(timezone.utc)
            metric_value = MetricValue(
                metric_type=metric_type,
                value=value,
                unit=unit,
                timestamp=timestamp,
                tags=tags or [],
                metadata=metadata or {}
            )

            # 添加到内存缓冲区
            self.metrics_buffer.append(metric_value)

            # 更新实时统计
            self.real_time_stats[metric_type.value].append(value)

            # 异步保存到数据库
            asyncio.create_task(self._save_metric_to_db(metric_value))

        except Exception as e:
            logger.error(f"Failed to record metric: {e}")

    async def _save_metric_to_db(self, metric_value: MetricValue):
        """保存指标到数据库"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO performance_metrics
                (metric_type, value, unit, timestamp, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                metric_value.metric_type.value,
                metric_value.value,
                metric_value.unit,
                metric_value.timestamp.isoformat(),
                json.dumps(metric_value.tags),
                json.dumps(metric_value.metadata)
            ))
            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Failed to save metric to database: {e}")

    def record_api_performance(self, metrics: APIPerformanceMetrics):
        """记录API性能指标"""
        try:
            self.api_metrics_buffer.append(metrics)

            # 异步保存到数据库
            asyncio.create_task(self._save_api_metrics_to_db(metrics))

            # 检查性能阈值
            asyncio.create_task(self._check_api_thresholds(metrics))

        except Exception as e:
            logger.error(f"Failed to record API performance: {e}")

    async def _save_api_metrics_to_db(self, metrics: APIPerformanceMetrics):
        """保存API性能指标到数据库"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO api_performance
                (endpoint, method, response_time_ms, status_code, request_size_bytes,
                 response_size_bytes, user_agent, session_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.endpoint,
                metrics.method,
                metrics.response_time_ms,
                metrics.status_code,
                metrics.request_size_bytes,
                metrics.response_size_bytes,
                metrics.user_agent,
                metrics.session_id,
                metrics.timestamp.isoformat()
            ))
            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Failed to save API metrics to database: {e}")

    def record_accuracy_metrics(self, metrics: AccuracyMetrics):
        """记录准确性指标"""
        try:
            self.accuracy_metrics_buffer.append(metrics)

            # 异步保存到数据库
            asyncio.create_task(self._save_accuracy_metrics_to_db(metrics))

        except Exception as e:
            logger.error(f"Failed to record accuracy metrics: {e}")

    async def _save_accuracy_metrics_to_db(self, metrics: AccuracyMetrics):
        """保存准确性指标到数据库"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO accuracy_metrics
                (task_type, model_name, ground_truth, prediction, accuracy_score,
                 precision, recall, f1_score, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.task_type,
                metrics.model_name,
                metrics.ground_truth,
                metrics.prediction,
                metrics.accuracy_score,
                metrics.precision,
                metrics.recall,
                metrics.f1_score,
                metrics.timestamp.isoformat()
            ))
            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Failed to save accuracy metrics to database: {e}")

    def start_system_monitoring(self, interval: int = 30):
        """启动系统资源监控"""
        if self.system_monitoring:
            return

        self.system_monitoring = True

        def monitor_system():
            while self.system_monitoring:
                try:
                    # 获取系统指标
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage('/')
                    network_io = psutil.net_io_counters()

                    system_metrics = SystemResourceMetrics(
                        cpu_percent=cpu_percent,
                        memory_percent=memory.percent,
                        memory_available_gb=memory.available / (1024**3),
                        disk_usage_percent=disk.percent,
                        network_io={
                            "bytes_sent": network_io.bytes_sent,
                            "bytes_recv": network_io.bytes_recv,
                            "packets_sent": network_io.packets_sent,
                            "packets_recv": network_io.packets_recv
                        },
                        timestamp=datetime.now(timezone.utc)
                    )

                    # 记录指标
                    self.record_metric(MetricType.RESPONSE_TIME, cpu_percent, "percent", ["system", "cpu"])
                    self.record_metric(MetricType.RESPONSE_TIME, memory.percent, "percent", ["system", "memory"])

                    # 异步保存详细系统指标
                    asyncio.create_task(self._save_system_metrics(system_metrics))

                    time.sleep(interval)

                except Exception as e:
                    logger.error(f"System monitoring error: {e}")
                    time.sleep(interval)

        self.system_monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        self.system_monitor_thread.start()
        logger.info("✅ System monitoring started")

    async def _save_system_metrics(self, metrics: SystemResourceMetrics):
        """保存系统资源指标"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO system_resources
                (cpu_percent, memory_percent, memory_available_gb, disk_usage_percent,
                 network_io, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                metrics.cpu_percent,
                metrics.memory_percent,
                metrics.memory_available_gb,
                metrics.disk_usage_percent,
                json.dumps(metrics.network_io),
                metrics.timestamp.isoformat()
            ))
            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Failed to save system metrics: {e}")

    def add_threshold_rule(self, rule: PerformanceThresholdRule):
        """添加性能阈值规则"""
        self.threshold_rules[rule.rule_id] = rule

        # 保存到数据库
        asyncio.create_task(self._save_threshold_rule(rule))

    async def _save_threshold_rule(self, rule: PerformanceThresholdRule):
        """保存阈值规则到数据库"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO performance_thresholds
                (rule_id, metric_type, threshold_value, comparison_operator,
                 aggregation_window_minutes, alert_cooldown_minutes, enabled, notification_config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rule.rule_id,
                rule.metric_type.value,
                rule.threshold_value,
                rule.comparison_operator,
                int(rule.aggregation_window.total_seconds() / 60),
                int(rule.alert_cooldown.total_seconds() / 60),
                rule.enabled,
                json.dumps(rule.notification_config)
            ))
            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Failed to save threshold rule: {e}")

    async def _check_api_thresholds(self, metrics: APIPerformanceMetrics):
        """检查API性能阈值"""
        for rule in self.threshold_rules.values():
            if not rule.enabled:
                continue

            # 检查响应时间阈值
            if rule.metric_type == MetricType.RESPONSE_TIME:
                await self._evaluate_threshold(rule, metrics.response_time_ms, {
                    "endpoint": metrics.endpoint,
                    "method": metrics.method,
                    "session_id": metrics.session_id
                })

    async def _evaluate_threshold(self, rule: PerformanceThresholdRule, value: float, context: Dict[str, Any]):
        """评估阈值"""
        try:
            # 检查冷却时间
            if rule.rule_id in self.alert_cooldowns:
                if datetime.now(timezone.utc) < self.alert_cooldowns[rule.rule_id]:
                    return

            # 评估条件
            threshold_met = False
            if rule.comparison_operator == ">":
                threshold_met = value > rule.threshold_value
            elif rule.comparison_operator == "<":
                threshold_met = value < rule.threshold_value
            elif rule.comparison_operator == ">=":
                threshold_met = value >= rule.threshold_value
            elif rule.comparison_operator == "<=":
                threshold_met = value <= rule.threshold_value
            elif rule.comparison_operator == "==":
                threshold_met = abs(value - rule.threshold_value) < 0.001

            if threshold_met:
                # 触发警报
                await self._trigger_alert(rule, value, context)
                # 设置冷却时间
                self.alert_cooldowns[rule.rule_id] = datetime.now(timezone.utc) + rule.alert_cooldown

        except Exception as e:
            logger.error(f"Failed to evaluate threshold {rule.rule_id}: {e}")

    async def _trigger_alert(self, rule: PerformanceThresholdRule, value: float, context: Dict[str, Any]):
        """触发警报"""
        try:
            # 记录到数据库
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO performance_alerts
                (rule_id, metric_value, threshold_value, triggered_at, context_data)
                VALUES (?, ?, ?, ?, ?)
            """, (
                rule.rule_id,
                value,
                rule.threshold_value,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(context)
            ))
            self.conn.commit()

            # 发送通知（这里可以集成邮件、Slack等通知系统）
            logger.warning(f"Performance alert triggered: {rule.rule_id} - Value: {value}, Threshold: {rule.threshold_value}")

            # 可以在这里添加实际的通知逻辑
            await self._send_notification(rule, value, context)

        except Exception as e:
            logger.error(f"Failed to trigger alert: {e}")

    async def _send_notification(self, rule: PerformanceThresholdRule, value: float, context: Dict[str, Any]):
        """发送通知"""
        # 这里可以实现各种通知方式
        # 例如：邮件、Slack、Webhook等
        notification_config = rule.notification_config

        if notification_config.get("webhook"):
            # 发送Webhook通知
            pass

        if notification_config.get("email"):
            # 发送邮件通知
            pass

        logger.info(f"Notification sent for rule {rule.rule_id}")

    async def _periodic_aggregation(self):
        """定期聚合指标"""
        while True:
            try:
                for interval in self.aggregation_intervals:
                    await self._aggregate_metrics(interval)

                # 每小时执行一次聚合
                await asyncio.sleep(3600)

            except Exception as e:
                logger.error(f"Error in periodic aggregation: {e}")
                await asyncio.sleep(300)  # 出错时5分钟后重试

    async def _aggregate_metrics(self, interval: timedelta):
        """聚合指定时间间隔的指标"""
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - interval

            cursor = self.conn.cursor()

            # 获取指标类型
            cursor.execute("SELECT DISTINCT metric_type FROM performance_metrics")
            metric_types = [row[0] for row in cursor.fetchall()]

            for metric_type_str in metric_types:
                try:
                    metric_type = MetricType(metric_type_str)

                    # 获取时间范围内的指标
                    cursor.execute("""
                        SELECT value FROM performance_metrics
                        WHERE metric_type = ? AND timestamp >= ? AND timestamp < ?
                    """, (metric_type.value, start_time.isoformat(), end_time.isoformat()))

                    values = [row[0] for row in cursor.fetchall()]

                    if values:
                        # 计算各种聚合值
                        aggregations = [
                            (MetricAggregation.MEAN, statistics.mean(values)),
                            (MetricAggregation.MEDIAN, statistics.median(values)),
                            (MetricAggregation.MIN, min(values)),
                            (MetricAggregation.MAX, max(values))
                        ]

                        if len(values) > 1:
                            aggregations.append((MetricAggregation.P95, np.percentile(values, 95)))
                            aggregations.append((MetricAggregation.P99, np.percentile(values, 99)))

                        # 保存聚合结果
                        for agg_type, agg_value in aggregations:
                            await self._save_aggregated_metric(
                                metric_type, agg_type, agg_value, len(values),
                                start_time, end_time
                            )

                except Exception as e:
                    logger.error(f"Error aggregating metric {metric_type_str}: {e}")

        except Exception as e:
            logger.error(f"Error in metric aggregation: {e}")

    async def _save_aggregated_metric(self,
                                    metric_type: MetricType,
                                    aggregation_type: MetricAggregation,
                                    value: float,
                                    count: int,
                                    period_start: datetime,
                                    period_end: datetime):
        """保存聚合指标"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO aggregated_metrics
                (metric_type, aggregation_type, value, count, period_start, period_end)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                metric_type.value,
                aggregation_type.value,
                value,
                count,
                period_start.isoformat(),
                period_end.isoformat()
            ))
            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Failed to save aggregated metric: {e}")

    async def _periodic_threshold_check(self):
        """定期检查阈值"""
        while True:
            try:
                # 每5分钟检查一次阈值
                await asyncio.sleep(300)

                # 检查各种实时统计的阈值
                for metric_type_str, values in self.real_time_stats.items():
                    if not values:
                        continue

                    try:
                        metric_type = MetricType(metric_type_str)
                        current_value = values[-1]  # 最新值

                        # 检查对应的阈值规则
                        for rule in self.threshold_rules.values():
                            if rule.metric_type == metric_type and rule.enabled:
                                await self._evaluate_threshold(rule, current_value, {
                                    "metric_type": metric_type.value,
                                    "recent_values": list(values)[-10:]  # 最近10个值
                                })

                    except ValueError:
                        continue  # 忽略无效的指标类型

            except Exception as e:
                logger.error(f"Error in periodic threshold check: {e}")

    def get_real_time_stats(self, metric_type: MetricType) -> Dict[str, float]:
        """获取实时统计信息"""
        values = list(self.real_time_stats[metric_type.value])
        if not values:
            return {}

        return {
            "current": values[-1],
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "count": len(values)
        }

    async def get_performance_report(self,
                                   start_time: datetime,
                                   end_time: datetime,
                                   metric_types: Optional[List[MetricType]] = None) -> Dict[str, Any]:
        """获取性能报告"""
        try:
            cursor = self.conn.cursor()

            report = {
                "period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "metrics": {},
                "api_performance": {},
                "accuracy_metrics": {},
                "system_resources": {}
            }

            # 获取原始指标
            if metric_types is None:
                cursor.execute("SELECT DISTINCT metric_type FROM performance_metrics")
                metric_types = [MetricType(row[0]) for row in cursor.fetchall()]

            for metric_type in metric_types:
                cursor.execute("""
                    SELECT value, timestamp FROM performance_metrics
                    WHERE metric_type = ? AND timestamp >= ? AND timestamp < ?
                    ORDER BY timestamp
                """, (metric_type.value, start_time.isoformat(), end_time.isoformat()))

                rows = cursor.fetchall()
                if rows:
                    values = [row[0] for row in rows]
                    report["metrics"][metric_type.value] = {
                        "count": len(values),
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values)
                    }

            # 获取API性能统计
            cursor.execute("""
                SELECT endpoint, method, AVG(response_time_ms) as avg_response_time,
                       COUNT(*) as request_count, SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as error_count
                FROM api_performance
                WHERE timestamp >= ? AND timestamp < ?
                GROUP BY endpoint, method
            """, (start_time.isoformat(), end_time.isoformat()))

            for row in cursor.fetchall():
                endpoint, method, avg_response_time, request_count, error_count = row
                key = f"{method} {endpoint}"
                report["api_performance"][key] = {
                    "avg_response_time_ms": avg_response_time,
                    "request_count": request_count,
                    "error_count": error_count,
                    "error_rate": error_count / request_count if request_count > 0 else 0
                }

            return report

        except Exception as e:
            logger.error(f"Failed to get performance report: {e}")
            return {}

    def stop_system_monitoring(self):
        """停止系统监控"""
        self.system_monitoring = False
        if self.system_monitor_thread:
            self.system_monitor_thread.join(timeout=5)
        logger.info("System monitoring stopped")

    def close(self):
        """关闭连接"""
        try:
            self.stop_system_monitoring()
            if self.conn:
                self.conn.close()
                logger.info("Performance tracker connection closed")
        except Exception as e:
            logger.error(f"Error closing performance tracker: {e}")

# 全局实例
_performance_tracker = None

def get_performance_tracker() -> PerformanceTracker:
    """获取性能跟踪器实例"""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = PerformanceTracker()
    return _performance_tracker