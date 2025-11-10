#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Dashboard Manager
实时仪表板管理器

Comprehensive real-time dashboard system for monitoring system health,
performance metrics, cost analysis, and manufacturing-specific indicators.
"""

import sqlite3
import json
import logging
import asyncio
import websockets
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics
import uuid

from .langfuse_integration import get_langfuse_integration
from .performance_tracker import get_performance_tracker, MetricType
from .cost_analyzer import get_cost_analyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DashboardType(Enum):
    """仪表板类型"""
    SYSTEM_HEALTH = "system_health"
    PERFORMANCE = "performance"
    COST_ANALYSIS = "cost_analysis"
    USER_BEHAVIOR = "user_behavior"
    MANUFACTURING_METRICS = "manufacturing_metrics"
    ALERT_CENTER = "alert_center"

class AlertSeverity(Enum):
    """警报严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class WidgetType(Enum):
    """小部件类型"""
    METRIC_CARD = "metric_card"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    TABLE = "table"
    GAUGE = "gauge"
    HEATMAP = "heatmap"
    ALERT_LIST = "alert_list"

@dataclass
class DashboardWidget:
    """仪表板小部件"""
    widget_id: str
    widget_type: WidgetType
    title: str
    position: Dict[str, int]  # {"x": 0, "y": 0, "w": 4, "h": 3}
    data_source: str
    config: Dict[str, Any] = field(default_factory=dict)
    refresh_interval: int = 30  # seconds
    visible: bool = True

@dataclass
class Dashboard:
    """仪表板配置"""
    dashboard_id: str
    name: str
    description: str
    dashboard_type: DashboardType
    widgets: List[DashboardWidget]
    layout: Dict[str, Any] = field(default_factory=dict)
    is_public: bool = False
    created_by: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class Alert:
    """警报"""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    source: str
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RealTimeMetric:
    """实时指标"""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    trend: Optional[str] = None  # "up", "down", "stable"
    change_percent: Optional[float] = None
    status: str = "normal"  # "normal", "warning", "critical"

class DashboardManager:
    """仪表板管理器"""

    def __init__(self,
                 db_path: str = "knowledge_base.db",
                 websocket_port: int = 8765,
                 enable_websocket: bool = True):
        """
        初始化仪表板管理器

        Args:
            db_path: SQLite数据库路径
            websocket_port: WebSocket端口
            enable_websocket: 是否启用WebSocket
        """
        self.db_path = db_path
        self.websocket_port = websocket_port
        self.enable_websocket = enable_websocket

        # 获取其他组件实例
        self.langfuse_integration = get_langfuse_integration()
        self.performance_tracker = get_performance_tracker()
        self.cost_analyzer = get_cost_analyzer()

        # 内存存储
        self.dashboards: Dict[str, Dashboard] = {}
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.active_alerts: Dict[str, Alert] = {}
        self.metric_cache: Dict[str, RealTimeMetric] = {}

        # WebSocket服务器
        self.websocket_server = None
        self.websocket_thread = None

        # 初始化数据库
        self._init_database()

        # 启动后台任务
        self._start_background_tasks()

        # 创建默认仪表板
        self._create_default_dashboards()

    def _init_database(self):
        """初始化数据库表"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute("PRAGMA foreign_keys = ON")

            self.conn.executescript("""
                -- 仪表板配置表
                CREATE TABLE IF NOT EXISTS dashboards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dashboard_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    dashboard_type TEXT NOT NULL,
                    widgets TEXT NOT NULL,
                    layout TEXT,
                    is_public BOOLEAN DEFAULT 0,
                    created_by TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- 警报表
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    severity TEXT NOT NULL,
                    source TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    acknowledged BOOLEAN DEFAULT 0,
                    resolved BOOLEAN DEFAULT 0,
                    acknowledged_by TEXT,
                    acknowledged_at DATETIME,
                    resolved_by TEXT,
                    resolved_at DATETIME,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- 实时指标缓存表
                CREATE TABLE IF NOT EXISTS realtime_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT UNIQUE NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT,
                    timestamp DATETIME NOT NULL,
                    trend TEXT,
                    change_percent REAL,
                    status TEXT DEFAULT 'normal',
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- 用户偏好设置表
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    dashboard_id TEXT,
                    preferences TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, dashboard_id)
                );

                -- 创建索引
                CREATE INDEX IF NOT EXISTS idx_dashboards_type ON dashboards(dashboard_type);
                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
                CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
                CREATE INDEX IF NOT EXISTS idx_realtime_metrics_name ON realtime_metrics(metric_name);
            """)

            logger.info("✅ Dashboard database initialized")

        except sqlite3.Error as e:
            logger.error(f"Failed to initialize dashboard database: {e}")
            raise

    def _start_background_tasks(self):
        """启动后台任务"""
        # 启动WebSocket服务器
        if self.enable_websocket:
            self._start_websocket_server()

        # 启动数据更新任务
        asyncio.create_task(self._update_metrics_loop())
        asyncio.create_task(self._check_alerts_loop())
        asyncio.create_task(self._cleanup_old_data())

    def _start_websocket_server(self):
        """启动WebSocket服务器"""
        async def handle_client(websocket, path):
            """处理WebSocket客户端连接"""
            try:
                self.connected_clients.add(websocket)
                logger.info(f"New dashboard client connected: {websocket.remote_address}")

                # 发送初始数据
                await self._send_initial_data(websocket)

                # 保持连接并处理消息
                async for message in websocket:
                    await self._handle_client_message(websocket, message)

            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Dashboard client disconnected: {websocket.remote_address}")
            except Exception as e:
                logger.error(f"Error handling dashboard client: {e}")
            finally:
                self.connected_clients.discard(websocket)

        def run_server():
            """运行WebSocket服务器"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            start_server = websockets.serve(
                handle_client,
                "localhost",
                self.websocket_port,
                ping_interval=20,
                ping_timeout=10
            )

            loop.run_until_complete(start_server)
            logger.info(f"Dashboard WebSocket server started on port {self.websocket_port}")
            loop.run_forever()

        self.websocket_thread = threading.Thread(target=run_server, daemon=True)
        self.websocket_thread.start()

    async def _send_initial_data(self, websocket):
        """发送初始数据给新连接的客户端"""
        try:
            # 发送所有仪表板
            dashboards_data = [self._dashboard_to_dict(dashboard) for dashboard in self.dashboards.values()]

            initial_data = {
                "type": "initial_data",
                "dashboards": dashboards_data,
                "alerts": [asdict(alert) for alert in self.active_alerts.values()],
                "metrics": {name: asdict(metric) for name, metric in self.metric_cache.items()},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            await websocket.send(json.dumps(initial_data))

        except Exception as e:
            logger.error(f"Error sending initial data: {e}")

    async def _handle_client_message(self, websocket, message):
        """处理客户端消息"""
        try:
            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "get_dashboard":
                dashboard_id = data.get("dashboard_id")
                if dashboard_id in self.dashboards:
                    response = {
                        "type": "dashboard_data",
                        "dashboard": self._dashboard_to_dict(self.dashboards[dashboard_id]),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    await websocket.send(json.dumps(response))

            elif message_type == "acknowledge_alert":
                alert_id = data.get("alert_id")
                await self.acknowledge_alert(alert_id, "dashboard_user")

            elif message_type == "resolve_alert":
                alert_id = data.get("alert_id")
                await self.resolve_alert(alert_id, "dashboard_user")

            elif message_type == "refresh_metrics":
                await self._send_metrics_update()

        except Exception as e:
            logger.error(f"Error handling client message: {e}")

    async def _update_metrics_loop(self):
        """更新指标循环"""
        while True:
            try:
                await self._update_all_metrics()
                await self._broadcast_metrics_update()
                await asyncio.sleep(5)  # 每5秒更新一次

            except Exception as e:
                logger.error(f"Error in metrics update loop: {e}")
                await asyncio.sleep(10)

    async def _update_all_metrics(self):
        """更新所有指标"""
        try:
            # 系统性能指标
            await self._update_system_metrics()

            # 制造业特定指标
            await self._update_manufacturing_metrics()

            # 成本指标
            await self._update_cost_metrics()

            # 用户活动指标
            await self._update_user_activity_metrics()

        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    async def _update_system_metrics(self):
        """更新系统指标"""
        try:
            # API响应时间
            api_stats = self.performance_tracker.get_real_time_stats(MetricType.RESPONSE_TIME)
            if api_stats:
                self._update_metric("api_response_time", api_stats["current"], "ms")

            # CPU使用率
            cpu_stats = self.performance_tracker.get_real_time_stats(MetricType.RESPONSE_TIME)
            if cpu_stats:
                self._update_metric("cpu_usage", cpu_stats["current"], "%")

            # 内存使用率
            memory_stats = self.performance_tracker.get_real_time_stats(MetricType.RESPONSE_TIME)
            if memory_stats:
                self._update_metric("memory_usage", memory_stats["current"], "%")

            # 错误率
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(CASE WHEN status_code >= 400 THEN 1 END) * 100.0 / COUNT(*) as error_rate
                FROM api_performance
                WHERE timestamp >= datetime('now', '-5 minutes')
            """)

            result = cursor.fetchone()
            if result and result[0] is not None:
                self._update_metric("error_rate", result[0], "%")

        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")

    async def _update_manufacturing_metrics(self):
        """更新制造业指标"""
        try:
            # 获取今日制造业成本指标
            today = datetime.now(timezone.utc).date()
            manufacturing_metrics = await self.cost_analyzer.get_manufacturing_cost_metrics(datetime.combine(today, datetime.min.time()))

            self._update_metric("cost_per_quote", manufacturing_metrics.cost_per_quote, "USD")
            self._update_metric("cost_per_document", manufacturing_metrics.cost_per_document_processed, "USD")
            self._update_metric("rag_query_cost", manufacturing_metrics.cost_rag_per_query, "USD")

            # 获取今日处理数量
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(DISTINCT session_id) as active_sessions,
                    COUNT(*) as total_interactions
                FROM observability_traces
                WHERE DATE(timestamp) = date('now')
            """)

            result = cursor.fetchone()
            if result:
                self._update_metric("active_sessions", result[0] or 0, "count")
                self._update_metric("total_interactions", result[1] or 0, "count")

        except Exception as e:
            logger.error(f"Error updating manufacturing metrics: {e}")

    async def _update_cost_metrics(self):
        """更新成本指标"""
        try:
            # 今日成本
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT SUM(total_cost) as daily_cost
                FROM cost_records
                WHERE DATE(timestamp) = date('now')
            """)

            result = cursor.fetchone()
            if result and result[0] is not None:
                self._update_metric("daily_cost", result[0], "USD")

            # 本月累计成本
            cursor.execute("""
                SELECT SUM(total_cost) as monthly_cost
                FROM cost_records
                WHERE timestamp >= date('now', '-30 days')
            """)

            result = cursor.fetchone()
            if result and result[0] is not None:
                self._update_metric("monthly_cost", result[0], "USD")

        except Exception as e:
            logger.error(f"Error updating cost metrics: {e}")

    async def _update_user_activity_metrics(self):
        """更新用户活动指标"""
        try:
            cursor = self.conn.cursor()

            # 今日活跃用户数
            cursor.execute("""
                SELECT COUNT(DISTINCT user_id) as active_users
                FROM observability_traces
                WHERE DATE(timestamp) = date('now') AND user_id IS NOT NULL
            """)

            result = cursor.fetchone()
            if result and result[0] is not None:
                self._update_metric("active_users", result[0], "count")

            # 平均会话时长
            cursor.execute("""
                SELECT AVG(julianday(MAX(timestamp)) - julianday(MIN(timestamp))) * 24 * 60 as avg_session_minutes
                FROM observability_traces
                WHERE DATE(timestamp) = date('now')
                GROUP BY session_id
            """)

            result = cursor.fetchone()
            if result and result[0] is not None:
                self._update_metric("avg_session_duration", result[0], "minutes")

        except Exception as e:
            logger.error(f"Error updating user activity metrics: {e}")

    def _update_metric(self, metric_name: str, value: float, unit: str):
        """更新单个指标"""
        try:
            timestamp = datetime.now(timezone.utc)

            # 计算趋势和变化百分比
            trend = None
            change_percent = None
            status = "normal"

            if metric_name in self.metric_cache:
                old_metric = self.metric_cache[metric_name]
                if value > old_metric.value:
                    trend = "up"
                    change_percent = ((value - old_metric.value) / old_metric.value) * 100 if old_metric.value != 0 else 0
                elif value < old_metric.value:
                    trend = "down"
                    change_percent = ((old_metric.value - value) / old_metric.value) * 100 if old_metric.value != 0 else 0
                else:
                    trend = "stable"
                    change_percent = 0

                # 确定状态
                if metric_name == "error_rate" and value > 5:
                    status = "critical"
                elif metric_name == "error_rate" and value > 2:
                    status = "warning"
                elif metric_name == "cpu_usage" and value > 90:
                    status = "critical"
                elif metric_name == "cpu_usage" and value > 80:
                    status = "warning"
                elif metric_name == "memory_usage" and value > 90:
                    status = "critical"
                elif metric_name == "memory_usage" and value > 80:
                    status = "warning"

            metric = RealTimeMetric(
                metric_name=metric_name,
                value=value,
                unit=unit,
                timestamp=timestamp,
                trend=trend,
                change_percent=change_percent,
                status=status
            )

            self.metric_cache[metric_name] = metric

            # 保存到数据库
            self._save_metric_to_db(metric)

        except Exception as e:
            logger.error(f"Error updating metric {metric_name}: {e}")

    def _save_metric_to_db(self, metric: RealTimeMetric):
        """保存指标到数据库"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO realtime_metrics
                (metric_name, value, unit, timestamp, trend, change_percent, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.metric_name,
                metric.value,
                metric.unit,
                metric.timestamp.isoformat(),
                metric.trend,
                metric.change_percent,
                metric.status
            ))
            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Failed to save metric to database: {e}")

    async def _broadcast_metrics_update(self):
        """广播指标更新给所有客户端"""
        if not self.connected_clients:
            return

        try:
            message = {
                "type": "metrics_update",
                "metrics": {name: asdict(metric) for name, metric in self.metric_cache.items()},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            await self._broadcast_message(message)

        except Exception as e:
            logger.error(f"Error broadcasting metrics update: {e}")

    async def _broadcast_message(self, message: Dict[str, Any]):
        """广播消息给所有连接的客户端"""
        if not self.connected_clients:
            return

        message_str = json.dumps(message)
        disconnected_clients = set()

        for client in self.connected_clients:
            try:
                await client.send(message_str)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error sending message to client: {e}")
                disconnected_clients.add(client)

        # 移除断开的客户端
        self.connected_clients -= disconnected_clients

    def _create_default_dashboards(self):
        """创建默认仪表板"""
        try:
            # 系统健康仪表板
            system_health_dashboard = Dashboard(
                dashboard_id="system_health",
                name="System Health",
                description="Real-time system health monitoring",
                dashboard_type=DashboardType.SYSTEM_HEALTH,
                widgets=[
                    DashboardWidget(
                        widget_id="cpu_usage",
                        widget_type=WidgetType.GAUGE,
                        title="CPU Usage",
                        position={"x": 0, "y": 0, "w": 6, "h": 4},
                        data_source="cpu_usage",
                        config={"min": 0, "max": 100, "thresholds": {"warning": 80, "critical": 90}}
                    ),
                    DashboardWidget(
                        widget_id="memory_usage",
                        widget_type=WidgetType.GAUGE,
                        title="Memory Usage",
                        position={"x": 6, "y": 0, "w": 6, "h": 4},
                        data_source="memory_usage",
                        config={"min": 0, "max": 100, "thresholds": {"warning": 80, "critical": 90}}
                    ),
                    DashboardWidget(
                        widget_id="api_response_time",
                        widget_type=WidgetType.METRIC_CARD,
                        title="API Response Time",
                        position={"x": 0, "y": 4, "w": 4, "h": 2},
                        data_source="api_response_time"
                    ),
                    DashboardWidget(
                        widget_id="error_rate",
                        widget_type=WidgetType.METRIC_CARD,
                        title="Error Rate",
                        position={"x": 4, "y": 4, "w": 4, "h": 2},
                        data_source="error_rate"
                    ),
                    DashboardWidget(
                        widget_id="active_sessions",
                        widget_type=WidgetType.METRIC_CARD,
                        title="Active Sessions",
                        position={"x": 8, "y": 4, "w": 4, "h": 2},
                        data_source="active_sessions"
                    )
                ],
                is_public=True
            )

            self.dashboards["system_health"] = system_health_dashboard
            self._save_dashboard_to_db(system_health_dashboard)

            # 制造业指标仪表板
            manufacturing_dashboard = Dashboard(
                dashboard_id="manufacturing_metrics",
                name="Manufacturing Metrics",
                description="Manufacturing-specific performance indicators",
                dashboard_type=DashboardType.MANUFACTURING_METRICS,
                widgets=[
                    DashboardWidget(
                        widget_id="cost_per_quote",
                        widget_type=WidgetType.METRIC_CARD,
                        title="Cost per Quote",
                        position={"x": 0, "y": 0, "w": 4, "h": 3},
                        data_source="cost_per_quote",
                        config={"currency": "USD"}
                    ),
                    DashboardWidget(
                        widget_id="cost_per_document",
                        widget_type=WidgetType.METRIC_CARD,
                        title="Cost per Document",
                        position={"x": 4, "y": 0, "w": 4, "h": 3},
                        data_source="cost_per_document",
                        config={"currency": "USD"}
                    ),
                    DashboardWidget(
                        widget_id="rag_query_cost",
                        widget_type=WidgetType.METRIC_CARD,
                        title="RAG Query Cost",
                        position={"x": 8, "y": 0, "w": 4, "h": 3},
                        data_source="rag_query_cost",
                        config={"currency": "USD"}
                    ),
                    DashboardWidget(
                        widget_id="total_interactions",
                        widget_type=WidgetType.METRIC_CARD,
                        title="Total Interactions",
                        position={"x": 0, "y": 3, "w": 4, "h": 2},
                        data_source="total_interactions"
                    ),
                    DashboardWidget(
                        widget_id="active_users",
                        widget_type=WidgetType.METRIC_CARD,
                        title="Active Users",
                        position={"x": 4, "y": 3, "w": 4, "h": 2},
                        data_source="active_users"
                    ),
                    DashboardWidget(
                        widget_id="daily_cost",
                        widget_type=WidgetType.METRIC_CARD,
                        title="Daily Cost",
                        position={"x": 8, "y": 3, "w": 4, "h": 2},
                        data_source="daily_cost",
                        config={"currency": "USD"}
                    )
                ],
                is_public=True
            )

            self.dashboards["manufacturing_metrics"] = manufacturing_dashboard
            self._save_dashboard_to_db(manufacturing_dashboard)

            # 警报中心仪表板
            alert_dashboard = Dashboard(
                dashboard_id="alert_center",
                name="Alert Center",
                description="System alerts and notifications",
                dashboard_type=DashboardType.ALERT_CENTER,
                widgets=[
                    DashboardWidget(
                        widget_id="alert_list",
                        widget_type=WidgetType.ALERT_LIST,
                        title="Active Alerts",
                        position={"x": 0, "y": 0, "w": 12, "h": 6},
                        data_source="alerts",
                        config={"auto_refresh": 10}
                    )
                ],
                is_public=True
            )

            self.dashboards["alert_center"] = alert_dashboard
            self._save_dashboard_to_db(alert_dashboard)

            logger.info("✅ Default dashboards created successfully")

        except Exception as e:
            logger.error(f"Failed to create default dashboards: {e}")

    def _save_dashboard_to_db(self, dashboard: Dashboard):
        """保存仪表板到数据库"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO dashboards
                (dashboard_id, name, description, dashboard_type, widgets, layout, is_public, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                dashboard.dashboard_id,
                dashboard.name,
                dashboard.description,
                dashboard.dashboard_type.value,
                json.dumps([asdict(widget) for widget in dashboard.widgets]),
                json.dumps(dashboard.layout),
                dashboard.is_public,
                dashboard.created_by
            ))
            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Failed to save dashboard to database: {e}")

    def _dashboard_to_dict(self, dashboard: Dashboard) -> Dict[str, Any]:
        """将仪表板转换为字典"""
        return {
            "dashboard_id": dashboard.dashboard_id,
            "name": dashboard.name,
            "description": dashboard.description,
            "dashboard_type": dashboard.dashboard_type.value,
            "widgets": [asdict(widget) for widget in dashboard.widgets],
            "layout": dashboard.layout,
            "is_public": dashboard.is_public,
            "created_by": dashboard.created_by,
            "created_at": dashboard.created_at.isoformat()
        }

    async def create_alert(self,
                          title: str,
                          description: str,
                          severity: AlertSeverity,
                          source: str,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """创建警报"""
        try:
            alert_id = str(uuid.uuid4())
            alert = Alert(
                alert_id=alert_id,
                title=title,
                description=description,
                severity=severity,
                source=source,
                timestamp=datetime.now(timezone.utc),
                metadata=metadata or {}
            )

            self.active_alerts[alert_id] = alert
            await self._save_alert_to_db(alert)

            # 广播新警报
            await self._broadcast_alert(alert)

            logger.info(f"Alert created: {title} ({severity.value})")
            return alert_id

        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            raise

    async def _save_alert_to_db(self, alert: Alert):
        """保存警报到数据库"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO alerts
                (alert_id, title, description, severity, source, timestamp,
                 acknowledged, resolved, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_id,
                alert.title,
                alert.description,
                alert.severity.value,
                alert.source,
                alert.timestamp.isoformat(),
                alert.acknowledged,
                alert.resolved,
                json.dumps(alert.metadata)
            ))
            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Failed to save alert to database: {e}")

    async def _broadcast_alert(self, alert: Alert):
        """广播警报给所有客户端"""
        try:
            message = {
                "type": "new_alert",
                "alert": asdict(alert),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            await self._broadcast_message(message)

        except Exception as e:
            logger.error(f"Error broadcasting alert: {e}")

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """确认警报"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now(timezone.utc)

                await self._save_alert_to_db(alert)

                # 广播更新
                await self._broadcast_alert_update(alert)

                logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")

        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")

    async def resolve_alert(self, alert_id: str, resolved_by: str):
        """解决警报"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_by = resolved_by
                alert.resolved_at = datetime.now(timezone.utc)

                await self._save_alert_to_db(alert)

                # 广播更新
                await self._broadcast_alert_update(alert)

                # 从活动警报中移除
                del self.active_alerts[alert_id]

                logger.info(f"Alert {alert_id} resolved by {resolved_by}")

        except Exception as e:
            logger.error(f"Failed to resolve alert: {e}")

    async def _broadcast_alert_update(self, alert: Alert):
        """广播警报更新"""
        try:
            message = {
                "type": "alert_update",
                "alert": asdict(alert),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            await self._broadcast_message(message)

        except Exception as e:
            logger.error(f"Error broadcasting alert update: {e}")

    async def _check_alerts_loop(self):
        """检查警报循环"""
        while True:
            try:
                # 每分钟检查一次警报条件
                await self._check_system_alerts()
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Error in alerts check loop: {e}")
                await asyncio.sleep(60)

    async def _check_system_alerts(self):
        """检查系统警报条件"""
        try:
            # 检查CPU使用率
            if "cpu_usage" in self.metric_cache:
                cpu_metric = self.metric_cache["cpu_usage"]
                if cpu_metric.value > 95:
                    await self.create_alert(
                        title="Critical CPU Usage",
                        description=f"CPU usage is at {cpu_metric.value:.1f}%",
                        severity=AlertSeverity.CRITICAL,
                        source="system_monitor",
                        metadata={"cpu_usage": cpu_metric.value}
                    )
                elif cpu_metric.value > 85:
                    await self.create_alert(
                        title="High CPU Usage",
                        description=f"CPU usage is at {cpu_metric.value:.1f}%",
                        severity=AlertSeverity.HIGH,
                        source="system_monitor",
                        metadata={"cpu_usage": cpu_metric.value}
                    )

            # 检查内存使用率
            if "memory_usage" in self.metric_cache:
                memory_metric = self.metric_cache["memory_usage"]
                if memory_metric.value > 95:
                    await self.create_alert(
                        title="Critical Memory Usage",
                        description=f"Memory usage is at {memory_metric.value:.1f}%",
                        severity=AlertSeverity.CRITICAL,
                        source="system_monitor",
                        metadata={"memory_usage": memory_metric.value}
                    )

            # 检查错误率
            if "error_rate" in self.metric_cache:
                error_metric = self.metric_cache["error_rate"]
                if error_metric.value > 10:
                    await self.create_alert(
                        title="High Error Rate",
                        description=f"Error rate is at {error_metric.value:.1f}%",
                        severity=AlertSeverity.HIGH,
                        source="system_monitor",
                        metadata={"error_rate": error_metric.value}
                    )

        except Exception as e:
            logger.error(f"Error checking system alerts: {e}")

    async def _cleanup_old_data(self):
        """清理旧数据"""
        while True:
            try:
                # 每天清理一次超过30天的数据
                await asyncio.sleep(24 * 3600)

                cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)

                # 清理旧的警报表记录
                cursor = self.conn.cursor()
                cursor.execute("""
                    DELETE FROM alerts
                    WHERE timestamp < ? AND resolved = 1
                """, (cutoff_date.isoformat(),))

                deleted_count = cursor.rowcount
                if deleted_count > 0:
                    self.conn.commit()
                    logger.info(f"Cleaned up {deleted_count} old resolved alerts")

            except Exception as e:
                logger.error(f"Error cleaning up old data: {e}")

    async def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """获取仪表板数据"""
        try:
            if dashboard_id not in self.dashboards:
                return {"error": "Dashboard not found"}

            dashboard = self.dashboards[dashboard_id]

            # 获取小部件数据
            widget_data = {}
            for widget in dashboard.widgets:
                if widget.data_source in self.metric_cache:
                    metric = self.metric_cache[widget.data_source]
                    widget_data[widget.widget_id] = asdict(metric)

            return {
                "dashboard": self._dashboard_to_dict(dashboard),
                "widget_data": widget_data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {"error": str(e)}

    def close(self):
        """关闭连接"""
        try:
            if self.conn:
                self.conn.close()
                logger.info("Dashboard manager connection closed")
        except Exception as e:
            logger.error(f"Error closing dashboard manager: {e}")

# 全局实例
_dashboard_manager = None

def get_dashboard_manager() -> DashboardManager:
    """获取仪表板管理器实例"""
    global _dashboard_manager
    if _dashboard_manager is None:
        _dashboard_manager = DashboardManager()
    return _dashboard_manager