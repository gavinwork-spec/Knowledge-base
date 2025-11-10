#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Alert System for Manufacturing Knowledge Base
制造业知识库高级警报系统

Comprehensive alerting system with anomaly detection, pattern recognition,
and intelligent notification routing for manufacturing-specific scenarios.
"""

import sqlite3
import json
import logging
import asyncio
import smtplib
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics
import numpy as np
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

from .langfuse_integration import get_langfuse_integration
from .performance_tracker import get_performance_tracker, MetricType
from .dashboard_manager import AlertSeverity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlertCategory(Enum):
    """警报类别"""
    SYSTEM_PERFORMANCE = "system_performance"
    API_HEALTH = "api_health"
    COST_MANAGEMENT = "cost_management"
    USER_EXPERIENCE = "user_experience"
    KNOWLEDGE_QUALITY = "knowledge_quality"
    MANUFACTURING_OPERATIONS = "manufacturing_operations"
    SECURITY = "security"
    DATA_INTEGRITY = "data_integrity"

class NotificationChannel(Enum):
    """通知渠道"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    DASHBOARD = "dashboard"

class AnomalyDetectionType(Enum):
    """异常检测类型"""
    STATISTICAL = "statistical"
    THRESHOLD = "threshold"
    PATTERN = "pattern"
    MACHINE_LEARNING = "machine_learning"

@dataclass
class AlertRule:
    """警报规则"""
    rule_id: str
    name: str
    description: str
    category: AlertCategory
    metric_name: str
    detection_type: AnomalyDetectionType
    threshold_value: Optional[float]
    threshold_operator: Optional[str]  # ">", "<", ">=", "<=", "=="
    time_window_minutes: int
    evaluation_frequency_minutes: int
    notification_channels: List[NotificationChannel]
    notification_recipients: List[str]
    enabled: bool = True
    cooldown_minutes: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlertIncident:
    """警报事件"""
    incident_id: str
    rule_id: str
    severity: AlertSeverity
    title: str
    description: str
    metric_value: float
    threshold_value: Optional[float]
    context_data: Dict[str, Any]
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    notifications_sent: List[str] = field(default_factory=list)

@dataclass
class ManufacturingAlertContext:
    """制造业警报上下文"""
    equipment_model: Optional[str] = None
    process_type: Optional[str] = None
    standard_reference: Optional[str] = None
    quality_metric: Optional[str] = None
    customer_impact: Optional[str] = None
    production_schedule_impact: Optional[str] = None

class AnomalyDetector:
    """异常检测器"""

    def __init__(self):
        self.historical_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.baseline_stats: Dict[str, Dict[str, float]] = {}

    def add_data_point(self, metric_name: str, value: float, timestamp: datetime):
        """添加数据点"""
        self.historical_data[metric_name].append({
            'value': value,
            'timestamp': timestamp
        })

        # 更新基线统计
        if len(self.historical_data[metric_name]) >= 30:
            values = [point['value'] for point in self.historical_data[metric_name]]
            self.baseline_stats[metric_name] = {
                'mean': statistics.mean(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'p95': np.percentile(values, 95),
                'p5': np.percentile(values, 5)
            }

    def detect_statistical_anomaly(self,
                                 metric_name: str,
                                 current_value: float,
                                 z_score_threshold: float = 3.0) -> Tuple[bool, float]:
        """检测统计异常"""
        if metric_name not in self.baseline_stats:
            return False, 0.0

        stats = self.baseline_stats[metric_name]
        if stats['std'] == 0:
            return False, 0.0

        z_score = abs(current_value - stats['mean']) / stats['std']
        is_anomaly = z_score > z_score_threshold

        return is_anomaly, z_score

    def detect_pattern_anomaly(self,
                             metric_name: str,
                             current_value: float,
                             time_window_minutes: int = 60) -> Tuple[bool, str]:
        """检测模式异常"""
        if len(self.historical_data[metric_name]) < 10:
            return False, "insufficient_data"

        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=time_window_minutes)
        recent_values = [
            point['value'] for point in self.historical_data[metric_name]
            if point['timestamp'] >= cutoff_time
        ]

        if len(recent_values) < 3:
            return False, "insufficient_recent_data"

        # 检测趋势异常
        if len(recent_values) >= 5:
            trend_slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
            if abs(trend_slope) > (self.baseline_stats[metric_name]['std'] * 0.1):
                return True, f"trend_anomaly_slope_{trend_slope:.3f}"

        # 检测波动性异常
        recent_std = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
        baseline_std = self.baseline_stats[metric_name]['std']
        if recent_std > baseline_std * 3:
            return True, f"volatility_anomaly_std_{recent_std:.3f}"

        return False, "no_anomaly"

class AlertSystem:
    """警报系统"""

    def __init__(self,
                 db_path: str = "knowledge_base.db",
                 email_config: Optional[Dict[str, Any]] = None,
                 slack_config: Optional[Dict[str, Any]] = None):
        """
        初始化警报系统

        Args:
            db_path: SQLite数据库路径
            email_config: 邮件配置
            slack_config: Slack配置
        """
        self.db_path = db_path
        self.email_config = email_config or {}
        self.slack_config = slack_config or {}

        # 获取其他组件
        self.langfuse_integration = get_langfuse_integration()
        self.performance_tracker = get_performance_tracker()

        # 异常检测器
        self.anomaly_detector = AnomalyDetector()

        # 内存存储
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_incidents: Dict[str, AlertIncident] = {}
        self.alert_cooldowns: Dict[str, datetime] = {}

        # 初始化数据库
        self._init_database()

        # 加载默认规则
        self._load_default_rules()

        # 启动后台任务
        self._start_background_tasks()

    def _init_database(self):
        """初始化数据库表"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute("PRAGMA foreign_keys = ON")

            self.conn.executescript("""
                -- 警报规则表
                CREATE TABLE IF NOT EXISTS alert_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    category TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    detection_type TEXT NOT NULL,
                    threshold_value REAL,
                    threshold_operator TEXT,
                    time_window_minutes INTEGER NOT NULL,
                    evaluation_frequency_minutes INTEGER NOT NULL,
                    notification_channels TEXT,
                    notification_recipients TEXT,
                    enabled BOOLEAN DEFAULT 1,
                    cooldown_minutes INTEGER DEFAULT 30,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- 警报事件表
                CREATE TABLE IF NOT EXISTS alert_incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id TEXT UNIQUE NOT NULL,
                    rule_id TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    metric_value REAL,
                    threshold_value REAL,
                    context_data TEXT,
                    created_at DATETIME NOT NULL,
                    acknowledged_at DATETIME,
                    acknowledged_by TEXT,
                    resolved_at DATETIME,
                    resolved_by TEXT,
                    notifications_sent TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- 警报通知历史表
                CREATE TABLE IF NOT EXISTS alert_notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    status TEXT NOT NULL,
                    sent_at DATETIME NOT NULL,
                    response_data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- 制造业警报上下文表
                CREATE TABLE IF NOT EXISTS manufacturing_alert_contexts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id TEXT UNIQUE NOT NULL,
                    equipment_model TEXT,
                    process_type TEXT,
                    standard_reference TEXT,
                    quality_metric TEXT,
                    customer_impact TEXT,
                    production_schedule_impact TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- 创建索引
                CREATE INDEX IF NOT EXISTS idx_alert_rules_category ON alert_rules(category);
                CREATE INDEX IF NOT EXISTS idx_alert_rules_enabled ON alert_rules(enabled);
                CREATE INDEX IF NOT EXISTS idx_incidents_severity ON alert_incidents(severity);
                CREATE INDEX IF NOT EXISTS idx_incidents_created ON alert_incidents(created_at);
                CREATE INDEX IF NOT EXISTS idx_notifications_incident ON alert_notifications(incident_id);
            """)

            logger.info("✅ Alert system database initialized")

        except sqlite3.Error as e:
            logger.error(f"Failed to initialize alert system database: {e}")
            raise

    def _load_default_rules(self):
        """加载默认警报规则"""
        default_rules = [
            # 系统性能规则
            AlertRule(
                rule_id="cpu_usage_critical",
                name="Critical CPU Usage",
                description="Alert when CPU usage exceeds 95%",
                category=AlertCategory.SYSTEM_PERFORMANCE,
                metric_name="cpu_usage",
                detection_type=AnomalyDetectionType.THRESHOLD,
                threshold_value=95.0,
                threshold_operator=">",
                time_window_minutes=5,
                evaluation_frequency_minutes=1,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.DASHBOARD],
                notification_recipients=["admin@company.com"]
            ),
            AlertRule(
                rule_id="error_rate_high",
                name="High Error Rate",
                description="Alert when error rate exceeds 10%",
                category=AlertCategory.API_HEALTH,
                metric_name="error_rate",
                detection_type=AnomalyDetectionType.THRESHOLD,
                threshold_value=10.0,
                threshold_operator=">",
                time_window_minutes=10,
                evaluation_frequency_minutes=2,
                notification_channels=[NotificationChannel.SLACK, NotificationChannel.DASHBOARD],
                notification_recipients=["devops@company.com"]
            ),
            AlertRule(
                rule_id="cost_spike",
                name="Daily Cost Spike",
                description="Alert when daily cost increases by 50% compared to baseline",
                category=AlertCategory.COST_MANAGEMENT,
                metric_name="daily_cost",
                detection_type=AnomalyDetectionType.STATISTICAL,
                threshold_value=1.5,
                threshold_operator="*",
                time_window_minutes=60,
                evaluation_frequency_minutes=30,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.DASHBOARD],
                notification_recipients=["finance@company.com"]
            ),
            # 制造业特定规则
            AlertRule(
                rule_id="quote_accuracy_low",
                name="Low Quote Accuracy",
                description="Alert when quote accuracy drops below 80%",
                category=AlertCategory.MANUFACTURING_OPERATIONS,
                metric_name="quote_accuracy",
                detection_type=AnomalyDetectionType.THRESHOLD,
                threshold_value=80.0,
                threshold_operator="<",
                time_window_minutes=60,
                evaluation_frequency_minutes=15,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
                notification_recipients=["sales@company.com", "quality@company.com"]
            ),
            AlertRule(
                rule_id="customer_satisfaction_low",
                name="Low Customer Satisfaction",
                description="Alert when average customer satisfaction drops below 3.0",
                category=AlertCategory.USER_EXPERIENCE,
                metric_name="customer_satisfaction",
                detection_type=AnomalyDetectionType.THRESHOLD,
                threshold_value=3.0,
                threshold_operator="<",
                time_window_minutes=120,
                evaluation_frequency_minutes=30,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.DASHBOARD],
                notification_recipients=["support@company.com", "management@company.com"]
            ),
            AlertRule(
                rule_id="document_processing_failure",
                name="Document Processing Failures",
                description="Alert when document processing success rate drops below 90%",
                category=AlertCategory.KNOWLEDGE_QUALITY,
                metric_name="document_processing_success_rate",
                detection_type=AnomalyDetectionType.THRESHOLD,
                threshold_value=90.0,
                threshold_operator="<",
                time_window_minutes=30,
                evaluation_frequency_minutes=5,
                notification_channels=[NotificationChannel.SLACK, NotificationChannel.DASHBOARD],
                notification_recipients["tech@company.com"]
            )
        ]

        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
            self._save_rule_to_db(rule)

        logger.info(f"✅ Loaded {len(default_rules)} default alert rules")

    def _save_rule_to_db(self, rule: AlertRule):
        """保存规则到数据库"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO alert_rules
                (rule_id, name, description, category, metric_name,
                 detection_type, threshold_value, threshold_operator,
                 time_window_minutes, evaluation_frequency_minutes,
                 notification_channels, notification_recipients,
                 enabled, cooldown_minutes, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rule.rule_id,
                rule.name,
                rule.description,
                rule.category.value,
                rule.metric_name,
                rule.detection_type.value,
                rule.threshold_value,
                rule.threshold_operator,
                rule.time_window_minutes,
                rule.evaluation_frequency_minutes,
                json.dumps([ch.value for ch in rule.notification_channels]),
                json.dumps(rule.notification_recipients),
                rule.enabled,
                rule.cooldown_minutes,
                json.dumps(rule.metadata)
            ))
            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Failed to save alert rule to database: {e}")

    def _start_background_tasks(self):
        """启动后台任务"""
        asyncio.create_task(self._alert_evaluation_loop())
        asyncio.create_task(self._incident_resolution_monitor())
        asyncio.create_task(self._cleanup_old_incidents())

    async def _alert_evaluation_loop(self):
        """警报评估循环"""
        while True:
            try:
                await self._evaluate_all_rules()
                await asyncio.sleep(60)  # 每分钟评估一次

            except Exception as e:
                logger.error(f"Error in alert evaluation loop: {e}")
                await asyncio.sleep(60)

    async def _evaluate_all_rules(self):
        """评估所有警报规则"""
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue

            try:
                await self._evaluate_rule(rule)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.rule_id}: {e}")

    async def _evaluate_rule(self, rule: AlertRule):
        """评估单个警报规则"""
        # 检查冷却时间
        if rule.rule_id in self.alert_cooldowns:
            if datetime.now(timezone.utc) < self.alert_cooldowns[rule.rule_id]:
                return

        # 获取当前指标值
        current_value = await self._get_metric_value(rule.metric_name, rule.time_window_minutes)
        if current_value is None:
            return

        # 添加到异常检测器
        self.anomaly_detector.add_data_point(rule.metric_name, current_value, datetime.now(timezone.utc))

        # 根据检测类型评估
        should_alert = False
        alert_context = {}

        if rule.detection_type == AnomalyDetectionType.THRESHOLD:
            should_alert = self._evaluate_threshold_rule(rule, current_value)
            alert_context["threshold_value"] = rule.threshold_value

        elif rule.detection_type == AnomalyDetectionType.STATISTICAL:
            is_anomaly, z_score = self.anomaly_detector.detect_statistical_anomaly(rule.metric_name, current_value)
            should_alert = is_anomaly
            alert_context["z_score"] = z_score

        elif rule.detection_type == AnomalyDetectionType.PATTERN:
            is_anomaly, pattern_info = self.anomaly_detector.detect_pattern_anomaly(
                rule.metric_name, current_value, rule.time_window_minutes
            )
            should_alert = is_anomaly
            alert_context["pattern_info"] = pattern_info

        if should_alert:
            await self._create_alert_incident(rule, current_value, alert_context)

    def _evaluate_threshold_rule(self, rule: AlertRule, current_value: float) -> bool:
        """评估阈值规则"""
        if rule.threshold_operator == ">":
            return current_value > rule.threshold_value
        elif rule.threshold_operator == "<":
            return current_value < rule.threshold_value
        elif rule.threshold_operator == ">=":
            return current_value >= rule.threshold_value
        elif rule.threshold_operator == "<=":
            return current_value <= rule.threshold_value
        elif rule.threshold_operator == "==":
            return abs(current_value - rule.threshold_value) < 0.001
        return False

    async def _get_metric_value(self, metric_name: str, time_window_minutes: int) -> Optional[float]:
        """获取指标值"""
        try:
            cursor = self.conn.cursor()

            # 从性能跟踪器获取实时指标
            if metric_name in ["cpu_usage", "memory_usage", "error_rate"]:
                stats = self.performance_tracker.get_real_time_stats(MetricType.RESPONSE_TIME)
                return stats.get("current", 0)

            # 从数据库获取其他指标
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=time_window_minutes)

            if metric_name == "daily_cost":
                cursor.execute("""
                    SELECT SUM(total_cost) as daily_cost
                    FROM cost_records
                    WHERE DATE(timestamp) = DATE('now')
                """)
                result = cursor.fetchone()
                return result[0] if result and result[0] is not None else 0

            elif metric_name == "quote_accuracy":
                cursor.execute("""
                    SELECT AVG(accuracy_score) as avg_accuracy
                    FROM accuracy_metrics
                    WHERE timestamp >= ? AND task_type = 'quote_generation'
                """, (cutoff_time.isoformat(),))
                result = cursor.fetchone()
                return result[0] if result and result[0] is not None else 100

            elif metric_name == "customer_satisfaction":
                cursor.execute("""
                    SELECT AVG(satisfaction_score) as avg_satisfaction
                    FROM user_feedback
                    WHERE timestamp >= ?
                """, (cutoff_time.isoformat(),))
                result = cursor.fetchone()
                return result[0] if result and result[0] is not None else 5

            elif metric_name == "document_processing_success_rate":
                cursor.execute("""
                    SELECT
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
                    FROM observability_traces
                    WHERE timestamp >= ? AND interaction_type = 'document_processing'
                """, (cutoff_time.isoformat(),))
                result = cursor.fetchone()
                return result[0] if result and result[0] is not None else 100

            return 0.0

        except Exception as e:
            logger.error(f"Failed to get metric value for {metric_name}: {e}")
            return None

    async def _create_alert_incident(self,
                                    rule: AlertRule,
                                    metric_value: float,
                                    context_data: Dict[str, Any]):
        """创建警报事件"""
        try:
            incident_id = f"incident_{rule.rule_id}_{int(datetime.now().timestamp())}"

            # 确定严重性
            severity = self._determine_severity(rule, metric_value, context_data)

            # 生成标题和描述
            title, description = self._generate_alert_content(rule, metric_value, context_data)

            incident = AlertIncident(
                incident_id=incident_id,
                rule_id=rule.rule_id,
                severity=severity,
                title=title,
                description=description,
                metric_value=metric_value,
                threshold_value=rule.threshold_value,
                context_data=context_data,
                created_at=datetime.now(timezone.utc)
            )

            self.active_incidents[incident_id] = incident
            await self._save_incident_to_db(incident)

            # 设置冷却时间
            self.alert_cooldowns[rule.rule_id] = datetime.now(timezone.utc) + timedelta(minutes=rule.cooldown_minutes)

            # 发送通知
            await self._send_notifications(incident, rule)

            # 记录到LangFuse
            await self._log_incident_to_langfuse(incident)

            logger.warning(f"Alert incident created: {title} ({severity.value})")

        except Exception as e:
            logger.error(f"Failed to create alert incident: {e}")

    def _determine_severity(self,
                          rule: AlertRule,
                          metric_value: float,
                          context_data: Dict[str, Any]) -> AlertSeverity:
        """确定警报严重性"""
        if rule.category == AlertCategory.SYSTEM_PERFORMANCE:
            if rule.metric_name == "cpu_usage":
                if metric_value > 98:
                    return AlertSeverity.CRITICAL
                elif metric_value > 95:
                    return AlertSeverity.HIGH
                else:
                    return AlertSeverity.MEDIUM

        elif rule.category == AlertCategory.MANUFACTURING_OPERATIONS:
            if rule.metric_name == "quote_accuracy":
                if metric_value < 60:
                    return AlertSeverity.CRITICAL
                elif metric_value < 75:
                    return AlertSeverity.HIGH
                else:
                    return AlertSeverity.MEDIUM

        elif rule.category == AlertCategory.COST_MANAGEMENT:
            if "z_score" in context_data and context_data["z_score"] > 4:
                return AlertSeverity.HIGH
            else:
                return AlertSeverity.MEDIUM

        return AlertSeverity.MEDIUM

    def _generate_alert_content(self,
                              rule: AlertRule,
                              metric_value: float,
                              context_data: Dict[str, Any]) -> Tuple[str, str]:
        """生成警报内容"""
        title = f"{rule.name}: {metric_value:.2f}"

        description_parts = [
            f"Alert triggered for {rule.description}",
            f"Current value: {metric_value:.2f}",
        ]

        if rule.threshold_value is not None:
            description_parts.append(f"Threshold: {rule.threshold_value}")

        if "z_score" in context_data:
            description_parts.append(f"Statistical anomaly (Z-score: {context_data['z_score']:.2f})")

        if "pattern_info" in context_data:
            description_parts.append(f"Pattern anomaly: {context_data['pattern_info']}")

        description = "\n".join(description_parts)

        return title, description

    async def _send_notifications(self, incident: AlertIncident, rule: AlertRule):
        """发送通知"""
        for channel in rule.notification_channels:
            for recipient in rule.notification_recipients:
                try:
                    if channel == NotificationChannel.EMAIL:
                        await self._send_email_notification(incident, recipient)
                    elif channel == NotificationChannel.SLACK:
                        await self._send_slack_notification(incident, recipient)
                    elif channel == NotificationChannel.WEBHOOK:
                        await self._send_webhook_notification(incident, recipient)

                    incident.notifications_sent.append(f"{channel.value}:{recipient}")
                    await self._log_notification(incident.incident_id, channel.value, recipient, "sent")

                except Exception as e:
                    logger.error(f"Failed to send {channel.value} notification to {recipient}: {e}")
                    await self._log_notification(incident.incident_id, channel.value, recipient, f"failed:{e}")

    async def _send_email_notification(self, incident: AlertIncident, recipient: str):
        """发送邮件通知"""
        if not self.email_config:
            logger.warning("Email configuration not provided")
            return

        try:
            msg = MimeMultipart()
            msg['From'] = self.email_config['from']
            msg['To'] = recipient
            msg['Subject'] = f"[Alert] {incident.title}"

            body = f"""
Alert Details:
- Title: {incident.title}
- Severity: {incident.severity.value.upper()}
- Description: {incident.description}
- Metric Value: {incident.metric_value}
- Threshold: {incident.threshold_value}
- Created: {incident.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}

Context Data:
{json.dumps(incident.context_data, indent=2)}

Please investigate and take appropriate action.
            """

            msg.attach(MimeText(body, 'plain'))

            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['username'], self.email_config['password'])
                server.send_message(msg)

            logger.info(f"Email notification sent to {recipient}")

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            raise

    async def _send_slack_notification(self, incident: AlertIncident, webhook_url: str):
        """发送Slack通知"""
        if not webhook_url:
            webhook_url = self.slack_config.get('webhook_url')
            if not webhook_url:
                logger.warning("Slack webhook URL not provided")
                return

        try:
            color_map = {
                AlertSeverity.LOW: "good",
                AlertSeverity.MEDIUM: "warning",
                AlertSeverity.HIGH: "danger",
                AlertSeverity.CRITICAL: "danger"
            }

            payload = {
                "attachments": [
                    {
                        "color": color_map.get(incident.severity, "warning"),
                        "title": f"🚨 Alert: {incident.title}",
                        "text": incident.description,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": incident.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Metric Value",
                                "value": f"{incident.metric_value:.2f}",
                                "short": True
                            },
                            {
                                "title": "Created",
                                "value": incident.created_at.strftime('%Y-%m-%d %H:%M:%S UTC'),
                                "short": True
                            }
                        ],
                        "footer": "Manufacturing Knowledge Base Alert System",
                        "ts": int(incident.created_at.timestamp())
                    }
                ]
            }

            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            logger.info(f"Slack notification sent")

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            raise

    async def _send_webhook_notification(self, incident: AlertIncident, webhook_url: str):
        """发送Webhook通知"""
        try:
            payload = {
                "incident_id": incident.incident_id,
                "title": incident.title,
                "description": incident.description,
                "severity": incident.severity.value,
                "metric_value": incident.metric_value,
                "threshold_value": incident.threshold_value,
                "context_data": incident.context_data,
                "created_at": incident.created_at.isoformat()
            }

            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            logger.info(f"Webhook notification sent to {webhook_url}")

        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            raise

    async def _log_notification(self,
                               incident_id: str,
                               channel: str,
                               recipient: str,
                               status: str):
        """记录通知历史"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO alert_notifications
                (incident_id, channel, recipient, status, sent_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                incident_id,
                channel,
                recipient,
                status,
                datetime.now(timezone.utc).isoformat()
            ))
            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Failed to log notification: {e}")

    async def _save_incident_to_db(self, incident: AlertIncident):
        """保存事件到数据库"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO alert_incidents
                (incident_id, rule_id, severity, title, description,
                 metric_value, threshold_value, context_data, created_at,
                 acknowledged_at, acknowledged_by, resolved_at, resolved_by, notifications_sent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                incident.incident_id,
                incident.rule_id,
                incident.severity.value,
                incident.title,
                incident.description,
                incident.metric_value,
                incident.threshold_value,
                json.dumps(incident.context_data),
                incident.created_at.isoformat(),
                incident.acknowledged_at.isoformat() if incident.acknowledged_at else None,
                incident.acknowledged_by,
                incident.resolved_at.isoformat() if incident.resolved_at else None,
                incident.resolved_by,
                json.dumps(incident.notifications_sent)
            ))
            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Failed to save incident to database: {e}")

    async def _log_incident_to_langfuse(self, incident: AlertIncident):
        """记录事件到LangFuse"""
        try:
            trace_data = {
                "incident_id": incident.incident_id,
                "title": incident.title,
                "severity": incident.severity.value,
                "description": incident.description,
                "metric_value": incident.metric_value,
                "threshold_value": incident.threshold_value,
                "context_data": incident.context_data
            }

            # 这里可以调用LangFuse集成记录事件
            # await self.langfuse_integration.create_trace(trace_data)
            logger.debug(f"Incident logged to LangFuse: {incident.incident_id}")

        except Exception as e:
            logger.warning(f"Failed to log incident to LangFuse: {e}")

    async def acknowledge_incident(self, incident_id: str, acknowledged_by: str) -> bool:
        """确认警报事件"""
        try:
            if incident_id not in self.active_incidents:
                return False

            incident = self.active_incidents[incident_id]
            incident.acknowledged_by = acknowledged_by
            incident.acknowledged_at = datetime.now(timezone.utc)

            await self._save_incident_to_db(incident)

            logger.info(f"Incident {incident_id} acknowledged by {acknowledged_by}")
            return True

        except Exception as e:
            logger.error(f"Failed to acknowledge incident: {e}")
            return False

    async def resolve_incident(self, incident_id: str, resolved_by: str) -> bool:
        """解决警报事件"""
        try:
            if incident_id not in self.active_incidents:
                return False

            incident = self.active_incidents[incident_id]
            incident.resolved_by = resolved_by
            incident.resolved_at = datetime.now(timezone.utc)

            await self._save_incident_to_db(incident)

            # 从活动事件中移除
            del self.active_incidents[incident_id]

            logger.info(f"Incident {incident_id} resolved by {resolved_by}")
            return True

        except Exception as e:
            logger.error(f"Failed to resolve incident: {e}")
            return False

    async def get_active_incidents(self) -> List[Dict[str, Any]]:
        """获取活动警报事件"""
        return [asdict(incident) for incident in self.active_incidents.values()]

    async def get_incident_history(self,
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None,
                                 severity: Optional[AlertSeverity] = None,
                                 category: Optional[AlertCategory] = None) -> List[Dict[str, Any]]:
        """获取警报事件历史"""
        try:
            cursor = self.conn.cursor()

            query = """
                SELECT ai.*, ar.name as rule_name, ar.category
                FROM alert_incidents ai
                JOIN alert_rules ar ON ai.rule_id = ar.rule_id
                WHERE 1=1
            """
            params = []

            if start_date:
                query += " AND ai.created_at >= ?"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND ai.created_at <= ?"
                params.append(end_date.isoformat())

            if severity:
                query += " AND ai.severity = ?"
                params.append(severity.value)

            if category:
                query += " AND ar.category = ?"
                params.append(category.value)

            query += " ORDER BY ai.created_at DESC"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            incidents = []
            for row in rows:
                incident_data = {
                    "incident_id": row[1],
                    "rule_id": row[2],
                    "severity": row[3],
                    "title": row[4],
                    "description": row[5],
                    "metric_value": row[6],
                    "threshold_value": row[7],
                    "context_data": json.loads(row[8]) if row[8] else {},
                    "created_at": row[9],
                    "acknowledged_at": row[10],
                    "acknowledged_by": row[11],
                    "resolved_at": row[12],
                    "resolved_by": row[13],
                    "rule_name": row[14],
                    "category": row[15]
                }
                incidents.append(incident_data)

            return incidents

        except Exception as e:
            logger.error(f"Failed to get incident history: {e}")
            return []

    async def _incident_resolution_monitor(self):
        """事件解决监控"""
        while True:
            try:
                # 每小时检查一次长时间未解决的事件
                await asyncio.sleep(3600)
                await self._check_stale_incidents()

            except Exception as e:
                logger.error(f"Error in incident resolution monitor: {e}")
                await asyncio.sleep(300)

    async def _check_stale_incidents(self):
        """检查过期事件"""
        try:
            stale_threshold = datetime.now(timezone.utc) - timedelta(hours=24)

            for incident in self.active_incidents.values():
                if incident.created_at < stale_threshold:
                    # 发送升级通知
                    await self._escalate_stale_incident(incident)

        except Exception as e:
            logger.error(f"Failed to check stale incidents: {e}")

    async def _escalate_stale_incident(self, incident: AlertIncident):
        """升级过期事件"""
        try:
            escalation_title = f"ESCALATED: {incident.title} (Unresolved for >24h)"
            escalation_description = f"""
Original Alert (Unresolved):
- Title: {incident.title}
- Created: {incident.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
- Severity: {incident.severity.value}

This incident has been unresolved for over 24 hours and requires immediate attention.
            """

            # 创建升级通知
            await self._send_escalation_notification(incident, escalation_title, escalation_description)

            logger.warning(f"Stale incident escalated: {incident.incident_id}")

        except Exception as e:
            logger.error(f"Failed to escalate stale incident: {e}")

    async def _send_escalation_notification(self,
                                           incident: AlertIncident,
                                           title: str,
                                           description: str):
        """发送升级通知"""
        try:
            # 这里可以发送给管理层或不同的通知渠道
            escalation_recipients = ["management@company.com", "oncall@company.com"]

            for recipient in escalation_recipients:
                if self.email_config:
                    await self._send_email_alert(recipient, title, description)

        except Exception as e:
            logger.error(f"Failed to send escalation notification: {e}")

    async def _send_email_alert(self, recipient: str, title: str, description: str):
        """发送邮件警报"""
        # 实现邮件发送逻辑
        pass

    async def _cleanup_old_incidents(self):
        """清理旧事件"""
        while True:
            try:
                # 每天清理一次超过30天的已解决事件
                await asyncio.sleep(24 * 3600)

                cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)

                cursor = self.conn.cursor()
                cursor.execute("""
                    DELETE FROM alert_incidents
                    WHERE resolved_at IS NOT NULL AND resolved_at < ?
                """, (cutoff_date.isoformat(),))

                deleted_count = cursor.rowcount
                if deleted_count > 0:
                    self.conn.commit()
                    logger.info(f"Cleaned up {deleted_count} old resolved incidents")

            except Exception as e:
                logger.error(f"Failed to cleanup old incidents: {e}")

    def close(self):
        """关闭连接"""
        try:
            if self.conn:
                self.conn.close()
                logger.info("Alert system connection closed")
        except Exception as e:
            logger.error(f"Error closing alert system: {e}")

# 全局实例
_alert_system = None

def get_alert_system() -> AlertSystem:
    """获取警报系统实例"""
    global _alert_system
    if _alert_system is None:
        _alert_system = AlertSystem()
    return _alert_system