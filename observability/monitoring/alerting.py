"""
Alerting system for observability.

Monitors metrics and health status, generates alerts for anomalies,
and supports multiple notification channels.
"""

import time
import asyncio
import threading
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from collections import defaultdict, deque
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

from pydantic import BaseModel, Field, validator

from ..core.metrics import get_metrics_collector
from ..core.logging import get_logger, log_system_event
from .health_checker import get_health_checker, HealthStatus


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    """Alert notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    LOG = "log"
    CONSOLE = "console"


class AlertStatus(str, Enum):
    """Alert status"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class TriggerType(str, Enum):
    """Types of alert triggers"""
    THRESHOLD = "threshold"
    RATE_OF_CHANGE = "rate_of_change"
    ANOMALY_DETECTION = "anomaly_detection"
    HEALTH_CHECK = "health_check"
    PREDICTIVE = "predictive"
    MANUAL = "manual"


@dataclass
class AlertCondition:
    """Alert condition configuration"""
    name: str
    metric_name: str
    trigger_type: TriggerType
    threshold: Optional[float] = None
    comparison: str = "greater_than"  # greater_than, less_than, equals
    evaluation_window: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    cooldown_period: timedelta = field(default_factory=lambda: timedelta(minutes=10))
    severity: AlertSeverity = AlertSeverity.WARNING
    enabled: bool = True
    labels: Dict[str, str] = field(default_factory=dict)


class AlertRule(BaseModel):
    """Alert rule configuration"""
    name: str
    description: str
    conditions: List[AlertCondition]
    channels: List[AlertChannel] = Field(default_factory=lambda: [AlertChannel.EMAIL])
    severity: AlertSeverity = AlertSeverity.WARNING
    enabled: bool = True
    cooldown_period: timedelta = Field(default_factory=lambda: timedelta(minutes=5))
    max_alerts_per_hour: int = 10
    escalation_rules: Dict[str, str] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)

    @validator('conditions')
    def validate_conditions(cls, v):
        if not v:
            raise ValueError("At least one condition must be specified")
        return v


class Alert(BaseModel):
    """Alert instance"""
    id: str = Field(default_factory=lambda: str(int(time.time() * 1000)))
    rule_name: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.ACTIVE
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    current_value: Optional[float] = None
    previous_value: Optional[float] = None
    labels: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    triggered_at: datetime = Field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    last_sent: Optional[datetime] = None
    sent_count: int = 0
    channels_sent: List[AlertChannel] = Field(default_factory=list)
    acknowledgments: List[Dict[str, Any]] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class NotificationResult(BaseModel):
    """Result of alert notification"""
    success: bool
    channel: AlertChannel
    message: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class NotificationChannel:
    """Base class for notification channels"""

    def __init__(self, channel_type: AlertChannel, config: Dict[str, Any]):
        self.channel_type = channel_type
        self.config = config
        self.logger = get_logger()

    def send_notification(self, alert: Alert) -> NotificationResult:
        """Send alert notification"""
        raise NotImplementedError("Subclasses must implement send_notification")

    def test_connection(self) -> bool:
        """Test connection to notification channel"""
        return True


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(AlertChannel.EMAIL, config)
        self.smtp_server = config.get("smtp_server", "localhost")
        self.smtp_port = config.get("smtp_port", 587)
        self.username = config.get("username", "")
        self.password = config.get("password", "")
        self.from_email = config.get("from_email", "alerts@knowledgebase.com")
        self.to_emails = config.get("to_emails", [])
        self.use_tls = config.get("use_tls", True)

    def send_notification(self, alert: Alert) -> NotificationResult:
        """Send email alert"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ", ".join(self.to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"

            # Create HTML body
            html_body = f"""
            <html>
            <body>
                <h2>Knowledge Base Alert</h2>
                <h3>{alert.title}</h3>
                <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
                <p><strong>Description:</strong> {alert.description}</p>
                <p><strong>Triggered:</strong> {alert.triggered_at}</p>

                {f'<p><strong>Current Value:</strong> {alert.current_value}</p>' if alert.current_value else ''}
                {f'<p><strong>Threshold:</strong> {alert.threshold}</p>' if alert.threshold else ''}

                <h4>Labels:</h4>
                <ul>
                {"".join(f"<li>{k}: {v}</li>" for k, v in alert.labels.items())}
                </ul>

                <h4>Metadata:</h4>
                <pre>{json.dumps(alert.metadata, indent=2)}</pre>

                <hr>
                <p><small>Alert ID: {alert.id} | Rule: {alert.rule_name}</small></p>
            </body>
            </html>
            """

            msg.attach(MIMEText(html_body, "html"))

            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.use_tls:
                server.starttls()

            if self.username and self.password:
                server.login(self.username, self.password)

            text = msg.as_string()
            server.sendmail(self.from_email, self.to_emails, text)

            server.quit()

            return NotificationResult(
                success=True,
                channel=self.channel_type,
                message="Email sent successfully"
            )

        except Exception as e:
            self.logger.error(f"Failed to send email alert: {str(e)}")
            return NotificationResult(
                success=False,
                channel=self.channel_type,
                error=str(e)
            )

    def test_connection(self) -> bool:
        """Test email connection"""
        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.use_tls:
                server.starttls()
            server.quit()
            return True
        except Exception:
            return False


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(AlertChannel.SLACK, config)
        self.webhook_url = config.get("webhook_url")
        self.channel = config.get("channel", "#alerts")
        self.username = config.get("username", "Knowledge Base Bot")

    def send_notification(self, alert: Alert) -> NotificationResult:
        """Send Slack alert"""
        try:
            import requests

            color = self._get_color_for_severity(alert.severity)

            payload = {
                "channel": self.channel,
                "username": self.username,
                "icon_emoji": self._get_emoji_for_severity(alert.severity),
                "attachments": [{
                    "color": color,
                    "title": alert.title,
                    "text": alert.description,
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.severity.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Rule",
                            "value": alert.rule_name,
                            "short": True
                        },
                        {
                            "title": "Triggered",
                            "value": alert.triggered_at.strftime("%Y-%m-%d %H:%M:%S"),
                            "short": True
                        }
                    ],
                    "footer": f"Alert ID: {alert.id}",
                    "ts": alert.triggered_at.timestamp()
                }]
            }

            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            return NotificationResult(
                success=True,
                channel=self.channel_type,
                message="Slack notification sent successfully"
            )

        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {str(e)}")
            return NotificationResult(
                success=False,
                channel=self.channel_type,
                error=str(e)
            )

    def _get_color_for_severity(self, severity: AlertSeverity) -> str:
        """Get Slack color for severity level"""
        colors = {
            AlertSeverity.INFO: "good",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "danger",
            AlertSeverity.CRITICAL: "#ff0000"
        }
        return colors.get(severity, "warning")

    def _get_emoji_for_severity(self, severity: AlertSeverity) -> str:
        """Get Slack emoji for severity level"""
        emojis = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨"
        }
        return emojis.get(severity, "⚠️")

    def test_connection(self) -> bool:
        """Test Slack webhook connection"""
        try:
            import requests
            response = requests.post(
                self.webhook_url,
                json={"text": "Test connection from Knowledge Base"},
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False


class WebhookNotificationChannel(NotificationChannel):
    """Webhook notification channel"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(AlertChannel.WEBHOOK, config)
        self.webhook_url = config.get("webhook_url")
        self.headers = config.get("headers", {})
        self.timeout = config.get("timeout", 10)

    def send_notification(self, alert: Alert) -> NotificationResult:
        """Send webhook alert"""
        try:
            import requests

            payload = {
                "alert_id": alert.id,
                "rule_name": alert.rule_name,
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "metric_value": alert.current_value,
                "threshold": alert.threshold,
                "labels": alert.labels,
                "metadata": alert.metadata,
                "triggered_at": alert.triggered_at.isoformat(),
                "tags": alert.tags
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()

            return NotificationResult(
                success=True,
                channel=self.channel_type,
                message="Webhook notification sent successfully"
            )

        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {str(e)}")
            return NotificationResult(
                success=False,
                channel=self.channel_type,
                error=str(e)
            )

    def test_connection(self) -> bool:
        """Test webhook connection"""
        try:
            import requests
            response = requests.post(
                self.webhook_url,
                json={"test": True},
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False


class LogNotificationChannel(NotificationChannel):
    """Log-based notification channel"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(AlertChannel.LOG, config or {})
        self.logger = get_logger()

    def send_notification(self, alert: Alert) -> NotificationResult:
        """Send alert to log"""
        try:
            self.logger.error(
                f"ALERT [{alert.severity.value.upper()}] {alert.title}",
                extra={
                    "alert_id": alert.id,
                    "rule_name": alert.rule_name,
                    "description": alert.description,
                    "metric_value": alert.current_value,
                    "threshold": alert.threshold,
                    "labels": alert.labels,
                    "metadata": alert.metadata,
                    "triggered_at": alert.triggered_at,
                    "tags": alert.tags
                }
            )

            return NotificationResult(
                success=True,
                channel=self.channel_type,
                message="Alert logged successfully"
            )

        except Exception as e:
            return NotificationResult(
                success=False,
                channel=self.channel_type,
                error=str(e)
            )

    def test_connection(self) -> bool:
        """Test log channel (always true)"""
        return True


class AlertManager:
    """
    Manages alert rules, conditions, and notifications.

    Monitors metrics and health status, generates alerts when conditions are met,
    and manages alert lifecycle with cooldowns and suppression.
    """

    def __init__(
        self,
        max_active_alerts: int = 1000,
        default_channels: List[AlertChannel] = None,
        alert_retention_days: int = 30
    ):
        self.logger = get_logger()
        self.metrics = get_metrics_collector()
        self.health_checker = get_health_checker()

        self.max_active_alerts = max_active_alerts
        self.alert_retention_days = alert_retention_days
        self.default_channels = default_channels or [AlertChannel.EMAIL, AlertChannel.LOG]

        # Alert management
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.suppressed_alerts: Set[str] = set()

        # Notification channels
        self.notification_channels: Dict[AlertChannel, NotificationChannel] = {}
        self._initialize_default_channels()

        # Alert rate limiting
        self.alert_rate_limits: Dict[str, List[datetime]] = defaultdict(list)

        # Thread safety
        self.lock = threading.RLock()

        # Start alert processing
        self._start_alert_processing()

    def _initialize_default_channels(self):
        """Initialize default notification channels"""
        # Email channel (if configured)
        try:
            email_config = {
                "smtp_server": "localhost",
                "smtp_port": 587,
                "from_email": "alerts@knowledgebase.com",
                "to_emails": ["admin@knowledgebase.com"],
                "use_tls": True
            }
            self.notification_channels[AlertChannel.EMAIL] = EmailNotificationChannel(email_config)
        except Exception as e:
            self.logger.warning(f"Email channel not configured: {e}")

        # Slack channel (if configured)
        try:
            slack_config = {
                "webhook_url": os.getenv("SLACK_WEBHOOK_URL"),
                "channel": "#alerts"
            }
            if slack_config["webhook_url"]:
                self.notification_channels[AlertChannel.SLACK] = SlackNotificationChannel(slack_config)
        except Exception as e:
            self.logger.warning(f"Slack channel not configured: {e}")

        # Log channel (always available)
        self.notification_channels[AlertChannel.LOG] = LogNotificationChannel()

    def add_alert_rule(self, rule: AlertRule) -> str:
        """Add a new alert rule"""
        with self.lock:
            rule_id = f"{rule.name}_{int(time.time())}"
            self.alert_rules[rule_id] = rule

            self.logger.info(f"Added alert rule: {rule.name} (ID: {rule_id})")
            return rule_id

    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule"""
        with self.lock:
            if rule_id in self.alert_rules:
                rule = self.alert_rules.pop(rule_id)
                self.logger.info(f"Removed alert rule: {rule.name} (ID: {rule_id})")

                # Resolve any active alerts from this rule
                alerts_to_resolve = [
                    alert_id for alert_id, alert in self.active_alerts.items()
                    if alert.rule_name == rule.name
                ]

                for alert_id in alerts_to_resolve:
                    self.resolve_alert(alert_id)

                return True
            return False

    def get_alert_rules(self) -> List[AlertRule]:
        """Get all alert rules"""
        with self.lock:
            return list(self.alert_rules.values())

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get a specific alert by ID"""
        with self.lock:
            return self.active_alerts.get(alert_id)

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        with self.lock:
            return list(self.active_alerts.values())

    def get_alert_history(
        self,
        limit: int = 100,
        severity: Optional[AlertSeverity] = None,
        rule_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Alert]:
        """Get alert history with filtering"""
        with self.lock:
            history = list(self.alert_history)

        # Apply filters
        if severity:
            history = [a for a in history if a.severity == severity]
        if rule_name:
            history = [a for a in history if a.rule_name == rule_name]

        if start_time:
            history = [a for a in history if a.triggered_at >= start_time]
        if end_time:
            history = [a for a in history if a.triggered_at <= end_time]

        return history[:limit]

    def acknowledge_alert(self, alert_id: str, user_id: str, comment: str) -> bool:
        """Acknowledge an alert"""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.acknowledged_at = datetime.now()
                alert.acknowledgments.append({
                    "user_id": user_id,
                    "comment": comment,
                    "timestamp": datetime.now()
                })

                self.logger.info(f"Alert acknowledged: {alert_id} by {user_id}")

                # Update metrics
                self.metrics.increment_counter(
                    "alerts_acknowledged",
                    labels={
                        "rule_name": alert.rule_name,
                        "severity": alert.severity.value
                    }
                )

                return True
            return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts.pop(alert_id)
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()

                # Move to history
                self.alert_history.append(alert)

                # Update metrics
                self.metrics.increment_counter(
                    "alerts_resolved",
                    labels={
                        "rule_name": alert.rule_name,
                        "severity": alert.severity.value
                    }
                )

                self.logger.info(f"Alert resolved: {alert_id} ({alert.title})")
                return True
            return False

    def suppress_alert(self, alert_id: str, duration: Optional[timedelta] = None) -> bool:
        """Suppress an alert for a duration"""
        with self.lock:
            self.suppressed_alerts.add(alert_id)

            if duration:
                # Schedule un-suppression
                threading.Timer(duration.total_seconds(), lambda: self.unsuppress_alert(alert_id)).start()

            self.logger.info(f"Alert suppressed: {alert_id}")
            return True

    def unsuppress_alert(self, alert_id: str) -> bool:
        """Unsuppress an alert"""
        with self.lock:
            if alert_id in self.suppressed_alerts:
                self.suppressed_alerts.remove(alert_id)
                self.logger.info(f"Alert unsuppressed: {alert_id}")
                return True
            return False

    def trigger_manual_alert(
        self,
        title: str,
        description: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        channels: Optional[List[AlertChannel]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Trigger a manual alert"""
        with self.lock:
            alert_id = str(int(time.time() * 1000))

            alert = Alert(
                rule_name="manual",
                title=title,
                description=description,
                severity=severity,
                metadata=metadata or {},
                tags=tags or []
            )

            return self._process_alert(alert)

    def check_metric_thresholds(self):
        """Check metric thresholds and trigger alerts"""
        metrics = self.metrics.get_all_metrics()

        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue

            for condition in rule.conditions:
                try:
                    metric = metrics.get(condition.metric_name)
                    if metric:
                        current_value = self._get_metric_value(metric)
                        if self._evaluate_condition(current_value, condition):
                            alert = Alert(
                                rule_name=rule.name,
                                title=f"Alert: {rule.name}",
                                description=rule.description,
                                severity=rule.severity,
                                metric_value=current_value,
                                threshold=condition.threshold,
                                current_value=current_value,
                                labels=condition.labels
                            )

                            alert.rule_name = rule.name
                            alert.tags.extend(rule.tags)

                            self._process_alert(alert)

                except Exception as e:
                    self.logger.error(f"Error checking condition {condition.name}: {str(e)}")

    def _get_metric_value(self, metric) -> Optional[float]:
        """Extract numeric value from metric"""
        if hasattr(metric, 'value'):
            return float(metric.value)
        elif hasattr(metric, 'count'):
            return float(metric.count)
        elif hasattr(metric, 'sum'):
            return float(metric.sum)
        return None

    def _evaluate_condition(self, value: float, condition: AlertCondition) -> bool:
        """Evaluate if condition is triggered"""
        if condition.threshold is None:
            return False

        comparison = condition.comparison.lower()

        if comparison == "greater_than":
            return value > condition.threshold
        elif comparison == "less_than":
            return value < condition.threshold
        elif comparison == "equals":
            return abs(value - condition.threshold) < 0.001
        elif comparison == "greater_than_or_equal":
            return value >= condition.threshold
        elif comparison == "less_than_or_equal":
            return value <= condition.threshold

        return False

    def _process_alert(self, alert: Alert) -> str:
        """Process an alert and send notifications"""
        alert_id = alert.id

        # Check rate limits
        if self._is_rate_limited(alert):
            self.logger.warning(f"Alert rate limited: {alert_id}")
            return alert_id

        # Check if alert is suppressed
        if alert_id in self.suppressed_alerts:
            self.logger.debug(f"Alert suppressed: {alert_id}")
            return alert_id

        # Check if similar alert is already active
        if self._is_duplicate_alert(alert):
            self.logger.debug(f"Duplicate alert detected: {alert_id}")
            return alert_id

        # Check if maximum active alerts reached
        if len(self.active_alerts) >= self.max_active_alerts:
            self.logger.warning(f"Maximum active alerts reached: {self.max_active_alerts}")
            return alert_id

        # Add to active alerts
        self.active_alerts[alert_id] = alert

        # Send notifications
        channels_to_use = self.default_channels
        if alert.channels_sent:
            channels_to_use = alert.channels_sent

        notification_results = []
        for channel in channels_to_use:
            if channel in self.notification_channels:
                try:
                    result = self.notification_channels[channel].send_notification(alert)
                    notification_results.append(result)
                    if result.success:
                        alert.channels_sent.append(channel)
                except Exception as e:
                    self.logger.error(f"Failed to send {channel.value} notification: {str(e)}")

        # Update metrics
        self.metrics.increment_counter(
            "alerts_triggered",
            labels={
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "channels": ",".join(c.value for c in notification_results if r.success)
            }
        )

        # Log the alert
        log_system_event(
            message=f"Alert triggered: {alert.title}",
            metadata={
                "alert_id": alert_id,
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "channels_notified": len([r for r in notification_results if r.success])
            }
        )

        return alert_id

    def _is_rate_limited(self, alert: Alert) -> bool:
        """Check if alert is rate limited"""
        rate_limit_key = f"{alert.rule_name}:{alert.severity.value}"
        current_time = datetime.now()

        # Clean old rate limit entries
        cutoff_time = current_time - timedelta(hours=1)
        self.alert_rate_limits[rate_limit_key] = [
            ts for ts in self.alert_rate_limits[rate_limit_key]
            if ts > cutoff_time
        ]

        # Check if too many alerts in the last hour
        if len(self.alert_rate_limits[rate_limit_key]) >= 10:
            return True

        # Add current time to rate limit tracking
        self.alert_rate_limits[rate_limit_key].append(current_time)
        return False

    def _is_duplicate_alert(self, alert: Alert) -> bool:
        """Check if similar alert is already active"""
        for existing_alert in self.active_alerts.values():
            if (existing_alert.rule_name == alert.rule_name and
                abs(existing_alert.metric_value - (alert.current_value or 0)) < 0.01 and
                (datetime.now() - existing_alert.triggered_at).total_seconds() < 300):
                return True
        return False

    def _start_alert_processing(self):
        """Start background alert processing"""
        def process_alerts():
            while True:
                try:
                    # Check metric thresholds
                    self.check_metric_thresholds()

                    # Check health status
                    health = self.health_checker.get_system_health()
                    if health and health.status == HealthStatus.UNHEALTHY:
                        self.trigger_manual_alert(
                            title="System Health Critical",
                            description=f"System health score: {health.overall_score:.1f}",
                            severity=AlertSeverity.CRITICAL,
                            metadata={"health_score": health.overall_score}
                        )

                    # Sleep before next check
                    time.sleep(60)  # Check every minute

                except Exception as e:
                    self.logger.error(f"Error in alert processing: {e}")
                    time.sleep(60)

        thread = threading.Thread(target=process_alerts, daemon=True)
        thread.start()
        self.logger.info("Alert processing started")

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status"""
        with self.lock:
            return {
                "total_rules": len(self.alert_rules),
                "active_rules": len([r for r in self.alert_rules.values() if r.enabled]),
                "active_alerts": len(self.active_alerts),
                "suppressed_alerts": len(self.suppressed_alerts),
                "alert_channels": list(self.notification_channels.keys()),
                "rate_limited_rules": len(self.alert_rate_limits),
                "recent_alerts": len(self.alert_history),
                "top_alerts": [
                    {
                        "id": alert.id,
                        "title": alert.title,
                        "severity": alert.severity.value,
                        "rule_name": alert.rule_name,
                        "triggered_at": alert.triggered_at.isoformat(),
                        "sent_count": alert.sent_count,
                        "channels": [c.value for c in alert.channels_sent]
                    }
                    for alert in sorted(
                        self.alert_history,
                        key=lambda x: x.triggered_at,
                        reverse=True
                    )[:10]
                ]
            }


# Global alert manager instance
_global_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance"""
    global _global_alert_manager
    if _global_alert_manager is None:
        _global_alert_manager = AlertManager()
    return _global_alert_manager


def configure_alert_manager(**kwargs) -> AlertManager:
    """Configure global alert manager"""
    global _global_alert_manager
    _global_alert_manager = AlertManager(**kwargs)
    return _global_alert_manager


import os