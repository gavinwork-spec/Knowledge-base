#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Observability Orchestrator for Manufacturing Knowledge Base
制造业知识库可观测性编排器

Main orchestrator that coordinates all observability components including LangFuse integration,
performance tracking, cost analysis, dashboards, user analytics, alerting, and manufacturing metrics.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

# Import all observability components
from .core.langfuse_integration import create_langfuse_integration, LangFuseIntegration
from .core.ai_interaction_logger import get_ai_logger, AIInteractionLogger
from .core.performance_tracker import get_performance_tracker, PerformanceTracker
from .core.cost_analyzer import get_cost_analyzer, CostAnalyzer
from .core.dashboard_manager import get_dashboard_manager, DashboardManager
from .core.user_analytics import get_user_analytics, UserAnalytics
from .core.alert_system import get_alert_system, AlertSystem
from .core.manufacturing_metrics import get_manufacturing_metrics_collector, ManufacturingMetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ObservabilityConfig:
    """可观测性配置"""
    db_path: str = "knowledge_base.db"
    enable_langfuse: bool = True
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: Optional[str] = None
    enable_dashboard: bool = True
    dashboard_websocket_port: int = 8765
    enable_alerts: bool = True
    email_config: Optional[Dict[str, Any]] = None
    slack_config: Optional[Dict[str, Any]] = None
    enable_detailed_logging: bool = True
    cost_tracking_enabled: bool = True
    user_analytics_enabled: bool = True
    manufacturing_metrics_enabled: bool = True

@dataclass
class SystemHealthReport:
    """系统健康报告"""
    timestamp: datetime
    overall_status: str  # "healthy", "warning", "critical"
    component_status: Dict[str, str]
    active_alerts_count: int
    performance_metrics: Dict[str, float]
    cost_metrics: Dict[str, float]
    user_metrics: Dict[str, float]
    manufacturing_metrics: Dict[str, float]
    recommendations: List[str]

class ObservabilityOrchestrator:
    """可观测性编排器"""

    def __init__(self, config: ObservabilityConfig):
        """
        初始化可观测性编排器

        Args:
            config: 可观测性配置
        """
        self.config = config
        self.initialized = False
        self.startup_time = datetime.now(timezone.utc)

        # 组件实例
        self.langfuse_integration: Optional[LangFuseIntegration] = None
        self.ai_logger: Optional[AIInteractionLogger] = None
        self.performance_tracker: Optional[PerformanceTracker] = None
        self.cost_analyzer: Optional[CostAnalyzer] = None
        self.dashboard_manager: Optional[DashboardManager] = None
        self.user_analytics: Optional[UserAnalytics] = None
        self.alert_system: Optional[AlertSystem] = None
        self.manufacturing_metrics: Optional[ManufacturingMetricsCollector] = None

        logger.info("Observability Orchestrator created with configuration")

    async def initialize(self):
        """初始化所有可观测性组件"""
        try:
            logger.info("🚀 Initializing Observability Orchestrator...")

            # 1. 初始化LangFuse集成
            if self.config.enable_langfuse:
                self.langfuse_integration = await create_langfuse_integration({
                    "db_path": self.config.db_path,
                    "enable_langfuse": self.config.enable_langfuse,
                    "langfuse_public_key": self.config.langfuse_public_key,
                    "langfuse_secret_key": self.config.langfuse_secret_key,
                    "langfuse_host": self.config.langfuse_host
                })
                logger.info("✅ LangFuse integration initialized")

            # 2. 初始化AI交互日志记录器
            if self.config.enable_detailed_logging:
                self.ai_logger = get_ai_logger()
                logger.info("✅ AI interaction logger initialized")

            # 3. 初始化性能跟踪器
            self.performance_tracker = get_performance_tracker()
            logger.info("✅ Performance tracker initialized")

            # 4. 初始化成本分析器
            if self.config.cost_tracking_enabled:
                self.cost_analyzer = get_cost_analyzer()
                logger.info("✅ Cost analyzer initialized")

            # 5. 初始化仪表板管理器
            if self.config.enable_dashboard:
                self.dashboard_manager = DashboardManager(
                    db_path=self.config.db_path,
                    websocket_port=self.config.dashboard_websocket_port
                )
                logger.info("✅ Dashboard manager initialized")

            # 6. 初始化用户分析系统
            if self.config.user_analytics_enabled:
                self.user_analytics = get_user_analytics()
                logger.info("✅ User analytics initialized")

            # 7. 初始化警报系统
            if self.config.enable_alerts:
                self.alert_system = AlertSystem(
                    db_path=self.config.db_path,
                    email_config=self.config.email_config,
                    slack_config=self.config.slack_config
                )
                logger.info("✅ Alert system initialized")

            # 8. 初始化制造业指标收集器
            if self.config.manufacturing_metrics_enabled:
                self.manufacturing_metrics = get_manufacturing_metrics_collector()
                logger.info("✅ Manufacturing metrics collector initialized")

            # 9. 启动后台监控任务
            await self._start_monitoring_tasks()

            self.initialized = True
            logger.info("🎉 Observability Orchestrator initialization completed successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize observability orchestrator: {e}")
            raise

    async def _start_monitoring_tasks(self):
        """启动后台监控任务"""
        asyncio.create_task(self._system_health_monitor())
        asyncio.create_task(self._component_health_check())
        asyncio.create_task(self._metrics_aggregation())

    @asynccontextmanager
    async def trace_interaction(self,
                              session_id: str,
                              user_id: Optional[str] = None,
                              interaction_type: str = "unknown",
                              metadata: Optional[Dict[str, Any]] = None):
        """跟踪AI交互的上下文管理器"""
        if not self.initialized:
            await self.initialize()

        if self.langfuse_integration:
            async with self.langfuse_integration.trace_interaction(
                session_id=session_id,
                user_id=user_id,
                interaction_type=interaction_type,
                input_data=metadata or {}
            ) as trace_data:
                yield trace_data
        else:
            yield None

    async def log_ai_interaction(self,
                               session_id: str,
                               user_id: Optional[str],
                               agent_type: str,
                               model_provider: str,
                               model_name: str,
                               prompt: str,
                               response: str,
                               context: Optional[Dict[str, Any]] = None,
                               performance_data: Optional[Dict[str, Any]] = None):
        """记录AI交互"""
        try:
            if not self.initialized:
                await self.initialize()

            # 记录到AI日志
            if self.ai_logger:
                # 创建AI交互对象
                from .core.ai_interaction_logger import AIInteraction, ModelProvider, AgentType

                interaction = AIInteraction(
                    interaction_id=f"ai_{session_id}_{int(datetime.now().timestamp())}",
                    session_id=session_id,
                    user_id=user_id,
                    agent_type=AgentType(agent_type),
                    model_provider=ModelProvider(model_provider),
                    model_name=model_name,
                    prompt=prompt,
                    response=response,
                    context=context or {},
                    performance_metrics=None,  # 将在下面创建
                    cost_metrics=None,  # 将由成本分析器计算
                    manufacturing_metrics=None,
                    timestamp=datetime.now(timezone.utc),
                    success=True
                )

                # 添加性能数据
                if performance_data:
                    from .core.ai_interaction_logger import PerformanceMetrics
                    interaction.performance_metrics = PerformanceMetrics(
                        response_time_ms=performance_data.get('response_time_ms', 0),
                        token_count=performance_data.get('token_count', 0),
                        model_name=model_name,
                        prompt_tokens=performance_data.get('prompt_tokens'),
                        completion_tokens=performance_data.get('completion_tokens')
                    )

                self.ai_logger.log_ai_interaction(interaction)

            # 记录成本
            if self.cost_analyzer and performance_data:
                prompt_tokens = performance_data.get('prompt_tokens', 0)
                completion_tokens = performance_data.get('completion_tokens', 0)
                await self.cost_analyzer.record_api_cost(
                    session_id=session_id,
                    user_id=user_id,
                    provider=model_provider,
                    model=model_name,
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    metadata=context
                )

            # 记录用户行为
            if self.user_analytics:
                await self.user_analytics.track_user_query(
                    session_id=session_id,
                    user_id=user_id,
                    query=prompt,
                    response=response,
                    satisfaction_score=performance_data.get('satisfaction_score') if performance_data else None
                )

            logger.debug(f"AI interaction logged: {session_id}")

        except Exception as e:
            logger.error(f"Failed to log AI interaction: {e}")

    async def track_user_query(self,
                             session_id: str,
                             user_id: Optional[str],
                             query: str,
                             response: str,
                             satisfaction_score: Optional[int] = None,
                             feedback: Optional[str] = None,
                             documents_accessed: Optional[List[str]] = None):
        """跟踪用户查询"""
        try:
            if not self.initialized:
                await self.initialize()

            if self.user_analytics:
                await self.user_analytics.track_user_query(
                    session_id=session_id,
                    user_id=user_id,
                    query=query,
                    response=response,
                    satisfaction_score=satisfaction_score,
                    feedback_comment=feedback,
                    documents_accessed=documents_accessed
                )

        except Exception as e:
            logger.error(f"Failed to track user query: {e}")

    async def record_manufacturing_event(self,
                                        event_type: str,
                                        data: Dict[str, Any]):
        """记录制造业事件"""
        try:
            if not self.initialized:
                await self.initialize()

            if event_type == "quote_created":
                from .core.manufacturing_metrics import QuoteMetrics, QuoteStatus
                quote_metrics = QuoteMetrics(
                    quote_id=data["quote_id"],
                    customer_id=data["customer_id"],
                    part_number=data["part_number"],
                    quantity=data["quantity"],
                    quoted_price=data["quoted_price"],
                    actual_cost=data.get("actual_cost"),
                    margin_percentage=data.get("margin_percentage"),
                    processing_time_minutes=data["processing_time_minutes"],
                    accuracy_score=data["accuracy_score"],
                    revision_count=data.get("revision_count", 0),
                    status=QuoteStatus(data.get("status", "pending")),
                    created_at=datetime.now(timezone.utc),
                    metadata=data.get("metadata", {})
                )
                await self.manufacturing_metrics.record_quote_metrics(quote_metrics)

            elif event_type == "quality_inspection":
                from .core.manufacturing_metrics import QualityMetrics, QualityCheckResult, ManufacturingProcessType
                quality_metrics = QualityMetrics(
                    inspection_id=data["inspection_id"],
                    part_number=data["part_number"],
                    batch_id=data["batch_id"],
                    process_type=ManufacturingProcessType(data["process_type"]),
                    total_inspected=data["total_inspected"],
                    passed_count=data["passed_count"],
                    failed_count=data.get("failed_count", 0),
                    rework_count=data.get("rework_count", 0),
                    defect_rate=data["defect_rate"],
                    first_pass_yield=data["first_pass_yield"],
                    inspection_time_minutes=data["inspection_time_minutes"],
                    result=QualityCheckResult(data["result"]),
                    inspector_id=data["inspector_id"],
                    timestamp=datetime.now(timezone.utc)
                )
                await self.manufacturing_metrics.record_quality_metrics(quality_metrics)

            elif event_type == "customer_feedback":
                from .core.manufacturing_metrics import CustomerSatisfactionMetrics, CustomerSatisfactionLevel
                satisfaction_metrics = CustomerSatisfactionMetrics(
                    feedback_id=data["feedback_id"],
                    customer_id=data["customer_id"],
                    order_id=data["order_id"],
                    quote_id=data.get("quote_id"),
                    overall_satisfaction=CustomerSatisfactionLevel(data["overall_satisfaction"]),
                    quality_satisfaction=CustomerSatisfactionLevel(data["quality_satisfaction"]),
                    delivery_satisfaction=CustomerSatisfactionLevel(data["delivery_satisfaction"]),
                    service_satisfaction=CustomerSatisfactionLevel(data["service_satisfaction"]),
                    price_satisfaction=CustomerSatisfactionLevel(data["price_satisfaction"]),
                    nps_score=data.get("nps_score"),
                    feedback_text=data.get("feedback_text", ""),
                    timestamp=datetime.now(timezone.utc)
                )
                await self.manufacturing_metrics.record_customer_satisfaction_metrics(satisfaction_metrics)

            elif event_type == "document_processed":
                from .core.manufacturing_metrics import DocumentProcessingMetrics, DocumentProcessingStatus
                doc_metrics = DocumentProcessingMetrics(
                    processing_id=data["processing_id"],
                    document_type=data["document_type"],
                    file_size_mb=data.get("file_size_mb", 0),
                    processing_time_seconds=data["processing_time_seconds"],
                    success=data["success"],
                    error_message=data.get("error_message"),
                    extracted_entities=data.get("extracted_entities", 0),
                    processing_accuracy=data.get("processing_accuracy", 0),
                    status=DocumentProcessingStatus(data.get("status", "success")),
                    timestamp=datetime.now(timezone.utc)
                )
                await self.manufacturing_metrics.record_document_processing_metrics(doc_metrics)

            logger.debug(f"Manufacturing event recorded: {event_type}")

        except Exception as e:
            logger.error(f"Failed to record manufacturing event: {e}")

    async def get_system_health(self) -> SystemHealthReport:
        """获取系统健康报告"""
        try:
            if not self.initialized:
                await self.initialize()

            timestamp = datetime.now(timezone.utc)
            component_status = {}
            performance_metrics = {}
            cost_metrics = {}
            user_metrics = {}
            manufacturing_metrics = {}
            active_alerts_count = 0
            recommendations = []

            # 检查各个组件状态
            if self.langfuse_integration:
                component_status["langfuse"] = "healthy"
            else:
                component_status["langfuse"] = "disabled"

            if self.performance_tracker:
                component_status["performance_tracker"] = "healthy"
                # 获取性能指标
                performance_metrics = self.performance_tracker.get_real_time_stats(
                    self.performance_tracker.MetricType.RESPONSE_TIME
                )
            else:
                component_status["performance_tracker"] = "error"

            if self.cost_analyzer:
                component_status["cost_analyzer"] = "healthy"
                # 获取成本指标
                cost_metrics = await self.cost_analyzer.get_manufacturing_cost_metrics(timestamp)
                cost_metrics = asdict(cost_metrics)
            else:
                component_status["cost_analyzer"] = "disabled"

            if self.user_analytics:
                component_status["user_analytics"] = "healthy"
                # 获取用户指标
                recent_analytics = await self.user_analytics.get_user_analytics_report(
                    start_date=timestamp - timedelta(days=7),
                    end_date=timestamp
                )
                if recent_analytics and "basic_statistics" in recent_analytics:
                    user_metrics = recent_analytics["basic_statistics"]
            else:
                component_status["user_analytics"] = "disabled"

            if self.manufacturing_metrics:
                component_status["manufacturing_metrics"] = "healthy"
                manufacturing_metrics = self.manufacturing_metrics.get_real_time_metrics()
            else:
                component_status["manufacturing_metrics"] = "disabled"

            if self.alert_system:
                component_status["alert_system"] = "healthy"
                active_alerts = await self.alert_system.get_active_incidents()
                active_alerts_count = len(active_alerts)
            else:
                component_status["alert_system"] = "disabled"

            if self.dashboard_manager:
                component_status["dashboard_manager"] = "healthy"
            else:
                component_status["dashboard_manager"] = "disabled"

            # 确定整体状态
            error_components = [k for k, v in component_status.items() if v == "error"]
            warning_components = [k for k, v in component_status.items() if v == "warning"]

            if error_components:
                overall_status = "critical"
                recommendations.append(f"Critical issues in components: {', '.join(error_components)}")
            elif warning_components or active_alerts_count > 5:
                overall_status = "warning"
                if active_alerts_count > 5:
                    recommendations.append(f"High number of active alerts: {active_alerts_count}")
            else:
                overall_status = "healthy"

            return SystemHealthReport(
                timestamp=timestamp,
                overall_status=overall_status,
                component_status=component_status,
                active_alerts_count=active_alerts_count,
                performance_metrics=performance_metrics,
                cost_metrics=cost_metrics,
                user_metrics=user_metrics,
                manufacturing_metrics=manufacturing_metrics,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return SystemHealthReport(
                timestamp=datetime.now(timezone.utc),
                overall_status="error",
                component_status={},
                active_alerts_count=0,
                performance_metrics={},
                cost_metrics={},
                user_metrics={},
                manufacturing_metrics={},
                recommendations=["Failed to generate health report"]
            )

    async def get_observability_summary(self) -> Dict[str, Any]:
        """获取可观测性摘要"""
        try:
            if not self.initialized:
                await self.initialize()

            health_report = await self.get_system_health()

            summary = {
                "system_health": {
                    "overall_status": health_report.overall_status,
                    "timestamp": health_report.timestamp.isoformat(),
                    "uptime_hours": (datetime.now(timezone.utc) - self.startup_time).total_seconds() / 3600,
                    "active_alerts": health_report.active_alerts_count
                },
                "components": {
                    "langfuse_integration": self.langfuse_integration is not None,
                    "ai_interaction_logger": self.ai_logger is not None,
                    "performance_tracker": self.performance_tracker is not None,
                    "cost_analyzer": self.cost_analyzer is not None,
                    "dashboard_manager": self.dashboard_manager is not None,
                    "user_analytics": self.user_analytics is not None,
                    "alert_system": self.alert_system is not None,
                    "manufacturing_metrics": self.manufacturing_metrics is not None
                },
                "performance": health_report.performance_metrics,
                "costs": health_report.cost_metrics,
                "user_analytics": health_report.user_metrics,
                "manufacturing": health_report.manufacturing_metrics,
                "recommendations": health_report.recommendations
            }

            return summary

        except Exception as e:
            logger.error(f"Failed to get observability summary: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def _system_health_monitor(self):
        """系统健康监控"""
        while True:
            try:
                # 每5分钟检查一次系统健康
                await asyncio.sleep(300)

                health_report = await self.get_system_health()

                # 如果状态是critical，触发警报
                if health_report.overall_status == "critical" and self.alert_system:
                    await self.alert_system._create_alert_incident(
                        rule=None,  # 系统健康警报不需要规则
                        metric_value=0,
                        context_data={
                            "system_health": health_report.overall_status,
                            "error_components": [k for k, v in health_report.component_status.items() if v == "error"]
                        }
                    )

            except Exception as e:
                logger.error(f"Error in system health monitor: {e}")
                await asyncio.sleep(60)

    async def _component_health_check(self):
        """组件健康检查"""
        while True:
            try:
                # 每10分钟检查一次组件健康
                await asyncio.sleep(600)

                # 检查各组件连接状态
                components_to_check = [
                    ("performance_tracker", self.performance_tracker),
                    ("cost_analyzer", self.cost_analyzer),
                    ("user_analytics", self.user_analytics),
                    ("manufacturing_metrics", self.manufacturing_metrics)
                ]

                for name, component in components_to_check:
                    if component:
                        try:
                            # 简单的健康检查 - 尝试获取一些基本数据
                            if hasattr(component, 'get_real_time_stats'):
                                component.get_real_time_stats()
                            elif hasattr(component, 'get_real_time_metrics'):
                                component.get_real_time_metrics()
                            logger.debug(f"Component {name} health check passed")
                        except Exception as e:
                            logger.warning(f"Component {name} health check failed: {e}")

            except Exception as e:
                logger.error(f"Error in component health check: {e}")
                await asyncio.sleep(120)

    async def _metrics_aggregation(self):
        """指标聚合"""
        while True:
            try:
                # 每小时执行一次指标聚合
                await asyncio.sleep(3600)

                # 这里可以添加跨组件的指标聚合逻辑
                # 例如：计算综合的系统效率指标等

                logger.debug("Metrics aggregation completed")

            except Exception as e:
                logger.error(f"Error in metrics aggregation: {e}")
                await asyncio.sleep(300)

    async def shutdown(self):
        """关闭可观测性编排器"""
        try:
            logger.info("Shutting down Observability Orchestrator...")

            # 关闭各个组件
            if self.langfuse_integration:
                self.langfuse_integration.close()
                logger.info("LangFuse integration closed")

            if self.performance_tracker:
                self.performance_tracker.close()
                logger.info("Performance tracker closed")

            if self.cost_analyzer:
                self.cost_analyzer.close()
                logger.info("Cost analyzer closed")

            if self.dashboard_manager:
                self.dashboard_manager.close()
                logger.info("Dashboard manager closed")

            if self.user_analytics:
                self.user_analytics.close()
                logger.info("User analytics closed")

            if self.alert_system:
                self.alert_system.close()
                logger.info("Alert system closed")

            if self.manufacturing_metrics:
                self.manufacturing_metrics.close()
                logger.info("Manufacturing metrics closed")

            self.initialized = False
            logger.info("✅ Observability Orchestrator shutdown completed")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def __del__(self):
        """析构函数"""
        if self.initialized:
            # 注意：这里不能使用await，因为析构函数不是异步的
            logger.warning("ObservabilityOrchestrator was not properly shutdown")

# 全局实例
_orchestrator = None

async def create_observability_orchestrator(config: Optional[ObservabilityConfig] = None) -> ObservabilityOrchestrator:
    """创建可观测性编排器实例"""
    global _orchestrator

    if _orchestrator is None:
        if config is None:
            config = ObservabilityConfig()

        _orchestrator = ObservabilityOrchestrator(config)
        await _orchestrator.initialize()

    return _orchestrator

def get_observability_orchestrator() -> ObservabilityOrchestrator:
    """获取可观测性编排器实例"""
    global _orchestrator
    if _orchestrator is None:
        raise RuntimeError("Observability orchestrator not initialized. Call create_observability_orchestrator() first.")
    return _orchestrator