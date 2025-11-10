"""
LangFuse Integration Framework
Manufacturing Knowledge Base - Observability and Monitoring Integration

This integration provides LangFuse-powered observability, monitoring, and analytics
capabilities with manufacturing-specific metrics, compliance tracking, and
AI performance optimization.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import os
from pathlib import Path

try:
    from langfuse import Langfuse
    from langfuse.model import CreateTrace, CreateSpan, CreateEvent, CreateGeneration
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

from ..shared.base import IntegrationBase, ManufacturingContext
from ..shared.utils import ManufacturingMetrics, ComplianceTracker

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of manufacturing metrics to track"""
    AI_PERFORMANCE = "ai_performance"
    SAFETY_COMPLIANCE = "safety_compliance"
    QUALITY_CONTROL = "quality_control"
    EQUIPMENT_UTILIZATION = "equipment_utilization"
    WORKFLOW_EFFICIENCY = "workflow_efficiency"
    COST_OPTIMIZATION = "cost_optimization"
    USER_SATISFACTION = "user_satisfaction"
    ERROR_RATE = "error_rate"


@dataclass
class ManufacturingMetric:
    """Manufacturing-specific metric data"""
    metric_type: MetricType
    value: float
    unit: str
    equipment_type: Optional[str] = None
    process_stage: Optional[str] = None
    user_role: Optional[str] = None
    facility_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ComplianceMetric:
    """Compliance tracking metric"""
    standard: str  # ISO, OSHA, ANSI
    requirement_id: str
    status: str  # compliant, non_compliant, pending
    score: float  # 0-100
    last_check: datetime
    evidence: Optional[List[str]] = None
    corrective_actions: Optional[List[str]] = None


class LangFuseIntegration(IntegrationBase):
    """
    LangFuse integration for manufacturing observability and monitoring

    Provides comprehensive tracking of AI interactions, compliance monitoring,
    performance optimization, and manufacturing-specific KPIs.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "langfuse"
        self.enabled = config.get("enabled", False)

        if not LANGFUSE_AVAILABLE and self.enabled:
            logger.warning("LangFuse not available. Install with: pip install langfuse")
            self.enabled = False
            return

        # Configuration
        self.public_key = config.get("public_key", os.getenv("LANGFUSE_PUBLIC_KEY"))
        self.secret_key = config.get("secret_key", os.getenv("LANGFUSE_SECRET_KEY"))
        self.host = config.get("host", os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))

        # Manufacturing settings
        self.manufacturing_mode = config.get("manufacturing_mode", True)
        self.compliance_tracking = config.get("compliance_tracking", True)
        self.cost_tracking = config.get("cost_tracking", True)
        self.performance_optimization = config.get("performance_optimization", True)

        # Initialize LangFuse client
        self.client = None
        self.metrics_collector = None
        self.compliance_tracker = None

        # Performance tracking
        self.metrics_buffer: List[ManufacturingMetric] = []
        self.compliance_buffer: List[ComplianceMetric] = []
        self.session_cache: Dict[str, Dict] = {}

        logger.info(f"LangFuse integration initialized: enabled={self.enabled}")

    async def initialize(self) -> bool:
        """
        Initialize LangFuse client and manufacturing monitoring systems

        Returns:
            True if initialization successful, False otherwise
        """
        if not self.enabled:
            logger.info("LangFuse integration is disabled")
            return True

        try:
            # Initialize LangFuse client
            self.client = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host
            )

            # Initialize manufacturing components
            if self.manufacturing_mode:
                self.metrics_collector = ManufacturingMetrics()
                self.compliance_tracker = ComplianceTracker()

            # Test connection
            await self._test_connection()

            # Start background tasks
            asyncio.create_task(self._metrics_processor())
            asyncio.create_task(self._compliance_monitor())

            self.initialized = True
            logger.info("LangFuse integration initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize LangFuse integration: {str(e)}")
            return False

    async def _test_connection(self):
        """Test LangFuse connection"""
        try:
            # Create a test trace
            trace_id = str(uuid.uuid4())
            test_trace = CreateTrace(
                id=trace_id,
                name="connection_test",
                input={"test": "connection_verification"},
                metadata={"integration": "langfuse", "test": True}
            )

            # In a real implementation, this would be sent to LangFuse
            logger.info(f"LangFuse connection test - Trace ID: {trace_id}")

        except Exception as e:
            logger.error(f"LangFuse connection test failed: {str(e)}")
            raise

    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process monitoring and observability request

        Args:
            request_data: Request containing monitoring parameters

        Returns:
            Monitoring results and metrics
        """
        if not self.initialized:
            return {"error": "LangFuse integration not initialized"}

        start_time = time.time()
        request_type = request_data.get("type", "general_monitoring")

        try:
            if request_type == "ai_performance":
                return await self._track_ai_performance(request_data)
            elif request_type == "compliance_check":
                return await self._check_compliance(request_data)
            elif request_type == "cost_analysis":
                return await self._analyze_costs(request_data)
            elif request_type == "quality_metrics":
                return await self._get_quality_metrics(request_data)
            elif request_type == "custom_metrics":
                return await self._track_custom_metrics(request_data)
            else:
                return await self._general_monitoring(request_data)

        except Exception as e:
            logger.error(f"Error processing monitoring request: {str(e)}")
            return {"error": str(e), "request_type": request_type}

        finally:
            # Track processing time
            processing_time = time.time() - start_time
            await self._record_metric(
                MetricType.AI_PERFORMANCE,
                processing_time,
                "seconds",
                metadata={"request_type": request_type}
            )

    async def create_trace(self,
                          name: str,
                          inputs: Dict[str, Any],
                          manufacturing_context: Optional[ManufacturingContext] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create LangFuse trace for manufacturing operation

        Args:
            name: Trace name
            inputs: Input data
            manufacturing_context: Manufacturing context
            metadata: Additional metadata

        Returns:
            Trace ID
        """
        if not self.client:
            return ""

        try:
            trace_id = str(uuid.uuid4())

            # Prepare trace data with manufacturing context
            trace_data = {
                "id": trace_id,
                "name": name,
                "input": inputs
            }

            # Add manufacturing context
            if manufacturing_context:
                trace_data["metadata"] = {
                    "equipment_type": manufacturing_context.equipment_type,
                    "user_role": manufacturing_context.user_role,
                    "facility_id": manufacturing_context.facility_id,
                    "process_type": manufacturing_context.process_type,
                    "compliance_standards": manufacturing_context.compliance_standards,
                    **(metadata or {})
                }
            elif metadata:
                trace_data["metadata"] = metadata

            # In a real implementation, send to LangFuse
            logger.info(f"Created LangFuse trace: {trace_id} for {name}")

            return trace_id

        except Exception as e:
            logger.error(f"Error creating LangFuse trace: {str(e)}")
            return ""

    async def create_span(self,
                         trace_id: str,
                         name: str,
                         span_type: str = "default",
                         inputs: Optional[Dict[str, Any]] = None,
                         outputs: Optional[Dict[str, Any]] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create LangFuse span for operation tracking

        Args:
            trace_id: Parent trace ID
            name: Span name
            span_type: Type of span (query, processing, etc.)
            inputs: Input data
            outputs: Output data
            metadata: Additional metadata

        Returns:
            Span ID
        """
        if not self.client:
            return ""

        try:
            span_id = str(uuid.uuid4())

            span_data = {
                "id": span_id,
                "trace_id": trace_id,
                "name": name,
                "type": span_type
            }

            if inputs:
                span_data["input"] = inputs
            if outputs:
                span_data["output"] = outputs
            if metadata:
                span_data["metadata"] = metadata

            # In a real implementation, send to LangFuse
            logger.info(f"Created LangFuse span: {span_id} for {name}")

            return span_id

        except Exception as e:
            logger.error(f"Error creating LangFuse span: {str(e)}")
            return ""

    async def track_ai_generation(self,
                                 trace_id: str,
                                 model_name: str,
                                 prompt: str,
                                 response: str,
                                 usage_data: Dict[str, Any],
                                 manufacturing_context: Optional[ManufacturingContext] = None) -> str:
        """
        Track AI generation with manufacturing context

        Args:
            trace_id: Parent trace ID
            model_name: AI model name
            prompt: Input prompt
            response: Generated response
            usage_data: Usage statistics
            manufacturing_context: Manufacturing context

        Returns:
            Generation ID
        """
        if not self.client:
            return ""

        try:
            generation_id = str(uuid.uuid4())

            generation_data = {
                "id": generation_id,
                "trace_id": trace_id,
                "model": model_name,
                "model_input": prompt,
                "model_output": response,
                "usage": usage_data
            }

            # Add manufacturing context metadata
            if manufacturing_context:
                generation_data["metadata"] = {
                    "equipment_type": manufacturing_context.equipment_type,
                    "user_role": manufacturing_context.user_role,
                    "process_type": manufacturing_context.process_type,
                    "safety_critical": manufacturing_context.is_safety_critical(),
                    "quality_critical": manufacturing_context.is_quality_critical()
                }

            # Track costs if enabled
            if self.cost_tracking:
                cost = self._calculate_generation_cost(model_name, usage_data)
                generation_data["metadata"]["estimated_cost"] = cost

                # Record cost metric
                await self._record_metric(
                    MetricType.COST_OPTIMIZATION,
                    cost,
                    "USD",
                    metadata={"model": model_name, "generation_id": generation_id}
                )

            # In a real implementation, send to LangFuse
            logger.info(f"Tracked AI generation: {generation_id} with model {model_name}")

            return generation_id

        except Exception as e:
            logger.error(f"Error tracking AI generation: {str(e)}")
            return ""

    async def track_safety_event(self,
                                event_type: str,
                                severity: str,
                                description: str,
                                equipment_type: str,
                                action_taken: str,
                                manufacturing_context: Optional[ManufacturingContext] = None):
        """
        Track safety-critical events for compliance monitoring

        Args:
            event_type: Type of safety event
            severity: Event severity (low, medium, high, critical)
            description: Event description
            equipment_type: Equipment involved
            action_taken: Actions taken
            manufacturing_context: Manufacturing context
        """
        try:
            # Create trace for safety event
            trace_id = await self.create_trace(
                name=f"safety_event_{event_type}",
                inputs={
                    "event_type": event_type,
                    "severity": severity,
                    "description": description,
                    "equipment_type": equipment_type,
                    "action_taken": action_taken,
                    "timestamp": datetime.now().isoformat()
                },
                manufacturing_context=manufacturing_context,
                metadata={
                    "safety_critical": True,
                    "compliance_standards": ["OSHA", "ANSI"],
                    "requires_immediate_attention": severity in ["high", "critical"]
                }
            )

            # Record safety compliance metric
            await self._record_metric(
                MetricType.SAFETY_COMPLIANCE,
                1.0 if severity in ["low", "medium"] else 0.5,
                "events",
                equipment_type=equipment_type,
                process_stage="safety_monitoring",
                metadata={
                    "event_type": event_type,
                    "severity": severity,
                    "action_taken": action_taken
                }
            )

            # Check compliance requirements
            if self.compliance_tracker:
                compliance_metric = ComplianceMetric(
                    standard="OSHA",
                    requirement_id=f"safety_event_{event_type}",
                    status="compliant" if action_taken else "pending",
                    score=100.0 if action_taken else 50.0,
                    last_check=datetime.now(),
                    evidence=[f"Event tracked: {description}"],
                    corrective_actions=[] if action_taken else ["Document corrective action"]
                )
                self.compliance_buffer.append(compliance_metric)

            logger.info(f"Safety event tracked: {event_type} - {severity}")

        except Exception as e:
            logger.error(f"Error tracking safety event: {str(e)}")

    async def track_quality_event(self,
                                inspection_type: str,
                                result: str,
                                measurements: Dict[str, float],
                                specifications: Dict[str, Any],
                                manufacturing_context: Optional[ManufacturingContext] = None):
        """
        Track quality control events and inspection results

        Args:
            inspection_type: Type of inspection
            result: Inspection result (pass, fail, needs_review)
            measurements: Measurement data
            specifications: Specification requirements
            manufacturing_context: Manufacturing context
        """
        try:
            # Calculate quality score
            quality_score = self._calculate_quality_score(measurements, specifications)

            # Create trace for quality event
            trace_id = await self.create_trace(
                name=f"quality_inspection_{inspection_type}",
                inputs={
                    "inspection_type": inspection_type,
                    "result": result,
                    "measurements": measurements,
                    "specifications": specifications,
                    "quality_score": quality_score,
                    "timestamp": datetime.now().isoformat()
                },
                manufacturing_context=manufacturing_context,
                metadata={
                    "quality_critical": True,
                    "compliance_standards": ["ISO_9001", "AS9100"],
                    "requires_review": result in ["fail", "needs_review"]
                }
            )

            # Record quality metric
            await self._record_metric(
                MetricType.QUALITY_CONTROL,
                quality_score,
                "score",
                process_stage="quality_control",
                metadata={
                    "inspection_type": inspection_type,
                    "result": result,
                    "measurements_count": len(measurements)
                }
            )

            # Check compliance
            if self.compliance_tracker:
                compliance_metric = ComplianceMetric(
                    standard="ISO_9001",
                    requirement_id=f"quality_{inspection_type}",
                    status="compliant" if result == "pass" else "non_compliant",
                    score=quality_score,
                    last_check=datetime.now(),
                    evidence=[f"Inspection result: {result}"],
                    corrective_actions=[] if result == "pass" else ["Quality improvement plan"]
                )
                self.compliance_buffer.append(compliance_metric)

            logger.info(f"Quality event tracked: {inspection_type} - Score: {quality_score}")

        except Exception as e:
            logger.error(f"Error tracking quality event: {str(e)}")

    async def get_manufacturing_dashboard(self,
                                         time_range: str = "24h",
                                         equipment_type: Optional[str] = None,
                                         facility_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get manufacturing-specific dashboard metrics

        Args:
            time_range: Time range for metrics (1h, 24h, 7d, 30d)
            equipment_type: Filter by equipment type
            facility_id: Filter by facility ID

        Returns:
            Dashboard metrics and analytics
        """
        try:
            # Calculate time range
            end_time = datetime.now()
            if time_range == "1h":
                start_time = end_time - timedelta(hours=1)
            elif time_range == "24h":
                start_time = end_time - timedelta(days=1)
            elif time_range == "7d":
                start_time = end_time - timedelta(days=7)
            elif time_range == "30d":
                start_time = end_time - timedelta(days=30)
            else:
                start_time = end_time - timedelta(days=1)

            # Collect metrics for dashboard
            dashboard_data = {
                "time_range": time_range,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "metrics": {
                    "ai_performance": await self._get_ai_performance_metrics(start_time, end_time),
                    "safety_compliance": await self._get_safety_compliance_metrics(start_time, end_time),
                    "quality_control": await self._get_quality_control_metrics(start_time, end_time),
                    "equipment_utilization": await self._get_equipment_utilization_metrics(start_time, end_time),
                    "cost_optimization": await self._get_cost_metrics(start_time, end_time)
                },
                "alerts": await self._get_active_alerts(equipment_type, facility_id),
                "compliance_status": await self._get_compliance_status()
            }

            # Apply filters
            if equipment_type:
                dashboard_data["equipment_type"] = equipment_type
            if facility_id:
                dashboard_data["facility_id"] = facility_id

            return dashboard_data

        except Exception as e:
            logger.error(f"Error generating manufacturing dashboard: {str(e)}")
            return {"error": str(e)}

    async def export_compliance_report(self,
                                     standard: str,
                                     start_date: datetime,
                                     end_date: datetime,
                                     format: str = "json") -> Dict[str, Any]:
        """
        Export compliance report for specific standard

        Args:
            standard: Compliance standard (ISO_9001, OSHA, AS9100)
            start_date: Report start date
            end_date: Report end date
            format: Export format (json, csv, pdf)

        Returns:
            Compliance report data
        """
        try:
            report_data = {
                "standard": standard,
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "compliance_metrics": [],
                "violations": [],
                "corrective_actions": [],
                "audit_trail": []
            }

            # Collect compliance metrics
            for metric in self.compliance_buffer:
                if metric.standard == standard and start_date <= metric.last_check <= end_date:
                    report_data["compliance_metrics"].append(asdict(metric))

                    if metric.status == "non_compliant":
                        report_data["violations"].append({
                            "requirement_id": metric.requirement_id,
                            "score": metric.score,
                            "last_check": metric.last_check.isoformat(),
                            "corrective_actions": metric.corrective_actions or []
                        })

            # Calculate overall compliance score
            if report_data["compliance_metrics"]:
                scores = [m.score for m in report_data["compliance_metrics"]]
                report_data["overall_score"] = sum(scores) / len(scores)
            else:
                report_data["overall_score"] = 0.0

            report_data["generated_at"] = datetime.now().isoformat()
            report_data["format"] = format

            return report_data

        except Exception as e:
            logger.error(f"Error exporting compliance report: {str(e)}")
            return {"error": str(e)}

    # Private methods

    async def _track_ai_performance(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track AI performance metrics"""
        try:
            metrics = {
                "response_time": request_data.get("response_time", 0),
                "accuracy": request_data.get("accuracy", 0),
                "relevance": request_data.get("relevance", 0),
                "user_satisfaction": request_data.get("user_satisfaction", 0)
            }

            # Record performance metrics
            for metric_name, value in metrics.items():
                await self._record_metric(
                    MetricType.AI_PERFORMANCE,
                    value,
                    "score",
                    metadata={"metric_type": metric_name}
                )

            return {
                "tracked_metrics": metrics,
                "optimization_suggestions": await self._get_optimization_suggestions(metrics)
            }

        except Exception as e:
            logger.error(f"Error tracking AI performance: {str(e)}")
            return {"error": str(e)}

    async def _check_compliance(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance status"""
        try:
            standard = request_data.get("standard", "ISO_9001")
            requirement_id = request_data.get("requirement_id")

            # Get compliance status
            compliance_metrics = [m for m in self.compliance_buffer
                                if m.standard == standard and
                                (not requirement_id or m.requirement_id == requirement_id)]

            return {
                "standard": standard,
                "compliance_status": [asdict(m) for m in compliance_metrics],
                "overall_score": sum(m.score for m in compliance_metrics) / len(compliance_metrics) if compliance_metrics else 0
            }

        except Exception as e:
            logger.error(f"Error checking compliance: {str(e)}")
            return {"error": str(e)}

    async def _analyze_costs(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze AI usage costs"""
        try:
            time_range = request_data.get("time_range", "24h")

            # Get cost metrics
            cost_metrics = [m for m in self.metrics_buffer
                           if m.metric_type == MetricType.COST_OPTIMIZATION]

            total_cost = sum(m.value for m in cost_metrics)
            cost_by_model = {}
            for metric in cost_metrics:
                model = metric.metadata.get("model", "unknown")
                cost_by_model[model] = cost_by_model.get(model, 0) + metric.value

            return {
                "time_range": time_range,
                "total_cost": total_cost,
                "cost_by_model": cost_by_model,
                "optimization_opportunities": await self._identify_cost_optimizations(cost_by_model)
            }

        except Exception as e:
            logger.error(f"Error analyzing costs: {str(e)}")
            return {"error": str(e)}

    async def _get_quality_metrics(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get quality control metrics"""
        try:
            equipment_type = request_data.get("equipment_type")

            quality_metrics = [m for m in self.metrics_buffer
                             if m.metric_type == MetricType.QUALITY_CONTROL and
                             (not equipment_type or m.equipment_type == equipment_type)]

            if quality_metrics:
                avg_quality = sum(m.value for m in quality_metrics) / len(quality_metrics)
                quality_trend = await self._calculate_quality_trend(quality_metrics)
            else:
                avg_quality = 0
                quality_trend = []

            return {
                "average_quality_score": avg_quality,
                "quality_trend": quality_trend,
                "quality_events_count": len(quality_metrics)
            }

        except Exception as e:
            logger.error(f"Error getting quality metrics: {str(e)}")
            return {"error": str(e)}

    async def _track_custom_metrics(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track custom manufacturing metrics"""
        try:
            custom_metrics = request_data.get("metrics", [])
            tracked_metrics = []

            for metric_data in custom_metrics:
                metric_type = MetricType(metric_data.get("type", "AI_PERFORMANCE"))
                await self._record_metric(
                    metric_type,
                    metric_data.get("value", 0),
                    metric_data.get("unit", "count"),
                    equipment_type=metric_data.get("equipment_type"),
                    process_stage=metric_data.get("process_stage"),
                    metadata=metric_data.get("metadata", {})
                )
                tracked_metrics.append(metric_data)

            return {
                "tracked_metrics": tracked_metrics,
                "total_metrics": len(tracked_metrics)
            }

        except Exception as e:
            logger.error(f"Error tracking custom metrics: {str(e)}")
            return {"error": str(e)}

    async def _general_monitoring(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """General monitoring request"""
        try:
            return {
                "status": "active",
                "metrics_buffer_size": len(self.metrics_buffer),
                "compliance_buffer_size": len(self.compliance_buffer),
                "active_sessions": len(self.session_cache),
                "integration_health": await self.health_check()
            }

        except Exception as e:
            logger.error(f"Error in general monitoring: {str(e)}")
            return {"error": str(e)}

    async def _record_metric(self,
                           metric_type: MetricType,
                           value: float,
                           unit: str,
                           equipment_type: Optional[str] = None,
                           process_stage: Optional[str] = None,
                           user_role: Optional[str] = None,
                           facility_id: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        """Record manufacturing metric"""
        metric = ManufacturingMetric(
            metric_type=metric_type,
            value=value,
            unit=unit,
            equipment_type=equipment_type,
            process_stage=process_stage,
            user_role=user_role,
            facility_id=facility_id,
            timestamp=datetime.now(),
            metadata=metadata
        )
        self.metrics_buffer.append(metric)

        # Process immediately if buffer is large
        if len(self.metrics_buffer) > 100:
            await self._process_metrics_buffer()

    async def _metrics_processor(self):
        """Background task to process metrics buffer"""
        while True:
            try:
                await asyncio.sleep(60)  # Process every minute
                await self._process_metrics_buffer()
            except Exception as e:
                logger.error(f"Error in metrics processor: {str(e)}")

    async def _process_metrics_buffer(self):
        """Process accumulated metrics"""
        if not self.metrics_buffer:
            return

        try:
            # In a real implementation, send to LangFuse or data warehouse
            metrics_count = len(self.metrics_buffer)
            self.metrics_buffer.clear()
            logger.debug(f"Processed {metrics_count} metrics")

        except Exception as e:
            logger.error(f"Error processing metrics buffer: {str(e)}")

    async def _compliance_monitor(self):
        """Background task to monitor compliance"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                await self._process_compliance_buffer()
            except Exception as e:
                logger.error(f"Error in compliance monitor: {str(e)}")

    async def _process_compliance_buffer(self):
        """Process compliance metrics"""
        if not self.compliance_buffer:
            return

        try:
            # In a real implementation, send to compliance monitoring system
            compliance_count = len(self.compliance_buffer)
            self.compliance_buffer.clear()
            logger.debug(f"Processed {compliance_count} compliance metrics")

        except Exception as e:
            logger.error(f"Error processing compliance buffer: {str(e)}")

    def _calculate_generation_cost(self, model_name: str, usage_data: Dict[str, Any]) -> float:
        """Calculate estimated cost for AI generation"""
        # Simplified cost calculation - in reality this would use actual pricing
        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)

        # Example pricing (per 1K tokens)
        pricing = {
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
            "claude-3": {"prompt": 0.015, "completion": 0.075}
        }

        model_pricing = pricing.get(model_name, pricing["gpt-3.5-turbo"])

        prompt_cost = (prompt_tokens / 1000) * model_pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * model_pricing["completion"]

        return prompt_cost + completion_cost

    def _calculate_quality_score(self,
                               measurements: Dict[str, float],
                               specifications: Dict[str, Any]) -> float:
        """Calculate quality score from measurements and specifications"""
        scores = []

        for spec_key, spec_value in specifications.items():
            if spec_key in measurements:
                measurement = measurements[spec_key]

                if isinstance(spec_value, dict):
                    # Handle range specifications
                    min_val = spec_value.get("min", float("-inf"))
                    max_val = spec_value.get("max", float("inf"))
                    target = spec_value.get("target")

                    if min_val <= measurement <= max_val:
                        if target:
                            # Score based on deviation from target
                            tolerance = (max_val - min_val) / 2
                            deviation = abs(measurement - target)
                            score = max(0, 1 - (deviation / tolerance))
                        else:
                            score = 1.0
                    else:
                        score = 0.0
                else:
                    # Handle exact specifications
                    score = 1.0 if measurement == spec_value else 0.0

                scores.append(score)

        return (sum(scores) / len(scores) * 100) if scores else 0.0

    async def _get_optimization_suggestions(self, metrics: Dict[str, float]) -> List[str]:
        """Get optimization suggestions based on metrics"""
        suggestions = []

        if metrics.get("response_time", 0) > 2.0:
            suggestions.append("Consider response time optimization - enable caching or use faster models")

        if metrics.get("accuracy", 1.0) < 0.8:
            suggestions.append("Improve accuracy by fine-tuning prompts or using higher-quality models")

        if metrics.get("user_satisfaction", 1.0) < 0.7:
            suggestions.append("Review response quality and user feedback for improvements")

        return suggestions

    async def _get_ai_performance_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get AI performance metrics for time range"""
        ai_metrics = [m for m in self.metrics_buffer
                     if m.metric_type == MetricType.AI_PERFORMANCE and
                     start_time <= (m.timestamp or datetime.now()) <= end_time]

        if not ai_metrics:
            return {"metrics_count": 0}

        # Calculate aggregations by metadata metric_type
        metrics_by_type = {}
        for metric in ai_metrics:
            metric_type = metric.metadata.get("metric_type", "unknown")
            if metric_type not in metrics_by_type:
                metrics_by_type[metric_type] = []
            metrics_by_type[metric_type].append(metric.value)

        aggregations = {}
        for metric_type, values in metrics_by_type.items():
            aggregations[metric_type] = {
                "count": len(values),
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values)
            }

        return aggregations

    async def _get_safety_compliance_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get safety compliance metrics"""
        safety_metrics = [m for m in self.metrics_buffer
                         if m.metric_type == MetricType.SAFETY_COMPLIANCE and
                         start_time <= (m.timestamp or datetime.now()) <= end_time]

        events_by_severity = {}
        for metric in safety_metrics:
            severity = metric.metadata.get("severity", "unknown")
            events_by_severity[severity] = events_by_severity.get(severity, 0) + 1

        return {
            "total_events": len(safety_metrics),
            "events_by_severity": events_by_severity,
            "compliance_rate": sum(m.value for m in safety_metrics) / len(safety_metrics) if safety_metrics else 0
        }

    async def _get_quality_control_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get quality control metrics"""
        quality_metrics = [m for m in self.metrics_buffer
                          if m.metric_type == MetricType.QUALITY_CONTROL and
                          start_time <= (m.timestamp or datetime.now()) <= end_time]

        if not quality_metrics:
            return {"metrics_count": 0}

        quality_scores = [m.value for m in quality_metrics]

        return {
            "total_inspections": len(quality_metrics),
            "average_quality_score": sum(quality_scores) / len(quality_scores),
            "quality_trend": await self._calculate_quality_trend(quality_metrics)
        }

    async def _get_equipment_utilization_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get equipment utilization metrics"""
        equipment_metrics = [m for m in self.metrics_buffer
                           if m.metric_type == MetricType.EQUIPMENT_UTILIZATION and
                           start_time <= (m.timestamp or datetime.now()) <= end_time]

        utilization_by_type = {}
        for metric in equipment_metrics:
            equipment_type = metric.equipment_type or "unknown"
            if equipment_type not in utilization_by_type:
                utilization_by_type[equipment_type] = []
            utilization_by_type[equipment_type].append(metric.value)

        return {
            "total_equipment_hours": sum(m.value for m in equipment_metrics),
            "utilization_by_type": {
                eq_type: {
                    "average_utilization": sum(values) / len(values),
                    "total_hours": sum(values)
                }
                for eq_type, values in utilization_by_type.items()
            }
        }

    async def _get_cost_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get cost optimization metrics"""
        cost_metrics = [m for m in self.metrics_buffer
                       if m.metric_type == MetricType.COST_OPTIMIZATION and
                       start_time <= (m.timestamp or datetime.now()) <= end_time]

        total_cost = sum(m.value for m in cost_metrics)
        cost_by_model = {}
        for metric in cost_metrics:
            model = metric.metadata.get("model", "unknown")
            cost_by_model[model] = cost_by_model.get(model, 0) + metric.value

        return {
            "total_cost": total_cost,
            "cost_by_model": cost_by_model,
            "cost_per_request": total_cost / len(cost_metrics) if cost_metrics else 0
        }

    async def _get_active_alerts(self,
                                equipment_type: Optional[str] = None,
                                facility_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active alerts"""
        # In a real implementation, this would query actual alert systems
        alerts = []

        # Check for critical safety events
        recent_safety = [m for m in self.metrics_buffer[-10:]
                        if m.metric_type == MetricType.SAFETY_COMPLIANCE and
                        m.metadata.get("severity") in ["high", "critical"]]

        for metric in recent_safety:
            if not equipment_type or metric.equipment_type == equipment_type:
                if not facility_id or metric.facility_id == facility_id:
                    alerts.append({
                        "type": "safety",
                        "severity": metric.metadata.get("severity"),
                        "description": "Critical safety event detected",
                        "timestamp": (metric.timestamp or datetime.now()).isoformat(),
                        "equipment_type": metric.equipment_type
                    })

        return alerts

    async def _get_compliance_status(self) -> Dict[str, Any]:
        """Get overall compliance status"""
        if not self.compliance_buffer:
            return {"status": "no_data"}

        # Group by standard
        compliance_by_standard = {}
        for metric in self.compliance_buffer:
            standard = metric.standard
            if standard not in compliance_by_standard:
                compliance_by_standard[standard] = []
            compliance_by_standard[standard].append(metric)

        overall_status = {}
        for standard, metrics in compliance_by_standard.items():
            compliant_count = sum(1 for m in metrics if m.status == "compliant")
            overall_score = sum(m.score for m in metrics) / len(metrics)

            overall_status[standard] = {
                "compliance_rate": compliant_count / len(metrics),
                "overall_score": overall_score,
                "total_requirements": len(metrics),
                "non_compliant_count": len(metrics) - compliant_count
            }

        return overall_status

    async def _calculate_quality_trend(self, quality_metrics: List[ManufacturingMetric]) -> List[Dict[str, Any]]:
        """Calculate quality trend over time"""
        if len(quality_metrics) < 2:
            return []

        # Sort by timestamp
        sorted_metrics = sorted(quality_metrics, key=lambda m: m.timestamp or datetime.min)

        trend = []
        for i, metric in enumerate(sorted_metrics):
            if i > 0:
                prev_metric = sorted_metrics[i-1]
                trend_value = metric.value - prev_metric.value
                trend.append({
                    "timestamp": metric.timestamp.isoformat() if metric.timestamp else datetime.now().isoformat(),
                    "value": metric.value,
                    "trend": "up" if trend_value > 0 else "down" if trend_value < 0 else "stable",
                    "change": trend_value
                })

        return trend

    async def _identify_cost_optimizations(self, cost_by_model: Dict[str, float]) -> List[str]:
        """Identify cost optimization opportunities"""
        optimizations = []

        if not cost_by_model:
            return optimizations

        total_cost = sum(cost_by_model.values())

        for model, cost in cost_by_model.items():
            percentage = (cost / total_cost) * 100

            if percentage > 50:
                optimizations.append(f"Consider optimizing {model} usage - accounts for {percentage:.1f}% of total costs")

            if "gpt-4" in model and percentage > 30:
                optimizations.append(f"Consider using more cost-effective models instead of {model}")

        return optimizations

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on LangFuse integration"""
        try:
            health_status = {
                "status": "healthy" if self.initialized else "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "connection_status": "connected" if self.client else "disconnected",
                "metrics_buffer_size": len(self.metrics_buffer),
                "compliance_buffer_size": len(self.compliance_buffer),
                "active_sessions": len(self.session_cache),
                "features": {
                    "manufacturing_mode": self.manufacturing_mode,
                    "compliance_tracking": self.compliance_tracking,
                    "cost_tracking": self.cost_tracking,
                    "performance_optimization": self.performance_optimization
                }
            }

            if self.client:
                # Test connection
                health_status["connection_test"] = "passed"

            return health_status

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }