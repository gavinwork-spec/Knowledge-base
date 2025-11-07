"""
Core logging system for observability.

Provides structured logging for all AI interactions, performance metrics,
user actions, and system events with correlation IDs and trace context.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading
from functools import wraps

import structlog
from pydantic import BaseModel, Field


class LogLevel(str, Enum):
    """Log levels with structured values"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventType(str, Enum):
    """Types of events that can be logged"""
    AI_INTERACTION = "ai_interaction"
    PERFORMANCE_METRIC = "performance_metric"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    ERROR_EVENT = "error_event"
    BUSINESS_EVENT = "business_event"


class InteractionType(str, Enum):
    """Types of AI interactions"""
    SEARCH_QUERY = "search_query"
    DOCUMENT_INDEX = "document_index"
    PERSONALIZED_SEARCH = "personalized_search"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    CHAT_INTERFACE = "chat_interface"
    RECOMMENDATION = "recommendation"


class MetricCategory(str, Enum):
    """Categories of performance metrics"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    BUSINESS_METRIC = "business_metric"


@dataclass
class TraceContext:
    """Trace context for distributed tracing"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None


class LogEvent(BaseModel):
    """Structured log event with all necessary context"""
    timestamp: datetime = Field(default_factory=datetime.now)
    level: LogLevel
    event_type: EventType
    message: str
    trace_context: Optional[Dict[str, str]] = None
    user_context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    duration_ms: Optional[float] = None
    error: Optional[Dict[str, Any]] = None
    tags: List[str] = Field(default_factory=list)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AIInteractionLog(BaseModel):
    """Specific structure for AI interaction logs"""
    interaction_type: InteractionType
    query: str
    response_length: int
    model_used: Optional[str] = None
    tokens_used: Optional[Dict[str, int]] = None
    cost_estimate: Optional[float] = None
    confidence_score: Optional[float] = None
    results_count: int = 0
    filters_applied: List[str] = Field(default_factory=list)
    personalization_applied: bool = False


class PerformanceMetric(BaseModel):
    """Performance metric structure"""
    metric_name: str
    value: float
    unit: str
    category: MetricCategory
    tags: Dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class UserAction(BaseModel):
    """User action tracking structure"""
    action_type: str
    user_id: str
    session_id: str
    resource_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


# Thread-local storage for trace context
_trace_context = threading.local()


class ObservabilityLogger:
    """
    Main logger class for observability with structured logging,
    trace context management, and correlation IDs.
    """

    def __init__(
        self,
        service_name: str = "knowledge-base-api",
        environment: str = "development",
        enable_console_logging: bool = True,
        log_file: Optional[str] = None,
        enable_json_logging: bool = True
    ):
        self.service_name = service_name
        self.environment = environment
        self.enable_json_logging = enable_json_logging

        # Configure structlog
        self._configure_structlog(enable_console_logging, log_file)

        # Get logger instance
        self.logger = structlog.get_logger(service_name)

        # Event storage for analytics
        self._event_buffer: List[LogEvent] = []
        self._buffer_lock = threading.Lock()

    def _configure_structlog(self, enable_console: bool, log_file: Optional[str]):
        """Configure structlog with appropriate processors"""

        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]

        if self.enable_json_logging:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())

        # Configure standard logging
        logging.basicConfig(
            format="%(message)s",
            stream=open(log_file, 'a') if log_file else None,
            level=logging.INFO
        )

        # Configure structlog
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    def _get_trace_context(self) -> Optional[TraceContext]:
        """Get current trace context from thread-local storage"""
        return getattr(_trace_context, 'context', None)

    def _set_trace_context(self, context: TraceContext):
        """Set trace context in thread-local storage"""
        _trace_context.context = context

    def _extract_context(self) -> Dict[str, Any]:
        """Extract current context for logging"""
        trace_context = self._get_trace_context()
        context = {
            "service": self.service_name,
            "environment": self.environment,
        }

        if trace_context:
            context.update({
                "trace_id": trace_context.trace_id,
                "span_id": trace_context.span_id,
                "parent_span_id": trace_context.parent_span_id,
                "user_id": trace_context.user_id,
                "session_id": trace_context.session_id,
                "request_id": trace_context.request_id,
            })

        return context

    def _create_log_event(
        self,
        level: LogLevel,
        event_type: EventType,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        error: Optional[Exception] = None
    ) -> LogEvent:
        """Create a structured log event"""
        return LogEvent(
            level=level,
            event_type=event_type,
            message=message,
            trace_context=self._extract_context(),
            metadata=metadata or {},
            duration_ms=duration_ms,
            error={
                "type": type(error).__name__,
                "message": str(error),
                "stack_trace": traceback.format_exc()
            } if error else None
        )

    def _log_event(self, event: LogEvent):
        """Log a structured event"""
        # Add to buffer for analytics
        with self._buffer_lock:
            self._event_buffer.append(event)
            # Keep buffer size manageable
            if len(self._event_buffer) > 10000:
                self._event_buffer = self._event_buffer[-5000:]

        # Log using structlog
        log_dict = asdict(event)
        log_level = getattr(logging, event.level.value)

        self.logger.log(log_level, event.message, extra=log_dict)

    def debug(self, message: str, event_type: EventType = EventType.SYSTEM_EVENT, **kwargs):
        """Log debug message"""
        event = self._create_log_event(LogLevel.DEBUG, event_type, message, **kwargs)
        self._log_event(event)

    def info(self, message: str, event_type: EventType = EventType.SYSTEM_EVENT, **kwargs):
        """Log info message"""
        event = self._create_log_event(LogLevel.INFO, event_type, message, **kwargs)
        self._log_event(event)

    def warning(self, message: str, event_type: EventType = EventType.SYSTEM_EVENT, **kwargs):
        """Log warning message"""
        event = self._create_log_event(LogLevel.WARNING, event_type, message, **kwargs)
        self._log_event(event)

    def error(self, message: str, error: Optional[Exception] = None, event_type: EventType = EventType.ERROR_EVENT, **kwargs):
        """Log error message"""
        event = self._create_log_event(LogLevel.ERROR, event_type, message, error=error, **kwargs)
        self._log_event(event)

    def critical(self, message: str, error: Optional[Exception] = None, event_type: EventType = EventType.ERROR_EVENT, **kwargs):
        """Log critical message"""
        event = self._create_log_event(LogLevel.CRITICAL, event_type, message, error=error, **kwargs)
        self._log_event(event)

    def log_ai_interaction(
        self,
        interaction_type: InteractionType,
        query: str,
        response_length: int = 0,
        model_used: Optional[str] = None,
        tokens_used: Optional[Dict[str, int]] = None,
        cost_estimate: Optional[float] = None,
        confidence_score: Optional[float] = None,
        results_count: int = 0,
        filters_applied: Optional[List[str]] = None,
        personalization_applied: bool = False,
        duration_ms: Optional[float] = None,
        error: Optional[Exception] = None
    ):
        """Log AI interaction with detailed metrics"""

        ai_log = AIInteractionLog(
            interaction_type=interaction_type,
            query=query,
            response_length=response_length,
            model_used=model_used,
            tokens_used=tokens_used,
            cost_estimate=cost_estimate,
            confidence_score=confidence_score,
            results_count=results_count,
            filters_applied=filters_applied or [],
            personalization_applied=personalization_applied
        )

        level = LogLevel.ERROR if error else LogLevel.INFO
        event_type = EventType.ERROR_EVENT if error else EventType.AI_INTERACTION

        message = f"AI {interaction_type.value}: {query[:100]}{'...' if len(query) > 100 else ''}"

        event = self._create_log_event(
            level=level,
            event_type=event_type,
            message=message,
            metadata={"ai_interaction": ai_log.dict()},
            duration_ms=duration_ms,
            error=error
        )

        self._log_event(event)

    def log_performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str,
        category: MetricCategory,
        tags: Optional[Dict[str, str]] = None
    ):
        """Log performance metric"""

        metric = PerformanceMetric(
            metric_name=metric_name,
            value=value,
            unit=unit,
            category=category,
            tags=tags or {}
        )

        message = f"Performance metric {metric_name}: {value} {unit}"

        event = self._create_log_event(
            level=LogLevel.INFO,
            event_type=EventType.PERFORMANCE_METRIC,
            message=message,
            metadata={"performance_metric": metric.dict()}
        )

        self._log_event(event)

    def log_user_action(
        self,
        action_type: str,
        user_id: str,
        session_id: str,
        resource_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log user action for analytics"""

        action = UserAction(
            action_type=action_type,
            user_id=user_id,
            session_id=session_id,
            resource_id=resource_id,
            metadata=metadata or {}
        )

        message = f"User action: {action_type} by user {user_id}"

        event = self._create_log_event(
            level=LogLevel.INFO,
            event_type=EventType.USER_ACTION,
            message=message,
            metadata={"user_action": action.dict()}
        )

        self._log_event(event)

    def get_recent_events(self, limit: int = 100, event_type: Optional[EventType] = None) -> List[LogEvent]:
        """Get recent events from buffer"""
        with self._buffer_lock:
            events = self._event_buffer.copy()

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events[-limit:]

    @contextmanager
    def trace_context(self, trace_id: Optional[str] = None, **kwargs):
        """Context manager for trace context"""
        current_trace = self._get_trace_context()

        # Create new trace context
        new_trace = TraceContext(
            trace_id=trace_id or str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            parent_span_id=current_trace.span_id if current_trace else None,
            **kwargs
        )

        # Set new context
        self._set_trace_context(new_trace)

        start_time = time.time()
        try:
            yield new_trace
        finally:
            duration_ms = (time.time() - start_time) * 1000
            # Restore previous context
            if current_trace:
                self._set_trace_context(current_trace)
            else:
                delattr(_trace_context, 'context')


# Global logger instance
_global_logger: Optional[ObservabilityLogger] = None


def get_logger() -> ObservabilityLogger:
    """Get global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = ObservabilityLogger()
    return _global_logger


def configure_logger(**kwargs):
    """Configure global logger"""
    global _global_logger
    _global_logger = ObservabilityLogger(**kwargs)


# Convenience functions
def log_ai_interaction(**kwargs):
    """Log AI interaction using global logger"""
    get_logger().log_ai_interaction(**kwargs)


def log_performance_metric(**kwargs):
    """Log performance metric using global logger"""
    get_logger().log_performance_metric(**kwargs)


def log_user_action(**kwargs):
    """Log user action using global logger"""
    get_logger().log_user_action(**kwargs)


def log_system_event(message: str, level: LogLevel = LogLevel.INFO, **kwargs):
    """Log system event using global logger"""
    logger = get_logger()
    if level == LogLevel.DEBUG:
        logger.debug(message, **kwargs)
    elif level == LogLevel.INFO:
        logger.info(message, **kwargs)
    elif level == LogLevel.WARNING:
        logger.warning(message, **kwargs)
    elif level == LogLevel.ERROR:
        logger.error(message, **kwargs)
    elif level == LogLevel.CRITICAL:
        logger.critical(message, **kwargs)


# Decorators for automatic tracing
def trace_function(func):
    """Decorator to automatically trace function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        with logger.trace_context():
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.debug(
                    f"Function {func.__name__} completed",
                    duration_ms=duration_ms,
                    metadata={"function": func.__name__, "args_count": len(args)}
                )
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"Function {func.__name__} failed",
                    error=e,
                    duration_ms=duration_ms,
                    metadata={"function": func.__name__, "args_count": len(args)}
                )
                raise
    return wrapper


def trace_async_function(func):
    """Decorator to automatically trace async function execution"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger = get_logger()
        with logger.trace_context():
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.debug(
                    f"Async function {func.__name__} completed",
                    duration_ms=duration_ms,
                    metadata={"function": func.__name__, "args_count": len(args)}
                )
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"Async function {func.__name__} failed",
                    error=e,
                    duration_ms=duration_ms,
                    metadata={"function": func.__name__, "args_count": len(args)}
                )
                raise
    return wrapper


import traceback