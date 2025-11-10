"""
LangFuse Integration Framework
Manufacturing Knowledge Base - Observability and Monitoring Integration

This integration provides LangFuse-powered observability, monitoring, and analytics
capabilities with manufacturing-specific metrics, compliance tracking, and
AI performance optimization.
"""

from .integration import (
    LangFuseIntegration,
    ManufacturingMetric,
    ComplianceMetric,
    MetricType
)

__version__ = "2.0.0"
__all__ = [
    "LangFuseIntegration",
    "ManufacturingMetric",
    "ComplianceMetric",
    "MetricType",
]