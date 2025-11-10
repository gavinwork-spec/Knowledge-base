"""
Shared Integration Utilities
Manufacturing Knowledge Base - Common Integration Components

This module provides shared utilities, base classes, and common functionality
for all open-source component integrations.
"""

from .base import IntegrationBase, ManufacturingContext
from .config import IntegrationConfig, ConfigManager
from .monitoring import IntegrationMonitor, HealthCheck
from .errors import (
    IntegrationError,
    ConfigurationError,
    ManufacturingContextError,
    HealthCheckError
)

__version__ = "1.0.0"
__all__ = [
    # Base classes
    "IntegrationBase",
    "ManufacturingContext",

    # Configuration
    "IntegrationConfig",
    "ConfigManager",

    # Monitoring
    "IntegrationMonitor",
    "HealthCheck",

    # Errors
    "IntegrationError",
    "ConfigurationError",
    "ManufacturingContextError",
    "HealthCheckError",
]