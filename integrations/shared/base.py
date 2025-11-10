"""
Base Classes for Integrations
Manufacturing Knowledge Base - Integration Foundation
"""

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .config import IntegrationConfig
from .monitoring import IntegrationMonitor
from .errors import IntegrationError, ManufacturingContextError

logger = logging.getLogger(__name__)


class IntegrationStatus(Enum):
    """Integration status enumeration"""
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


@dataclass
class ManufacturingDomain:
    """Manufacturing domain configuration"""
    industry: str = "manufacturing"
    sub_industry: str = "general"
    compliance_standards: List[str] = None
    safety_regulations: List[str] = None
    quality_standards: List[str] = None

    def __post_init__(self):
        if self.compliance_standards is None:
            self.compliance_standards = ["ISO_9001", "OSHA"]
        if self.safety_regulations is None:
            self.safety_regulations = ["ANSI_Z535", "OSHA_1910"]
        if self.quality_standards is None:
            self.quality_standards = ["AS9100", "IATF_16949"]


@dataclass
class ManufacturingContext:
    """Manufacturing-specific context for integrations"""
    domain: ManufacturingDomain
    user_role: str = "operator"
    equipment_type: Optional[str] = None
    process_type: Optional[str] = None
    work_order_id: Optional[str] = None
    facility_id: Optional[str] = None
    shift_id: Optional[str] = None

    def get_context_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for API calls"""
        return {
            "domain": self.domain.__dict__,
            "user_role": self.user_role,
            "equipment_type": self.equipment_type,
            "process_type": self.process_type,
            "work_order_id": self.work_order_id,
            "facility_id": self.facility_id,
            "shift_id": self.shift_id,
        }


class IntegrationBase(ABC):
    """
    Base class for all integrations.
    Provides common functionality and interface for open-source component integrations.
    """

    def __init__(self, name: str, config: IntegrationConfig):
        self.name = name
        self.config = config
        self.status = IntegrationStatus.INITIALIZING
        self.manufacturing_context = self._initialize_manufacturing_context()
        self.monitor = IntegrationMonitor(name, config)
        self.start_time = None
        self._shutdown_event = asyncio.Event()

        # Performance metrics
        self.metrics = {
            "requests_processed": 0,
            "errors_encountered": 0,
            "average_response_time": 0.0,
            "last_health_check": None,
        }

        logger.info(f"Initialized {name} integration")

    def _initialize_manufacturing_context(self) -> ManufacturingContext:
        """Initialize manufacturing context from configuration"""
        try:
            domain_config = self.config.get("manufacturing.domain", {})
            domain = ManufacturingDomain(**domain_config)

            return ManufacturingContext(
                domain=domain,
                user_role=self.config.get("manufacturing.user_role", "operator"),
                equipment_type=self.config.get("manufacturing.equipment_type"),
                process_type=self.config.get("manufacturing.process_type"),
            )
        except Exception as e:
            raise ManufacturingContextError(f"Failed to initialize manufacturing context: {e}")

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the integration.
        Must be implemented by each integration component.

        Returns:
            bool: True if initialization successful
        """
        pass

    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Shutdown the integration gracefully.
        Must be implemented by each integration component.

        Returns:
            bool: True if shutdown successful
        """
        pass

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for the integration.

        Returns:
            Dict[str, Any]: Health check results
        """
        try:
            health_status = {
                "integration_name": self.name,
                "status": self.status.value,
                "uptime": self._get_uptime(),
                "metrics": self.metrics.copy(),
                "last_check": datetime.now().isoformat(),
            }

            # Integration-specific health check
            integration_health = await self._integration_health_check()
            health_status.update(integration_health)

            # Update last health check timestamp
            self.metrics["last_health_check"] = datetime.now().isoformat()

            return health_status

        except Exception as e:
            logger.error(f"Health check failed for {self.name}: {e}")
            return {
                "integration_name": self.name,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    @abstractmethod
    async def _integration_health_check(self) -> Dict[str, Any]:
        """
        Integration-specific health check.
        Must be implemented by each integration.

        Returns:
            Dict[str, Any]: Integration-specific health metrics
        """
        pass

    def _get_uptime(self) -> float:
        """Calculate uptime in seconds"""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0

    async def process_request(
        self,
        request_data: Any,
        context: Optional[ManufacturingContext] = None
    ) -> Any:
        """
        Process a request with manufacturing context.

        Args:
            request_data: The request data to process
            context: Optional manufacturing context override

        Returns:
            Any: Processed response
        """
        start_time = time.time()

        try:
            # Use provided context or default
            processing_context = context or self.manufacturing_context

            # Log the request
            self.monitor.log_request(request_data, processing_context)

            # Process the request
            response = await self._process_with_context(request_data, processing_context)

            # Update metrics
            response_time = time.time() - start_time
            self._update_metrics(response_time, success=True)

            # Log the response
            self.monitor.log_response(response, response_time)

            return response

        except Exception as e:
            response_time = time.time() - start_time
            self._update_metrics(response_time, success=False)
            logger.error(f"Request processing failed for {self.name}: {e}")
            raise IntegrationError(f"Request processing failed: {e}")

    @abstractmethod
    async def _process_with_context(
        self,
        request_data: Any,
        context: ManufacturingContext
    ) -> Any:
        """
        Process request with manufacturing context.
        Must be implemented by each integration.

        Args:
            request_data: Request data to process
            context: Manufacturing context

        Returns:
            Any: Processed response
        """
        pass

    def _update_metrics(self, response_time: float, success: bool):
        """Update internal metrics"""
        self.metrics["requests_processed"] += 1

        if not success:
            self.metrics["errors_encountered"] += 1

        # Update average response time
        total_requests = self.metrics["requests_processed"]
        current_avg = self.metrics["average_response_time"]
        self.metrics["average_response_time"] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )

    def update_manufacturing_context(self, **kwargs):
        """
        Update manufacturing context with new values.

        Args:
            **kwargs: Context parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.manufacturing_context, key):
                setattr(self.manufacturing_context, key, value)
            else:
                logger.warning(f"Unknown context parameter: {key}")

    def get_integration_info(self) -> Dict[str, Any]:
        """Get integration information and status"""
        return {
            "name": self.name,
            "status": self.status.value,
            "config": self.config.get_public_config(),
            "manufacturing_context": self.manufacturing_context.get_context_dict(),
            "metrics": self.metrics.copy(),
            "uptime": self._get_uptime(),
        }

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.shutdown()


class IntegrationManager:
    """
    Manager for multiple integrations.
    Provides centralized management and coordination.
    """

    def __init__(self):
        self.integrations: Dict[str, IntegrationBase] = {}
        self.config_manager = None
        self.global_monitor = None

    async def initialize(self, config_path: str):
        """Initialize all integrations from configuration"""
        from .config import ConfigManager

        try:
            # Load configuration
            self.config_manager = ConfigManager(config_path)
            await self.config_manager.load_config()

            # Initialize integrations
            integration_configs = self.config_manager.get_integration_configs()

            for name, config in integration_configs.items():
                if config.get("enabled", False):
                    await self.add_integration(name, config)

            logger.info(f"Initialized {len(self.integrations)} integrations")

        except Exception as e:
            logger.error(f"Failed to initialize integration manager: {e}")
            raise

    async def add_integration(self, name: str, config: IntegrationConfig):
        """Add a new integration"""
        try:
            # Import integration class based on name
            integration_class = self._get_integration_class(name)

            # Create integration instance
            integration = integration_class(name, config)

            # Initialize integration
            success = await integration.initialize()

            if success:
                self.integrations[name] = integration
                logger.info(f"Added integration: {name}")
            else:
                logger.error(f"Failed to initialize integration: {name}")

        except Exception as e:
            logger.error(f"Failed to add integration {name}: {e}")
            raise

    def _get_integration_class(self, name: str):
        """Get integration class by name"""
        # This would be implemented to dynamically import integration classes
        # For now, return a placeholder
        if name == "langchain":
            from ..langchain import LangChainIntegration
            return LangChainIntegration
        elif name == "lobechat":
            from ..lobechat import LobeChatIntegration
            return LobeChatIntegration
        elif name == "xagent":
            from ..xagent import XAgentIntegration
            return XAgentIntegration
        elif name == "langfuse":
            from ..langfuse import LangFuseIntegration
            return LangFuseIntegration
        else:
            raise ValueError(f"Unknown integration: {name}")

    def get_integration(self, name: str) -> IntegrationBase:
        """Get integration by name"""
        if name not in self.integrations:
            raise ValueError(f"Integration not found: {name}")
        return self.integrations[name]

    async def shutdown_all(self):
        """Shutdown all integrations"""
        logger.info("Shutting down all integrations")

        for name, integration in self.integrations.items():
            try:
                await integration.shutdown()
                logger.info(f"Shutdown integration: {name}")
            except Exception as e:
                logger.error(f"Failed to shutdown integration {name}: {e}")

        self.integrations.clear()
        logger.info("All integrations shutdown")

    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all integrations"""
        health_results = {}

        for name, integration in self.integrations.items():
            try:
                health_results[name] = await integration.health_check()
            except Exception as e:
                health_results[name] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }

        return health_results

    def get_status(self) -> Dict[str, Any]:
        """Get status of all integrations"""
        return {
            "total_integrations": len(self.integrations),
            "integrations": {
                name: integration.get_integration_info()
                for name, integration in self.integrations.items()
            },
            "timestamp": datetime.now().isoformat(),
        }