#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAgent Configuration Manager
XAgent配置管理器

This module manages XAgent configurations, loading YAML files,
migrating existing agents, and providing configuration validation.
"""

import yaml
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler('data/processed/xagent_config_manager.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class XAgentConfigManager:
    """Manages XAgent configurations and migrations"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.agents_config = {}
        self.workflows_config = {}
        self.orchestrator_config = {}
        self.migration_log = []

    async def load_configurations(self):
        """Load all XAgent configurations"""
        logger.info("📁 Loading XAgent configurations")

        # Load main XAgent manufacturing configuration
        xagent_config_path = self.config_dir / "xagent_manufacturing_agents.yaml"
        if xagent_config_path.exists():
            await self._load_xagent_config(xagent_config_path)

        # Migrate existing YAML agent configurations
        await self._migrate_existing_agents()

        # Validate configurations
        await self._validate_configurations()

        logger.info(f"✅ Loaded configurations for {len(self.agents_config)} agents")

    async def _load_xagent_config(self, config_path: Path):
        """Load XAgent manufacturing configuration"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Extract different sections
            self.orchestrator_config = config.get('orchestrator', {})
            self.agents_config = config.get('agents', {})
            self.workflows_config = config.get('workflows', {})

            logger.info(f"✅ Loaded XAgent config from {config_path}")

        except Exception as e:
            logger.error(f"❌ Failed to load XAgent config: {e}")
            raise

    async def _migrate_existing_agents(self):
        """Migrate existing YAML agent configurations to XAgent patterns"""
        logger.info("🔄 Migrating existing agent configurations to XAgent patterns")

        existing_agent_files = list(self.config_dir.glob("*_agent.yaml"))

        for agent_file in existing_agent_files:
            if agent_file.name != "xagent_manufacturing_agents.yaml":
                await self._migrate_single_agent(agent_file)

        logger.info(f"✅ Migrated {len(existing_agent_files)} existing agent configurations")

    async def _migrate_single_agent(self, agent_file: Path):
        """Migrate a single agent configuration to XAgent pattern"""
        try:
            with open(agent_file, 'r', encoding='utf-8') as f:
                old_config = yaml.safe_load(f)

            # Create XAgent-compatible configuration
            xagent_config = await self._convert_to_xagent_format(old_config, agent_file.stem)

            # Store migrated configuration
            self.agents_config[xagent_config['id']] = xagent_config

            # Log migration
            self.migration_log.append({
                "original_file": str(agent_file),
                "agent_id": xagent_config['id'],
                "migration_time": datetime.now().isoformat(),
                "status": "success"
            })

            logger.info(f"✅ Migrated agent: {xagent_config['id']}")

        except Exception as e:
            logger.error(f"❌ Failed to migrate {agent_file}: {e}")
            self.migration_log.append({
                "original_file": str(agent_file),
                "error": str(e),
                "migration_time": datetime.now().isoformat(),
                "status": "failed"
            })

    async def _convert_to_xagent_format(self, old_config: Dict, agent_id: str) -> Dict[str, Any]:
        """Convert legacy agent configuration to XAgent format"""

        # Determine agent role based on configuration
        role_mapping = {
            "customer_ingest_agent": "supply_chain_coordinator",
            "analysis_agent": "knowledge_manager",
            "reminder_agent": "performance_monitor",
            "parse_documents_agent": "knowledge_manager",
            "learn_from_updates_agent": "knowledge_manager",
            "quote_strategy_agent": "process_optimizer",
            "classify_drawings_agent": "quality_controller",
            "analyze_factory_trends_agent": "process_optimizer"
        }

        role = role_mapping.get(agent_id, "knowledge_manager")

        # Map actions to capabilities
        capabilities_mapping = {
            "run_customer_ingestion": "document_processing",
            "run_data_analysis": "data_analysis",
            "validate_data_quality": "quality_inspection",
            "parse_document": "document_processing",
            "build_embedding_index": "knowledge_synthesis",
            "analyze_trends": "data_analysis",
            "generate_recommendations": "process_optimization"
        }

        capabilities = []
        if 'actions' in old_config:
            for action in old_config['actions']:
                action_name = action.get('name', '')
                if action_name in capabilities_mapping:
                    capabilities.append(capabilities_mapping[action_name])

        # Ensure unique capabilities
        capabilities = list(set(capabilities)) or ["document_processing"]

        # Create XAgent configuration
        xagent_config = {
            "id": agent_id,
            "class": self._get_agent_class(role),
            "name": old_config.get('description', agent_id.replace('_', ' ').title()),
            "role": role,
            "description": old_config.get('description', f"Migrated from legacy configuration"),

            "capabilities": capabilities,

            "config": {
                "legacy_settings": old_config.get('settings', {}),
                "legacy_triggers": old_config.get('triggers', []),
                "legacy_actions": old_config.get('actions', []),
                "legacy_notifications": old_config.get('notifications', {}),
                "migration_timestamp": datetime.now().isoformat()
            },

            "skills": {},

            "triggers": self._convert_triggers(old_config.get('triggers', [])),

            "performance_metrics": {
                "tasks_completed": 0,
                "average_task_time": 0.0,
                "success_rate": 1.0,
                "last_activity": datetime.now().isoformat()
            }
        }

        # Add skills based on capabilities
        for capability in capabilities:
            skill_name = f"{role}_{capability}"
            xagent_config["skills"][skill_name] = {
                "proficiency": 0.7,
                "experience": 0,
                "last_used": datetime.now().isoformat(),
                "success_rate": 1.0
            }

        return xagent_config

    def _get_agent_class(self, role: str) -> str:
        """Get appropriate agent class for role"""
        class_mapping = {
            "safety_inspector": "ManufacturingSafetyInspector",
            "quality_controller": "QualityController",
            "maintenance_technician": "MaintenanceTechnician",
            "production_manager": "ProductionManager",
            "supply_chain_coordinator": "SupplyChainCoordinator",
            "compliance_auditor": "ComplianceAuditor",
            "knowledge_manager": "KnowledgeManager",
            "process_optimizer": "ProcessOptimizer",
            "inventory_manager": "InventoryManager",
            "risk_analyzer": "RiskAnalyzer",
            "performance_monitor": "PerformanceMonitor"
        }

        return class_mapping.get(role, "GenericAgent")

    def _convert_triggers(self, old_triggers: List[Dict]) -> List[Dict]:
        """Convert legacy triggers to XAgent format"""
        converted_triggers = []

        for trigger in old_triggers:
            xagent_trigger = {
                "type": trigger.get("type", "manual"),
                "priority": "normal"
            }

            if trigger.get("type") == "schedule":
                xagent_trigger.update({
                    "cron": trigger.get("cron"),
                    "timezone": trigger.get("timezone", "UTC")
                })
            elif trigger.get("type") == "file_added":
                xagent_trigger.update({
                    "path": trigger.get("path"),
                    "patterns": trigger.get("patterns", []),
                    "recursive": trigger.get("recursive", False)
                })
            elif trigger.get("type") == "manual":
                xagent_trigger.update({
                    "command": trigger.get("command", "run")
                })

            converted_triggers.append(xagent_trigger)

        return converted_triggers

    async def _validate_configurations(self):
        """Validate all loaded configurations"""
        logger.info("✅ Validating XAgent configurations")

        validation_errors = []

        # Validate orchestrator config
        if not self.orchestrator_config:
            validation_errors.append("Missing orchestrator configuration")

        # Validate agent configurations
        for agent_id, agent_config in self.agents_config.items():
            errors = await self._validate_agent_config(agent_id, agent_config)
            if errors:
                validation_errors.extend([f"{agent_id}: {error}" for error in errors])

        # Validate workflow configurations
        for workflow_id, workflow_config in self.workflows_config.items():
            errors = await self._validate_workflow_config(workflow_id, workflow_config)
            if errors:
                validation_errors.extend([f"{workflow_id}: {error}" for error in errors])

        if validation_errors:
            logger.error(f"❌ Configuration validation failed: {validation_errors}")
            raise ValueError("Configuration validation failed")
        else:
            logger.info("✅ All configurations validated successfully")

    async def _validate_agent_config(self, agent_id: str, config: Dict) -> List[str]:
        """Validate a single agent configuration"""
        errors = []

        required_fields = ["id", "name", "role", "capabilities"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        # Validate capabilities
        if "capabilities" in config:
            valid_capabilities = [
                "data_analysis", "document_processing", "rag_query",
                "safety_check", "quality_inspection", "predictive_maintenance",
                "process_optimization", "anomaly_detection", "workflow_orchestration",
                "real_time_monitoring", "compliance_checking", "knowledge_synthesis"
            ]

            for capability in config["capabilities"]:
                if capability not in valid_capabilities:
                    errors.append(f"Invalid capability: {capability}")

        return errors

    async def _validate_workflow_config(self, workflow_id: str, config: Dict) -> List[str]:
        """Validate a single workflow configuration"""
        errors = []

        if "steps" not in config:
            errors.append("Missing workflow steps")

        return errors

    def get_agent_config(self, agent_id: str) -> Optional[Dict]:
        """Get configuration for a specific agent"""
        return self.agents_config.get(agent_id)

    def get_all_agent_configs(self) -> Dict[str, Dict]:
        """Get all agent configurations"""
        return self.agents_config

    def get_orchestrator_config(self) -> Dict:
        """Get orchestrator configuration"""
        return self.orchestrator_config

    def get_workflow_config(self, workflow_id: str) -> Optional[Dict]:
        """Get configuration for a specific workflow"""
        return self.workflows_config.get(workflow_id)

    def get_migration_log(self) -> List[Dict]:
        """Get migration log"""
        return self.migration_log

    async def save_configuration(self, config_type: str, config: Dict) -> bool:
        """Save configuration to file"""
        try:
            if config_type == "orchestrator":
                config_file = self.config_dir / "xagent_orchestrator.yaml"
                self.orchestrator_config = config
            elif config_type == "agents":
                config_file = self.config_dir / "xagent_agents.yaml"
                self.agents_config = config
            elif config_type == "workflows":
                config_file = self.config_dir / "xagent_workflows.yaml"
                self.workflows_config = config
            else:
                logger.error(f"Unknown configuration type: {config_type}")
                return False

            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

            logger.info(f"💾 Saved {config_type} configuration to {config_file}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to save {config_type} configuration: {e}")
            return False

    async def reload_configurations(self):
        """Reload all configurations from files"""
        logger.info("🔄 Reloading XAgent configurations")
        self.agents_config.clear()
        self.workflows_config.clear()
        self.orchestrator_config.clear()
        await self.load_configurations()

    def get_system_overview(self) -> Dict[str, Any]:
        """Get system configuration overview"""
        return {
            "orchestrator": {
                "configured": bool(self.orchestrator_config),
                "version": self.orchestrator_config.get("version", "unknown"),
                "max_concurrent_tasks": self.orchestrator_config.get("max_concurrent_tasks", 10)
            },
            "agents": {
                "total": len(self.agents_config),
                "configured_roles": list(set(agent.get("role") for agent in self.agents_config.values())),
                "total_capabilities": len(set(cap for agent in self.agents_config.values() for cap in agent.get("capabilities", []))),
                "migrated": len(self.migration_log)
            },
            "workflows": {
                "total": len(self.workflows_config),
                "configured": len(self.workflows_config) > 0
            },
            "migration": {
                "total_migrated": len(self.migration_log),
                "successful": len([log for log in self.migration_log if log.get("status") == "success"]),
                "failed": len([log for log in self.migration_log if log.get("status") == "failed"])
            }
        }

async def main():
    """Main function to test configuration manager"""
    logger.info("🚀 Testing XAgent Configuration Manager")

    config_manager = XAgentConfigManager()

    # Load configurations
    await config_manager.load_configurations()

    # Get system overview
    overview = config_manager.get_system_overview()
    logger.info(f"📊 System Overview: {json.dumps(overview, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())