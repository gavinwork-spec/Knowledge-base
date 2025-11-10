"""
YAML Configuration Compatibility Layer
XAgent-inspired compatibility layer that bridges legacy YAML agent configurations
with the new advanced multi-agent orchestration system.
"""

import asyncio
import json
import logging
import yaml
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import os
import re

# Import new agent framework
from multi_agent_orchestrator import BaseAgent, AgentTask, AgentCapability
from multi_agent_system.agents.specialized_agents import EnhancedBaseAgent
from multi_agent_system.agents.coordinator_agent import CoordinatorAgent
from multi_agent_system.agents.trend_predictor_agent import TrendPredictorAgent
from multi_agent_system.agents.customer_insights_agent import CustomerInsightsAgent
from multi_agent_system.agents.specialized_agents import DocumentProcessorAgent
from multi_agent_system.marketplace.agent_marketplace import AgentMarketplace, AgentRegistration, AgentType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigFormat(Enum):
    """Configuration format versions"""
    LEGACY = "legacy"           # Original YAML format
    TRANSITION = "transition"     # Hybrid format for migration
    ENHANCED = "enhanced"        # New enhanced YAML format
    JSON = "json"               # JSON format


@dataclass
class LegacyAgentConfig:
    """Legacy YAML agent configuration structure"""
    name: str
    description: str
    version: str
    author: str
    triggers: Dict[str, Any] = field(default_factory=dict)
    actions: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    tools_allowed: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MigratedAgentConfig:
    """Migrated agent configuration with enhanced features"""
    agent_id: str
    name: str
    description: str
    version: str
    author: str
    agent_type: str
    capabilities: List[str]
    enhanced_features: Dict[str, Any]
    legacy_config: LegacyAgentConfig
    migration_date: datetime
    compatibility_score: float
    recommended_enhancements: List[str]
    migration_path: str


class YAMLCompatibilityLayer:
    """Compatibility layer for YAML agent configurations"""

    def __init__(self, config_path: str = "agent_configs"):
        self.config_path = Path(config_path)
        self.migration_db_path = self.config_path / "migrations.db"
        self.migration_history: List[Dict[str, Any]] = []
        self.migration_rules = self._load_migration_rules()
        self.capability_mapping = self._load_capability_mapping()

        # Initialize database
        self._initialize_migration_db()

        # Load existing configurations
        self.legacy_configs: Dict[str, LegacyAgentConfig] = {}
        self.migrated_configs: Dict[str, MigratedAgentConfig] = {}

    def _initialize_migration_db(self):
        """Initialize migration tracking database"""
        conn = sqlite3.connect(self.migration_db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS migrations (
                migration_id TEXT PRIMARY KEY,
                original_config_path TEXT NOT NULL,
                config_name TEXT NOT NULL,
                original_format TEXT NOT NULL,
                migrated_format TEXT NOT NULL,
                migration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN DEFAULT FALSE,
                error_message TEXT,
                enhancements_applied TEXT,
                compatibility_score REAL,
                metadata TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS capability_mappings (
                legacy_capability TEXT PRIMARY KEY,
                new_capability TEXT NOT NULL,
                agent_type TEXT,
                confidence REAL DEFAULT 0.0,
                custom_mapping TEXT DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def _load_migration_rules(self) -> Dict[str, Any]:
        """Load migration rules for different agent types"""
        return {
            'learning_agent': {
                'target_type': 'DocumentProcessor',
                'capabilities': ['document_processing', 'text_extraction', 'metadata_extraction', 'file_system_monitor'],
                'enhancements': ['multi_modal_processing', 'ocr_support', 'batch_processing'],
                'priority': 'high'
            },
            'trend_analyst': {
                'target_type': 'TrendPredictor',
                'capabilities': ['trend_prediction', 'time_series_forecasting', 'market_analysis', 'anomaly_detection'],
                'enhancements': ['advanced_forecasting', 'seasonal_analysis', 'market_signal_detection'],
                'priority': 'high'
            },
            'price_analyst': {
                'target_type': 'PriceAnalyzer',
                'capabilities': ['price_analysis', 'market_comparison', 'competitor_analysis', 'trend_analysis'],
                'enhancements': ['ml_pricing', 'risk_assessment', 'optimization_recommendations'],
                'priority': 'high'
            },
            'customer_analyst': {
                'target_type': 'CustomerInsights',
                'capabilities': ['customer_segmentation', 'behavioral_analysis', 'churn_prediction', 'lifetime_value_analysis'],
                'enhancements': ['ml_segmentation', 'predictive_insights', 'personalization'],
                'priority': 'medium'
            },
            'coordinator': {
                'target_type': 'Coordinator',
                'capabilities': ['task_decomposition', 'agent_orchestration', 'result_aggregation', 'complex_planning'],
                'enhancements': ['adaptive_execution', 'collaborative_coordination', 'load_balancing'],
                'priority': 'critical'
            },
            'default': {
                'target_type': 'EnhancedBaseAgent',
                'capabilities': ['general_processing', 'task_execution', 'communication'],
                'enhancements': ['enhanced_capabilities', 'performance_monitoring'],
                'priority': 'low'
            }
        }

    def _load_capability_mapping(self) -> Dict[str, str]:
        """Load legacy capability to new capability mapping"""
        return {
            # Document processing capabilities
            'file_scanning': 'document_processing',
            'document_processing': 'document_processing',
            'text_extraction': 'text_extraction',
            'pdf_processing': 'document_processing',
            'excel_processing': 'document_processing',
            'metadata_extraction': 'metadata_extraction',
            'content_analysis': 'text_extraction',
            'indexing': 'document_processing',

            # Data analysis capabilities
            'data_processing': 'data_analysis',
            'statistical_analysis': 'data_analysis',
            'analytics': 'data_analysis',
            'reporting': 'data_analysis',
            'data_mining': 'data_analysis',

            # System operations
            'file_system_access': 'file_system_access',
            'database_access': 'database_access',
            'api_calls': 'communication',
            'system_monitoring': 'performance_monitoring',

            # Learning capabilities
            'automated_learning': 'knowledge_extraction',
            'knowledge_management': 'knowledge_extraction',
            'ml_training': 'model_training',
            'pattern_recognition': 'pattern_detection',

            # Quality control
            'validation': 'quality_control',
            'testing': 'quality_control',
            'verification': 'quality_control',
            'compliance': 'quality_control',

            # Communication
            'notifications': 'communication',
            'email_sending': 'communication',
            'webhook_calls': 'communication',
            'messaging': 'communication'
        }

    async def load_legacy_configurations(self) -> Dict[str, LegacyAgentConfig]:
        """Load legacy YAML configurations"""
        configs = {}

        # Find all YAML files
        yaml_files = list(self.config_path.glob("**/*.yaml")) + list(self.config_path.glob("**/*.yml"))

        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)

                if config_data and 'name' in config_data:
                    # Parse legacy configuration
                    legacy_config = self._parse_legacy_config(config_data)
                    config_key = f"{legacy_config.name}_{legacy_config.version}"
                    configs[config_key] = legacy_config

                    logger.info(f"Loaded legacy config: {config_key}")

            except Exception as e:
                logger.error(f"Error loading YAML file {yaml_file}: {e}")

        self.legacy_configs = configs
        return configs

    def _parse_legacy_config(self, config_data: Dict[str, Any]) -> LegacyAgentConfig:
        """Parse legacy configuration data"""
        return LegacyAgentConfig(
            name=config_data.get('name', ''),
            description=config_data.get('description', ''),
            version=config_data.get('version', '1.0.0'),
            author=config_data.get('author', 'Unknown'),
            triggers=config_data.get('triggers', {}),
            actions=config_data.get('actions', {}),
            config=config_data.get('config', {}),
            tools_allowed=config_data.get('tools_allowed', {}),
            resources=config_data.get('resources', {}),
            metadata=config_data.get('metadata', {})
        )

    async def migrate_all_configs(self, marketplace: AgentMarketplace) -> Dict[str, MigratedAgentConfig]:
        """Migrate all legacy configurations to enhanced format"""
        migrated = {}

        for config_key, legacy_config in self.legacy_configs.items():
            try:
                # Determine target agent type
                target_type = self._determine_target_type(legacy_config)

                # Map capabilities
                capabilities = self._map_capabilities(legacy_config)

                # Generate enhanced features
                enhanced_features = await self._generate_enhanced_features(legacy_config, target_type)

                # Calculate compatibility score
                compatibility_score = self._calculate_compatibility_score(legacy_config)

                # Generate recommendations
                recommendations = self._generate_recommendations(legacy_config, target_type)

                # Create migrated configuration
                agent_id = self._generate_agent_id(legacy_config.name, legacy_config.version)

                migrated_config = MigratedAgentConfig(
                    agent_id=agent_id,
                    name=legacy_config.name,
                    description=legacy_config.description,
                    version=legacy_config.version,
                    author=legacy_config.author,
                    agent_type=target_type,
                    capabilities=capabilities,
                    enhanced_features=enhanced_features,
                    legacy_config=legacy_config,
                    migration_date=datetime.now(),
                    compatibility_score=compatibility_score,
                    recommended_enhancements=recommendations,
                    migration_path="automatic"
                )

                # Register in marketplace
                await self._register_migrated_agent(migrated_config, marketplace)

                migrated[config_key] = migrated_config

                # Record migration
                await self._record_migration(legacy_config, migrated_config, success=True)

                logger.info(f"Successfully migrated config: {config_key} -> {agent_id}")

            except Exception as e:
                logger.error(f"Error migrating config {config_key}: {e}")
                await self._record_migration(legacy_config, None, success=False, error_message=str(e))

        self.migrated_configs = migrated
        return migrated

    def _determine_target_type(self, legacy_config: LegacyAgentConfig) -> str:
        """Determine target agent type for migration"""
        config_name = legacy_config.name.lower()
        config_desc = legacy_config.description.lower()
        config_author = legacy_config.author.lower()

        # Check for specific agent types in config name or description
        if any(keyword in config_name for keyword in ['learn', 'learning', 'train', 'automated']):
            return 'DocumentProcessor'
        elif any(keyword in config_name for keyword in ['trend', 'forecast', 'predict', 'analyze', 'analytics']):
            if any(keyword in config_desc for keyword in ['price', 'cost', 'quote', 'market']):
                return 'PriceAnalyzer'
            else:
                return 'TrendPredictor'
        elif any(keyword in config_name for keyword in ['customer', 'client', 'crm', 'insight', 'segment']):
            return 'CustomerInsights'
        elif any(keyword in config_name for keyword in ['coordinate', 'orchestrator', 'manager', 'coordinator']):
            return 'Coordinator'

        # Check based on actions
        actions = legacy_config.actions.get('primary', []) + legacy_config.actions.get('secondary', [])
        action_texts = ' '.join([str(action) for action in actions]).lower()

        if any(keyword in action_texts for keyword in ['document', 'pdf', 'file', 'process', 'parse']):
            return 'DocumentProcessor'
        elif any(keyword in action_texts for keyword in ['learn', 'update', 'embed', 'index']):
            return 'DocumentProcessor'
        elif any(keyword in action_texts for keyword in ['trend', 'analysis', 'forecast', 'predict']):
            if any(keyword in action_texts for keyword in ['price', 'cost', 'market']):
                return 'PriceAnalyzer'
            else:
                return 'TrendPredictor'
        elif any(keyword in action_texts for keyword in ['customer', 'segment', 'insight']):
            return 'CustomerInsights'

        # Check based on tools
        tools = legacy_config.tools_allowed.get('FileSystem', [])
        tool_texts = ' '.join(tools).lower()

        if 'document' in tool_texts or 'pdf' in tool_texts:
            return 'DocumentProcessor'

        # Use migration rules
        rule_key = config_name.replace(' ', '_').lower()
        if rule_key in self.migration_rules:
            return self.migration_rules[rule_key]['target_type']

        return 'EnhancedBaseAgent'

    def _map_capabilities(self, legacy_config: LegacyAgentConfig) -> List[str]:
        """Map legacy capabilities to new capability system"""
        capabilities = set()

        # Map based on tools allowed
        for tool_class, tools in legacy_config.tools_allowed.items():
            for tool in tools:
                if tool in self.capability_mapping:
                    capabilities.add(self.capability_mapping[tool])

        # Map based on actions
        actions = legacy_config.actions.get('primary', []) + legacy_config.actions.get('secondary', [])
        for action in actions:
            if isinstance(action, dict):
                command = action.get('command', '')
            else:
                command = str(action)

            # Extract capabilities from command
            if 'document' in command.lower() or 'pdf' in command.lower() or 'process' in command.lower():
                capabilities.update(['document_processing', 'text_extraction'])
            if 'learning' in command.lower() or 'train' in command.lower():
                capabilities.update(['knowledge_extraction', 'ml_training'])
            if 'analyze' in command.lower() or 'report' in command.lower():
                capabilities.add('data_analysis')
            if 'database' in command.lower() or 'sql' in command.lower():
                capabilities.add('database_access')

        # Map based on config
        if legacy_config.config.get('learning', {}).get('auto_rebuild_embeddings', False):
            capabilities.add('knowledge_extraction')

        if 'search' in str(legacy_config.actions).lower():
            capabilities.add('search')

        return list(capabilities)

    async def _generate_enhanced_features(self, legacy_config: LegacyAgentConfig, target_type: str) -> Dict[str, Any]:
        """Generate enhanced features for migrated agent"""
        enhanced_features = {
            'migration_version': '1.0',
            'original_config_hash': self._calculate_config_hash(legacy_config),
            'enhancement_applied': True,
            'legacy_compatibility': True
        }

        # Add type-specific enhancements
        if target_type == 'DocumentProcessor':
            enhanced_features.update({
                'multi_modal_support': True,
                'ocr_capability': True,
                'batch_processing': True,
                'format_detection': True,
                'quality_validation': True
            })
        elif target_type == 'PriceAnalyzer':
            enhanced_features.update({
                'ml_pricing_models': True,
                'market_intelligence': True,
                'risk_assessment': True,
                'competitor_tracking': True,
                'price_optimization': True
            })
        elif target_type == 'TrendPredictor':
            enhanced_features.update({
                'time_series_forecasting': True,
                'seasonal_analysis': True,
                'anomaly_detection': True,
                'market_signal_detection': True,
                'predictive_modeling': True
            })
        elif target_type == 'CustomerInsights':
            enhanced_features.update({
                'behavioral_segmentation': True,
                'predictive_analytics': True,
                'personalization': True,
                'lifetime_value_calculation': True,
                'churn_prediction': True
            })
        elif target_type == 'Coordinator':
            enhanced_features.update({
                'adaptive_execution': True,
                'collaborative_coordination': True,
                'load_balancing': True,
                'resource_optimization': True,
                'failure_recovery': True
            })

        # Preserve legacy triggers as enhanced features
        if legacy_config.triggers:
            enhanced_features['legacy_triggers'] = legacy_config.triggers
            enhanced_features['trigger_compatibility'] = True

        # Preserve legacy actions
        if legacy_config.actions:
            enhanced_features['legacy_actions'] = legacy_config.actions
            enhanced_features['action_compatibility'] = True

        return enhanced_features

    def _calculate_compatibility_score(self, legacy_config: LegacyAgentConfig) -> float:
        """Calculate compatibility score between legacy and enhanced format"""
        score = 0.0

        # Base score for having basic structure
        if legacy_config.name and legacy_config.version:
            score += 0.2

        # Score for having triggers
        if legacy_config.triggers:
            score += 0.2

        # Score for having actions
        if legacy_config.actions:
            score += 0.2

        # Score for having proper configuration
        if legacy_config.config:
            score += 0.1

        # Score for having tools configuration
        if legacy_config.tools_allowed:
            score += 0.1

        # Score for having resource limits
        if legacy_config.resources:
            score += 0.1

        # Score for having metadata
        if legacy_config.metadata:
            score += 0.1

        return min(score, 1.0)

    def _generate_recommendations(self, legacy_config: LegacyAgentConfig, target_type: str) -> List[str]:
        """Generate recommendations for enhanced agent improvements"""
        recommendations = []

        # General recommendations
        recommendations.append("Consider adding performance monitoring for better observability")
        recommendations.append("Implement proper error handling and retry mechanisms")
        recommendations.append("Add comprehensive logging and debugging capabilities")

        # Type-specific recommendations
        if target_type == 'DocumentProcessor':
            recommendations.extend([
                "Add support for additional document formats (images, presentations)",
                "Implement advanced OCR and text extraction capabilities",
                "Add document classification and metadata extraction",
                "Implement batch processing for improved efficiency"
            ])
        elif target_type == 'PriceAnalyzer':
            recommendations.extend([
                "Implement advanced machine learning models for price prediction",
                "Add real-time market data integration",
                "Implement competitor analysis and comparison features",
                "Add price optimization and recommendation capabilities"
            ])
        elif target_type == 'TrendPredictor':
            recommendations.extend([
                "Implement advanced time series forecasting models",
                "Add seasonal decomposition and analysis",
                "Implement anomaly detection for early warning",
                "Add market signal identification and analysis"
            ])
        elif target_type == 'CustomerInsights':
            recommendations.extend([
                "Implement advanced customer segmentation algorithms",
                "Add predictive analytics for customer behavior",
                "Implement personalized recommendation systems",
                "Add customer lifetime value calculation"
            ])
        elif target_type == 'Coordinator':
            recommendations.extend([
                "Implement advanced task decomposition algorithms",
                "Add collaborative coordination capabilities",
                "Implement intelligent load balancing",
                "Add failure recovery and resilience features"
            ])

        # Legacy-specific recommendations
        if not legacy_config.config.get('database', {}).get('connection_timeout'):
            recommendations.append("Configure database connection timeout for better reliability")

        if not legacy_config.resources.get('max_cpu_time'):
            recommendations.append("Set reasonable resource limits to prevent system overload")

        return recommendations

    def _generate_agent_id(self, name: str, version: str) -> str:
        """Generate unique agent ID from name and version"""
        # Clean up the name
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())
        clean_version = re.sub(r'[^a-zA-Z0-9_]', '_', version.lower())

        # Combine name and version
        return f"{clean_name}_{clean_version}_{int(datetime.now().timestamp())}"

    def _calculate_config_hash(self, config: LegacyAgentConfig) -> str:
        """Calculate hash for configuration integrity"""
        config_str = json.dumps(asdict(config), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    async def _register_migrated_agent(self, migrated_config: MigratedConfig, marketplace: AgentMarketplace):
        """Register migrated agent in marketplace"""
        try:
            # Create agent registration
            registration = AgentRegistration(
                agent_id=migrated_config.agent_id,
                name=migrated_config.name,
                description=f"Migrated from legacy config: {migrated_config.legacy_config.name}",
                agent_type=AgentType(migrated_config.agent_type.lower()),
                version=migrated_config.version,
                author=f"Legacy Agent - {migrated_config.author}",
                contact_info={
                    'email': f"legacy.{migrated_config.author.lower().replace(' ', '.replace()]}@example.com",
                    'system': 'legacy_migration'
                },
                capabilities=[],
                resource_limits=migrated_config.legacy_config.resources,
                metadata={
                    'migration_info': {
                        'original_config': asdict(migrated_config.legacy_config),
                        'migration_date': migrated_config.migration_date.isoformat(),
                        'compatibility_score': migrated_config.compatibility_score,
                        'enhancements_applied': len(migrated_config.recommended_enhancements)
                    }
                },
                status=AgentStatus.REGISTERED
            )

            # Convert string capabilities to AgentCapability objects
            for cap_name in migrated_config.capabilities:
                try:
                    capability = AgentCapability(cap_name)
                    registration.capabilities.append(capability)
                except:
                    logger.warning(f"Unknown capability: {cap_name}")

            success = marketplace.registry.register_agent(registration)
            return success

        except Exception as e:
            logger.error(f"Error registering migrated agent {migrated_config.agent_id}: {e}")
            return False

    async def _record_migration(self, legacy_config: LegacyAgentConfig,
                               migrated_config: Optional[MigratedConfig],
                               success: bool,
                               error_message: str = None):
        """Record migration in database"""
        try:
            conn = sqlite3.connect(self.migration_db_path)
            cursor = conn.cursor()

            migration_id = str(uuid.uuid4())

            enhancements_applied = json.dumps(migrated_config.recommended_enhancements) if migrated_config else []

            cursor.execute('''
                INSERT INTO migrations
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                migration_id,
                str(legacy_config.name),
                f"{legacy_config.name}_{legacy_config.version}",
                ConfigFormat.LEGACY.value,
                ConfigFormat.ENHANCED.value,
                datetime.now().isoformat(),
                success,
                error_message,
                enhancements_applied,
                migrated_config.compatibility_score if migrated_config else 0.0,
                json.dumps({'migration_metadata': 'legacy_to_enhanced_v1'})
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error recording migration: {e}")

    def get_migration_summary(self) -> Dict[str, Any]:
        """Get migration summary and statistics"""
        try:
            conn = sqlite3.connect(self.migration_db_path)
            cursor = conn.cursor()

            # Get migration statistics
            cursor.execute('''
                SELECT COUNT(*) as total_migrations,
                       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_migrations,
                       AVG(compatibility_score) as avg_compatibility
                FROM migrations
            ''')

            stats = cursor.fetchone()

            # Get migration details
            cursor.execute('''
                SELECT original_config_path, original_format, migrated_format,
                       migration_date, success, error_message
                FROM migrations
                ORDER BY migration_date DESC
                LIMIT 10
            ''')

            recent_migrations = cursor.fetchall()

            # Get capability mappings
            cursor.execute('SELECT COUNT(*) as total_mappings FROM capability_mappings')
            mapping_count = cursor.fetchone()[0]

            conn.close()

            return {
                'total_legacy_configs': len(self.legacy_configs),
                'migrated_configs': len(self.migrated_configs),
                'database_migrations': {
                    'total': stats[0] if stats[0] else 0,
                    'successful': stats[1] if stats[1] else 0,
                    'average_compatibility': stats[2] if stats[2] else 0.0
                },
                'capability_mappings': mapping_count,
                'recent_migrations': recent_migrations,
                'migration_status': 'completed' if stats[0] == stats[1] else 'in_progress',
                'success_rate': (stats[1] / stats[0] * 100) if stats[0] > 0 else 0
            }

        except Exception as e:
            logger.error(f"Error getting migration summary: {e}")
            return {'error': str(e)}

    def validate_configuration(self, config_path: str) -> Dict[str, Any]:
        """Validate YAML configuration for compatibility"""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            validation_result = {
                'valid': True,
                'issues': [],
                'warnings': [],
                'recommendations': []
            }

            # Check required fields
            required_fields = ['name', 'description', 'version', 'author']
            for field in required_fields:
                if not config_data.get(field):
                    validation_result['issues'].append(f"Missing required field: {field}")
                    validation_result['valid'] = False

            # Validate triggers
            if 'triggers' in config_data:
                triggers = config_data['triggers']
                if isinstance(triggers, dict):
                    for trigger_type, trigger_config in triggers.items():
                        if trigger_type == 'scheduled' and 'cron' in trigger_config:
                            # Basic cron validation
                            cron_expr = trigger_config['cron']
                            if not self._validate_cron_expression(cron_expr):
                                validation_result['issues'].append(f"Invalid cron expression: {cron_expr}")
                                validation_result['valid'] = False

            # Validate actions
            if 'actions' in config_data:
                actions = config_data['actions']
                if isinstance(actions, dict):
                    for action_type, action_list in actions.items():
                        if isinstance(action_list, list):
                            for action in action_list:
                                if isinstance(action, dict) and 'command' in action:
                                    command = action['command']
                                    if not self._validate_command(command):
                                        validation_result['issues'].append(f"Invalid command: {command}")
                                        validation_result['warnings'].append(f"Command may need manual verification: {command}")

            # Generate recommendations
            if validation_result['valid']:
                if not config_data.get('config', {}).get('database', {}).get('connection_timeout'):
                    validation_result['recommendations'].append("Consider adding database connection timeout for better reliability")

                if not config_data.get('resources', {}).get('max_cpu_time'):
                    validation_result['recommendations'].append("Consider setting CPU time limits to prevent resource exhaustion")

            return validation_result

        except Exception as e:
            return {
                'valid': False,
                'issues': [f"Error reading configuration: {e}"],
                'warnings': [],
                'recommendations': []
            }

    def _validate_cron_expression(self, cron_expr: str) -> bool:
        """Basic cron expression validation"""
        # Very basic validation - in production, use a proper cron library
        parts = cron_expr.split()
        if len(parts) != 5:
            return False

        # Check each part for basic validity
        try:
            # Check if parts contain valid patterns
            return all(
                part.replace('*', '').isdigit() or
                part.replace(',', '').replace('-', '/').isdigit() or
                part in ['*', '?']
                for part in parts
            )
        except:
            return False

    def _validate_command(self, command: str) -> bool:
        """Basic command validation"""
        # Very basic validation - in production, implement comprehensive security checks
        if not command:
            return False

        # Check for dangerous patterns
        dangerous_patterns = [
            'rm -rf /',
            'sudo rm',
            'format',
            'fdisk',
            'mkfs'
        ]

        return not any(pattern in command for pattern in dangerous_patterns)


# Factory function
def create_yaml_compatibility_layer(config_path: str = "agent_configs") -> YAMLCompatibilityLayer:
    """Create a YAML compatibility layer"""
    return YAMLCompatibilityLayer(config_path)


# Usage example
if __name__ == "__main__":
    async def test_yaml_compatibility():
        # Create compatibility layer
        compatibility = create_yaml_compatibility_layer()

        # Load legacy configurations
        configs = await compatibility.load_legacy_configurations()
        print(f"Loaded {len(configs)} legacy configurations")

        # Create marketplace for registration
        marketplace = create_agent_marketplace()
        await marketplace.start()

        # Migrate all configurations
        migrated = await compatibility.migrate_all_configs(marketplace)
        print(f"Migrated {len(migrated)} configurations")

        # Get migration summary
        summary = compatibility.get_migration_summary()
        print(f"Migration summary: {summary}")

        await marketplace.stop()

    asyncio.run(test_yaml_compatibility())