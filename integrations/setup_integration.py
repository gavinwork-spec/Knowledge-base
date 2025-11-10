#!/usr/bin/env python3
"""
Manufacturing Knowledge Base - Integration Setup Script
Setup and configure open-source component integrations

This script provides automated setup and configuration for all integration components
including LangChain, LobeChat, XAgent, and LangFuse.
"""

import asyncio
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add integrations directory to path
sys.path.insert(0, str(Path(__file__).parent))

from shared.config import ConfigManager
from shared.base import IntegrationManager

logger = logging.getLogger(__name__)


class IntegrationSetup:
    """Integration setup and configuration manager"""

    def __init__(self, config_path: str, env: str = "production"):
        self.config_path = config_path
        self.env = env
        self.config_manager = None
        self.integration_manager = IntegrationManager()

    async def setup_integrations(self) -> bool:
        """Setup all integrations"""
        try:
            logger.info("Starting integration setup process")

            # Load configuration
            await self._load_configuration()

            # Validate environment
            await self._validate_environment()

            # Setup each integration
            setup_results = {}

            # LangChain Integration
            setup_results["langchain"] = await self._setup_langchain()

            # LobeChat Integration
            setup_results["lobechat"] = await self._setup_lobechat()

            # XAgent Integration
            setup_results["xagent"] = await self._setup_xagent()

            # LangFuse Integration
            setup_results["langfuse"] = await self._setup_langfuse()

            # Initialize integration manager
            await self._initialize_manager()

            # Verify setup
            verification_result = await self._verify_setup()

            if verification_result:
                logger.info("All integrations setup successfully")
                return True
            else:
                logger.error("Integration setup verification failed")
                return False

        except Exception as e:
            logger.error(f"Integration setup failed: {e}")
            return False

    async def _load_configuration(self):
        """Load integration configuration"""
        try:
            self.config_manager = ConfigManager(self.config_path)
            await self.config_manager.load_config()

            # Override with environment-specific configuration
            env_config_path = self.config_path.replace(".yaml", f"_{self.env}.yaml")
            if os.path.exists(env_config_path):
                await self.config_manager.load_config(env_config_path)

            logger.info("Configuration loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    async def _validate_environment(self):
        """Validate environment and dependencies"""
        logger.info("Validating environment")

        # Check required Python packages
        required_packages = [
            "langchain",
            "openai",
            "chromadb",
            "redis",
            "sqlalchemy"
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            logger.error(f"Missing required packages: {missing_packages}")
            print(f"\n❌ Missing packages: {missing_packages}")
            print("Install with: pip install " + " ".join(missing_packages))
            return False

        # Check environment variables
        required_env_vars = [
            "OPENAI_API_KEY",
            "REDIS_PASSWORD",
            "ENCRYPTION_KEY",
            "JWT_SECRET"
        ]

        missing_env_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_env_vars.append(var)

        if missing_env_vars:
            logger.error(f"Missing environment variables: {missing_env_vars}")
            print(f"\n⚠️  Missing environment variables: {missing_env_vars}")
            print("Set these in your environment or .env file")
            return False

        logger.info("Environment validation passed")
        return True

    async def _setup_langchain(self) -> bool:
        """Setup LangChain integration"""
        try:
            logger.info("Setting up LangChain integration")

            config = self.config_manager.get_integration_config("langchain")
            if not config.get("enabled", False):
                logger.info("LangChain integration disabled, skipping setup")
                return True

            # Import LangChain integration
            from langchain import LangChainIntegration

            # Create integration instance
            langchain_integration = LangChainIntegration("langchain", config)

            # Initialize integration
            success = await langchain_integration.initialize()

            if success:
                # Add to integration manager
                self.integration_manager.integrations["langchain"] = langchain_integration
                logger.info("LangChain integration setup completed")
                return True
            else:
                logger.error("LangChain integration initialization failed")
                return False

        except Exception as e:
            logger.error(f"LangChain setup failed: {e}")
            return False

    async def _setup_lobechat(self) -> bool:
        """Setup LobeChat integration"""
        try:
            logger.info("Setting up LobeChat integration")

            config = self.config_manager.get_integration_config("lobechat")
            if not config.get("enabled", False):
                logger.info("LobeChat integration disabled, skipping setup")
                return True

            # Import LobeChat integration
            from lobechat import LobeChatIntegration

            # Create integration instance
            lobechat_integration = LobeChatIntegration("lobechat", config)

            # Initialize integration
            success = await lobechat_integration.initialize()

            if success:
                # Add to integration manager
                self.integration_manager.integrations["lobechat"] = lobechat_integration
                logger.info("LobeChat integration setup completed")
                return True
            else:
                logger.error("LobeChat integration initialization failed")
                return False

        except Exception as e:
            logger.error(f"LobeChat setup failed: {e}")
            return False

    async def _setup_xagent(self) -> bool:
        """Setup XAgent integration"""
        try:
            logger.info("Setting up XAgent integration")

            config = self.config_manager.get_integration_config("xagent")
            if not config.get("enabled", False):
                logger.info("XAgent integration disabled, skipping setup")
                return True

            # Import XAgent integration
            from xagent import XAgentIntegration

            # Create integration instance
            xagent_integration = XAgentIntegration("xagent", config)

            # Initialize integration
            success = await xagent_integration.initialize()

            if success:
                # Add to integration manager
                self.integration_manager.integrations["xagent"] = xagent_integration
                logger.info("XAgent integration setup completed")
                return True
            else:
                logger.error("XAgent integration initialization failed")
                return False

        except Exception as e:
            logger.error(f"XAgent setup failed: {e}")
            return False

    async def _setup_langfuse(self) -> bool:
        """Setup LangFuse integration"""
        try:
            logger.info("Setting up LangFuse integration")

            config = self.config_manager.get_integration_config("langfuse")
            if not config.get("enabled", False):
                logger.info("LangFuse integration disabled, skipping setup")
                return True

            # Import LangFuse integration
            from langfuse import LangFuseIntegration

            # Create integration instance
            langfuse_integration = LangFuseIntegration("langfuse", config)

            # Initialize integration
            success = await langfuse_integration.initialize()

            if success:
                # Add to integration manager
                self.integration_manager.integrations["langfuse"] = langfuse_integration
                logger.info("LangFuse integration setup completed")
                return True
            else:
                logger.error("LangFuse integration initialization failed")
                return False

        except Exception as e:
            logger.error(f"LangFuse setup failed: {e}")
            return False

    async def _initialize_manager(self):
        """Initialize integration manager with loaded integrations"""
        try:
            logger.info("Initializing integration manager")

            # The integration manager should already have integrations added
            # Just perform any final initialization steps

            logger.info("Integration manager initialized")

        except Exception as e:
            logger.error(f"Manager initialization failed: {e}")
            raise

    async def _verify_setup(self) -> bool:
        """Verify all integrations are working correctly"""
        try:
            logger.info("Verifying integration setup")

            # Perform health check on all integrations
            health_results = await self.integration_manager.health_check_all()

            all_healthy = True
            for integration_name, health in health_results.items():
                if health.get("status") == "error":
                    logger.error(f"Integration {integration_name} health check failed: {health.get('error')}")
                    all_healthy = False
                else:
                    logger.info(f"Integration {integration_name} health check passed")

            return all_healthy

        except Exception as e:
            logger.error(f"Setup verification failed: {e}")
            return False

    async def run_health_check(self) -> Dict[str, Any]:
        """Run health check on all integrations"""
        if not self.integration_manager.integrations:
            logger.warning("No integrations initialized")
            return {"status": "no_integrations"}

        return await self.integration_manager.health_check_all()

    async def shutdown(self):
        """Shutdown all integrations"""
        logger.info("Shutting down all integrations")
        await self.integration_manager.shutdown_all()


async def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup Manufacturing Knowledge Base Integrations")
    parser.add_argument(
        "--config",
        default="integrations/config/integrations.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--env",
        choices=["development", "staging", "production"],
        default="production",
        help="Environment to setup"
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run health check after setup"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("integration_setup.log")
        ]
    )

    # Create setup instance
    setup = IntegrationSetup(args.config, args.env)

    try:
        # Run setup
        success = await setup.setup_integrations()

        if success:
            print("\n✅ Integration setup completed successfully!")
            print("\n📋 Setup Summary:")
            setup_status = setup.integration_manager.get_status()
            print(f"   Total Integrations: {setup_status['total_integrations']}")

            for name, info in setup_status['integrations'].items():
                status = info.get('status', 'unknown')
                print(f"   - {name}: {status}")

            # Run health check if requested
            if args.health_check:
                print("\n🏥 Running health checks...")
                health_results = await setup.run_health_check()

                for integration, health in health_results.items():
                    status = health.get('status', 'unknown')
                    if status == "error":
                        print(f"   ❌ {integration}: {health.get('error', 'Unknown error')}")
                    else:
                        print(f"   ✅ {integration}: Healthy")

            print("\n🚀 Your Manufacturing Knowledge Base is now ready with enhanced AI capabilities!")
            print("\n📚 Next Steps:")
            print("1. Test the integrations with sample queries")
            print("2. Configure your environment variables")
            print("3. Start the application servers")
            print("4. Monitor the integration health and performance")

        else:
            print("\n❌ Integration setup failed!")
            print("Check the logs for detailed error information.")
            return 1

    except KeyboardInterrupt:
        print("\n⏹️ Setup interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Setup failed with error: {e}")
        logger.exception("Setup failed")
        return 1
    finally:
        # Cleanup
        await setup.shutdown()

    return 0


if __name__ == "__main__":
    # Run the setup
    exit_code = asyncio.run(main())
    sys.exit(exit_code)