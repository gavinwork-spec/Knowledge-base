#!/usr/bin/env python3
"""
微服务部署脚本
用于部署和管理知识库微服务架构
"""

import asyncio
import logging
import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import argparse

import docker
from docker.types import ServiceMode
import aiohttp
import asyncpg
import redis.asyncio as redis

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MicroservicesDeployer:
    """微服务部署器"""

    def __init__(self, project_dir: str = "."):
        self.project_dir = Path(project_dir).resolve()
        self.docker_client = None
        self.compose_file = self.project_dir / "docker-compose.yml"
        self.services = [
            "redis", "postgres", "document-service", "search-service",
            "agent-service", "notify-service", "user-service", "api-gateway"
        ]
        self.monitoring_services = ["prometheus", "grafana", "elasticsearch", "kibana"]
        self.storage_services = ["minio"]

    async def initialize(self):
        """初始化部署器"""
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
            logger.info("Connected to Docker daemon")
        except Exception as e:
            logger.error(f"Failed to connect to Docker: {e}")
            raise

    async def deploy_infrastructure(self):
        """部署基础设施（Redis和PostgreSQL）"""
        logger.info("Starting infrastructure deployment...")

        # 启动Redis和PostgreSQL
        cmd = [
            "docker-compose", "-f", str(self.compose_file),
            "up", "-d", "redis", "postgres"
        ]

        try:
            process = subprocess.run(
                cmd,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Infrastructure started successfully")
            logger.info(process.stdout)

            # 等待服务就绪
            await self._wait_for_infrastructure()

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start infrastructure: {e}")
            logger.error(e.stderr)
            raise

    async def deploy_services(self):
        """部署微服务"""
        logger.info("Starting microservices deployment...")

        # 启动核心服务
        cmd = [
            "docker-compose", "-f", str(self.compose_file),
            "up", "-d"
        ] + self.services

        try:
            process = subprocess.run(
                cmd,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Microservices started successfully")
            logger.info(process.stdout)

            # 等待服务就绪
            await self._wait_for_services()

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start microservices: {e}")
            logger.error(e.stderr)
            raise

    async def deploy_monitoring(self):
        """部署监控服务"""
        logger.info("Starting monitoring services...")

        cmd = [
            "docker-compose", "-f", str(self.compose_file),
            "up", "-d"
        ] + self.monitoring_services

        try:
            process = subprocess.run(
                cmd,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Monitoring services started successfully")
            logger.info(process.stdout)

        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to start monitoring services: {e}")
            logger.warning(e.stderr)

    async def run_migration(self):
        """运行数据库迁移"""
        logger.info("Running database migration...")

        cmd = [
            "docker-compose", "-f", str(self.compose_file),
            "--profile", "migration", "run", "--rm", "migration",
            "python", "microservices/database_migration.py"
        ]

        try:
            process = subprocess.run(
                cmd,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 5分钟超时
            )
            logger.info("Database migration completed successfully")
            logger.info(process.stdout)

        except subprocess.CalledProcessError as e:
            logger.error(f"Database migration failed: {e}")
            logger.error(e.stderr)
            raise
        except subprocess.TimeoutExpired:
            logger.error("Database migration timed out")
            raise

    async def _wait_for_infrastructure(self):
        """等待基础设施就绪"""
        logger.info("Waiting for infrastructure to be ready...")

        # 等待Redis
        redis_ready = False
        for i in range(30):  # 30秒超时
            try:
                redis_client = redis.from_url("redis://localhost:6379")
                await redis_client.ping()
                redis_ready = True
                break
            except:
                await asyncio.sleep(1)

        if not redis_ready:
            raise Exception("Redis failed to start")
        logger.info("Redis is ready")

        # 等待PostgreSQL
        postgres_ready = False
        for i in range(30):  # 30秒超时
            try:
                conn = await asyncpg.connect(
                    "postgresql://postgres:postgres@localhost:5432/knowledge_base"
                )
                await conn.close()
                postgres_ready = True
                break
            except:
                await asyncio.sleep(1)

        if not postgres_ready:
            raise Exception("PostgreSQL failed to start")
        logger.info("PostgreSQL is ready")

    async def _wait_for_services(self):
        """等待微服务就绪"""
        logger.info("Waiting for microservices to be ready...")

        service_ports = {
            "document-service": 8003,
            "search-service": 8004,
            "agent-service": 8005,
            "notify-service": 8006,
            "user-service": 8007,
            "api-gateway": 8000
        }

        for service_name, port in service_ports.items():
            await self._wait_for_http_service(f"http://localhost:{port}/health", service_name)

    async def _wait_for_http_service(self, url: str, service_name: str, timeout: int = 60):
        """等待HTTP服务就绪"""
        logger.info(f"Waiting for {service_name} at {url}...")

        async with aiohttp.ClientSession() as session:
            for i in range(timeout):
                try:
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            logger.info(f"{service_name} is ready")
                            return True
                except:
                    pass

                await asyncio.sleep(1)

            raise Exception(f"{service_name} failed to start within {timeout} seconds")

    async def check_health(self):
        """检查所有服务健康状态"""
        logger.info("Checking service health...")

        service_ports = {
            "redis": 6379,
            "postgres": 5432,
            "document-service": 8003,
            "search-service": 8004,
            "agent-service": 8005,
            "notify-service": 8006,
            "user-service": 8007,
            "api-gateway": 8000
        }

        health_status = {}

        # 检查Redis
        try:
            redis_client = redis.from_url("redis://localhost:6379")
            await redis_client.ping()
            health_status["redis"] = "healthy"
        except Exception as e:
            health_status["redis"] = f"unhealthy: {e}"

        # 检查PostgreSQL
        try:
            conn = await asyncpg.connect(
                "postgresql://postgres:postgres@localhost:5432/knowledge_base"
            )
            await conn.close()
            health_status["postgres"] = "healthy"
        except Exception as e:
            health_status["postgres"] = f"unhealthy: {e}"

        # 检查HTTP服务
        async with aiohttp.ClientSession() as session:
            for service_name, port in service_ports.items():
                if service_name in ["redis", "postgres"]:
                    continue

                try:
                    async with session.get(f"http://localhost:{port}/health", timeout=5) as response:
                        if response.status == 200:
                            health_status[service_name] = "healthy"
                        else:
                            health_status[service_name] = f"unhealthy: HTTP {response.status}"
                except Exception as e:
                    health_status[service_name] = f"unhealthy: {e}"

        # 输出健康状态
        logger.info("Health Status Report:")
        for service, status in health_status.items():
            logger.info(f"  {service}: {status}")

        return health_status

    async def get_service_logs(self, service_name: str, lines: int = 50):
        """获取服务日志"""
        try:
            cmd = [
                "docker-compose", "-f", str(self.compose_file),
                "logs", "--tail", str(lines), service_name
            ]

            process = subprocess.run(
                cmd,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                check=True
            )

            return process.stdout

        except subprocess.CalledProcessError as e:
            return f"Failed to get logs for {service_name}: {e}"

    async def stop_services(self):
        """停止所有服务"""
        logger.info("Stopping all services...")

        cmd = [
            "docker-compose", "-f", str(self.compose_file),
            "down"
        ]

        try:
            process = subprocess.run(
                cmd,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("All services stopped successfully")
            logger.info(process.stdout)

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stop services: {e}")
            logger.error(e.stderr)

    async def restart_services(self, services: Optional[List[str]] = None):
        """重启指定服务"""
        if not services:
            services = self.services

        logger.info(f"Restarting services: {', '.join(services)}")

        cmd = [
            "docker-compose", "-f", str(self.compose_file),
            "restart"
        ] + services

        try:
            process = subprocess.run(
                cmd,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Services restarted successfully")
            logger.info(process.stdout)

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to restart services: {e}")
            logger.error(e.stderr)

    async def show_service_status(self):
        """显示服务状态"""
        logger.info("Getting service status...")

        cmd = [
            "docker-compose", "-f", str(self.compose_file),
            "ps"
        ]

        try:
            process = subprocess.run(
                cmd,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Service Status:")
            logger.info(process.stdout)

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get service status: {e}")
            logger.error(e.stderr)

    async def run_full_deployment(self):
        """执行完整部署流程"""
        logger.info("Starting full deployment process...")

        try:
            # 1. 部署基础设施
            await self.deploy_infrastructure()

            # 2. 运行数据库迁移
            await self.run_migration()

            # 3. 部署微服务
            await self.deploy_services()

            # 4. 部署监控服务（可选）
            await self.deploy_monitoring()

            # 5. 检查健康状态
            await self.check_health()

            logger.info("Full deployment completed successfully!")

            # 显示访问信息
            await self.show_access_info()

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise

    async def show_access_info(self):
        """显示服务访问信息"""
        logger.info("\n" + "="*60)
        logger.info("服务访问信息")
        logger.info("="*60)

        access_info = {
            "API 网关": "http://localhost:8000",
            "文档服务": "http://localhost:8003",
            "搜索服务": "http://localhost:8004",
            "Agent 服务": "http://localhost:8005",
            "通知服务": "http://localhost:8006",
            "用户服务": "http://localhost:8007",
            "前端应用": "http://localhost:3000",
            "Redis": "localhost:6379",
            "PostgreSQL": "localhost:5432",
            "Prometheus": "http://localhost:9090",
            "Grafana": "http://localhost:3001 (admin/admin)",
            "Kibana": "http://localhost:5601",
            "MinIO": "http://localhost:9000 (minioadmin/minioadmin123)"
        }

        for service, url in access_info.items():
            logger.info(f"  {service:12} : {url}")

        logger.info("\n健康检查端点:")
        for service in ["document-service", "search-service", "agent-service", "notify-service", "user-service", "api-gateway"]:
            port = {
                "document-service": 8003,
                "search-service": 8004,
                "agent-service": 8005,
                "notify-service": 8006,
                "user-service": 8007,
                "api-gateway": 8000
            }[service]
            logger.info(f"  {service}: http://localhost:{port}/health")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="知识库微服务部署工具")
    parser.add_argument(
        "action",
        choices=[
            "deploy", "deploy-infra", "deploy-services", "deploy-monitoring",
            "migrate", "health", "status", "logs", "stop", "restart", "full"
        ],
        help="要执行的操作"
    )
    parser.add_argument(
        "--service",
        help="指定服务名称（用于logs和restart操作）"
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=50,
        help="日志行数（默认50）"
    )
    parser.add_argument(
        "--project-dir",
        default=".",
        help="项目目录路径（默认当前目录）"
    )

    args = parser.parse_args()

    deployer = MicroservicesDeployer(args.project_dir)
    await deployer.initialize()

    try:
        if args.action == "deploy-infra":
            await deployer.deploy_infrastructure()
        elif args.action == "deploy-services":
            await deployer.deploy_services()
        elif args.action == "deploy-monitoring":
            await deployer.deploy_monitoring()
        elif args.action == "migrate":
            await deployer.run_migration()
        elif args.action == "health":
            await deployer.check_health()
        elif args.action == "status":
            await deployer.show_service_status()
        elif args.action == "logs":
            if not args.service:
                logger.error("请指定服务名称: --service <service-name>")
                return
            logs = await deployer.get_service_logs(args.service, args.lines)
            print(logs)
        elif args.action == "stop":
            await deployer.stop_services()
        elif args.action == "restart":
            services = [args.service] if args.service else None
            await deployer.restart_services(services)
        elif args.action == "full":
            await deployer.run_full_deployment()

    except KeyboardInterrupt:
        logger.info("操作被用户中断")
    except Exception as e:
        logger.error(f"操作失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())