#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Interaction Logger
AI交互日志记录器

Comprehensive logging system for all AI interactions, agent executions,
and RAG operations with detailed manufacturing context tracking.
"""

import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import time
import traceback
import functools

from .langfuse_integration import (
    LangFuseIntegration,
    InteractionType,
    AgentType,
    PerformanceMetrics,
    CostMetrics,
    ManufacturingMetrics,
    TraceData,
    get_langfuse_integration
)

# Configure logging
logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ModelProvider(Enum):
    """模型提供商"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    AZURE = "azure"
    GOOGLE = "google"

@dataclass
class AIInteraction:
    """AI交互数据"""
    interaction_id: str
    session_id: str
    user_id: Optional[str]
    agent_type: AgentType
    model_provider: ModelProvider
    model_name: str
    prompt: str
    response: str
    context: Dict[str, Any]
    performance_metrics: PerformanceMetrics
    cost_metrics: Optional[CostMetrics]
    manufacturing_metrics: Optional[ManufacturingMetrics]
    timestamp: datetime
    success: bool
    error_info: Optional[Dict[str, Any]] = None

@dataclass
class AgentExecution:
    """代理执行数据"""
    execution_id: str
    agent_type: AgentType
    session_id: str
    user_id: Optional[str]
    task_description: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    execution_steps: List[Dict[str, Any]]
    performance_metrics: PerformanceMetrics
    manufacturing_context: Dict[str, Any]
    timestamp: datetime
    success: bool
    error_info: Optional[Dict[str, Any]] = None

@dataclass
class RAGInteraction:
    """RAG交互数据"""
    interaction_id: str
    session_id: str
    user_id: Optional[str]
    query: str
    retrieved_documents: List[Dict[str, Any]]
    generated_response: str
    retrieval_strategy: str
    performance_metrics: PerformanceMetrics
    manufacturing_entities: List[str]
    citation_count: int
    timestamp: datetime
    success: bool

class AIInteractionLogger:
    """AI交互日志记录器"""

    def __init__(self,
                 langfuse_integration: Optional[LangFuseIntegration] = None,
                 enable_detailed_logging: bool = True,
                 log_level: LogLevel = LogLevel.INFO):
        """
        初始化AI交互日志记录器

        Args:
            langfuse_integration: LangFuse集成实例
            enable_detailed_logging: 是否启用详细日志
            log_level: 日志级别
        """
        self.langfuse_integration = langfuse_integration or get_langfuse_integration()
        self.enable_detailed_logging = enable_detailed_logging
        self.log_level = log_level

        # 设置日志记录器
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(getattr(logging, log_level.value))

        # 成本计算配置
        self.cost_config = self._init_cost_config()

    def _init_cost_config(self) -> Dict[str, Dict[str, float]]:
        """初始化成本配置"""
        return {
            "openai": {
                "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
                "gpt-4-turbo": {"input": 0.01, "output": 0.03},
                "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
                "text-embedding-3-small": {"input": 0.00002, "output": 0}
            },
            "anthropic": {
                "claude-3-opus": {"input": 0.015, "output": 0.075},
                "claude-3-sonnet": {"input": 0.003, "output": 0.015},
                "claude-3-haiku": {"input": 0.00025, "output": 0.00125}
            }
        }

    def log_ai_interaction(self, interaction: AIInteraction):
        """记录AI交互"""
        try:
            if self.enable_detailed_logging:
                self.logger.info(f"AI Interaction - {interaction.agent_type.value}: {interaction.prompt[:100]}...")

            # 创建跟踪数据
            trace_data = TraceData(
                trace_id=interaction.interaction_id,
                session_id=interaction.session_id,
                user_id=interaction.user_id,
                interaction_type=InteractionType.QUERY,
                agent_type=interaction.agent_type,
                timestamp=interaction.timestamp,
                input_data={
                    "prompt": interaction.prompt,
                    "context": interaction.context,
                    "model_provider": interaction.model_provider.value,
                    "model_name": interaction.model_name
                },
                output_data={
                    "response": interaction.response,
                    "success": interaction.success
                },
                performance_metrics=interaction.performance_metrics,
                cost_metrics=interaction.cost_metrics,
                manufacturing_metrics=interaction.manufacturing_metrics,
                tags=["ai_interaction", interaction.agent_type.value, interaction.model_provider.value],
                metadata={
                    "success": interaction.success,
                    "error_info": interaction.error_info
                }
            )

            # 异步保存跟踪数据
            asyncio.create_task(self.langfuse_integration.create_trace(trace_data))

        except Exception as e:
            self.logger.error(f"Failed to log AI interaction: {e}")

    def log_agent_execution(self, execution: AgentExecution):
        """记录代理执行"""
        try:
            if self.enable_detailed_logging:
                self.logger.info(f"Agent Execution - {execution.agent_type.value}: {execution.task_description}")

            trace_data = TraceData(
                trace_id=execution.execution_id,
                session_id=execution.session_id,
                user_id=execution.user_id,
                interaction_type=InteractionType.AGENT_EXECUTION,
                agent_type=execution.agent_type,
                timestamp=execution.timestamp,
                input_data={
                    "task_description": execution.task_description,
                    "input_data": execution.input_data,
                    "manufacturing_context": execution.manufacturing_context
                },
                output_data={
                    "output_data": execution.output_data,
                    "execution_steps": execution.execution_steps,
                    "success": execution.success
                },
                performance_metrics=execution.performance_metrics,
                tags=["agent_execution", execution.agent_type.value],
                metadata={
                    "success": execution.success,
                    "error_info": execution.error_info,
                    "execution_steps_count": len(execution.execution_steps)
                }
            )

            asyncio.create_task(self.langfuse_integration.create_trace(trace_data))

        except Exception as e:
            self.logger.error(f"Failed to log agent execution: {e}")

    def log_rag_interaction(self, rag_interaction: RAGInteraction):
        """记录RAG交互"""
        try:
            if self.enable_detailed_logging:
                self.logger.info(f"RAG Interaction: {rag_interaction.query[:100]}... (Documents: {len(rag_interaction.retrieved_documents)})")

            trace_data = TraceData(
                trace_id=rag_interaction.interaction_id,
                session_id=rag_interaction.session_id,
                user_id=rag_interaction.user_id,
                interaction_type=InteractionType.RAG_RETRIEVAL,
                agent_type=AgentType.KNOWLEDGE_RETRIEVER,
                timestamp=rag_interaction.timestamp,
                input_data={
                    "query": rag_interaction.query,
                    "retrieval_strategy": rag_interaction.retrieval_strategy,
                    "manufacturing_entities": rag_interaction.manufacturing_entities
                },
                output_data={
                    "response": rag_interaction.generated_response,
                    "retrieved_documents_count": len(rag_interaction.retrieved_documents),
                    "citation_count": rag_interaction.citation_count,
                    "success": rag_interaction.success
                },
                performance_metrics=rag_interaction.performance_metrics,
                tags=["rag_retrieval", rag_interaction.retrieval_strategy],
                metadata={
                    "retrieved_documents": rag_interaction.retrieved_documents[:3],  # 只保存前3个文档
                    "success": rag_interaction.success,
                    "manufacturing_entities_count": len(rag_interaction.manufacturing_entities)
                }
            )

            asyncio.create_task(self.langfuse_integration.create_trace(trace_data))

        except Exception as e:
            self.logger.error(f"Failed to log RAG interaction: {e}")

    def calculate_cost(self,
                      model_provider: ModelProvider,
                      model_name: str,
                      prompt_tokens: int,
                      completion_tokens: int) -> CostMetrics:
        """计算API调用成本"""
        try:
            provider_config = self.cost_config.get(model_provider.value, {})
            model_config = provider_config.get(model_name, {"input": 0.0, "output": 0.0})

            input_cost = (prompt_tokens / 1000) * model_config["input"]
            output_cost = (completion_tokens / 1000) * model_config["output"]
            total_cost = input_cost + output_cost

            total_tokens = prompt_tokens + completion_tokens
            cost_per_token = total_cost / total_tokens if total_tokens > 0 else 0.0

            return CostMetrics(
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost,
                cost_per_token=cost_per_token,
                api_call_count=1
            )

        except Exception as e:
            self.logger.warning(f"Failed to calculate cost: {e}")
            return CostMetrics(
                input_cost=0.0,
                output_cost=0.0,
                total_cost=0.0,
                api_call_count=1
            )

    def create_interaction_decorator(self,
                                   agent_type: AgentType,
                                   model_provider: ModelProvider,
                                   model_name: str):
        """创建交互装饰器"""
        def decorator(func: Callable):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._log_interaction_async(
                    func, agent_type, model_provider, model_name, *args, **kwargs
                )

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self._log_interaction_sync(
                    func, agent_type, model_provider, model_name, *args, **kwargs
                )

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    async def _log_interaction_async(self,
                                   func: Callable,
                                   agent_type: AgentType,
                                   model_provider: ModelProvider,
                                   model_name: str,
                                   *args, **kwargs):
        """异步函数交互日志"""
        start_time = time.time()
        interaction_id = str(uuid.uuid4())

        # 提取会话和用户信息
        session_id = kwargs.get("session_id", "unknown")
        user_id = kwargs.get("user_id")
        prompt = str(args[0]) if args else str(kwargs.get("prompt", ""))

        try:
            result = await func(*args, **kwargs)

            # 计算性能指标
            response_time_ms = (time.time() - start_time) * 1000
            response = str(result)

            # 估算token数量（简化版本）
            prompt_tokens = len(prompt.split())
            response_tokens = len(response.split())

            performance_metrics = PerformanceMetrics(
                response_time_ms=response_time_ms,
                token_count=prompt_tokens + response_tokens,
                model_name=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=response_tokens
            )

            # 计算成本
            cost_metrics = self.calculate_cost(
                model_provider, model_name, prompt_tokens, response_tokens
            )

            # 创建交互记录
            interaction = AIInteraction(
                interaction_id=interaction_id,
                session_id=session_id,
                user_id=user_id,
                agent_type=agent_type,
                model_provider=model_provider,
                model_name=model_name,
                prompt=prompt,
                response=response,
                context=kwargs.get("context", {}),
                performance_metrics=performance_metrics,
                cost_metrics=cost_metrics,
                manufacturing_metrics=None,  # 可以根据需要设置
                timestamp=datetime.now(timezone.utc),
                success=True
            )

            self.log_ai_interaction(interaction)
            return result

        except Exception as e:
            # 记录错误
            response_time_ms = (time.time() - start_time) * 1000

            performance_metrics = PerformanceMetrics(
                response_time_ms=response_time_ms,
                token_count=0,
                model_name=model_name,
                error_type=type(e).__name__,
                error_message=str(e)
            )

            interaction = AIInteraction(
                interaction_id=interaction_id,
                session_id=session_id,
                user_id=user_id,
                agent_type=agent_type,
                model_provider=model_provider,
                model_name=model_name,
                prompt=prompt,
                response="",
                context=kwargs.get("context", {}),
                performance_metrics=performance_metrics,
                cost_metrics=None,
                manufacturing_metrics=None,
                timestamp=datetime.now(timezone.utc),
                success=False,
                error_info={
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
            )

            self.log_ai_interaction(interaction)
            raise

    def _log_interaction_sync(self,
                            func: Callable,
                            agent_type: AgentType,
                            model_provider: ModelProvider,
                            model_name: str,
                            *args, **kwargs):
        """同步函数交互日志"""
        start_time = time.time()
        interaction_id = str(uuid.uuid4())

        session_id = kwargs.get("session_id", "unknown")
        user_id = kwargs.get("user_id")
        prompt = str(args[0]) if args else str(kwargs.get("prompt", ""))

        try:
            result = func(*args, **kwargs)

            response_time_ms = (time.time() - start_time) * 1000
            response = str(result)

            prompt_tokens = len(prompt.split())
            response_tokens = len(response.split())

            performance_metrics = PerformanceMetrics(
                response_time_ms=response_time_ms,
                token_count=prompt_tokens + response_tokens,
                model_name=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=response_tokens
            )

            cost_metrics = self.calculate_cost(
                model_provider, model_name, prompt_tokens, response_tokens
            )

            interaction = AIInteraction(
                interaction_id=interaction_id,
                session_id=session_id,
                user_id=user_id,
                agent_type=agent_type,
                model_provider=model_provider,
                model_name=model_name,
                prompt=prompt,
                response=response,
                context=kwargs.get("context", {}),
                performance_metrics=performance_metrics,
                cost_metrics=cost_metrics,
                manufacturing_metrics=None,
                timestamp=datetime.now(timezone.utc),
                success=True
            )

            self.log_ai_interaction(interaction)
            return result

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000

            performance_metrics = PerformanceMetrics(
                response_time_ms=response_time_ms,
                token_count=0,
                model_name=model_name,
                error_type=type(e).__name__,
                error_message=str(e)
            )

            interaction = AIInteraction(
                interaction_id=interaction_id,
                session_id=session_id,
                user_id=user_id,
                agent_type=agent_type,
                model_provider=model_provider,
                model_name=model_name,
                prompt=prompt,
                response="",
                context=kwargs.get("context", {}),
                performance_metrics=performance_metrics,
                cost_metrics=None,
                manufacturing_metrics=None,
                timestamp=datetime.now(timezone.utc),
                success=False,
                error_info={
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
            )

            self.log_ai_interaction(interaction)
            raise

# 便捷装饰器
def log_ai_interaction(agent_type: AgentType,
                      model_provider: ModelProvider,
                      model_name: str):
    """便捷的AI交互日志装饰器"""
    logger_instance = AIInteractionLogger()
    return logger_instance.create_interaction_decorator(agent_type, model_provider, model_name)

# 全局实例
_ai_logger = None

def get_ai_logger() -> AIInteractionLogger:
    """获取AI日志记录器实例"""
    global _ai_logger
    if _ai_logger is None:
        _ai_logger = AIInteractionLogger()
    return _ai_logger