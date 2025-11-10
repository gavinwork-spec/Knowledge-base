#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangFuse Integration for Manufacturing Knowledge Base
制造业知识库的LangFuse集成

This module provides comprehensive observability using LangFuse patterns for tracking
AI interactions, performance metrics, cost analysis, and user behavior in manufacturing contexts.
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import sqlite3
from contextlib import asynccontextmanager

# LangFuse imports (with fallback)
try:
    from langfuse import Langfuse
    from langfuse.model import CreateTrace, CreateSpan, CreateEvent
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logging.warning("LangFuse not available, using local storage fallback")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InteractionType(Enum):
    """交互类型枚举"""
    QUERY = "query"
    RESPONSE = "response"
    DOCUMENT_PROCESSING = "document_processing"
    AGENT_EXECUTION = "agent_execution"
    RAG_RETRIEVAL = "rag_retrieval"
    ERROR = "error"
    PERFORMANCE_METRIC = "performance_metric"

class AgentType(Enum):
    """代理类型枚举"""
    DOCUMENT_PROCESSOR = "document_processor"
    CUSTOMER_SERVICE = "customer_service"
    QUOTE_ANALYZER = "quote_analyzer"
    KNOWLEDGE_RETRIEVER = "knowledge_retriever"
    QUALITY_CHECKER = "quality_checker"
    MULTI_MODAL_PROCESSOR = "multi_modal_processor"

class MetricType(Enum):
    """指标类型枚举"""
    RESPONSE_TIME = "response_time"
    ACCURACY = "accuracy"
    COST = "cost"
    USER_SATISFACTION = "user_satisfaction"
    QUOTE_ACCURACY = "quote_accuracy"
    DOCUMENT_PROCESSING_SUCCESS = "document_processing_success"
    KNOWLEDGE_GAP = "knowledge_gap"
    TOKEN_USAGE = "token_usage"

@dataclass
class ManufacturingMetrics:
    """制造业特定指标"""
    quote_accuracy: Optional[float] = None
    document_processing_success_rate: Optional[float] = None
    customer_satisfaction_score: Optional[float] = None
    knowledge_gap_score: Optional[float] = None
    equipment_recognition_accuracy: Optional[float] = None
    standards_compliance_score: Optional[float] = None
    processing_time_per_document: Optional[float] = None
    cost_per_quote: Optional[float] = None

@dataclass
class PerformanceMetrics:
    """性能指标"""
    response_time_ms: float
    token_count: int
    model_name: str
    temperature: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cache_hit: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class CostMetrics:
    """成本指标"""
    input_cost: float
    output_cost: float
    total_cost: float
    currency: str = "USD"
    cost_per_token: Optional[float] = None
    api_call_count: int = 1

@dataclass
class UserBehaviorMetrics:
    """用户行为指标"""
    session_id: str
    user_id: Optional[str] = None
    query_count: int = 0
    avg_query_length: float = 0.0
    satisfaction_score: Optional[float] = None
    feedback_count: int = 0
    repeat_queries: int = 0
    knowledge_gaps_identified: List[str] = field(default_factory=list)

@dataclass
class TraceData:
    """跟踪数据"""
    trace_id: str
    session_id: str
    user_id: Optional[str]
    interaction_type: InteractionType
    agent_type: Optional[AgentType]
    timestamp: datetime
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    performance_metrics: Optional[PerformanceMetrics] = None
    cost_metrics: Optional[CostMetrics] = None
    manufacturing_metrics: Optional[ManufacturingMetrics] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class LangFuseIntegration:
    """LangFuse集成类"""

    def __init__(self,
                 db_path: str = "knowledge_base.db",
                 enable_langfuse: bool = True,
                 langfuse_public_key: Optional[str] = None,
                 langfuse_secret_key: Optional[str] = None,
                 langfuse_host: Optional[str] = None):
        """
        初始化LangFuse集成

        Args:
            db_path: SQLite数据库路径
            enable_langfuse: 是否启用LangFuse云服务
            langfuse_public_key: LangFuse公钥
            langfuse_secret_key: LangFuse私钥
            langfuse_host: LangFuse主机地址
        """
        self.db_path = db_path
        self.enable_langfuse = enable_langfuse and LANGFUSE_AVAILABLE

        # 初始化LangFuse客户端
        if self.enable_langfuse:
            try:
                self.langfuse = Langfuse(
                    public_key=langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY"),
                    secret_key=langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY"),
                    host=langfuse_host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
                )
                logger.info("✅ LangFuse client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize LangFuse client: {e}")
                self.enable_langfuse = False
                self.langfuse = None
        else:
            self.langfuse = None
            logger.info("Using local storage for observability data")

        # 本地数据库连接
        self.conn = None
        self._init_database()

    def _init_database(self):
        """初始化本地数据库表"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute("PRAGMA foreign_keys = ON")

            # 创建跟踪数据表
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS observability_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT UNIQUE NOT NULL,
                    session_id TEXT NOT NULL,
                    user_id TEXT,
                    interaction_type TEXT NOT NULL,
                    agent_type TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    input_data TEXT,
                    output_data TEXT,
                    performance_metrics TEXT,
                    cost_metrics TEXT,
                    manufacturing_metrics TEXT,
                    tags TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_traces_session_id ON observability_traces(session_id);
                CREATE INDEX IF NOT EXISTS idx_traces_user_id ON observability_traces(user_id);
                CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON observability_traces(timestamp);
                CREATE INDEX IF NOT EXISTS idx_traces_interaction_type ON observability_traces(interaction_type);

                -- 性能指标汇总表
                CREATE TABLE IF NOT EXISTS performance_metrics_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    interaction_type TEXT NOT NULL,
                    avg_response_time REAL,
                    total_interactions INTEGER,
                    error_rate REAL,
                    total_cost REAL,
                    manufacturing_metrics TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, interaction_type)
                );

                -- 用户行为分析表
                CREATE TABLE IF NOT EXISTS user_behavior_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT,
                    date DATE NOT NULL,
                    query_count INTEGER,
                    avg_query_length REAL,
                    satisfaction_score REAL,
                    feedback_count INTEGER,
                    repeat_queries INTEGER,
                    knowledge_gaps TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- 警报配置表
                CREATE TABLE IF NOT EXISTS alert_configurations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_name TEXT UNIQUE NOT NULL,
                    metric_type TEXT NOT NULL,
                    threshold_value REAL NOT NULL,
                    comparison_operator TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT 1,
                    notification_config TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- 警报历史表
                CREATE TABLE IF NOT EXISTS alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_name TEXT NOT NULL,
                    triggered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric_value REAL,
                    threshold_value REAL,
                    context_data TEXT,
                    resolved BOOLEAN DEFAULT 0,
                    resolved_at DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)

            logger.info("✅ Observability database initialized")

        except sqlite3.Error as e:
            logger.error(f"Failed to initialize observability database: {e}")
            raise

    async def create_trace(self, trace_data: TraceData) -> str:
        """创建跟踪记录"""
        try:
            # 保存到本地数据库
            await self._save_trace_to_db(trace_data)

            # 如果启用LangFuse，也发送到云服务
            if self.enable_langfuse and self.langfuse:
                await self._send_to_langfuse(trace_data)

            return trace_data.trace_id

        except Exception as e:
            logger.error(f"Failed to create trace: {e}")
            raise

    async def _save_trace_to_db(self, trace_data: TraceData):
        """保存跟踪数据到本地数据库"""
        try:
            cursor = self.conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO observability_traces
                (trace_id, session_id, user_id, interaction_type, agent_type,
                 timestamp, input_data, output_data, performance_metrics,
                 cost_metrics, manufacturing_metrics, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trace_data.trace_id,
                trace_data.session_id,
                trace_data.user_id,
                trace_data.interaction_type.value,
                trace_data.agent_type.value if trace_data.agent_type else None,
                trace_data.timestamp.isoformat(),
                json.dumps(trace_data.input_data),
                json.dumps(trace_data.output_data),
                json.dumps(asdict(trace_data.performance_metrics)) if trace_data.performance_metrics else None,
                json.dumps(asdict(trace_data.cost_metrics)) if trace_data.cost_metrics else None,
                json.dumps(asdict(trace_data.manufacturing_metrics)) if trace_data.manufacturing_metrics else None,
                json.dumps(trace_data.tags),
                json.dumps(trace_data.metadata)
            ))

            self.conn.commit()
            logger.debug(f"Saved trace {trace_data.trace_id} to local database")

        except sqlite3.Error as e:
            logger.error(f"Failed to save trace to database: {e}")
            raise

    async def _send_to_langfuse(self, trace_data: TraceData):
        """发送数据到LangFuse"""
        try:
            # 创建LangFuse trace
            trace = self.langfuse.trace(
                id=trace_data.trace_id,
                name=f"{trace_data.interaction_type.value}_{trace_data.agent_type.value if trace_data.agent_type else 'system'}",
                input=trace_data.input_data,
                output=trace_data.output_data,
                metadata={
                    "session_id": trace_data.session_id,
                    "user_id": trace_data.user_id,
                    "tags": trace_data.tags,
                    **trace_data.metadata
                }
            )

            # 添加性能指标
            if trace_data.performance_metrics:
                trace.event(
                    name="performance_metrics",
                    input=asdict(trace_data.performance_metrics)
                )

            # 添加成本指标
            if trace_data.cost_metrics:
                trace.event(
                    name="cost_metrics",
                    input=asdict(trace_data.cost_metrics)
                )

            # 添加制造业指标
            if trace_data.manufacturing_metrics:
                trace.event(
                    name="manufacturing_metrics",
                    input=asdict(trace_data.manufacturing_metrics)
                )

            logger.debug(f"Sent trace {trace_data.trace_id} to LangFuse")

        except Exception as e:
            logger.warning(f"Failed to send trace to LangFuse: {e}")

    @asynccontextmanager
    async def trace_interaction(self,
                               session_id: str,
                               user_id: Optional[str] = None,
                               interaction_type: InteractionType = InteractionType.QUERY,
                               agent_type: Optional[AgentType] = None,
                               input_data: Optional[Dict[str, Any]] = None,
                               tags: Optional[List[str]] = None):
        """上下文管理器用于跟踪交互"""
        trace_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        trace_data = TraceData(
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            interaction_type=interaction_type,
            agent_type=agent_type,
            timestamp=start_time,
            input_data=input_data or {},
            output_data={},
            tags=tags or []
        )

        try:
            yield trace_data
        except Exception as e:
            # 记录错误
            trace_data.performance_metrics = PerformanceMetrics(
                response_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                token_count=0,
                model_name="error",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            trace_data.interaction_type = InteractionType.ERROR
            trace_data.output_data = {"error": str(e)}
            raise
        finally:
            # 确保跟踪数据被保存
            await self.create_trace(trace_data)

    async def log_performance_metrics(self,
                                    session_id: str,
                                    user_id: Optional[str],
                                    metrics: PerformanceMetrics,
                                    context: Optional[Dict[str, Any]] = None):
        """记录性能指标"""
        trace_data = TraceData(
            trace_id=str(uuid.uuid4()),
            session_id=session_id,
            user_id=user_id,
            interaction_type=InteractionType.PERFORMANCE_METRIC,
            agent_type=None,
            timestamp=datetime.now(timezone.utc),
            input_data=context or {},
            output_data={},
            performance_metrics=metrics
        )

        await self.create_trace(trace_data)

    async def log_manufacturing_metrics(self,
                                      session_id: str,
                                      user_id: Optional[str],
                                      metrics: ManufacturingMetrics,
                                      context: Optional[Dict[str, Any]] = None):
        """记录制造业指标"""
        trace_data = TraceData(
            trace_id=str(uuid.uuid4()),
            session_id=session_id,
            user_id=user_id,
            interaction_type=InteractionType.PERFORMANCE_METRIC,
            agent_type=None,
            timestamp=datetime.now(timezone.utc),
            input_data=context or {},
            output_data={},
            manufacturing_metrics=metrics
        )

        await self.create_trace(trace_data)

    async def get_trace_history(self,
                              session_id: Optional[str] = None,
                              user_id: Optional[str] = None,
                              interaction_type: Optional[InteractionType] = None,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """获取跟踪历史"""
        try:
            cursor = self.conn.cursor()

            query = "SELECT * FROM observability_traces WHERE 1=1"
            params = []

            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)

            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)

            if interaction_type:
                query += " AND interaction_type = ?"
                params.append(interaction_type.value)

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # 转换为字典格式
            columns = [desc[0] for desc in cursor.description]
            results = []
            for row in rows:
                result = dict(zip(columns, row))
                # 解析JSON字段
                for json_field in ['input_data', 'output_data', 'performance_metrics',
                                  'cost_metrics', 'manufacturing_metrics', 'tags', 'metadata']:
                    if result[json_field]:
                        result[json_field] = json.loads(result[json_field])
                results.append(result)

            return results

        except sqlite3.Error as e:
            logger.error(f"Failed to get trace history: {e}")
            return []

    async def get_performance_summary(self,
                                    date: Optional[datetime] = None) -> Dict[str, Any]:
        """获取性能汇总"""
        try:
            if not date:
                date = datetime.now().date()

            cursor = self.conn.cursor()

            # 获取当日性能指标汇总
            cursor.execute("""
                SELECT
                    interaction_type,
                    AVG(CAST(JSON_EXTRACT(performance_metrics, '$.response_time_ms') AS REAL)) as avg_response_time,
                    COUNT(*) as total_interactions,
                    SUM(CASE WHEN performance_metrics LIKE '%"error_type"%' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as error_rate,
                    SUM(CAST(JSON_EXTRACT(cost_metrics, '$.total_cost') AS REAL)) as total_cost
                FROM observability_traces
                WHERE DATE(timestamp) = ? AND performance_metrics IS NOT NULL
                GROUP BY interaction_type
            """, (date.isoformat(),))

            summary_rows = cursor.fetchall()

            summary = {
                "date": date.isoformat(),
                "interaction_types": {}
            }

            for row in summary_rows:
                interaction_type, avg_response_time, total_interactions, error_rate, total_cost = row
                summary["interaction_types"][interaction_type] = {
                    "avg_response_time_ms": avg_response_time,
                    "total_interactions": total_interactions,
                    "error_rate_percent": error_rate,
                    "total_cost": total_cost
                }

            return summary

        except sqlite3.Error as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}

    def flush(self):
        """刷新缓冲区"""
        if self.langfuse:
            self.langfuse.flush()

    def close(self):
        """关闭连接"""
        try:
            if self.conn:
                self.conn.close()
                logger.info("Observability database connection closed")
        except Exception as e:
            logger.error(f"Error closing observability connection: {e}")

# 全局实例
_langfuse_integration = None

def get_langfuse_integration() -> LangFuseIntegration:
    """获取LangFuse集成实例"""
    global _langfuse_integration
    if _langfuse_integration is None:
        _langfuse_integration = LangFuseIntegration()
    return _langfuse_integration

async def create_langfuse_integration(config: Optional[Dict[str, Any]] = None) -> LangFuseIntegration:
    """创建LangFuse集成实例"""
    if config is None:
        config = {}

    integration = LangFuseIntegration(
        db_path=config.get("db_path", "knowledge_base.db"),
        enable_langfuse=config.get("enable_langfuse", True),
        langfuse_public_key=config.get("langfuse_public_key"),
        langfuse_secret_key=config.get("langfuse_secret_key"),
        langfuse_host=config.get("langfuse_host")
    )

    return integration