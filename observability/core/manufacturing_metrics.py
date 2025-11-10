#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manufacturing-Specific Metrics System
制造业特定指标系统

Specialized metrics tracking for manufacturing operations including quote accuracy,
document processing success rates, quality control effectiveness, and customer satisfaction.
"""

import sqlite3
import json
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics

from .langfuse_integration import get_langfuse_integration
from .performance_tracker import get_performance_tracker, MetricType
from .cost_analyzer import get_cost_analyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QuoteStatus(Enum):
    """报价状态"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVISED = "revised"
    EXPIRED = "expired"

class QualityCheckResult(Enum):
    """质量检查结果"""
    PASS = "pass"
    FAIL = "fail"
    REWORK_REQUIRED = "rework_required"
    CONDITIONAL_PASS = "conditional_pass"

class DocumentProcessingStatus(Enum):
    """文档处理状态"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    ERROR = "error"

class CustomerSatisfactionLevel(Enum):
    """客户满意度级别"""
    VERY_DISSATISFIED = 1
    DISSATISFIED = 2
    NEUTRAL = 3
    SATISFIED = 4
    VERY_SATISFIED = 5

class ManufacturingProcessType(Enum):
    """制造流程类型"""
    CNC_MACHINING = "cnc_machining"
    WELDING = "welding"
    ASSEMBLY = "assembly"
    QUALITY_INSPECTION = "quality_inspection"
    SURFACE_TREATMENT = "surface_treatment"
    HEAT_TREATMENT = "heat_treatment"
    TESTING = "testing"
    PACKAGING = "packaging"

@dataclass
class QuoteMetrics:
    """报价指标"""
    quote_id: str
    customer_id: str
    part_number: str
    quantity: int
    quoted_price: float
    actual_cost: float
    margin_percentage: float
    processing_time_minutes: float
    accuracy_score: float
    revision_count: int
    status: QuoteStatus
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityMetrics:
    """质量指标"""
    inspection_id: str
    part_number: str
    batch_id: str
    process_type: ManufacturingProcessType
    total_inspected: int
    passed_count: int
    failed_count: int
    rework_count: int
    defect_rate: float
    first_pass_yield: float
    inspection_time_minutes: float
    result: QualityCheckResult
    inspector_id: str
    timestamp: datetime

@dataclass
class DocumentProcessingMetrics:
    """文档处理指标"""
    processing_id: str
    document_type: str
    file_size_mb: float
    processing_time_seconds: float
    success: bool
    error_message: Optional[str]
    extracted_entities: int
    processing_accuracy: float
    status: DocumentProcessingStatus
    timestamp: datetime

@dataclass
class CustomerSatisfactionMetrics:
    """客户满意度指标"""
    feedback_id: str
    customer_id: str
    order_id: str
    quote_id: Optional[str]
    overall_satisfaction: CustomerSatisfactionLevel
    quality_satisfaction: CustomerSatisfactionLevel
    delivery_satisfaction: CustomerSatisfactionLevel
    service_satisfaction: CustomerSatisfactionLevel
    price_satisfaction: CustomerSatisfactionLevel
    nps_score: int  # Net Promoter Score
    feedback_text: str
    timestamp: datetime

@dataclass
class ProductionEfficiencyMetrics:
    """生产效率指标"""
    production_id: str
    work_order_id: str
    part_number: str
    process_type: ManufacturingProcessType
    planned_time_hours: float
    actual_time_hours: float
    efficiency_percentage: float
    downtime_minutes: float
    scrap_percentage: float
    rework_percentage: float
    operator_id: str
    timestamp: datetime

class ManufacturingMetricsCollector:
    """制造业指标收集器"""

    def __init__(self, db_path: str = "knowledge_base.db"):
        """
        初始化制造业指标收集器

        Args:
            db_path: SQLite数据库路径
        """
        self.db_path = db_path

        # 获取其他组件
        self.langfuse_integration = get_langfuse_integration()
        self.performance_tracker = get_performance_tracker()
        self.cost_analyzer = get_cost_analyzer()

        # 内存缓存
        self.quote_metrics_cache: deque = deque(maxlen=1000)
        self.quality_metrics_cache: deque = deque(maxlen=1000)
        self.customer_satisfaction_cache: deque = deque(maxlen=1000)
        self.document_processing_cache: deque = deque(maxlen=1000)
        self.production_efficiency_cache: deque = deque(maxlen=1000)

        # 实时计算值
        self.real_time_metrics = {}

        # 初始化数据库
        self._init_database()

        # 启动后台任务
        self._start_background_tasks()

    def _init_database(self):
        """初始化数据库表"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute("PRAGMA foreign_keys = ON")

            self.conn.executescript("""
                -- 报价指标表
                CREATE TABLE IF NOT EXISTS quote_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    quote_id TEXT UNIQUE NOT NULL,
                    customer_id TEXT NOT NULL,
                    part_number TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    quoted_price REAL NOT NULL,
                    actual_cost REAL,
                    margin_percentage REAL,
                    processing_time_minutes REAL NOT NULL,
                    accuracy_score REAL NOT NULL,
                    revision_count INTEGER DEFAULT 0,
                    status TEXT NOT NULL,
                    metadata TEXT,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- 质量指标表
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inspection_id TEXT UNIQUE NOT NULL,
                    part_number TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    process_type TEXT NOT NULL,
                    total_inspected INTEGER NOT NULL,
                    passed_count INTEGER NOT NULL,
                    failed_count INTEGER DEFAULT 0,
                    rework_count INTEGER DEFAULT 0,
                    defect_rate REAL NOT NULL,
                    first_pass_yield REAL NOT NULL,
                    inspection_time_minutes REAL NOT NULL,
                    result TEXT NOT NULL,
                    inspector_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL
                );

                -- 文档处理指标表
                CREATE TABLE IF NOT EXISTS document_processing_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    processing_id TEXT UNIQUE NOT NULL,
                    document_type TEXT NOT NULL,
                    file_size_mb REAL,
                    processing_time_seconds REAL NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    extracted_entities INTEGER DEFAULT 0,
                    processing_accuracy REAL,
                    status TEXT NOT NULL,
                    timestamp DATETIME NOT NULL
                );

                -- 客户满意度指标表
                CREATE TABLE IF NOT EXISTS customer_satisfaction_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_id TEXT UNIQUE NOT NULL,
                    customer_id TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    quote_id TEXT,
                    overall_satisfaction INTEGER NOT NULL,
                    quality_satisfaction INTEGER NOT NULL,
                    delivery_satisfaction INTEGER NOT NULL,
                    service_satisfaction INTEGER NOT NULL,
                    price_satisfaction INTEGER NOT NULL,
                    nps_score INTEGER,
                    feedback_text TEXT,
                    timestamp DATETIME NOT NULL
                );

                -- 生产效率指标表
                CREATE TABLE IF NOT EXISTS production_efficiency_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    production_id TEXT UNIQUE NOT NULL,
                    work_order_id TEXT NOT NULL,
                    part_number TEXT NOT NULL,
                    process_type TEXT NOT NULL,
                    planned_time_hours REAL NOT NULL,
                    actual_time_hours REAL NOT NULL,
                    efficiency_percentage REAL NOT NULL,
                    downtime_minutes REAL DEFAULT 0,
                    scrap_percentage REAL DEFAULT 0,
                    rework_percentage REAL DEFAULT 0,
                    operator_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL
                );

                -- 制造业KPI汇总表
                CREATE TABLE IF NOT EXISTS manufacturing_kpi_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    quote_accuracy_avg REAL,
                    quote_margin_avg REAL,
                    quality_first_pass_yield REAL,
                    defect_rate_avg REAL,
                    customer_satisfaction_avg REAL,
                    document_processing_success_rate REAL,
                    production_efficiency_avg REAL,
                    total_quotes INTEGER,
                    total_inspections INTEGER,
                    total_feedback INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                );

                -- 设备特定指标表
                CREATE TABLE IF NOT EXISTS equipment_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    equipment_id TEXT NOT NULL,
                    equipment_type TEXT NOT NULL,
                    utilization_rate REAL,
                    uptime_percentage REAL,
                    downtime_minutes REAL,
                    maintenance_count INTEGER,
                    production_count INTEGER,
                    quality_score REAL,
                    date DATE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(equipment_id, date)
                );

                -- 创建索引
                CREATE INDEX IF NOT EXISTS idx_quote_metrics_date ON quote_metrics(created_at);
                CREATE INDEX IF NOT EXISTS idx_quote_metrics_customer ON quote_metrics(customer_id);
                CREATE INDEX IF NOT EXISTS idx_quality_metrics_date ON quality_metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_quality_metrics_process ON quality_metrics(process_type);
                CREATE INDEX IF NOT EXISTS idx_satisfaction_metrics_date ON customer_satisfaction_metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_document_processing_date ON document_processing_metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_production_efficiency_date ON production_efficiency_metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_equipment_metrics_date ON equipment_metrics(date);
            """)

            logger.info("✅ Manufacturing metrics database initialized")

        except sqlite3.Error as e:
            logger.error(f"Failed to initialize manufacturing metrics database: {e}")
            raise

    def _start_background_tasks(self):
        """启动后台任务"""
        asyncio.create_task(self._calculate_real_time_metrics())
        asyncio.create_task(self._daily_kpi_aggregation())
        asyncio.create_task(self._generate_mfg_insights())

    async def record_quote_metrics(self, metrics: QuoteMetrics):
        """记录报价指标"""
        try:
            # 保存到数据库
            await self._save_quote_metrics(metrics)

            # 添加到缓存
            self.quote_metrics_cache.append(metrics)

            # 更新实时指标
            await self._update_quote_real_time_metrics(metrics)

            # 记录成本
            if metrics.actual_cost:
                await self.cost_analyzer.record_manufacturing_cost(
                    "quote_generation",
                    metrics.actual_cost,
                    1,
                    {
                        "quote_id": metrics.quote_id,
                        "customer_id": metrics.customer_id,
                        "part_number": metrics.part_number,
                        "quantity": metrics.quantity,
                        "accuracy_score": metrics.accuracy_score
                    }
                )

            logger.debug(f"Recorded quote metrics: {metrics.quote_id} (accuracy: {metrics.accuracy_score:.2f})")

        except Exception as e:
            logger.error(f"Failed to record quote metrics: {e}")

    async def _save_quote_metrics(self, metrics: QuoteMetrics):
        """保存报价指标到数据库"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO quote_metrics
                (quote_id, customer_id, part_number, quantity, quoted_price,
                 actual_cost, margin_percentage, processing_time_minutes,
                 accuracy_score, revision_count, status, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.quote_id,
                metrics.customer_id,
                metrics.part_number,
                metrics.quantity,
                metrics.quoted_price,
                metrics.actual_cost,
                metrics.margin_percentage,
                metrics.processing_time_minutes,
                metrics.accuracy_score,
                metrics.revision_count,
                metrics.status.value,
                json.dumps(metrics.metadata),
                metrics.created_at.isoformat()
            ))
            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Failed to save quote metrics: {e}")

    async def record_quality_metrics(self, metrics: QualityMetrics):
        """记录质量指标"""
        try:
            await self._save_quality_metrics(metrics)
            self.quality_metrics_cache.append(metrics)
            await self._update_quality_real_time_metrics(metrics)

            # 记录成本
            await self.cost_analyzer.record_manufacturing_cost(
                "quality_check",
                metrics.inspection_time_minutes * 2.0,  # 假设每小时成本$2
                metrics.total_inspected,
                {
                    "inspection_id": metrics.inspection_id,
                    "part_number": metrics.part_number,
                    "batch_id": metrics.batch_id,
                    "defect_rate": metrics.defect_rate
                }
            )

            logger.debug(f"Recorded quality metrics: {metrics.inspection_id} (FPY: {metrics.first_pass_yield:.2f})")

        except Exception as e:
            logger.error(f"Failed to record quality metrics: {e}")

    async def _save_quality_metrics(self, metrics: QualityMetrics):
        """保存质量指标到数据库"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO quality_metrics
                (inspection_id, part_number, batch_id, process_type,
                 total_inspected, passed_count, failed_count, rework_count,
                 defect_rate, first_pass_yield, inspection_time_minutes,
                 result, inspector_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.inspection_id,
                metrics.part_number,
                metrics.batch_id,
                metrics.process_type.value,
                metrics.total_inspected,
                metrics.passed_count,
                metrics.failed_count,
                metrics.rework_count,
                metrics.defect_rate,
                metrics.first_pass_yield,
                metrics.inspection_time_minutes,
                metrics.result.value,
                metrics.inspector_id,
                metrics.timestamp.isoformat()
            ))
            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Failed to save quality metrics: {e}")

    async def record_document_processing_metrics(self, metrics: DocumentProcessingMetrics):
        """记录文档处理指标"""
        try:
            await self._save_document_processing_metrics(metrics)
            self.document_processing_cache.append(metrics)
            await self._update_document_processing_real_time_metrics(metrics)

            logger.debug(f"Recorded document processing metrics: {metrics.processing_id} (success: {metrics.success})")

        except Exception as e:
            logger.error(f"Failed to record document processing metrics: {e}")

    async def _save_document_processing_metrics(self, metrics: DocumentProcessingMetrics):
        """保存文档处理指标到数据库"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO document_processing_metrics
                (processing_id, document_type, file_size_mb,
                 processing_time_seconds, success, error_message,
                 extracted_entities, processing_accuracy, status, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.processing_id,
                metrics.document_type,
                metrics.file_size_mb,
                metrics.processing_time_seconds,
                metrics.success,
                metrics.error_message,
                metrics.extracted_entities,
                metrics.processing_accuracy,
                metrics.status.value,
                metrics.timestamp.isoformat()
            ))
            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Failed to save document processing metrics: {e}")

    async def record_customer_satisfaction_metrics(self, metrics: CustomerSatisfactionMetrics):
        """记录客户满意度指标"""
        try:
            await self._save_customer_satisfaction_metrics(metrics)
            self.customer_satisfaction_cache.append(metrics)
            await self._update_satisfaction_real_time_metrics(metrics)

            logger.debug(f"Recorded customer satisfaction: {metrics.feedback_id} (overall: {metrics.overall_satisfaction})")

        except Exception as e:
            logger.error(f"Failed to record customer satisfaction metrics: {e}")

    async def _save_customer_satisfaction_metrics(self, metrics: CustomerSatisfactionMetrics):
        """保存客户满意度指标到数据库"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO customer_satisfaction_metrics
                (feedback_id, customer_id, order_id, quote_id,
                 overall_satisfaction, quality_satisfaction, delivery_satisfaction,
                 service_satisfaction, price_satisfaction, nps_score,
                 feedback_text, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.feedback_id,
                metrics.customer_id,
                metrics.order_id,
                metrics.quote_id,
                metrics.overall_satisfaction.value,
                metrics.quality_satisfaction.value,
                metrics.delivery_satisfaction.value,
                metrics.service_satisfaction.value,
                metrics.price_satisfaction.value,
                metrics.nps_score,
                metrics.feedback_text,
                metrics.timestamp.isoformat()
            ))
            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Failed to save customer satisfaction metrics: {e}")

    async def record_production_efficiency_metrics(self, metrics: ProductionEfficiencyMetrics):
        """记录生产效率指标"""
        try:
            await self._save_production_efficiency_metrics(metrics)
            self.production_efficiency_cache.append(metrics)
            await self._update_production_efficiency_real_time_metrics(metrics)

            logger.debug(f"Recorded production efficiency: {metrics.production_id} (efficiency: {metrics.efficiency_percentage:.1f}%)")

        except Exception as e:
            logger.error(f"Failed to record production efficiency metrics: {e}")

    async def _save_production_efficiency_metrics(self, metrics: ProductionEfficiencyMetrics):
        """保存生产效率指标到数据库"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO production_efficiency_metrics
                (production_id, work_order_id, part_number, process_type,
                 planned_time_hours, actual_time_hours, efficiency_percentage,
                 downtime_minutes, scrap_percentage, rework_percentage,
                 operator_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.production_id,
                metrics.work_order_id,
                metrics.part_number,
                metrics.process_type.value,
                metrics.planned_time_hours,
                metrics.actual_time_hours,
                metrics.efficiency_percentage,
                metrics.downtime_minutes,
                metrics.scrap_percentage,
                metrics.rework_percentage,
                metrics.operator_id,
                metrics.timestamp.isoformat()
            ))
            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Failed to save production efficiency metrics: {e}")

    async def _update_quote_real_time_metrics(self, metrics: QuoteMetrics):
        """更新报价实时指标"""
        try:
            # 计算最近7天的报价准确率
            recent_quotes = [q for q in self.quote_metrics_cache if
                           (datetime.now(timezone.utc) - q.created_at).days <= 7]

            if recent_quotes:
                accuracy_scores = [q.accuracy_score for q in recent_quotes]
                self.real_time_metrics["quote_accuracy_7d"] = statistics.mean(accuracy_scores)

                # 计算平均利润率
                margins = [q.margin_percentage for q in recent_quotes if q.margin_percentage is not None]
                if margins:
                    self.real_time_metrics["quote_margin_avg_7d"] = statistics.mean(margins)

        except Exception as e:
            logger.error(f"Failed to update quote real-time metrics: {e}")

    async def _update_quality_real_time_metrics(self, metrics: QualityMetrics):
        """更新质量实时指标"""
        try:
            # 计算最近24小时的首次通过率
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
            recent_quality = [q for q in self.quality_metrics_cache if q.timestamp >= cutoff_time]

            if recent_quality:
                fpy_scores = [q.first_pass_yield for q in recent_quality]
                self.real_time_metrics["quality_fpy_24h"] = statistics.mean(fpy_scores)

                defect_rates = [q.defect_rate for q in recent_quality]
                self.real_time_metrics["quality_defect_rate_24h"] = statistics.mean(defect_rates)

        except Exception as e:
            logger.error(f"Failed to update quality real-time metrics: {e}")

    async def _update_document_processing_real_time_metrics(self, metrics: DocumentProcessingMetrics):
        """更新文档处理实时指标"""
        try:
            # 计算最近1小时的成功率
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
            recent_docs = [d for d in self.document_processing_cache if d.timestamp >= cutoff_time]

            if recent_docs:
                success_count = sum(1 for d in recent_docs if d.success)
                success_rate = (success_count / len(recent_docs)) * 100
                self.real_time_metrics["document_processing_success_rate_1h"] = success_rate

                # 计算平均处理时间
                processing_times = [d.processing_time_seconds for d in recent_docs]
                self.real_time_metrics["document_processing_avg_time_1h"] = statistics.mean(processing_times)

        except Exception as e:
            logger.error(f"Failed to update document processing real-time metrics: {e}")

    async def _update_satisfaction_real_time_metrics(self, metrics: CustomerSatisfactionMetrics):
        """更新满意度实时指标"""
        try:
            # 计算最近30天的平均满意度
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=30)
            recent_satisfaction = [s for s in self.customer_satisfaction_cache if s.timestamp >= cutoff_time]

            if recent_satisfaction:
                overall_scores = [s.overall_satisfaction.value for s in recent_satisfaction]
                self.real_time_metrics["customer_satisfaction_30d"] = statistics.mean(overall_scores)

                # 计算NPS分数
                nps_scores = [s.nps_score for s in recent_satisfaction if s.nps_score is not None]
                if nps_scores:
                    self.real_time_metrics["nps_score_30d"] = statistics.mean(nps_scores)

        except Exception as e:
            logger.error(f"Failed to update satisfaction real-time metrics: {e}")

    async def _update_production_efficiency_real_time_metrics(self, metrics: ProductionEfficiencyMetrics):
        """更新生产效率实时指标"""
        try:
            # 计算最近24小时的平均效率
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
            recent_production = [p for p in self.production_efficiency_cache if p.timestamp >= cutoff_time]

            if recent_production:
                efficiency_scores = [p.efficiency_percentage for p in recent_production]
                self.real_time_metrics["production_efficiency_24h"] = statistics.mean(efficiency_scores)

                # 计算平均废品率
                scrap_rates = [p.scrap_percentage for p in recent_production]
                self.real_time_metrics["production_scrap_rate_24h"] = statistics.mean(scrap_rates)

        except Exception as e:
            logger.error(f"Failed to update production efficiency real-time metrics: {e}")

    async def _calculate_real_time_metrics(self):
        """计算实时指标"""
        while True:
            try:
                # 每分钟更新一次实时指标
                await asyncio.sleep(60)

                # 这里可以添加更多的实时指标计算逻辑
                # 例如：趋势分析、预测等

            except Exception as e:
                logger.error(f"Error in real-time metrics calculation: {e}")
                await asyncio.sleep(60)

    async def _daily_kpi_aggregation(self):
        """每日KPI聚合"""
        while True:
            try:
                # 每天凌晨1点执行聚合
                now = datetime.now(timezone.utc)
                next_run = now.replace(hour=1, minute=0, second=0, microsecond=0)
                if now > next_run:
                    next_run += timedelta(days=1)

                sleep_seconds = (next_run - now).total_seconds()
                await asyncio.sleep(sleep_seconds)

                # 聚合昨天的KPI
                yesterday = now - timedelta(days=1)
                await self._aggregate_daily_kpis(yesterday)

            except Exception as e:
                logger.error(f"Error in daily KPI aggregation: {e}")
                await asyncio.sleep(3600)  # 出错时1小时后重试

    async def _aggregate_daily_kpis(self, date: datetime):
        """聚合指定日期的KPI"""
        try:
            cursor = self.conn.cursor()
            date_str = date.strftime('%Y-%m-%d')

            # 报价KPI
            cursor.execute("""
                SELECT AVG(accuracy_score) as avg_accuracy,
                       AVG(margin_percentage) as avg_margin,
                       COUNT(*) as total_quotes
                FROM quote_metrics
                WHERE DATE(created_at) = ?
            """, (date_str,))

            quote_result = cursor.fetchone()
            quote_accuracy_avg = quote_result[0] if quote_result[0] else 0
            quote_margin_avg = quote_result[1] if quote_result[1] else 0
            total_quotes = quote_result[2] or 0

            # 质量KPI
            cursor.execute("""
                SELECT AVG(first_pass_yield) as avg_fpy,
                       AVG(defect_rate) as avg_defect_rate,
                       COUNT(*) as total_inspections
                FROM quality_metrics
                WHERE DATE(timestamp) = ?
            """, (date_str,))

            quality_result = cursor.fetchone()
            quality_first_pass_yield = quality_result[0] if quality_result[0] else 0
            defect_rate_avg = quality_result[1] if quality_result[1] else 0
            total_inspections = quality_result[2] or 0

            # 客户满意度KPI
            cursor.execute("""
                SELECT AVG(overall_satisfaction) as avg_satisfaction,
                       COUNT(*) as total_feedback
                FROM customer_satisfaction_metrics
                WHERE DATE(timestamp) = ?
            """, (date_str,))

            satisfaction_result = cursor.fetchone()
            customer_satisfaction_avg = satisfaction_result[0] if satisfaction_result[0] else 0
            total_feedback = satisfaction_result[1] or 0

            # 文档处理KPI
            cursor.execute("""
                SELECT
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                    COUNT(*) as total_processed
                FROM document_processing_metrics
                WHERE DATE(timestamp) = ?
            """, (date_str,))

            doc_result = cursor.fetchone()
            document_processing_success_rate = doc_result[0] if doc_result and doc_result[0] else 0

            # 生产效率KPI
            cursor.execute("""
                SELECT AVG(efficiency_percentage) as avg_efficiency
                FROM production_efficiency_metrics
                WHERE DATE(timestamp) = ?
            """, (date_str,))

            production_result = cursor.fetchone()
            production_efficiency_avg = production_result[0] if production_result[0] else 0

            # 保存聚合结果
            cursor.execute("""
                INSERT OR REPLACE INTO manufacturing_kpi_summary
                (date, quote_accuracy_avg, quote_margin_avg, quality_first_pass_yield,
                 defect_rate_avg, customer_satisfaction_avg, document_processing_success_rate,
                 production_efficiency_avg, total_quotes, total_inspections, total_feedback)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date_str,
                quote_accuracy_avg,
                quote_margin_avg,
                quality_first_pass_yield,
                defect_rate_avg,
                customer_satisfaction_avg,
                document_processing_success_rate,
                production_efficiency_avg,
                total_quotes,
                total_inspections,
                total_feedback
            ))

            self.conn.commit()
            logger.info(f"Daily KPI aggregation completed for {date_str}")

        except Exception as e:
            logger.error(f"Failed to aggregate daily KPIs: {e}")

    async def _generate_mfg_insights(self):
        """生成制造业洞察"""
        while True:
            try:
                # 每小时生成一次洞察
                await asyncio.sleep(3600)
                insights = await self._analyze_manufacturing_trends()

                # 记录洞察到日志或发送到仪表板
                for insight in insights:
                    logger.info(f"Manufacturing Insight: {insight}")

            except Exception as e:
                logger.error(f"Error in manufacturing insights generation: {e}")
                await asyncio.sleep(3600)

    async def _analyze_manufacturing_trends(self) -> List[str]:
        """分析制造业趋势"""
        insights = []

        try:
            # 分析报价准确率趋势
            if "quote_accuracy_7d" in self.real_time_metrics:
                current_accuracy = self.real_time_metrics["quote_accuracy_7d"]
                if current_accuracy < 85:
                    insights.append(f"Quote accuracy is below target at {current_accuracy:.1f}%")

            # 分析质量趋势
            if "quality_fpy_24h" in self.real_time_metrics:
                current_fpy = self.real_time_metrics["quality_fpy_24h"]
                if current_fpy < 95:
                    insights.append(f"First Pass Yield is concerning at {current_fpy:.1f}%")

            # 分析客户满意度趋势
            if "customer_satisfaction_30d" in self.real_time_metrics:
                current_satisfaction = self.real_time_metrics["customer_satisfaction_30d"]
                if current_satisfaction < 4.0:
                    insights.append(f"Customer satisfaction needs attention at {current_satisfaction:.1f}/5.0")

            # 分析生产效率趋势
            if "production_efficiency_24h" in self.real_time_metrics:
                current_efficiency = self.real_time_metrics["production_efficiency_24h"]
                if current_efficiency < 80:
                    insights.append(f"Production efficiency is below target at {current_efficiency:.1f}%")

        except Exception as e:
            logger.error(f"Error analyzing manufacturing trends: {e}")

        return insights

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """获取实时指标"""
        return self.real_time_metrics.copy()

    async def get_manufacturing_dashboard_data(self, days: int = 30) -> Dict[str, Any]:
        """获取制造业仪表板数据"""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            cursor = self.conn.cursor()

            # KPI趋势数据
            cursor.execute("""
                SELECT date, quote_accuracy_avg, quality_first_pass_yield,
                       customer_satisfaction_avg, document_processing_success_rate,
                       production_efficiency_avg
                FROM manufacturing_kpi_summary
                WHERE date >= DATE(?)
                ORDER BY date
            """, (cutoff_date.isoformat(),))

            kpi_trends = cursor.fetchall()

            # 当前指标
            current_metrics = self.get_real_time_metrics()

            # 最近的活动
            cursor.execute("""
                SELECT 'quote' as type, created_at as timestamp, quote_id as activity_id,
                       accuracy_score as metric_value
                FROM quote_metrics
                WHERE created_at >= ?
                UNION ALL
                SELECT 'quality' as type, timestamp as timestamp, inspection_id as activity_id,
                       first_pass_yield as metric_value
                FROM quality_metrics
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 10
            """, (cutoff_date.isoformat(), cutoff_date.isoformat()))

            recent_activities = cursor.fetchall()

            return {
                "period_days": days,
                "kpi_trends": [
                    {
                        "date": row[0],
                        "quote_accuracy": row[1],
                        "quality_fpy": row[2],
                        "customer_satisfaction": row[3],
                        "document_processing_success_rate": row[4],
                        "production_efficiency": row[5]
                    }
                    for row in kpi_trends
                ],
                "current_metrics": current_metrics,
                "recent_activities": [
                    {
                        "type": row[0],
                        "timestamp": row[1],
                        "activity_id": row[2],
                        "metric_value": row[3]
                    }
                    for row in recent_activities
                ]
            }

        except Exception as e:
            logger.error(f"Failed to get manufacturing dashboard data: {e}")
            return {}

    def close(self):
        """关闭连接"""
        try:
            if self.conn:
                self.conn.close()
                logger.info("Manufacturing metrics connection closed")
        except Exception as e:
            logger.error(f"Error closing manufacturing metrics: {e}")

# 全局实例
_manufacturing_metrics_collector = None

def get_manufacturing_metrics_collector() -> ManufacturingMetricsCollector:
    """获取制造业指标收集器实例"""
    global _manufacturing_metrics_collector
    if _manufacturing_metrics_collector is None:
        _manufacturing_metrics_collector = ManufacturingMetricsCollector()
    return _manufacturing_metrics_collector