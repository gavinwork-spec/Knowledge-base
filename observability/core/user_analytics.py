#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User Behavior Analytics & Knowledge Gap Identification
用户行为分析和知识缺口识别

Advanced analytics system for tracking user behavior, identifying patterns,
measuring satisfaction, and detecting knowledge gaps in the manufacturing knowledge base.
"""

import sqlite3
import json
import logging
import asyncio
import re
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, Counter
import statistics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import spacy

# 尝试加载spaCy模型
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available, using simplified text processing")

from .langfuse_integration import get_langfuse_integration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UserIntent(Enum):
    """用户意图"""
    INFORMATION_SEEKING = "information_seeking"
    PROBLEM_SOLVING = "problem_solving"
    PROCEDURE_GUIDANCE = "procedure_guidance"
    SPECIFICATION_LOOKUP = "specification_lookup"
    TRAINING_LEARNING = "training_learning"
    TROUBLESHOOTING = "troubleshooting"
    COMPLIANCE_CHECK = "compliance_check"

class SatisfactionLevel(Enum):
    """满意度级别"""
    VERY_DISSATISFIED = 1
    DISSATISFIED = 2
    NEUTRAL = 3
    SATISFIED = 4
    VERY_SATISFIED = 5

class KnowledgeGapType(Enum):
    """知识缺口类型"""
    MISSING_INFORMATION = "missing_information"
    OUTDATED_CONTENT = "outdated_content"
    UNCLEAR_PROCEDURES = "unclear_procedures"
    INSUFFICIENT_DETAIL = "insufficient_detail"
    INCORRECT_INFORMATION = "incorrect_information"
    EQUIPMENT_SPECIFIC = "equipment_specific"
    STANDARD_COMPLIANCE = "standard_compliance"

@dataclass
class UserSession:
    """用户会话"""
    session_id: str
    user_id: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]
    queries: List[str]
    responses: List[str]
    satisfaction_scores: List[int]
    feedback_comments: List[str]
    documents_accessed: List[str]
    session_duration_minutes: float
    query_count: int
    avg_query_length: float
    manufacturing_entities: List[str]

@dataclass
class UserBehaviorPattern:
    """用户行为模式"""
    user_id: str
    common_intents: List[Tuple[UserIntent, float]]
    preferred_content_types: List[str]
    frequent_entities: List[str]
    session_patterns: Dict[str, Any]
    satisfaction_trend: List[float]
    engagement_score: float

@dataclass
class KnowledgeGap:
    """知识缺口"""
    gap_id: str
    gap_type: KnowledgeGapType
    description: str
    affected_queries: List[str]
    frequency: int
    severity: str  # "low", "medium", "high", "critical"
    suggested_improvements: List[str]
    detected_at: datetime
    related_entities: List[str]

@dataclass
class UserFeedback:
    """用户反馈"""
    feedback_id: str
    user_id: Optional[str]
    session_id: str
    query: str
    response: str
    satisfaction_score: int
    feedback_comment: str
    improvement_suggestions: List[str]
    timestamp: datetime

class UserAnalytics:
    """用户分析系统"""

    def __init__(self,
                 db_path: str = "knowledge_base.db",
                 analysis_interval_hours: int = 1):
        """
        初始化用户分析系统

        Args:
            db_path: SQLite数据库路径
            analysis_interval_hours: 分析间隔（小时）
        """
        self.db_path = db_path
        self.analysis_interval_hours = analysis_interval_hours

        # 获取LangFuse集成
        self.langfuse_integration = get_langfuse_integration()

        # 文本处理组件
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )

        # 制造业特定模式
        self.manufacturing_patterns = {
            'equipment_models': [
                r'\bHAAS[-\s]?VF[0-9]+\b',
                r'\bDMG[-\s]?MORI[-\s]?DMU[-\s]?[0-9]+\b',
                r'\bFANUC[-\s]?0[iF]?[-\s]?[A-Z0-9]+\b',
                r'\bSIEMENS[-\s]?[0-9]+[A-Z]*\b'
            ],
            'standards': [
                r'\bISO[-\s]?[0-9]+(?::[0-9]+)?\b',
                r'\bANSI[-\s]?[A-Z]+[-\s]?[0-9.]+\b',
                r'\bAPI[-\s]?[0-9]+\b',
                r'\bAWS[-\s]?D[0-9.]+\b'
            ],
            'measurements': [
                r'\b[0-9]+(?:\.[0-9]+)?\s*(?:psi|MPa|bar|Pa)\b',
                r'\b[0-9]+(?:\.[0-9]+)?\s*(?:kg|g|lb|oz)\b',
                r'\b[0-9]+(?:\.[0-9]+)?\s*(?:mm|cm|m|in|ft)\b'
            ],
            'procedures': [
                r'\b(?:risk\s*assessment|lockout|tagout|PPE|safety\s*check)\b',
                r'\b(?:quality\s*inspection|dimensional\s*check|CMM)\b',
                r'\b(?:preventive\s*maintenance|PM|routine\s*check)\b'
            ]
        }

        # 初始化数据库
        self._init_database()

        # 启动后台分析任务
        self._start_background_tasks()

    def _init_database(self):
        """初始化数据库表"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute("PRAGMA foreign_keys = ON")

            self.conn.executescript("""
                -- 用户会话表
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    user_id TEXT,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME,
                    queries TEXT,
                    responses TEXT,
                    satisfaction_scores TEXT,
                    feedback_comments TEXT,
                    documents_accessed TEXT,
                    session_duration_minutes REAL,
                    query_count INTEGER,
                    avg_query_length REAL,
                    manufacturing_entities TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- 用户行为模式表
                CREATE TABLE IF NOT EXISTS user_behavior_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    common_intents TEXT,
                    preferred_content_types TEXT,
                    frequent_entities TEXT,
                    session_patterns TEXT,
                    satisfaction_trend TEXT,
                    engagement_score REAL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id)
                );

                -- 知识缺口表
                CREATE TABLE IF NOT EXISTS knowledge_gaps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gap_id TEXT UNIQUE NOT NULL,
                    gap_type TEXT NOT NULL,
                    description TEXT,
                    affected_queries TEXT,
                    frequency INTEGER DEFAULT 1,
                    severity TEXT,
                    suggested_improvements TEXT,
                    detected_at DATETIME NOT NULL,
                    related_entities TEXT,
                    resolved BOOLEAN DEFAULT 0,
                    resolved_at DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- 用户反馈表
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_id TEXT UNIQUE NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    query TEXT,
                    response TEXT,
                    satisfaction_score INTEGER,
                    feedback_comment TEXT,
                    improvement_suggestions TEXT,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- 用户意图表
                CREATE TABLE IF NOT EXISTS user_intents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    confidence REAL,
                    entities TEXT,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- 内容访问统计表
                CREATE TABLE IF NOT EXISTS content_access_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_id TEXT NOT NULL,
                    content_type TEXT,
                    access_count INTEGER DEFAULT 1,
                    user_count INTEGER DEFAULT 1,
                    avg_satisfaction REAL,
                    last_accessed DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(content_id)
                );

                -- 创建索引
                CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON user_sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON user_sessions(start_time);
                CREATE INDEX IF NOT EXISTS idx_gaps_severity ON knowledge_gaps(severity);
                CREATE INDEX IF NOT EXISTS idx_feedback_satisfaction ON user_feedback(satisfaction_score);
                CREATE INDEX IF NOT EXISTS idx_intents_timestamp ON user_intents(timestamp);
                CREATE INDEX IF NOT EXISTS idx_content_access_count ON content_access_stats(access_count);
            """)

            logger.info("✅ User analytics database initialized")

        except sqlite3.Error as e:
            logger.error(f"Failed to initialize user analytics database: {e}")
            raise

    def _start_background_tasks(self):
        """启动后台分析任务"""
        asyncio.create_task(self._periodic_session_analysis())
        asyncio.create_task(self._knowledge_gap_detection())
        asyncio.create_task(self._user_behavior_analysis())

    async def track_user_query(self,
                             session_id: str,
                             user_id: Optional[str],
                             query: str,
                             response: str,
                             satisfaction_score: Optional[int] = None,
                             feedback_comment: Optional[str] = None,
                             documents_accessed: Optional[List[str]] = None):
        """跟踪用户查询"""
        try:
            timestamp = datetime.now(timezone.utc)

            # 识别用户意图
            intent, confidence = self._identify_user_intent(query)
            entities = self._extract_manufacturing_entities(query)

            # 保存意图分析
            await self._save_user_intent(session_id, query, intent, confidence, entities)

            # 更新会话信息
            await self._update_session(session_id, user_id, query, response, satisfaction_score, feedback_comment, documents_accessed)

            # 更新内容访问统计
            if documents_accessed:
                await self._update_content_access_stats(documents_accessed, satisfaction_score)

            # 保存反馈
            if satisfaction_score is not None:
                await self._save_user_feedback(
                    session_id, user_id, query, response, satisfaction_score, feedback_comment
                )

            logger.debug(f"Tracked user query: {query[:50]}... (Intent: {intent.value})")

        except Exception as e:
            logger.error(f"Failed to track user query: {e}")

    def _identify_user_intent(self, query: str) -> Tuple[UserIntent, float]:
        """识别用户意图"""
        query_lower = query.lower()

        # 定义意图关键词
        intent_keywords = {
            UserIntent.INFORMATION_SEEKING: [
                'what is', 'tell me about', 'explain', 'describe', 'information about',
                'can you explain', 'what are', 'how does', 'define'
            ],
            UserIntent.PROCEDURE_GUIDANCE: [
                'how to', 'step by step', 'procedure', 'process', 'instructions',
                'guide', 'walkthrough', 'tutorial', 'method'
            ],
            UserIntent.PROBLEM_SOLVING: [
                'problem', 'issue', 'troubleshoot', 'fix', 'solve', 'error',
                'not working', 'failure', 'broken', 'malfunction'
            ],
            UserIntent.SPECIFICATION_LOOKUP: [
                'specification', 'specs', 'dimensions', 'tolerance', 'parameters',
                'technical data', 'measurements', 'requirements'
            ],
            UserIntent.COMPLIANCE_CHECK: [
                'compliance', 'standard', 'regulation', 'certification', 'audit',
                'ISO', 'ANSI', 'requirement', 'mandatory'
            ],
            UserIntent.TRAINING_LEARNING: [
                'learn', 'training', 'education', 'course', 'tutorial',
                'study', 'understand', 'master', 'skill'
            ]
        }

        # 计算每个意图的匹配分数
        intent_scores = {}
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            intent_scores[intent] = score

        # 找到最高分意图
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = min(intent_scores[best_intent] / 3.0, 1.0)  # 标准化置信度
        else:
            best_intent = UserIntent.INFORMATION_SEEKING
            confidence = 0.5

        return best_intent, confidence

    def _extract_manufacturing_entities(self, text: str) -> List[str]:
        """提取制造业实体"""
        entities = []

        for entity_type, patterns in self.manufacturing_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities.extend(matches)

        # 去重并返回
        return list(set(entities))

    async def _save_user_intent(self,
                              session_id: str,
                              query: str,
                              intent: UserIntent,
                              confidence: float,
                              entities: List[str]):
        """保存用户意图"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO user_intents
                (session_id, query, intent, confidence, entities, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                query,
                intent.value,
                confidence,
                json.dumps(entities),
                datetime.now(timezone.utc).isoformat()
            ))
            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Failed to save user intent: {e}")

    async def _update_session(self,
                            session_id: str,
                            user_id: Optional[str],
                            query: str,
                            response: str,
                            satisfaction_score: Optional[int],
                            feedback_comment: Optional[str],
                            documents_accessed: Optional[List[str]]):
        """更新会话信息"""
        try:
            cursor = self.conn.cursor()

            # 检查会话是否存在
            cursor.execute("""
                SELECT queries, responses, satisfaction_scores, feedback_comments,
                       documents_accessed, manufacturing_entities, start_time
                FROM user_sessions
                WHERE session_id = ?
            """, (session_id,))

            row = cursor.fetchone()

            if row:
                # 更新现有会话
                queries = json.loads(row[0]) if row[0] else []
                responses = json.loads(row[1]) if row[1] else []
                satisfaction_scores = json.loads(row[2]) if row[2] else []
                feedback_comments = json.loads(row[3]) if row[3] else []
                docs_accessed = json.loads(row[4]) if row[4] else []
                entities = json.loads(row[5]) if row[5] else []
                start_time = datetime.fromisoformat(row[6])

                queries.append(query)
                responses.append(response)
                if satisfaction_score is not None:
                    satisfaction_scores.append(satisfaction_score)
                if feedback_comment:
                    feedback_comments.append(feedback_comment)
                if documents_accessed:
                    docs_accessed.extend(documents_accessed)

                # 提取新的实体
                new_entities = self._extract_manufacturing_entities(query + " " + response)
                entities.extend(new_entities)
                entities = list(set(entities))  # 去重

                # 计算会话持续时间
                session_duration = (datetime.now(timezone.utc) - start_time).total_seconds() / 60

                # 计算平均查询长度
                avg_query_length = statistics.mean(len(q.split()) for q in queries) if queries else 0

                cursor.execute("""
                    UPDATE user_sessions
                    SET queries = ?, responses = ?, satisfaction_scores = ?,
                        feedback_comments = ?, documents_accessed = ?,
                        manufacturing_entities = ?, session_duration_minutes = ?,
                        query_count = ?, avg_query_length = ?
                    WHERE session_id = ?
                """, (
                    json.dumps(queries),
                    json.dumps(responses),
                    json.dumps(satisfaction_scores),
                    json.dumps(feedback_comments),
                    json.dumps(docs_accessed),
                    json.dumps(entities),
                    session_duration,
                    len(queries),
                    avg_query_length,
                    session_id
                ))

            else:
                # 创建新会话
                entities = self._extract_manufacturing_entities(query + " " + response)

                cursor.execute("""
                    INSERT INTO user_sessions
                    (session_id, user_id, start_time, queries, responses,
                     satisfaction_scores, feedback_comments, documents_accessed,
                     manufacturing_entities, session_duration_minutes, query_count, avg_query_length)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    user_id,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps([query]),
                    json.dumps([response]),
                    json.dumps([satisfaction_score]) if satisfaction_score is not None else json.dumps([]),
                    json.dumps([feedback_comment]) if feedback_comment else json.dumps([]),
                    json.dumps(documents_accessed) if documents_accessed else json.dumps([]),
                    json.dumps(entities),
                    0.0,  # 新会话持续时间为0
                    1,
                    len(query.split())
                ))

            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Failed to update session: {e}")

    async def _save_user_feedback(self,
                                session_id: str,
                                user_id: Optional[str],
                                query: str,
                                response: str,
                                satisfaction_score: int,
                                feedback_comment: Optional[str]):
        """保存用户反馈"""
        try:
            feedback_id = f"feedback_{session_id}_{int(datetime.now().timestamp())}"

            # 从反馈中提取改进建议
            improvement_suggestions = self._extract_improvement_suggestions(feedback_comment) if feedback_comment else []

            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO user_feedback
                (feedback_id, user_id, session_id, query, response,
                 satisfaction_score, feedback_comment, improvement_suggestions, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback_id,
                user_id,
                session_id,
                query,
                response,
                satisfaction_score,
                feedback_comment,
                json.dumps(improvement_suggestions),
                datetime.now(timezone.utc).isoformat()
            ))
            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Failed to save user feedback: {e}")

    def _extract_improvement_suggestions(self, feedback: str) -> List[str]:
        """从反馈中提取改进建议"""
        suggestions = []

        # 定义改进相关关键词
        improvement_keywords = [
            'add', 'include', 'need', 'missing', 'unclear', 'confusing',
            'wrong', 'incorrect', 'update', 'improve', 'better', 'more detail'
        ]

        feedback_lower = feedback.lower()
        sentences = feedback.split('.')

        for sentence in sentences:
            if any(keyword in sentence for keyword in improvement_keywords):
                cleaned_sentence = sentence.strip()
                if cleaned_sentence:
                    suggestions.append(cleaned_sentence)

        return suggestions

    async def _update_content_access_stats(self,
                                         documents_accessed: List[str],
                                         satisfaction_score: Optional[int]):
        """更新内容访问统计"""
        try:
            cursor = self.conn.cursor()

            for doc_id in documents_accessed:
                cursor.execute("""
                    INSERT INTO content_access_stats
                    (content_id, access_count, user_count, avg_satisfaction, last_accessed)
                    VALUES (?, 1, 1, ?, ?)
                    ON CONFLICT(content_id) DO UPDATE SET
                        access_count = access_count + 1,
                        avg_satisfaction = CASE
                            WHEN ? IS NOT NULL THEN
                                (avg_satisfaction * (SELECT COUNT(*) FROM user_feedback WHERE satisfaction_score IS NOT NULL) + ?) /
                                (SELECT COUNT(*) FROM user_feedback WHERE satisfaction_score IS NOT NULL) + 1
                            ELSE avg_satisfaction
                        END,
                        last_accessed = ?
                """, (
                    doc_id,
                    satisfaction_score,
                    datetime.now(timezone.utc).isoformat(),
                    satisfaction_score,
                    satisfaction_score,
                    datetime.now(timezone.utc).isoformat()
                ))

            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Failed to update content access stats: {e}")

    async def _periodic_session_analysis(self):
        """定期会话分析"""
        while True:
            try:
                await asyncio.sleep(self.analysis_interval_hours * 3600)
                await self._analyze_recent_sessions()
                await self._update_user_behavior_patterns()

            except Exception as e:
                logger.error(f"Error in periodic session analysis: {e}")
                await asyncio.sleep(300)  # 出错时5分钟后重试

    async def _analyze_recent_sessions(self):
        """分析最近的会话"""
        try:
            # 获取过去24小时的已完成会话
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)

            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM user_sessions
                WHERE start_time >= ? AND end_time IS NOT NULL
                ORDER BY start_time DESC
            """, (cutoff_time.isoformat(),))

            sessions = cursor.fetchall()

            # 分析每个会话
            for session_data in sessions:
                session = self._row_to_session(session_data)
                await self._analyze_session_patterns(session)

        except Exception as e:
            logger.error(f"Failed to analyze recent sessions: {e}")

    def _row_to_session(self, row) -> UserSession:
        """将数据库行转换为会话对象"""
        return UserSession(
            session_id=row[1],
            user_id=row[2],
            start_time=datetime.fromisoformat(row[3]),
            end_time=datetime.fromisoformat(row[4]) if row[4] else None,
            queries=json.loads(row[5]) if row[5] else [],
            responses=json.loads(row[6]) if row[6] else [],
            satisfaction_scores=json.loads(row[7]) if row[7] else [],
            feedback_comments=json.loads(row[8]) if row[8] else [],
            documents_accessed=json.loads(row[9]) if row[9] else [],
            session_duration_minutes=row[10],
            query_count=row[11],
            avg_query_length=row[12],
            manufacturing_entities=json.loads(row[13]) if row[13] else []
        )

    async def _analyze_session_patterns(self, session: UserSession):
        """分析会话模式"""
        try:
            # 分析满意度趋势
            if session.satisfaction_scores:
                satisfaction_trend = np.polyfit(range(len(session.satisfaction_scores)), session.satisfaction_scores, 1)[0]
            else:
                satisfaction_trend = 0

            # 计算参与度分数
            engagement_score = self._calculate_engagement_score(session)

            # 识别模式
            patterns = {
                "avg_satisfaction": statistics.mean(session.satisfaction_scores) if session.satisfaction_scores else 0,
                "satisfaction_trend": satisfaction_trend,
                "engagement_score": engagement_score,
                "query_complexity": session.avg_query_length,
                "session_length": session.session_duration_minutes,
                "document_usage": len(session.documents_accessed),
                "entity_usage": len(session.manufacturing_entities)
            }

            # 保存分析结果到会话模式中
            # 这里可以根据需要扩展

        except Exception as e:
            logger.error(f"Failed to analyze session patterns: {e}")

    def _calculate_engagement_score(self, session: UserSession) -> float:
        """计算用户参与度分数"""
        try:
            # 基于多个因素计算参与度
            query_factor = min(session.query_count / 10.0, 1.0)  # 查询数量因子
            duration_factor = min(session.session_duration_minutes / 30.0, 1.0)  # 会话时长因子
            document_factor = min(len(session.documents_accessed) / 5.0, 1.0)  # 文档访问因子
            entity_factor = min(len(session.manufacturing_entities) / 3.0, 1.0)  # 实体使用因子

            # 如果有满意度分数，包含满意度因子
            satisfaction_factor = 0
            if session.satisfaction_scores:
                avg_satisfaction = statistics.mean(session.satisfaction_scores)
                satisfaction_factor = avg_satisfaction / 5.0

            # 计算加权平均
            engagement_score = (
                query_factor * 0.3 +
                duration_factor * 0.2 +
                document_factor * 0.2 +
                entity_factor * 0.2 +
                satisfaction_factor * 0.1
            )

            return min(engagement_score, 1.0)

        except Exception as e:
            logger.error(f"Error calculating engagement score: {e}")
            return 0.0

    async def _update_user_behavior_patterns(self):
        """更新用户行为模式"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT DISTINCT user_id FROM user_sessions WHERE user_id IS NOT NULL")
            user_ids = [row[0] for row in cursor.fetchall()]

            for user_id in user_ids:
                pattern = await self._analyze_user_behavior(user_id)
                if pattern:
                    await self._save_user_behavior_pattern(pattern)

        except Exception as e:
            logger.error(f"Failed to update user behavior patterns: {e}")

    async def _analyze_user_behavior(self, user_id: str) -> Optional[UserBehaviorPattern]:
        """分析用户行为"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM user_sessions
                WHERE user_id = ?
                ORDER BY start_time DESC
                LIMIT 50
            """, (user_id,))

            sessions_data = cursor.fetchall()
            if not sessions_data:
                return None

            sessions = [self._row_to_session(row) for row in sessions_data]

            # 分析常见意图
            cursor.execute("""
                SELECT intent, AVG(confidence) as avg_confidence, COUNT(*) as count
                FROM user_intents ui
                JOIN user_sessions us ON ui.session_id = us.session_id
                WHERE us.user_id = ?
                GROUP BY intent
                ORDER BY count DESC
            """, (user_id,))

            intent_results = cursor.fetchall()
            common_intents = [(UserIntent(row[0]), row[1]) for row in intent_results]

            # 分析首选内容类型
            content_types = []
            for session in sessions:
                for doc_id in session.documents_accessed:
                    if '_' in doc_id:
                        content_type = doc_id.split('_')[0]
                        content_types.append(content_type)

            preferred_content_types = list(Counter(content_types).most_common(5))

            # 分析频繁实体
            all_entities = []
            for session in sessions:
                all_entities.extend(session.manufacturing_entities)

            frequent_entities = list(Counter(all_entities).most_common(10))

            # 分析满意度趋势
            all_satisfaction_scores = []
            for session in sessions:
                all_satisfaction_scores.extend(session.satisfaction_scores)

            # 计算参与度分数
            engagement_scores = [self._calculate_engagement_score(session) for session in sessions]
            avg_engagement_score = statistics.mean(engagement_scores) if engagement_scores else 0

            return UserBehaviorPattern(
                user_id=user_id,
                common_intents=common_intents,
                preferred_content_types=[ct[0] for ct in preferred_content_types],
                frequent_entities=[entity[0] for entity in frequent_entities],
                session_patterns={
                    "avg_session_duration": statistics.mean([s.session_duration_minutes for s in sessions if s.session_duration_minutes > 0]) if sessions else 0,
                    "avg_queries_per_session": statistics.mean([s.query_count for s in sessions]) if sessions else 0,
                    "total_sessions": len(sessions)
                },
                satisfaction_trend=all_satisfaction_scores,
                engagement_score=avg_engagement_score
            )

        except Exception as e:
            logger.error(f"Failed to analyze user behavior for {user_id}: {e}")
            return None

    async def _save_user_behavior_pattern(self, pattern: UserBehaviorPattern):
        """保存用户行为模式"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO user_behavior_patterns
                (user_id, common_intents, preferred_content_types, frequent_entities,
                 session_patterns, satisfaction_trend, engagement_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern.user_id,
                json.dumps([(intent.value, confidence) for intent, confidence in pattern.common_intents]),
                json.dumps(pattern.preferred_content_types),
                json.dumps(pattern.frequent_entities),
                json.dumps(pattern.session_patterns),
                json.dumps(pattern.satisfaction_trend),
                pattern.engagement_score
            ))
            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Failed to save user behavior pattern: {e}")

    async def _knowledge_gap_detection(self):
        """知识缺口检测"""
        while True:
            try:
                await asyncio.sleep(self.analysis_interval_hours * 3600 * 2)  # 每2倍分析间隔检测一次
                await self._detect_knowledge_gaps()

            except Exception as e:
                logger.error(f"Error in knowledge gap detection: {e}")
                await asyncio.sleep(3600)  # 出错时1小时后重试

    async def _detect_knowledge_gaps(self):
        """检测知识缺口"""
        try:
            # 分析低满意度查询
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT uf.query, uf.response, uf.satisfaction_score, uf.feedback_comment,
                       ui.entities, COUNT(*) as frequency
                FROM user_feedback uf
                JOIN user_intents ui ON uf.session_id = ui.session_id
                WHERE uf.satisfaction_score <= 2
                    AND uf.timestamp >= date('now', '-7 days')
                GROUP BY uf.query, uf.response
                HAVING frequency >= 2
                ORDER BY frequency DESC
            """)

            low_satisfaction_queries = cursor.fetchall()

            for query_data in low_satisfaction_queries:
                query, response, satisfaction_score, feedback, entities_str, frequency = query_data
                entities = json.loads(entities_str) if entities_str else []

                gap_type = self._classify_knowledge_gap(query, response, feedback, entities)
                severity = self._assess_gap_severity(satisfaction_score, frequency, gap_type)

                await self._create_knowledge_gap_record(
                    gap_type, query, response, entities, frequency, severity, feedback
                )

            # 分析无结果查询
            cursor.execute("""
                SELECT ui.query, ui.entities, COUNT(*) as frequency
                FROM user_intents ui
                LEFT JOIN user_feedback uf ON ui.session_id = uf.session_id
                WHERE uf.session_id IS NULL  -- 没有反馈的查询可能是无结果的
                    AND ui.timestamp >= date('now', '-3 days')
                GROUP BY ui.query
                HAVING frequency >= 3
                ORDER BY frequency DESC
            """)

            no_result_queries = cursor.fetchall()

            for query_data in no_result_queries:
                query, entities_str, frequency = query_data
                entities = json.loads(entities_str) if entities_str else []

                await self._create_knowledge_gap_record(
                    KnowledgeGapType.MISSING_INFORMATION, query, "", entities, frequency,
                    "medium", "No results found for this query"
                )

        except Exception as e:
            logger.error(f"Failed to detect knowledge gaps: {e}")

    def _classify_knowledge_gap(self,
                              query: str,
                              response: str,
                              feedback: Optional[str],
                              entities: List[str]) -> KnowledgeGapType:
        """分类知识缺口类型"""
        query_lower = query.lower()
        response_lower = response.lower() if response else ""
        feedback_lower = feedback.lower() if feedback else ""

        # 检查是否是信息缺失
        if not response or len(response) < 50:
            return KnowledgeGapType.MISSING_INFORMATION

        # 检查是否是程序不清晰
        if any(word in query_lower for word in ['how to', 'procedure', 'step by step', 'instructions']):
            if 'step' not in response_lower and 'procedure' not in response_lower:
                return KnowledgeGapType.UNCLEAR_PROCEDURES

        # 检查是否是细节不足
        if any(word in feedback_lower for word in ['more detail', 'not enough', 'incomplete', 'vague']):
            return KnowledgeGapType.INSUFFICIENT_DETAIL

        # 检查是否是设备特定问题
        if entities:
            equipment_entities = [e for e in entities if any(brand in e.upper() for brand in ['HAAS', 'DMG', 'FANUC', 'SIEMENS'])]
            if equipment_entities and 'equipment' not in response_lower:
                return KnowledgeGapType.EQUIPMENT_SPECIFIC

        # 检查是否是标准合规问题
        if any(word in query_lower for word in ['compliance', 'standard', 'iso', 'ansi', 'requirement']):
            return KnowledgeGapType.STANDARD_COMPLIANCE

        return KnowledgeGapType.OUTDATED_CONTENT

    def _assess_gap_severity(self,
                           satisfaction_score: int,
                           frequency: int,
                           gap_type: KnowledgeGapType) -> str:
        """评估缺口严重性"""
        base_severity = 5 - satisfaction_score  # 满意度越低，严重性越高
        frequency_factor = min(frequency / 10.0, 2.0)  # 频率因子

        # 特定缺口类型的权重
        type_weights = {
            KnowledgeGapType.MISSING_INFORMATION: 2.0,
            KnowledgeGapType.INCORRECT_INFORMATION: 3.0,
            KnowledgeGapType.EQUIPMENT_SPECIFIC: 1.5,
            KnowledgeGapType.STANDARD_COMPLIANCE: 2.5,
            KnowledgeGapType.UNCLEAR_PROCEDURES: 1.2,
            KnowledgeGapType.INSUFFICIENT_DETAIL: 1.0,
            KnowledgeGapType.OUTDATED_CONTENT: 1.3
        }

        severity_score = base_severity * frequency_factor * type_weights.get(gap_type, 1.0)

        if severity_score >= 8:
            return "critical"
        elif severity_score >= 6:
            return "high"
        elif severity_score >= 4:
            return "medium"
        else:
            return "low"

    async def _create_knowledge_gap_record(self,
                                         gap_type: KnowledgeGapType,
                                         query: str,
                                         response: str,
                                         entities: List[str],
                                         frequency: int,
                                         severity: str,
                                         feedback: Optional[str]):
        """创建知识缺口记录"""
        try:
            gap_id = f"gap_{gap_type.value}_{hash(query) % 10000}"

            # 生成描述和建议
            description = f"Knowledge gap detected in {gap_type.value.replace('_', ' ')}"
            if feedback:
                description += f": {feedback}"

            suggested_improvements = self._generate_improvement_suggestions(gap_type, query, entities)

            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO knowledge_gaps
                (gap_id, gap_type, description, affected_queries, frequency,
                 severity, suggested_improvements, detected_at, related_entities)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                gap_id,
                gap_type.value,
                description,
                json.dumps([query]),
                frequency,
                severity,
                json.dumps(suggested_improvements),
                datetime.now(timezone.utc).isoformat(),
                json.dumps(entities)
            ))
            self.conn.commit()

            logger.info(f"Knowledge gap detected: {gap_type.value} - {severity}")

        except sqlite3.Error as e:
            logger.error(f"Failed to create knowledge gap record: {e}")

    def _generate_improvement_suggestions(self,
                                        gap_type: KnowledgeGapType,
                                        query: str,
                                        entities: List[str]) -> List[str]:
        """生成改进建议"""
        suggestions = []

        if gap_type == KnowledgeGapType.MISSING_INFORMATION:
            suggestions.append("Add comprehensive content for this topic")
            suggestions.append("Create detailed documentation covering the query subject")
            if entities:
                suggestions.append(f"Include specific information about {', '.join(entities[:3])}")

        elif gap_type == KnowledgeGapType.UNCLEAR_PROCEDURES:
            suggestions.append("Rewrite procedures with clear step-by-step instructions")
            suggestions.append("Add visual aids and diagrams for better understanding")
            suggestions.append("Include troubleshooting steps and common issues")

        elif gap_type == KnowledgeGapType.INSUFFICIENT_DETAIL:
            suggestions.append("Add more detailed explanations and examples")
            suggestions.append("Include relevant technical specifications")
            suggestions.append("Provide context and background information")

        elif gap_type == KnowledgeGapType.EQUIPMENT_SPECIFIC:
            suggestions.append("Add equipment-specific procedures and guidelines")
            if entities:
                suggestions.append(f"Create dedicated content for {', '.join(entities[:2])}")
            suggestions.append("Include manufacturer specifications and best practices")

        elif gap_type == KnowledgeGapType.STANDARD_COMPLIANCE:
            suggestions.append("Update content to reflect current standards and regulations")
            suggestions.append("Include compliance checklists and audit procedures")
            suggestions.append("Add references to official documentation")

        elif gap_type == KnowledgeGapType.OUTDATED_CONTENT:
            suggestions.append("Review and update outdated information")
            suggestions.append("Add version information and last update dates")
            suggestions.append("Schedule regular content reviews")

        return suggestions

    async def get_user_analytics_report(self,
                                      user_id: Optional[str] = None,
                                      start_date: Optional[datetime] = None,
                                      end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """获取用户分析报告"""
        try:
            if not start_date:
                start_date = datetime.now(timezone.utc) - timedelta(days=30)
            if not end_date:
                end_date = datetime.now(timezone.utc)

            cursor = self.conn.cursor()

            # 基本统计
            cursor.execute("""
                SELECT
                    COUNT(DISTINCT user_id) as total_users,
                    COUNT(*) as total_sessions,
                    AVG(session_duration_minutes) as avg_session_duration,
                    AVG(query_count) as avg_queries_per_session,
                    AVG(CASE WHEN satisfaction_scores != '[]' THEN
                        json_extract(satisfaction_scores, '$[#-1]')
                    END) as avg_satisfaction
                FROM user_sessions
                WHERE start_time >= ? AND start_time <= ?
                    AND (user_id = ? OR ? IS NULL)
            """, (start_date.isoformat(), end_date.isoformat(), user_id, user_id))

            basic_stats = cursor.fetchone()

            # 意图分析
            cursor.execute("""
                SELECT ui.intent, COUNT(*) as count, AVG(ui.confidence) as avg_confidence
                FROM user_intents ui
                JOIN user_sessions us ON ui.session_id = us.session_id
                WHERE us.start_time >= ? AND us.start_time <= ?
                    AND (us.user_id = ? OR ? IS NULL)
                GROUP BY ui.intent
                ORDER BY count DESC
            """, (start_date.isoformat(), end_date.isoformat(), user_id, user_id))

            intent_stats = cursor.fetchall()

            # 知识缺口统计
            cursor.execute("""
                SELECT gap_type, severity, COUNT(*) as count
                FROM knowledge_gaps
                WHERE detected_at >= ? AND detected_at <= ?
                GROUP BY gap_type, severity
                ORDER BY count DESC
            """, (start_date.isoformat(), end_date.isoformat()))

            gap_stats = cursor.fetchall()

            # 内容访问统计
            cursor.execute("""
                SELECT content_type, SUM(access_count) as total_accesses,
                       AVG(avg_satisfaction) as avg_satisfaction
                FROM content_access_stats
                WHERE last_accessed >= ? AND last_accessed <= ?
                GROUP BY content_type
                ORDER BY total_accesses DESC
                LIMIT 10
            """, (start_date.isoformat(), end_date.isoformat()))

            content_stats = cursor.fetchall()

            report = {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "basic_statistics": {
                    "total_users": basic_stats[0] or 0,
                    "total_sessions": basic_stats[1] or 0,
                    "avg_session_duration_minutes": basic_stats[2] or 0,
                    "avg_queries_per_session": basic_stats[3] or 0,
                    "avg_satisfaction": basic_stats[4] or 0
                },
                "intent_analysis": [
                    {
                        "intent": row[0],
                        "count": row[1],
                        "avg_confidence": row[2]
                    }
                    for row in intent_stats
                ],
                "knowledge_gaps": [
                    {
                        "gap_type": row[0],
                        "severity": row[1],
                        "count": row[2]
                    }
                    for row in gap_stats
                ],
                "popular_content": [
                    {
                        "content_type": row[0],
                        "total_accesses": row[1],
                        "avg_satisfaction": row[2]
                    }
                    for row in content_stats
                ]
            }

            return report

        except Exception as e:
            logger.error(f"Failed to generate user analytics report: {e}")
            return {}

    def close(self):
        """关闭连接"""
        try:
            if self.conn:
                self.conn.close()
                logger.info("User analytics connection closed")
        except Exception as e:
            logger.error(f"Error closing user analytics: {e}")

# 全局实例
_user_analytics = None

def get_user_analytics() -> UserAnalytics:
    """获取用户分析系统实例"""
    global _user_analytics
    if _user_analytics is None:
        _user_analytics = UserAnalytics()
    return _user_analytics