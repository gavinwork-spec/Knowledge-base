"""
Advanced Query Decomposition System for Manufacturing RAG

Decomposes complex manufacturing queries into simpler, more answerable sub-queries
with context understanding and relationship mapping.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import sqlite3
import networkx as nx
from sentence_transformers import SentenceTransformer
import spacy
import jieba
import numpy as np

class QueryType(str, Enum):
    """Types of queries in manufacturing domain"""
    SIMPLE_FACT = "simple_fact"
    SPECIFICATION_LOOKUP = "specification_lookup"
    COMPARISON = "comparison"
    TROUBLESHOOTING = "troubleshooting"
    PROCEDURE = "procedure"
    CALCULATION = "calculation"
    RECOMMENDATION = "recommendation"
    ANALYSIS = "analysis"
    RELATIONSHIP = "relationship"

class QueryComplexity(str, Enum):
    """Complexity levels of queries"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

class SubQueryType(str, Enum):
    """Types of sub-queries after decomposition"""
    ENTITY_SPECIFIC = "entity_specific"
    ATTRIBUTE_QUERY = "attribute_query"
    COMPARISON_QUERY = "comparison_query"
    TEMPORAL_QUERY = "temporal_query"
    PROCEDURAL_STEP = "procedural_step"
    CONTEXTUAL_QUERY = "contextual_query"
    CONSTRAINT_QUERY = "constraint_query"

@dataclass
class QueryEntity:
    """Entity identified in a query"""
    entity_id: str
    entity_name: str
    entity_type: str  # "product", "equipment", "material", "process"
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    query_context: str = ""

@dataclass
class SubQuery:
    """A decomposed sub-query"""
    subquery_id: str
    original_query: str
    processed_query: str
    subquery_type: SubQueryType
    query_type: QueryType
    complexity: QueryComplexity
    entities: List[QueryEntity] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    priority: float = 1.0
    expected_answer_type: str = "text"

    # Execution results
    retrieval_queries: List[str] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0
    success: bool = False

@dataclass
class DecompositionPlan:
    """Plan for query decomposition"""
    query_id: str
    original_query: str
    query_type: QueryType
    complexity: QueryComplexity
    decomposition_strategy: str
    sub_queries: List[SubQuery] = field(default_factory=list)
    entities: List[QueryEntity] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    estimated_time: float = 0.0

class ManufacturingQueryDecomposer:
    """
    Advanced query decomposition system for manufacturing knowledge base.
    Decomposes complex queries into simpler, more answerable sub-queries.
    """

    def __init__(self,
                 db_path: str = "knowledge_base.db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 spacy_model: str = "en_core_web_sm"):

        self.db_path = db_path
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize NLP models
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            # Fallback to basic processing if spaCy not available
            self.nlp = None

        # Initialize Jieba for Chinese processing
        jieba.initialize()

        # Manufacturing domain knowledge
        self.manufacturing_entities = self._load_manufacturing_entities()
        self.query_patterns = self._load_query_patterns()
        self.relationship_patterns = self._load_relationship_patterns()

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize database for query decomposition"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Query decomposition results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_decompositions (
                query_id TEXT PRIMARY KEY,
                original_query TEXT NOT NULL,
                query_type TEXT NOT NULL,
                complexity TEXT NOT NULL,
                decomposition_strategy TEXT,
                sub_queries TEXT,
                entities TEXT,
                relationships TEXT,
                estimated_time REAL,
                execution_time REAL,
                success_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Entity knowledge base
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entity_knowledge (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                entity_name TEXT NOT NULL,
                canonical_name TEXT,
                synonyms TEXT,
                attributes TEXT,
                relationships TEXT,
                confidence REAL DEFAULT 1.0,
                mention_count INTEGER DEFAULT 0,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Query patterns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                pattern_regex TEXT NOT NULL,
                description TEXT,
                examples TEXT,
                success_rate REAL DEFAULT 0.0
            )
        ''')

        # Insert manufacturing query patterns
        self._insert_query_patterns()

        conn.commit()
        conn.close()

    def _load_manufacturing_entities(self) -> Dict[str, Dict[str, Any]]:
        """Load manufacturing entity definitions"""

        return {
            "products": {
                "patterns": [
                    r"螺栓|螺母|垫圈|销|键|轴承|密封件",
                    r"part\s+no[:：]\s*([A-Z0-9\-]+)",
                    r"型号\s*([A-Z0-9\-]+)",
                    r"(?:产品|product)\s*([A-Za-z0-9\s\-]+)"
                ],
                "properties": ["specification", "material", "dimensions", "tolerance", "grade"],
                "examples": ["M8x20螺栓", "304不锈钢螺母", "6204轴承"]
            },
            "equipment": {
                "patterns": [
                    r"设备|机器|工具|仪器|装置",
                    r"(?:CNC|数控)\s*([A-Za-z0-9\s]+)",
                    r"机床\s*([A-Za-z0-9]+)",
                    r"设备编号\s*([A-Z0-9\-]+)"
                ],
                "properties": ["model", "manufacturer", "capacity", "status", "maintenance"],
                "examples": ["CNC车床", "压力机", "测量仪器", "设备#001"]
            },
            "materials": {
                "patterns": [
                    r"材料|材质|原料|合金|化合物",
                    r"不锈钢\s*([0-9]+)",
                    r"铝\s*([0-9]+)",
                    r"钢材\s*([A-Z0-9]+)"
                ],
                "properties": ["grade", "composition", "hardness", "strength", "finish"],
                "examples": ["304不锈钢", "6061铝合金", "45号钢"]
            },
            "processes": {
                "patterns": [
                    r"工艺|流程|工序|操作|程序",
                    r"焊接|切割|钻孔|车削|铣削",
                    r"热处理|表面处理|检验|测试",
                    r"(?:加工|manufacturing|fabrication)\s*([A-Za-z0-9\s]+)"
                ],
                "properties": ["parameters", "equipment", "duration", "quality", "standards"],
                "examples": ["热处理工艺", "CNC加工流程", "表面处理程序"]
            },
            "quality": {
                "patterns": [
                    r"质量|检验|测试|认证|标准",
                    r"公差|精度|误差|缺陷",
                    r"合格|不合格|返工|报废",
                    r"Iso\s*([0-9]+)"
                ],
                "properties": ["standard", "criteria", "tolerance", "inspection"],
                "examples": ["ISO9001", "质量检验标准", "公差要求"]
            }
        }

    def _load_query_patterns(self) -> Dict[str, List[str]]:
        """Load query decomposition patterns"""

        return {
            "specification_lookup": [
                r"^(.+?)的规格是什么？",
                r"^(.+?)的技术参数",
                r"查询(.+?)的详细信息",
                r"(.+?)的规格说明"
            ],
            "comparison": [
                r"(.+?)和(.+?)的区别",
                r"(.+?)与(.+?)的对比",
                r"比较(.+?)和(.+?)",
                r"(.+?)和(.+?)哪个更好"
            ],
            "troubleshooting": [
                r"(.+?)出现了(.+?)问题",
                r"如何解决(.+?)的故障",
                r"(.+?)(.+?)的原因",
                r"(.+?)无法(.+？)"
            ],
            "procedure": [
                r"如何(.+？)",
                r"(.+?)的步骤",
                r"(.+?)的工艺流程",
                r"(.+?)的操作方法"
            ],
            "calculation": [
                r"计算(.+?)的(.+？)",
                r"(.+?)的(.+？)是多少",
                r"(.+?)需要多少(.+？)",
                r"(.+?)的成本是多少"
            ]
        }

    def _load_relationship_patterns(self) -> Dict[str, str]:
        """Load relationship patterns between entities"""

        return {
            "compatible_with": "兼容",
            "alternative_to": "替代",
            "part_of": "包含于",
            "used_with": "配合使用",
            "requires": "需要",
            "processed_by": "通过...处理",
            "follows": "遵循",
            "conforms_to": "符合"
        }

    def _insert_query_patterns(self):
        """Insert manufacturing query patterns into database"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            patterns_to_insert = []
            for pattern_type, patterns in self.query_patterns.items():
                for i, pattern in enumerate(patterns):
                    pattern_id = f"{pattern_type}_{i}"
                    patterns_to_insert.append((
                        pattern_id, pattern_type, pattern,
                        f"Pattern for {pattern_type}",
                        json.dumps([pattern]), 0.0
                    ))

            cursor.executemany('''
                INSERT INTO query_patterns
                (pattern_id, pattern_type, pattern_regex, description, examples, success_rate)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', patterns_to_insert)

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Error inserting query patterns: {e}")

    def decompose_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> DecompositionPlan:
        """
        Main method to decompose a complex query.
        """

        print(f"开始分解查询: {query}")

        # Analyze query
        query_analysis = self._analyze_query(query)

        # Determine decomposition strategy
        strategy = self._determine_decomposition_strategy(
            query_analysis['type'], query_analysis['complexity']
        )

        # Extract entities
        entities = self._extract_entities(query, context)

        # Decompose into sub-queries
        sub_queries = self._decompose_into_subqueries(
            query, query_analysis, strategy, entities
        )

        # Build decomposition plan
        plan = DecompositionPlan(
            query_id=f"query_{int(datetime.now().timestamp())}",
            original_query=query,
            query_type=query_analysis['type'],
            complexity=query_analysis['complexity'],
            decomposition_strategy=strategy,
            sub_queries=sub_queries,
            entities=entities,
            relationships=[],
            estimated_time=sum(sq.priority for sq in sub_queries) * 0.1
        )

        # Store decomposition for analytics
        self._store_decomposition_plan(plan)

        print(f"查询分解完成，生成 {len(sub_queries)} 个子查询")
        return plan

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine type and complexity"""

        # Basic analysis
        analysis = {
            'original_query': query,
            'query_length': len(query),
            'word_count': len(query.split()),
            'has_question_words': any(q in query for q in ['什么', '如何', '为什么', '哪里', 'when']),
            'has_comparison_words': any(c in query for c in ['比较', '对比', '区别', 'vs', 'versus']),
            'has_troubleshooting_words': any(t in query for t in ['问题', '故障', '错误', '无法', '解决']),
        }

        # Determine query type
        query_type = QueryType.SIMPLE_FACT
        max_confidence = 0

        for qtype, patterns in self.query_patterns.items():
            confidence = sum(1 for pattern in patterns if re.search(pattern, query, re.IGNORECASE))
            if confidence > max_confidence:
                query_type = QueryType(qtype)
                max_confidence = confidence

        analysis['type'] = query_type
        analysis['type_confidence'] = max_confidence

        # Determine complexity
        if analysis['word_count'] <= 5 and not analysis['has_comparison_words']:
            complexity = QueryComplexity.SIMPLE
        elif analysis['word_count'] <= 15 and len(self._extract_entities(query)) <= 2:
            complexity = QueryComplexity.MODERATE
        elif analysis['word_count'] <= 30 and len(self._extract_entities(query)) <= 5:
            complexity = QueryComplexity.COMPLEX
        else:
            complexity = QueryComplexity.VERY_COMPLEX

        analysis['complexity'] = complexity

        return analysis

    def _determine_decomposition_strategy(self, query_type: QueryType, complexity: QueryComplexity) -> str:
        """Determine decomposition strategy based on query type and complexity"""

        if query_type == QueryType.SIMPLE_FACT:
            return "no_decomposition"
        elif query_type == QueryType.SPECIFICATION_LOOKUP:
            return "entity_extraction"
        elif query_type == QueryType.COMPARISON:
            return "split_comparison"
        elif query_type == QueryType.TROUBLESHOOTING:
            return "causal_analysis"
        elif query_type == QueryType.PROCEDURE:
            return "step_decomposition"
        elif query_type == QueryType.CALCULATION:
            return "parameter_extraction"
        elif complexity == QueryComplexity.COMPLEX:
            return "entity_relationship"
        elif complexity == QueryComplexity.VERY_COMPLEX:
            return "graph_traversal"
        else:
            return "hybrid_decomposition"

    def _extract_entities(self, query: str, context: Optional[Dict[str, Any]]) -> List[QueryEntity]:
        """Extract manufacturing entities from query"""

        entities = []

        for entity_type, entity_config in self.manufacturing_entities.items():
            for pattern in entity_config['patterns']:
                matches = re.finditer(pattern, query, re.IGNORECASE)

                for match in matches:
                    entity_name = match.group(1) if match.groups() else match.group(0)

                    # Create entity ID
                    entity_id = hashlib.md5(f"{entity_type}_{entity_name}".encode()).hexdigest()

                    # Extract attributes if available
                    attributes = {}
                    for prop in entity_config['properties']:
                        prop_pattern = f"{prop}[:：]([^\\n]+)"
                        prop_match = re.search(prop_pattern, query, re.IGNORECASE)
                        if prop_match:
                            attributes[prop] = prop_match.group(1).strip()

                    entity = QueryEntity(
                        entity_id=entity_id,
                        entity_name=entity_name,
                        entity_type=entity_type,
                        attributes=attributes,
                        confidence=0.8,  # Default confidence
                        query_context=query
                    )

                    entities.append(entity)

        # Remove duplicates based on canonical names
        unique_entities = {}
        for entity in entities:
            canonical_name = self._get_canonical_name(entity.entity_name)
            if canonical_name not in unique_entities:
                entity.entity_name = canonical_name
                unique_entities[canonical_name] = entity

        return list(unique_entities.values())

    def _get_canonical_name(self, entity_name: str) -> str:
        """Get canonical name for entity"""

        # Simple canonicalization
        canonical = entity_name.lower().strip()

        # Remove common prefixes/suffixes
        canonical = re.sub(r'^(产品|型号|设备|材料|工艺).*?', '', canonical)
        canonical = re.sub(r'.*(产品|型号|设备|材料|工艺).*$', '', canonical)
        canonical = re.sub(r'[^\w\u4e00-\u9fff]+', '', canonical)

        return canonical.strip()

    def _decompose_into_subqueries(self,
                                    query: str,
                                    query_analysis: Dict[str, Any],
                                    strategy: str,
                                    entities: List[QueryEntity]) -> List[SubQuery]:
        """Decompose query into sub-queries based on strategy"""

        sub_queries = []

        if strategy == "no_decomposition":
            # Simple case - create single sub-query
            sub_query = SubQuery(
                subquery_id="sub_001",
                original_query=query,
                processed_query=query,
                subquery_type=SubQueryType.CONTEXTUAL_QUERY,
                query_type=query_analysis['type'],
                complexity=QueryComplexity.SIMPLE,
                entities=entities,
                priority=1.0,
                expected_answer_type="text"
            )
            sub_queries.append(sub_query)

        elif strategy == "entity_extraction":
            sub_queries = self._entity_extraction_decomposition(query, entities)

        elif strategy == "split_comparison":
            sub_queries = self._comparison_decomposition(query, entities)

        elif strategy == "causal_analysis":
            sub_queries = self._troubleshooting_decomposition(query, entities)

        elif strategy == "step_decomposition":
            sub_queries = self._procedural_decomposition(query, entities)

        elif strategy == "parameter_extraction":
            sub_queries = self._calculation_decomposition(query, entities)

        else:
            # Fallback to hybrid decomposition
            sub_queries = self._hybrid_decomposition(query, entities)

        return sub_queries

    def _entity_extraction_decomposition(self, query: str, entities: List[QueryEntity]) -> List[SubQuery]:
        """Decompose query by extracting entities"""

        sub_queries = []

        if not entities:
            # Fallback to contextual query
            sub_query = SubQuery(
                subquery_id="sub_001",
                original_query=query,
                processed_query=query,
                subquery_type=SubQueryType.CONTEXTUAL_QUERY,
                query_type=QueryType.SPECIFICATION_LOOKUP,
                complexity=QueryComplexity.SIMPLE,
                priority=1.0
            )
            sub_queries.append(sub_query)
            return sub_queries

        # Create entity-specific sub-queries
        for i, entity in enumerate(entities):
            entity_query = f"{entity.entity_name}的规格和参数"
            if entity.attributes:
                attributes_text = "、".join([f"{k}:{v}" for k, v in entity.attributes.items()])
                entity_query += f"，包含{attributes_text}"

            sub_query = SubQuery(
                subquery_id=f"sub_{i+1:03d}",
                original_query=query,
                processed_query=entity_query,
                subquery_type=SubQueryType.ENTITY_SPECIFIC,
                query_type=QueryType.SPECIFICATION_LOOKUP,
                complexity=QueryComplexity.SIMPLE,
                entities=[entity],
                priority=0.9 + (0.1 * i),  # Prioritize first entities
                expected_answer_type="structured"
            )
            sub_queries.append(sub_query)

        # Add contextual sub-query for completeness
        contextual_query = SubQuery(
            subquery_id=f"sub_{len(entities)+1:03d}",
            original_query=query,
            processed_query=query,
            subquery_type=SubQueryType.CONTEXTUAL_QUERY,
            query_type=QueryType.SIMPLE_FACT,
            complexity=QueryComplexity.SIMPLE,
            entities=entities,
            priority=0.5,
            expected_answer_type="summary"
        )
        sub_queries.append(contextual_query)

        return sub_queries

    def _comparison_decomposition(self, query: str, entities: List[QueryEntity]) -> List[SubQuery]:
        """Decompose comparison queries"""

        sub_queries = []

        # Find comparison indicators
        comparison_indicators = [
            (r"(.*?)\s*(和|与|vs|对比|比较|区别)\s*(.*?)", 3),
            (r"对比(.+?)和(.+?)(?:的|的|的差别|的区别)", 2),
            (r"(.+?)比(.+?)如何", 2)
        ]

        matched_indicators = []
        for pattern, num_groups in comparison_indicators:
            matches = list(re.finditer(pattern, query))
            if matches:
                matched_indicators.extend([(match, num_groups)])

        if not matched_indicators:
            # Fallback to simple entity comparison
            if len(entities) >= 2:
                for i in range(len(entities)-1):
                    comp_query = f"对比{entities[i].entity_name}和{entities[i+1].entity_name}"
                    sub_query = SubQuery(
                        subquery_id=f"comp_{i+1:03d}",
                        original_query=query,
                        processed_query=comp_query,
                        subquery_type=SubQueryType.COMPARISON_QUERY,
                        query_type=QueryType.COMPARISON,
                        complexity=QueryComplexity.MODERATE,
                        entities=[entities[i], entities[i+1]],
                        priority=0.8,
                        expected_answer_type="comparison_table"
                    )
                    sub_queries.append(sub_query)

            return sub_queries

        # Process matched comparisons
        for match, num_groups in matched_indicators:
            if num_groups >= 2:
                entity1 = match.group(1).strip()
                entity2 = match.group(2).strip() if len(match.groups()) > 1 else ""

                # Try to find matching entities
                matched_entities = []
                for entity in entities:
                    if entity.entity_name in entity1:
                        matched_entities.append(('entity1', entity))
                    elif entity.entity_name in entity2:
                        matched_entities.append(('entity2', entity))

                # Create comparison sub-query
                if len(matched_entities) >= 1:
                    comp_query = f"对比{entity1}和{entity2}"
                    sub_query = SubQuery(
                        subquery_id=f"comp_{len(sub_queries)+1:03d}",
                        original_query=query,
                        processed_query=comp_query,
                        subquery_type=SubQueryType.COMPARISON_QUERY,
                        query_type=QueryType.COMPARISON,
                        complexity=QueryComplexity.MODERATE,
                        entities=list(dict(matched_entities).values()),
                        priority=0.8,
                        expected_answer_type="comparison_analysis"
                    )
                    sub_queries.append(sub_query)

        return sub_queries

    def _troubleshooting_decomposition(self, query: str, entities: List[Entity]) -> List[SubQuery]:
        """Decompose troubleshooting queries"""

        sub_queries = []

        # Extract problem description
        problem_patterns = [
            r"(.+?)出现了(.+?)(?:问题|故障|错误)",
            r"(如何解决|怎么处理)(.+?)(?:的故障|的问题)",
            r"(.+?)(?:无法|不能)(.+?)",
            r"(.+?)(?:损坏|故障|失效)"
        ]

        matched_problems = []
        for pattern in problem_patterns:
            matches = list(re.finditer(pattern, query))
            matched_problems.extend(matches)

        if not matched_problems:
            # Generic troubleshooting query
            if entities:
                for entity in entities:
                    trouble_query = f"{entity.entity_name}故障排除方法"
                    sub_query = SubQuery(
                        subquery_id=f"trouble_{len(sub_queries)+1:03d}",
                        original_query=query,
                        processed_query=trouble_query,
                        subquery_type=SubQueryType.ENTITY_SPECIFIC,
                        query_type=QueryType.TROUBLESHOOTING,
                        complexity=QueryComplexity.MODERATE,
                        entities=[entity],
                        priority=0.8,
                        expected_answer_type="procedural_steps"
                    )
                    sub_queries.append(sub_query)

            return sub_queries

        # Process matched problems
        for match in matched_problems:
            if match.groups() and len(match.groups()) >= 2:
                subject = match.group(1).strip()
                problem = match.group(2).strip()

                # Try to find matching entity
                subject_entity = None
                for entity in entities:
                    if entity.entity_name in subject:
                        subject_entity = entity
                        break

                # Create troubleshooting sub-query
                if subject_entity:
                    trouble_query = f"{subject_entity}{problem}的解决方案"
                else:
                    trouble_query = f"{subject}{problem}的解决方法"

                sub_query = SubQuery(
                    subquery_id=f"trouble_{len(sub_queries)+1:03d}",
                    original_query=query,
                    processed_query=trouble_query,
                    subquery_type=SubQueryType.TROUBLESHOOTING,
                    query_type=QueryType.TROUBLESHOOTING,
                    complexity=QueryComplexity.COMPLEX,
                    entities=[subject_entity] if subject_entity else [],
                    priority=0.7,
                    expected_answer_type="troubleshooting_guide"
                )
                sub_queries.append(sub_query)

        return sub_queries

    def _procedural_decomposition(self, query: query, entities: List[Entity]) -> List[SubQuery]:
        """Decompose procedural queries"""

        sub_queries = []

        # Identify procedure type
        procedure_patterns = [
            r"如何(.+？)",
            r"(.+?)的步骤",
            r"(.+?)的流程",
            r"(.+?)的方法",
            r"(.+?)的操作规程",
            r"操作(.+？)"
        ]

        matched_procedures = []
        for pattern in procedure_patterns:
            matches = list(re.finditer(pattern, query))
            matched_procedures.extend(matches)

        if not matched_procedures:
            # Generic procedural query
            if entities:
                procedure_query = f"{entities[0].entity_name}的操作流程"
                sub_query = SubQuery(
                    subquery_id="proc_001",
                    original_query=query,
                    processed_query=procedure_query,
                    subquery_type=SubQueryType.PROCEDURAL_STEP,
                    query_type=QueryType.PROCEDURE,
                    complexity=QueryComplexity.MODERATE,
                    entities=entities,
                    priority=0.8,
                    expected_answer_type="step_by_step"
                )
                sub_queries.append(sub_query)

            return sub_queries

        # Process matched procedures
        for match in matched_procedures:
            procedure = match.group(1).strip()

            sub_query = SubQuery(
                subquery_id=f"proc_{len(sub_queries)+1:03d}",
                original_query=query,
                processed_query=procedure,
                subquery_type=SubQueryType.PROCEDURAL_STEP,
                query_type=QueryType.PROCEDURE,
                complexity=QueryComplexity.MODERATE,
                entities=entities,
                priority=0.8,
                expected_answer_type="instructional"
            )
            sub_queries.append(sub_query)

        return sub_queries

    def _calculation_decomposition(self, query: query, entities: List[EntityEntity]) -> List[SubQuery]:
        """Decompose calculation queries"""

        sub_queries = []

        # Identify calculation patterns
        calc_patterns = [
            r"计算(.+?)的(.+？)",
            r"(.+?)的(.+？)是多少",
            r"(.+?)需要多少(.+？)",
            r"(.+?)的成本是多少"
        ]

        matched_calcs = []
        for pattern in calc_patterns:
            matches = list(re.finditer(pattern, query))
            matched_calcs.extend(matches)

        if not matched_calcs:
            # Generic calculation query
            if entities:
                calc_query = f"{entities[0].entity_name}的相关计算"
                sub_query = SubQuery(
                    subquery_id="calc_001",
                    original_query=query,
                    processed_query=calc_query,
                    subquery_type=SubQueryType.CALCULATION_QUERY,
                    query_type=QueryType.CALCULATION,
                    complexity=QueryComplexity.MODERATE,
                    entities=entities,
                    priority=0.7,
                    expected_answer_type="numerical_result"
                )
                sub_queries.append(sub_query)

            return sub_queries

        # Process matched calculations
        for match in matched_calcs:
            if match.groups() and len(match.groups()) >= 2:
                subject = match.group(1).strip()
                calculation = match.group(2).strip()

                # Try to find matching entity
                subject_entity = None
                for entity in entities:
                    if entity.entity_name in subject:
                        subject_entity = entity
                        break

                # Create calculation sub-query
                if subject_entity:
                    calc_query = f"{subject_entity}{calculation}"
                else:
                    calc_query = f"计算{subject}{calculation}"

                sub_query = SubQuery(
                    subquery_id=f"calc_{len(sub_queries)+1:03d}",
                    original_query=query,
                    processed_query=calc_query,
                    subquery_type=SubQueryType.CALCULATION_QUERY,
                    query_type=QueryType.CALCULATION,
                    complexity=QueryComplexity.COMPLEX,
                    entities=[subject_entity] if subject_entity else []],
                    priority=0.7,
                    expected_answer_type="numerical"
                )
                sub_queries.append(sub_query)

        return sub_queries

    def _hybrid_decomposition(self, query: str, entities: List[QueryEntity]) -> List[SubQuery]:
        """Hybrid decomposition combining multiple strategies"""

        sub_queries = []

        # Start with entity extraction
        entity_subqueries = self._entity_extraction_decomposition(query, entities)
        sub_queries.extend(entity_subqueries)

        # Add contextual query for completeness
        if len(entity_subqueries) > 1:
            contextual_query = SubQuery(
                subquery_id="hybrid_001",
                original_query=query,
                processed_query=query,
                subquery_type=SubQueryType.CONTEXTUAL_QUERY,
                query_type=QueryType.ANALYSIS,
                complexity=QueryComplexity.COMPLEX,
                entities=entities,
                priority=0.6,
                expected_answer_type="comprehensive"
            )
            sub_queries.append(contextual_query)

        return sub_queries

    def _store_decomposition_plan(self, plan: DecompositionPlan):
        """Store decomposition plan for analytics"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO query_decompositions
                (query_id, original_query, query_type, complexity,
                 decomposition_strategy, sub_queries, entities,
                 relationships, estimated_time, execution_time, success_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                plan.query_id, plan.original_query, plan.query_type.value,
                plan.complexity.value, plan.decomposition_strategy,
                json.dumps([asdict(sq) for sq in plan.sub_queries]),
                json.dumps([asdict(entity) for entity in plan.entities]),
                json.dumps(plan.relationships),
                plan.estimated_time, 0.0, 0
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Error storing decomposition plan: {e}")

    def get_entity_suggestions(self, partial_entity: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get entity suggestions based on partial input"""

        suggestions = []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT entity_id, entity_type, entity_name, canonical_name,
                       synonyms, mention_count
                FROM entity_knowledge
                WHERE entity_name LIKE ? OR synonyms LIKE ?
                ORDER BY mention_count DESC, confidence DESC
                LIMIT ?
            ''', (f"%{partial_entity}%", f"%{partial_entity}%", limit))

            results = cursor.fetchall()
            conn.close()

            for result in results:
                suggestions.append({
                    'entity_id': result[0],
                    'entity_type': result[1],
                    'entity_name': result[2],
                    'canonical_name': result[3],
                    'synonyms': json.loads(result[4]) if result[4] else [],
                    'mention_count': result[5]
                })

        except Exception as e:
            print(f"Error getting entity suggestions: {e}")

        return suggestions

    def get_decomposition_analytics(self) -> Dict[str, Any]:
        """Get analytics on query decomposition performance"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = cursor.cursor()

            # Overall statistics
            cursor.execute('SELECT COUNT(*) FROM query_decompositions')
            total_decompositions = cursor.fetchone()[0]

            cursor.execute('SELECT AVG(execution_time) FROM query_decompositions WHERE execution_time > 0')
            avg_execution_time = cursor.fetchone()[0] or 0

            # By query type
            cursor.execute('''
                SELECT query_type, COUNT(*), AVG(execution_time)
                FROM query_decompositions
                WHERE execution_time > 0
                GROUP BY query_type
            ''')
            by_type_stats = dict(cursor.fetchall())

            # By complexity
            cursor.execute('''
                SELECT complexity, COUNT(*), AVG(execution_time)
                FROM query_decompositions
                WHERE execution_time > 0
                GROUP BY complexity
            ''')
            by_complexity_stats = dict(cursor.fetchall())

            # Success rate
            cursor.execute('''
                SELECT SUM(success_count), SUM(success_count) / NULLIF(COUNT(*), 0) * 100
                FROM query_decompositions
            ''')
            stats = cursor.fetchone()
            success_rate = stats[0] or 0

            conn.close()

            return {
                "total_decompositions": total_decompositions,
                "average_execution_time": avg_execution_time,
                "success_rate": success_rate,
                "by_query_type": by_type_stats,
                "by_complexity": by_complexity_stats,
                "popular_patterns": []  # TODO: Implement pattern analysis
            }

        except Exception as e:
            print(f"Error getting decomposition analytics: {e}")
            return {}

# Factory function for easy instantiation
def create_query_decomposer(db_path: str = "knowledge_base.db") -> ManufacturingQueryDecomposer:
    """Create a query decomposer instance"""
    return ManufacturingQueryDecomposer(db_path=db_path)

# Example usage and testing
if __name__ == "__main__":
    # Test query decomposer
    decomposer = create_query_decomposer()

    # Test various query types
    test_queries = [
        "不锈钢螺栓M8x20的规格是什么？",
        "304不锈钢和316不锈钢有什么区别？",
        "CNC车床出现了主轴振动过大的故障，如何解决？",
        "如何进行不锈钢零件的热处理工艺？",
        "计算一个M8x20螺栓的成本",
        "比较不同品牌的润滑脂性能",
        "查询设备#001的维护计划"
    ]

    for query in test_queries:
        print(f"\n=== 测试查询: {query} ===")
        plan = decomposer.decompose_query(query)

        print(f"查询类型: {plan.query_type.value}")
        print(f"复杂度: {plan.complexity.value}")
        print(f"分解策略: {plan.decomposition_strategy}")
        print(f"子查询数量: {len(plan.sub_queries)}")
        print(f"识别实体数量: {len(plan.entities)}")

        print(f"\n子查询详情:")
        for i, sub_query in enumerate(plan.sub_queries):
            print(f"{i+1}. {sub_query.processed_query}")
            print(f"   类型: {sub_query.subquery_type.value}")
            print(f"   优先级: {sub_query.priority}")
            print(f"   实体: {[e.entity_name for e in sub_query.entities]}")

    # Get analytics
        analytics = decomposer.get_decomposition_analytics()
        print(f"\n查询分解分析:")
        print(f"- 总分解次数: {analytics['total_decompositions']}")
        print(f"- 平均执行时间: {analytics['average_execution_time']:.3f}秒")
        print(f"- 成功率: {analytics['success_rate']:.1f}%")

    print(f"\n查询分解测试完成！")