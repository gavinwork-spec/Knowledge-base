"""
Advanced Conversation Memory System for Manufacturing RAG

Implements sophisticated conversation memory with context awareness,
entity tracking, and manufacturing-specific context management.
"""

import json
import re
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

class MemoryType(str, Enum):
    """Types of memory storage"""
    SHORT_TERM = "short_term"      # Current conversation
    LONG_TERM = "long_term"        # Historical context
    ENTITY_MEMORY = "entity"      # Entity knowledge
    PROCEDURAL_MEMORY = "procedural"  # Process patterns
    EPISODIC_MEMORY = "episodic"   # Specific episodes

class ContextType(str, Enum):
    """Types of context in manufacturing domain"""
    TECHNICAL_SPEC = "technical_spec"
    PROCESS_FLOW = "process_flow"
    QUALITY_CONTROL = "quality_control"
    EQUIPMENT = "equipment"
    MATERIALS = "materials"
    SAFETY = "safety"
    TROUBLESHOOTING = "troubleshooting"
    MAINTENANCE = "maintenance"
    PRODUCTION = "production"
    SUPPLIER = "supplier"

@dataclass
class EntityMention:
    """Represents an entity mentioned in conversation"""
    entity_id: str
    entity_type: str  # "product", "equipment", "material", "process", etc.
    entity_name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    mention_count: int = 1
    contexts: List[str] = field(default_factory=list)

@dataclass
class ConversationTurn:
    """Represents a single turn in conversation"""
    turn_id: str
    session_id: str
    timestamp: datetime
    user_query: str
    assistant_response: str
    context_used: List[str] = field(default_factory=list)
    entities_mentioned: List[EntityMention] = field(default_factory=list)
    topics_discussed: List[ContextType] = field(default_factory=list)
    user_intent: Optional[str] = None
    query_complexity: float = 0.0  # 0-1 scale
    satisfaction_score: Optional[float] = None

@dataclass
class ContextWindow:
    """Represents a window of context with metadata"""
    window_id: str
    session_id: str
    content: str
    context_type: ContextType
    relevance_score: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ManufacturingConversationMemory:
    """
    Advanced conversation memory system for manufacturing RAG applications.
    Maintains context awareness, entity tracking, and intelligent memory management.
    """

    def __init__(self,
                 db_path: str = "conversation_memory.db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 max_context_window: int = 10,
                 context_retention_days: int = 30):

        self.db_path = db_path
        self.embedding_model = SentenceTransformer(embedding_model)
        self.max_context_window = max_context_window
        self.context_retention_days = context_retention_days

        # Initialize database
        self._init_database()

        # Manufacturing domain knowledge
        self.manufacturing_entities = self._load_manufacturing_entities()
        self.context_patterns = self._load_context_patterns()
        self.intent_patterns = self._load_intent_patterns()

    def _init_database(self):
        """Initialize SQLite database for memory storage"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT,
                session_metadata TEXT
            )
        ''')

        # Conversation turns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_turns (
                turn_id TEXT PRIMARY KEY,
                session_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_query TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                context_used TEXT,
                entities_mentioned TEXT,
                topics_discussed TEXT,
                user_intent TEXT,
                query_complexity REAL DEFAULT 0.0,
                satisfaction_score REAL,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')

        # Context windows table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS context_windows (
                window_id TEXT PRIMARY KEY,
                session_id TEXT,
                content TEXT NOT NULL,
                context_type TEXT NOT NULL,
                relevance_score REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')

        # Entity knowledge table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entity_knowledge (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                entity_name TEXT NOT NULL,
                properties TEXT,
                confidence REAL DEFAULT 1.0,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                mention_count INTEGER DEFAULT 1,
                contexts TEXT,
                embedding_vector BLOB
            )
        ''')

        # User preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                preferred_context_types TEXT,
                entity_focus_areas TEXT,
                conversation_style TEXT,
                learning_preferences TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_turns_session_timestamp ON conversation_turns(session_id, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_context_session_type ON context_windows(session_id, context_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_name_type ON entity_knowledge(entity_name, entity_type)')

        conn.commit()
        conn.close()

    def _load_manufacturing_entities(self) -> Dict[str, Dict[str, Any]]:
        """Load manufacturing-specific entity patterns"""

        return {
            "products": {
                "patterns": [
                    r"产品\s*([A-Z0-9\-]+)",
                    r"型号\s*([A-Z0-9\-]+)",
                    r"part\s*no[:：]\s*([A-Z0-9\-]+)",
                    r"零件\s*([A-Z0-9\-]+)"
                ],
                "properties": ["specification", "material", "dimensions", "tolerance"]
            },
            "equipment": {
                "patterns": [
                    r"设备\s*([A-Z0-9\-]+)",
                    r"machine\s*([A-Z0-9\-]+)",
                    r"机器\s*([A-Z0-9\-]+)",
                    r"仪器\s*([A-Z0-9\-]+)"
                ],
                "properties": ["model", "manufacturer", "capacity", "status"]
            },
            "materials": {
                "patterns": [
                    r"材料\s*([A-Z0-9a-zA-Z]+)",
                    r"材质\s*([A-Z0-9a-zA-Z]+)",
                    r"steel\s*([A-Z0-9]+)",
                    r"aluminum\s*([A-Z0-9]+)"
                ],
                "properties": ["grade", "composition", "hardness", "finish"]
            },
            "processes": {
                "patterns": [
                    r"工艺\s*([A-Z0-9a-zA-Z]+)",
                    r"流程\s*([A-Z0-9a-zA-Z]+)",
                    r"procedure\s*([A-Z0-9a-zA-Z]+)",
                    r"工序\s*([A-Z0-9a-zA-Z]+)"
                ],
                "properties": ["parameters", "equipment", "duration", "quality"]
            }
        }

    def _load_context_patterns(self) -> Dict[ContextType, List[str]]:
        """Load manufacturing context patterns"""

        return {
            ContextType.TECHNICAL_SPEC: [
                r"规格|specification|spec|参数|parameter",
                r"尺寸|dimension|大小|size",
                r"公差|tolerance|精度|precision",
                r"材料|material|材质|composition"
            ],
            ContextType.PROCESS_FLOW: [
                r"工艺|process|procedure|流程",
                r"工序|workflow|steps|步骤",
                r"操作|operation|action|操作",
                r"生产线|production|line"
            ],
            ContextType.QUALITY_CONTROL: [
                r"质量|quality|检验|inspection",
                r"测试|test|testing|测试",
                r"标准|standard|criteria|标准",
                r"缺陷|defect|error|错误"
            ],
            ContextType.EQUIPMENT: [
                r"设备|equipment|machine|机器",
                r"工具|tool|instrument|仪器",
                r"维护|maintenance|保养|upkeep",
                r"故障|fault|breakdown|故障"
            ],
            ContextType.SAFETY: [
                r"安全|safety|保护|protection",
                r"危险|hazard|risk|风险",
                r"防护|protection|prevention|预防",
                r"规程|regulation|procedure|规程"
            ]
        }

    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load user intent patterns"""

        return {
            "search": [
                r"查找|搜索|search|find",
                r"什么|what|how|如何",
                r"哪里|where|when|何时"
            ],
            "compare": [
                r"比较|对比|compare|difference",
                r"优缺点|pros and cons|advantage",
                r"选择|choose|select|selection"
            ],
            "troubleshoot": [
                r"故障|问题|problem|issue",
                r"解决|solve|fix|repair",
                r"诊断|diagnose|check|检查"
            ],
            "specify": [
                r"规格|spec|specification|参数",
                r"要求|requirement|need|需求",
                r"标准|standard|criteria|标准"
            ]
        }

    def create_session(self, session_id: str, user_id: Optional[str] = None) -> bool:
        """Create a new conversation session"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO sessions
                (session_id, user_id, created_at, last_updated)
                VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ''', (session_id, user_id))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f"Error creating session: {e}")
            return False

    def add_conversation_turn(self,
                              session_id: str,
                              user_query: str,
                              assistant_response: str,
                              context_used: Optional[List[str]] = None,
                              satisfaction_score: Optional[float] = None) -> str:
        """Add a conversation turn to memory"""

        turn_id = f"{session_id}_{int(datetime.now().timestamp())}"

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Extract entities and analyze context
            entities = self._extract_entities(user_query + " " + assistant_response)
            topics = self._identify_contexts(user_query + " " + assistant_response)
            intent = self._detect_intent(user_query)
            complexity = self._calculate_query_complexity(user_query)

            # Store conversation turn
            cursor.execute('''
                INSERT INTO conversation_turns
                (turn_id, session_id, user_query, assistant_response,
                 context_used, entities_mentioned, topics_discussed,
                 user_intent, query_complexity, satisfaction_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                turn_id, session_id, user_query, assistant_response,
                json.dumps(context_used or []),
                json.dumps([asdict(entity) for entity in entities]),
                json.dumps([topic.value for topic in topics]),
                intent, complexity, satisfaction_score
            ))

            # Update entities in knowledge base
            for entity in entities:
                self._update_entity_knowledge(entity)

            # Update session timestamp
            cursor.execute('''
                UPDATE sessions SET last_updated = CURRENT_TIMESTAMP
                WHERE session_id = ?
            ''', (session_id,))

            # Store context window
            self._store_context_windows(session_id, user_query, assistant_response, topics)

            conn.commit()
            conn.close()

            return turn_id

        except Exception as e:
            print(f"Error adding conversation turn: {e}")
            return turn_id

    def get_conversation_context(self,
                                session_id: str,
                                max_turns: int = 10,
                                include_entities: bool = True) -> Dict[str, Any]:
        """Get conversation context for RAG retrieval"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get recent conversation turns
            cursor.execute('''
                SELECT * FROM conversation_turns
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (session_id, max_turns))

            turns = cursor.fetchall()

            # Get relevant context windows
            cursor.execute('''
                SELECT * FROM context_windows
                WHERE session_id = ? AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                ORDER BY relevance_score DESC
                LIMIT 5
            ''', (session_id,))

            contexts = cursor.fetchall()

            # Get entity knowledge if requested
            entities = []
            if include_entities:
                # Get recent entities from conversation
                recent_entities = set()
                for turn in turns:
                    if turn[6]:  # entities_mentioned
                        turn_entities = json.loads(turn[6])
                        for entity_data in turn_entities:
                            recent_entities.add(entity_data['entity_id'])

                # Get detailed entity information
                if recent_entities:
                    placeholders = ','.join('?' * len(recent_entities))
                    cursor.execute(f'''
                        SELECT * FROM entity_knowledge
                        WHERE entity_id IN ({placeholders})
                    ''', list(recent_entities))
                    entities = cursor.fetchall()

            conn.close()

            # Format response
            formatted_turns = []
            for turn in reversed(turns):  # Reverse to chronological order
                formatted_turns.append({
                    'turn_id': turn[0],
                    'timestamp': turn[2],
                    'user_query': turn[3],
                    'assistant_response': turn[4],
                    'context_used': json.loads(turn[5]) if turn[5] else [],
                    'entities_mentioned': json.loads(turn[6]) if turn[6] else [],
                    'topics_discussed': json.loads(turn[7]) if turn[7] else [],
                    'user_intent': turn[8],
                    'query_complexity': turn[9]
                })

            formatted_contexts = []
            for context in contexts:
                formatted_contexts.append({
                    'window_id': context[0],
                    'content': context[2],
                    'context_type': context[3],
                    'relevance_score': context[4],
                    'created_at': context[5],
                    'metadata': json.loads(context[7]) if context[7] else {}
                })

            formatted_entities = []
            for entity in entities:
                formatted_entities.append({
                    'entity_id': entity[0],
                    'entity_type': entity[1],
                    'entity_name': entity[2],
                    'properties': json.loads(entity[3]) if entity[3] else {},
                    'confidence': entity[4],
                    'mention_count': entity[6],
                    'contexts': json.loads(entity[7]) if entity[7] else []
                })

            return {
                'session_id': session_id,
                'conversation_turns': formatted_turns,
                'context_windows': formatted_contexts,
                'entity_knowledge': formatted_entities,
                'total_turns': len(formatted_turns),
                'context_coverage': self._calculate_context_coverage(formatted_contexts)
            }

        except Exception as e:
            print(f"Error getting conversation context: {e}")
            return {'session_id': session_id, 'conversation_turns': [], 'context_windows': [], 'entity_knowledge': []}

    def _extract_entities(self, text: str) -> List[EntityMention]:
        """Extract manufacturing entities from text"""

        entities = []
        text_lower = text.lower()

        for entity_type, entity_config in self.manufacturing_entities.items():
            for pattern in entity_config['patterns']:
                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    entity_name = match.group(1) if match.groups() else match.group(0)

                    # Create entity ID
                    entity_id = hashlib.md5(f"{entity_type}_{entity_name}".encode()).hexdigest()

                    entity = EntityMention(
                        entity_id=entity_id,
                        entity_type=entity_type,
                        entity_name=entity_name,
                        properties={prop: "" for prop in entity_config['properties']},
                        confidence=0.8,  # Default confidence
                        mention_count=1
                    )

                    entities.append(entity)

        return entities

    def _identify_contexts(self, text: str) -> List[ContextType]:
        """Identify manufacturing contexts from text"""

        text_lower = text.lower()
        detected_contexts = []

        for context_type, patterns in self.context_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    detected_contexts.append(context_type)
                    break

        return list(set(detected_contexts))  # Remove duplicates

    def _detect_intent(self, query: str) -> Optional[str]:
        """Detect user intent from query"""

        query_lower = query.lower()
        intent_scores = {}

        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            intent_scores[intent] = score

        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        return None

    def _calculate_query_complexity(self, query: str) -> float:
        """Calculate query complexity score (0-1)"""

        complexity = 0.0

        # Length complexity
        length_score = min(1.0, len(query) / 200.0)
        complexity += length_score * 0.3

        # Question complexity
        question_words = ['如何', '怎么', '什么', '为什么', 'where', 'when', 'how', 'what', 'why']
        question_count = sum(1 for word in question_words if word in query.lower())
        question_score = min(1.0, question_count / 3.0)
        complexity += question_score * 0.2

        # Technical complexity (manufacturing terms)
        tech_terms = ['规格', '工艺', '质量', '设备', '材料', '标准', 'procurement', 'specification']
        tech_count = sum(1 for term in tech_terms if term in query.lower())
        tech_score = min(1.0, tech_count / 2.0)
        complexity += tech_score * 0.3

        # Comparative complexity
        comparative_words = ['比较', '对比', 'difference', '区别', 'vs', 'versus', 'compare']
        comp_count = sum(1 for word in comparative_words if word in query.lower())
        comp_score = min(1.0, comp_count / 1.0)
        complexity += comp_score * 0.2

        return min(1.0, complexity)

    def _update_entity_knowledge(self, entity: EntityMention):
        """Update entity knowledge in database"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if entity exists
            cursor.execute('SELECT mention_count FROM entity_knowledge WHERE entity_id = ?', (entity.entity_id,))
            result = cursor.fetchone()

            if result:
                # Update existing entity
                new_count = result[0] + 1
                cursor.execute('''
                    UPDATE entity_knowledge
                    SET mention_count = ?, last_seen = CURRENT_TIMESTAMP, confidence = ?
                    WHERE entity_id = ?
                ''', (new_count, entity.confidence, entity.entity_id))
            else:
                # Insert new entity
                cursor.execute('''
                    INSERT INTO entity_knowledge
                    (entity_id, entity_type, entity_name, properties, confidence,
                     mention_count, contexts)
                    VALUES (?, ?, ?, ?, ?, 1, ?)
                ''', (
                    entity.entity_id, entity.entity_type, entity.entity_name,
                    json.dumps(entity.properties), entity.confidence,
                    json.dumps(entity.contexts)
                ))

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Error updating entity knowledge: {e}")

    def _store_context_windows(self,
                                session_id: str,
                                user_query: str,
                                assistant_response: str,
                                topics: List[ContextType]):
        """Store context windows for future retrieval"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create context windows for each topic
            combined_text = f"User: {user_query}\nAssistant: {assistant_response}"

            for topic in topics:
                window_id = f"{session_id}_{topic.value}_{int(datetime.now().timestamp())}"

                # Set expiration based on context type
                expires_at = datetime.now() + timedelta(days=self.context_retention_days)

                cursor.execute('''
                    INSERT INTO context_windows
                    (window_id, session_id, content, context_type,
                     relevance_score, expires_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    window_id, session_id, combined_text, topic.value,
                    1.0, expires_at, json.dumps({"source": "conversation"})
                ))

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Error storing context windows: {e}")

    def _calculate_context_coverage(self, contexts: List[Dict[str, Any]]) -> float:
        """Calculate context coverage score"""

        if not contexts:
            return 0.0

        context_types = set(ctx['context_type'] for ctx in contexts)
        total_types = len(ContextType)

        return len(context_types) / total_types

    def search_entity_knowledge(self,
                                entity_name: str,
                                entity_type: Optional[str] = None,
                                limit: int = 10) -> List[Dict[str, Any]]:
        """Search for entities in knowledge base"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if entity_type:
                cursor.execute('''
                    SELECT * FROM entity_knowledge
                    WHERE entity_name LIKE ? AND entity_type = ?
                    ORDER BY mention_count DESC, confidence DESC
                    LIMIT ?
                ''', (f"%{entity_name}%", entity_type, limit))
            else:
                cursor.execute('''
                    SELECT * FROM entity_knowledge
                    WHERE entity_name LIKE ?
                    ORDER BY mention_count DESC, confidence DESC
                    LIMIT ?
                ''', (f"%{entity_name}%", limit))

            results = cursor.fetchall()
            conn.close()

            formatted_results = []
            for result in results:
                formatted_results.append({
                    'entity_id': result[0],
                    'entity_type': result[1],
                    'entity_name': result[2],
                    'properties': json.loads(result[3]) if result[3] else {},
                    'confidence': result[4],
                    'first_seen': result[5],
                    'last_seen': result[6],
                    'mention_count': result[7],
                    'contexts': json.loads(result[8]) if result[8] else []
                })

            return formatted_results

        except Exception as e:
            print(f"Error searching entity knowledge: {e}")
            return []

    def cleanup_expired_memory(self, days_to_keep: int = 30) -> int:
        """Clean up expired memory entries"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            # Clean up old context windows
            cursor.execute('''
                DELETE FROM context_windows
                WHERE created_at < ?
            ''', (cutoff_date,))

            # Clean up very old conversation turns (keep longer for analytics)
            old_cutoff = datetime.now() - timedelta(days=90)
            cursor.execute('''
                DELETE FROM conversation_turns
                WHERE timestamp < ?
            ''', (old_cutoff,))

            deleted_count = cursor.rowcount

            conn.commit()
            conn.close()

            return deleted_count

        except Exception as e:
            print(f"Error cleaning up expired memory: {e}")
            return 0

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory usage statistics"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Session statistics
            cursor.execute('SELECT COUNT(*) FROM sessions')
            session_count = cursor.fetchone()[0]

            # Turn statistics
            cursor.execute('SELECT COUNT(*) FROM conversation_turns')
            turn_count = cursor.fetchone()[0]

            # Entity statistics
            cursor.execute('SELECT COUNT(*) FROM entity_knowledge')
            entity_count = cursor.fetchone()[0]

            # Context window statistics
            cursor.execute('SELECT COUNT(*) FROM context_windows')
            context_count = cursor.fetchone()[0]

            # Entity type distribution
            cursor.execute('''
                SELECT entity_type, COUNT(*)
                FROM entity_knowledge
                GROUP BY entity_type
            ''')
            entity_types = dict(cursor.fetchall())

            # Context type distribution
            cursor.execute('''
                SELECT context_type, COUNT(*)
                FROM context_windows
                GROUP BY context_type
            ''')
            context_types = dict(cursor.fetchall())

            conn.close()

            return {
                'total_sessions': session_count,
                'total_turns': turn_count,
                'total_entities': entity_count,
                'total_context_windows': context_count,
                'entity_type_distribution': entity_types,
                'context_type_distribution': context_types,
                'average_turns_per_session': turn_count / max(1, session_count)
            }

        except Exception as e:
            print(f"Error getting memory statistics: {e}")
            return {}

# Factory function for easy instantiation
def create_conversation_memory(db_path: str = "conversation_memory.db") -> ManufacturingConversationMemory:
    """Create a conversation memory instance"""
    return ManufacturingConversationMemory(db_path=db_path)

# Example usage and testing
if __name__ == "__main__":
    # Test conversation memory
    memory = create_conversation_memory()

    # Create a session
    session_id = "test_session_001"
    memory.create_session(session_id, user_id="test_user")

    # Add some conversation turns
    turn1 = memory.add_conversation_turn(
        session_id=session_id,
        user_query="请问不锈钢螺栓M8x20的规格是什么？",
        assistant_response="不锈钢螺栓M8x20的规格包括：直径8mm，长度20mm，材质为304不锈钢，强度等级为8.8级。",
        satisfaction_score=0.9
    )

    turn2 = memory.add_conversation_turn(
        session_id=session_id,
        user_query="这个螺栓的扭矩要求是多少？",
        assistant_response="根据标准规范，M8x20不锈钢螺栓的推荐扭矩为12-15 Nm，需要使用扭矩扳手进行紧固。",
        satisfaction_score=0.95
    )

    # Get conversation context
    context = memory.get_conversation_context(session_id)
    print(f"Conversation Context:")
    print(f"- Total turns: {context['total_turns']}")
    print(f"- Entities mentioned: {len(context['entity_knowledge'])}")
    print(f"- Context windows: {len(context['context_windows'])}")
    print(f"- Context coverage: {context['context_coverage']:.2f}")

    # Search for entities
    entities = memory.search_entity_knowledge("螺栓")
    print(f"\nFound {len(entities)} entities related to '螺栓':")
    for entity in entities[:3]:
        print(f"- {entity['entity_name']} ({entity['entity_type']}) - mentioned {entity['mention_count']} times")

    # Get statistics
    stats = memory.get_memory_statistics()
    print(f"\nMemory Statistics:")
    print(f"- Total sessions: {stats['total_sessions']}")
    print(f"- Total turns: {stats['total_turns']}")
    print(f"- Total entities: {stats['total_entities']}")
    print(f"- Average turns per session: {stats['average_turns_per_session']:.2f}")

    print(f"\nConversation memory test completed!")