#!/usr/bin/env python3
"""
Agent Result Synthesizer
Implements result aggregation, synthesis, and conflict resolution for multi-agent systems.
Provides intelligent merging of results from multiple agents with conflict detection
and resolution strategies.
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SynthesisStrategy(Enum):
    """Result synthesis strategies"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_BASED = "confidence_based"
    EXPERT_CONSENSUS = "expert_consensus"
    HIERARCHICAL_MERGE = "hierarchical_merge"
    CONFLICT_RESOLUTION = "conflict_resolution"
    TEMPORAL_SEQUENCE = "temporal_sequence"

class ConflictType(Enum):
    """Types of conflicts between agent results"""
    VALUE_CONFLICT = "value_conflict"
    FACTUAL_CONFLICT = "factual_conflict"
    INTERPRETATION_CONFLICT = "interpretation_conflict"
    TEMPORAL_CONFLICT = "temporal_conflict"
    SCOPE_CONFLICT = "scope_conflict"
    CERTAINTY_CONFLICT = "certainty_conflict"

class AggregationType(Enum):
    """Types of result aggregation"""
    NUMERICAL_AGGREGATION = "numerical_aggregation"
    TEXTUAL_AGGREGATION = "textual_aggregation"
    CATEGORICAL_AGGREGATION = "categorical_aggregation"
    LIST_AGGREGATION = "list_aggregation"
    STRUCTURED_AGGREGATION = "structured_aggregation"

@dataclass
class AgentResult:
    """Result from an agent execution"""
    agent_id: str
    task_id: str
    result_data: Dict[str, Any]
    confidence: float = 1.0
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    result_type: str = "general"
    reliability_score: float = 1.0
    evidence: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class Conflict:
    """Conflict between agent results"""
    conflict_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conflict_type: ConflictType = ConflictType.VALUE_CONFLICT
    conflicting_results: List[AgentResult] = field(default_factory=list)
    field_path: str = ""
    description: str = ""
    severity: float = 1.0
    resolution_strategy: Optional[str] = None
    resolved: bool = False
    resolution_result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SynthesisResult:
    """Result of synthesis process"""
    synthesis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    synthesized_data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    contributing_agents: List[str] = field(default_factory=list)
    synthesis_strategy: SynthesisStrategy = SynthesisStrategy.MAJORITY_VOTE
    conflicts_detected: List[Conflict] = field(default_factory=list)
    conflicts_resolved: List[Conflict] = field(default_factory=list)
    synthesis_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    quality_score: float = 0.0

class ConflictDetector:
    """Detects conflicts between agent results"""

    def __init__(self):
        self.conflict_detectors = {
            ConflictType.VALUE_CONFLICT: self._detect_value_conflicts,
            ConflictType.FACTUAL_CONFLICT: self._detect_factual_conflicts,
            ConflictType.INTERPRETATION_CONFLICT: self._detect_interpretation_conflicts,
            ConflictType.TEMPORAL_CONFLICT: self._detect_temporal_conflicts,
            ConflictType.SCOPE_CONFLICT: self._detect_scope_conflicts,
            ConflictType.CERTAINTY_CONFLICT: self._detect_certainty_conflicts
        }

    async def detect_conflicts(self, results: List[AgentResult]) -> List[Conflict]:
        """Detect all types of conflicts between results"""
        conflicts = []

        for conflict_type, detector in self.conflict_detectors.items():
            try:
                type_conflicts = await detector(results)
                conflicts.extend(type_conflicts)
            except Exception as e:
                logger.error(f"Error detecting {conflict_type} conflicts: {e}")

        return conflicts

    async def _detect_value_conflicts(self, results: List[AgentResult]) -> List[Conflict]:
        """Detect value conflicts between results"""
        conflicts = []

        # Group results by task and field
        task_fields = defaultdict(lambda: defaultdict(list))
        for result in results:
            for field_path, value in self._flatten_dict(result.result_data).items():
                task_fields[result.task_id][field_path].append((result, value))

        # Check for conflicts in each field
        for task_id, fields in task_fields.items():
            for field_path, values in fields.items():
                if len(values) > 1:
                    conflict = await self._check_value_conflict(field_path, values)
                    if conflict:
                        conflict.task_id = task_id
                        conflicts.append(conflict)

        return conflicts

    async def _check_value_conflict(self, field_path: str, values: List[Tuple[AgentResult, Any]]) -> Optional[Conflict]:
        """Check for conflicts in field values"""
        # Extract unique values
        unique_values = set(str(v[1]) for v in values)

        # If only one unique value, no conflict
        if len(unique_values) <= 1:
            return None

        # Determine if values are genuinely conflicting
        conflicting_values = [v for v in values]
        conflict = Conflict(
            conflict_type=ConflictType.VALUE_CONFLICT,
            field_path=field_path,
            description=f"Conflicting values found at {field_path}: {unique_values}",
            severity=self._calculate_conflict_severity(conflicting_values),
            conflicting_results=[v[0] for v in conflicting_values]
        )

        return conflict

    async def _detect_factual_conflicts(self, results: List[AgentResult]) -> List[Conflict]:
        """Detect factual conflicts between results"""
        conflicts = []

        # Look for statements that contradict each other
        for i, result1 in enumerate(results):
            for j, result2 in enumerate(results[i+1:], i+1):
                fact_conflicts = await self._compare_factual_statements(result1, result2)
                conflicts.extend(fact_conflicts)

        return conflicts

    async def _compare_factual_statements(self, result1: AgentResult, result2: AgentResult) -> List[Conflict]:
        """Compare factual statements between two results"""
        conflicts = []

        # Extract statements from results
        statements1 = self._extract_factual_statements(result1.result_data)
        statements2 = self._extract_factual_statements(result2.result_data)

        # Compare statements for contradictions
        for stmt1 in statements1:
            for stmt2 in statements2:
                if await self._are_contradictory(stmt1, stmt2):
                    conflict = Conflict(
                        conflict_type=ConflictType.FACTUAL_CONFLICT,
                        field_path=stmt1.get('field', ''),
                        description=f"Factual conflict: '{stmt1['statement']}' vs '{stmt2['statement']}'",
                        severity=0.8,
                        conflicting_results=[result1, result2]
                    )
                    conflicts.append(conflict)

        return conflicts

    async def _detect_interpretation_conflicts(self, results: List[AgentResult]) -> List[Conflict]:
        """Detect interpretation conflicts"""
        conflicts = []

        # Look for different interpretations of the same data
        interpretations = defaultdict(list)
        for result in results:
            if 'interpretation' in result.result_data:
                key = result.result_data.get('data_key', 'unknown')
                interpretations[key].append(result)

        for key, interp_results in interpretations.items():
            if len(interp_results) > 1:
                # Check if interpretations are significantly different
                interpretations_text = [r.result_data.get('interpretation', '') for r in interp_results]
                similarity_scores = self._calculate_text_similarity(interpretations_text)

                if min(similarity_scores) < 0.3:  # Low similarity indicates conflict
                    conflict = Conflict(
                        conflict_type=ConflictType.INTERPRETATION_CONFLICT,
                        field_path=f"interpretation.{key}",
                        description=f"Different interpretations for {key}",
                        severity=0.6,
                        conflicting_results=interp_results
                    )
                    conflicts.append(conflict)

        return conflicts

    async def _detect_temporal_conflicts(self, results: List[AgentResult]) -> List[Conflict]:
        """Detect temporal conflicts"""
        conflicts = []

        # Look for conflicting temporal information
        temporal_data = []
        for result in results:
            temporal_fields = self._extract_temporal_data(result.result_data)
            for field, temporal_info in temporal_fields.items():
                temporal_data.append((result, field, temporal_info))

        # Check for temporal inconsistencies
        for i, (result1, field1, time1) in enumerate(temporal_data):
            for j, (result2, field2, time2) in enumerate(temporal_data[i+1:], i+1):
                if field1 == field2 and self._are_temporally_conflicting(time1, time2):
                    conflict = Conflict(
                        conflict_type=ConflictType.TEMPORAL_CONFLICT,
                        field_path=field1,
                        description=f"Temporal conflict in {field1}: {time1} vs {time2}",
                        severity=0.7,
                        conflicting_results=[result1, result2]
                    )
                    conflicts.append(conflict)

        return conflicts

    async def _detect_scope_conflicts(self, results: List[AgentResult]) -> List[Conflict]:
        """Detect scope conflicts"""
        conflicts = []

        # Check for different scopes of analysis
        scopes = []
        for result in results:
            scope = result.result_data.get('scope', 'unknown')
            scopes.append((result, scope))

        # Check if scopes are incompatible
        unique_scopes = set(scope for _, scope in scopes)
        if len(unique_scopes) > 1:
            conflicting_results = [r for r, s in scopes]
            conflict = Conflict(
                conflict_type=ConflictType.SCOPE_CONFLICT,
                field_path="scope",
                description=f"Different analysis scopes: {unique_scopes}",
                severity=0.5,
                conflicting_results=conflicting_results
            )
            conflicts.append(conflict)

        return conflicts

    async def _detect_certainty_conflicts(self, results: List[AgentResult]) -> List[Conflict]:
        """Detect certainty conflicts"""
        conflicts = []

        # Check for conflicting certainty levels
        for result in results:
            confidence = result.confidence
            stated_certainty = result.result_data.get('certainty', 'unknown')

            # High confidence with low stated certainty or vice versa
            if confidence > 0.8 and stated_certainty in ['low', 'uncertain']:
                conflict = Conflict(
                    conflict_type=ConflictType.CERTAINTY_CONFLICT,
                    field_path="certainty",
                    description=f"Confidence mismatch: {confidence} vs {stated_certainty}",
                    severity=0.4,
                    conflicting_results=[result]
                )
                conflicts.append(conflict)

        return conflicts

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _extract_factual_statements(self, data: Dict) -> List[Dict]:
        """Extract factual statements from result data"""
        statements = []

        # Look for key-value pairs that represent facts
        for key, value in self._flatten_dict(data).items():
            if isinstance(value, (str, int, float, bool)):
                statement = {
                    'field': key,
                    'statement': f"{key}: {value}",
                    'value': value,
                    'type': type(value).__name__
                }
                statements.append(statement)

        return statements

    async def _are_contradictory(self, stmt1: Dict, stmt2: Dict) -> bool:
        """Check if two statements are contradictory"""
        # Simple contradiction detection
        if stmt1['field'] != stmt2['field']:
            return False

        value1, value2 = stmt1['value'], stmt2['value']

        # Check for direct contradictions
        contradictions = [
            (True, False), (False, True),
            ('yes', 'no'), ('no', 'yes'),
            ('present', 'absent'), ('absent', 'present')
        ]

        return (value1, value2) in contradictions

    def _calculate_text_similarity(self, texts: List[str]) -> List[float]:
        """Calculate similarity scores between texts"""
        if len(texts) < 2:
            return [1.0]

        try:
            vectorizer = TfidfVectorizer().fit_transform(texts)
            similarity_matrix = cosine_similarity(vectorizer)

            # Get upper triangle (excluding diagonal)
            similarities = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    similarities.append(similarity_matrix[i][j])

            return similarities if similarities else [1.0]
        except:
            return [0.5]  # Default similarity if calculation fails

    def _extract_temporal_data(self, data: Dict) -> Dict[str, Any]:
        """Extract temporal data from result"""
        temporal_data = {}

        for key, value in self._flatten_dict(data).items():
            if any(temporal_keyword in key.lower() for temporal_keyword in
                   ['time', 'date', 'timestamp', 'duration', 'period', 'when']):
                temporal_data[key] = value

        return temporal_data

    def _are_temporally_conflicting(self, time1: Any, time2: Any) -> bool:
        """Check if two temporal values conflict"""
        try:
            # Convert to datetime if possible
            if isinstance(time1, str):
                time1 = datetime.fromisoformat(time1.replace('Z', '+00:00'))
            if isinstance(time2, str):
                time2 = datetime.fromisoformat(time2.replace('Z', '+00:00'))

            if isinstance(time1, datetime) and isinstance(time2, datetime):
                # Check if times are significantly different
                time_diff = abs((time1 - time2).total_seconds())
                return time_diff > 3600  # 1 hour difference considered conflict

        except:
            pass

        return False

    def _calculate_conflict_severity(self, conflicting_values: List[Tuple[AgentResult, Any]]) -> float:
        """Calculate severity of a conflict"""
        # Base severity on number of conflicting agents and their reliability
        num_agents = len(conflicting_values)
        avg_reliability = sum(v[0].reliability_score for v in conflicting_values) / num_agents

        # More agents and higher reliability = higher severity
        severity = (num_agents / 10) * avg_reliability
        return min(severity, 1.0)

class ConflictResolver:
    """Resolves conflicts between agent results"""

    def __init__(self):
        self.resolution_strategies = {
            ConflictType.VALUE_CONFLICT: self._resolve_value_conflict,
            ConflictType.FACTUAL_CONFLICT: self._resolve_factual_conflict,
            ConflictType.INTERPRETATION_CONFLICT: self._resolve_interpretation_conflict,
            ConflictType.TEMPORAL_CONFLICT: self._resolve_temporal_conflict,
            ConflictType.SCOPE_CONFLICT: self._resolve_scope_conflict,
            ConflictType.CERTAINTY_CONFLICT: self._resolve_certainty_conflict
        }

    async def resolve_conflicts(self, conflicts: List[Conflict]) -> List[Conflict]:
        """Resolve all conflicts"""
        resolved_conflicts = []

        for conflict in conflicts:
            try:
                resolver = self.resolution_strategies.get(conflict.conflict_type)
                if resolver:
                    resolution = await resolver(conflict)
                    if resolution:
                        conflict.resolved = True
                        conflict.resolution_result = resolution
                        resolved_conflicts.append(conflict)
            except Exception as e:
                logger.error(f"Error resolving conflict {conflict.conflict_id}: {e}")

        return resolved_conflicts

    async def _resolve_value_conflict(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Resolve value conflict using voting or weighted average"""
        if not conflict.conflicting_results:
            return None

        values = [(r.confidence * r.reliability_score,
                   self._get_nested_value(r.result_data, conflict.field_path))
                  for r in conflict.conflicting_results]

        # Separate numeric and non-numeric values
        numeric_values = [(w, v) for w, v in values if isinstance(v, (int, float))]
        text_values = [(w, str(v)) for w, v in values if not isinstance(v, (int, float))]

        if numeric_values:
            # Use weighted average for numeric values
            total_weight = sum(w for w, _ in numeric_values)
            weighted_sum = sum(w * v for w, v in numeric_values)
            resolved_value = weighted_sum / total_weight if total_weight > 0 else 0
        else:
            # Use weighted voting for text values
            value_votes = defaultdict(float)
            for weight, value in text_values:
                value_votes[value] += weight

            # Select value with highest weight
            resolved_value = max(value_votes.items(), key=lambda x: x[1])[0]

        return {
            'resolved_value': resolved_value,
            'resolution_method': 'weighted_voting',
            'votes_cast': len(values),
            'confidence': sum(w for w, _ in values) / len(values)
        }

    async def _resolve_factual_conflict(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Resolve factual conflict using evidence and reliability"""
        if len(conflict.conflicting_results) < 2:
            return None

        # Choose result with highest reliability score
        best_result = max(conflict.conflicting_results, key=lambda r: r.reliability_score)

        return {
            'resolved_value': self._get_nested_value(best_result.result_data, conflict.field_path),
            'resolution_method': 'reliability_based',
            'chosen_agent': best_result.agent_id,
            'reliability_score': best_result.reliability_score
        }

    async def _resolve_interpretation_conflict(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Resolve interpretation conflict using synthesis"""
        interpretations = []
        for result in conflict.conflicting_results:
            interp = self._get_nested_value(result.result_data, conflict.field_path)
            if interp:
                interpretations.append((result.confidence, interp))

        if not interpretations:
            return None

        # Create synthesized interpretation
        synthesized = await self._synthesize_interpretations(interpretations)

        return {
            'resolved_value': synthesized,
            'resolution_method': 'synthesis',
            'source_count': len(interpretations)
        }

    async def _resolve_temporal_conflict(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Resolve temporal conflict"""
        temporal_values = []
        for result in conflict.conflicting_results:
            value = self._get_nested_value(result.result_data, conflict.field_path)
            temporal_values.append((result.confidence, value))

        # Choose most recent or most reliable temporal value
        if temporal_values:
            best_value = max(temporal_values, key=lambda x: x[0])
            return {
                'resolved_value': best_value[1],
                'resolution_method': 'confidence_based',
                'chosen_confidence': best_value[0]
            }

        return None

    async def _resolve_scope_conflict(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Resolve scope conflict by expanding to include all scopes"""
        scopes = []
        for result in conflict.conflicting_results:
            scope = self._get_nested_value(result.result_data, conflict.field_path)
            if scope:
                scopes.append(scope)

        if scopes:
            # Use union of all scopes
            resolved_scope = f"combined_scope_{'_'.join(sorted(set(scopes)))}"
            return {
                'resolved_value': resolved_scope,
                'resolution_method': 'scope_union',
                'original_scopes': scopes
            }

        return None

    async def _resolve_certainty_conflict(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Resolve certainty conflict by adjusting confidence"""
        if not conflict.conflicting_results:
            return None

        # Use the lower of agent confidence and stated certainty
        result = conflict.conflicting_results[0]
        agent_confidence = result.confidence
        stated_certainty = self._get_nested_value(result.result_data, conflict.field_path)

        # Map certainty levels to numeric values
        certainty_map = {
            'high': 0.8, 'certain': 0.9,
            'medium': 0.5, 'moderate': 0.5,
            'low': 0.2, 'uncertain': 0.1
        }

        certainty_numeric = certainty_map.get(str(stated_certainty).lower(), 0.5)
        resolved_confidence = min(agent_confidence, certainty_numeric)

        return {
            'resolved_value': resolved_confidence,
            'resolution_method': 'confidence_adjustment',
            'original_agent_confidence': agent_confidence,
            'original_stated_certainty': stated_certainty
        }

    def _get_nested_value(self, data: Dict, field_path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = field_path.split('.')
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    async def _synthesize_interpretations(self, interpretations: List[Tuple[float, str]]) -> str:
        """Synthesize multiple interpretations into one"""
        if not interpretations:
            return ""

        # Sort by confidence
        interpretations.sort(key=lambda x: x[0], reverse=True)

        # Take the highest confidence interpretation as base
        base_interpretation = interpretations[0][1]

        # Add qualifiers from other interpretations
        additional_aspects = []
        for confidence, interp in interpretations[1:]:
            if confidence > 0.6:  # Only include high-confidence additional interpretations
                additional_aspects.append(interp)

        if additional_aspects:
            return f"{base_interpretation} (Additional aspects: {'; '.join(additional_aspects)})"
        else:
            return base_interpretation

class ResultSynthesizer:
    """Main result synthesizer that coordinates conflict detection and resolution"""

    def __init__(self):
        self.conflict_detector = ConflictDetector()
        self.conflict_resolver = ConflictResolver()
        self.synthesis_strategies = {
            SynthesisStrategy.MAJORITY_VOTE: self._majority_vote_synthesis,
            SynthesisStrategy.WEIGHTED_AVERAGE: self._weighted_average_synthesis,
            SynthesisStrategy.CONFIDENCE_BASED: self._confidence_based_synthesis,
            SynthesisStrategy.EXPERT_CONSENSUS: self._expert_consensus_synthesis,
            SynthesisStrategy.HIERARCHICAL_MERGE: self._hierarchical_merge_synthesis,
            SynthesisStrategy.CONFLICT_RESOLUTION: self._conflict_resolution_synthesis,
            SynthesisStrategy.TEMPORAL_SEQUENCE: self._temporal_sequence_synthesis
        }

    async def synthesize_results(self,
                                results: List[AgentResult],
                                strategy: SynthesisStrategy = SynthesisStrategy.WEIGHTED_AVERAGE,
                                context: Optional[Dict] = None) -> SynthesisResult:
        """Synthesize multiple agent results into a single result"""

        if not results:
            return SynthesisResult(
                synthesized_data={},
                confidence=0.0,
                synthesis_strategy=strategy,
                quality_score=0.0
            )

        logger.info(f"Synthesizing {len(results)} results using {strategy.value} strategy")

        # Detect conflicts
        conflicts = await self.conflict_detector.detect_conflicts(results)

        # Resolve conflicts
        resolved_conflicts = await self.conflict_resolver.resolve_conflicts(conflicts)
        unresolved_conflicts = [c for c in conflicts if not c.resolved]

        # Apply synthesis strategy
        synthesizer = self.synthesis_strategies.get(strategy, self._weighted_average_synthesis)
        synthesized_data = await synthesizer(results, resolved_conflicts, context)

        # Calculate confidence and quality scores
        confidence = self._calculate_synthesis_confidence(results, synthesized_data, resolved_conflicts)
        quality_score = self._calculate_quality_score(results, synthesized_data, conflicts, resolved_conflicts)

        synthesis_result = SynthesisResult(
            synthesized_data=synthesized_data,
            confidence=confidence,
            contributing_agents=[r.agent_id for r in results],
            synthesis_strategy=strategy,
            conflicts_detected=conflicts,
            conflicts_resolved=resolved_conflicts,
            synthesis_metadata={
                'input_results_count': len(results),
                'conflicts_detected_count': len(conflicts),
                'conflicts_resolved_count': len(resolved_conflicts),
                'unresolved_conflicts_count': len(unresolved_conflicts),
                'context': context or {}
            },
            quality_score=quality_score
        )

        logger.info(f"Synthesis completed with confidence {confidence:.2f}, quality {quality_score:.2f}")
        return synthesis_result

    async def _majority_vote_synthesis(self,
                                     results: List[AgentResult],
                                     resolved_conflicts: List[Conflict],
                                     context: Optional[Dict]) -> Dict[str, Any]:
        """Synthesize using majority voting"""
        # Flatten all result data
        all_fields = defaultdict(list)
        for result in results:
            for field_path, value in self._flatten_dict(result.result_data).items():
                all_fields[field_path].append((result.agent_id, value))

        synthesized = {}
        for field_path, values in all_fields.items():
            # Count occurrences of each value
            value_counts = Counter(str(v) for _, v in values)

            # Get majority value
            majority_value, count = value_counts.most_common(1)[0]

            # Set in synthesized result
            self._set_nested_value(synthesized, field_path, values[0][1])

        return synthesized

    async def _weighted_average_synthesis(self,
                                         results: List[AgentResult],
                                         resolved_conflicts: List[Conflict],
                                         context: Optional[Dict]) -> Dict[str, Any]:
        """Synthesize using weighted averaging"""
        # Apply conflict resolutions first
        for conflict in resolved_conflicts:
            if conflict.resolution_result and conflict.field_path:
                self._set_nested_value(
                    self._get_base_synthesis(results),
                    conflict.field_path,
                    conflict.resolution_result.get('resolved_value')
                )

        # Weight fields by agent confidence and reliability
        field_weights = defaultdict(lambda: defaultdict(float))
        field_values = defaultdict(list)

        for result in results:
            weight = result.confidence * result.reliability_score
            for field_path, value in self._flatten_dict(result.result_data).items():
                if isinstance(value, (int, float)):
                    field_weights[field_path]['numeric'] += weight
                    field_values[field_path].append((weight, value))
                else:
                    field_weights[field_path][str(value)] += weight

        synthesized = {}
        for field_path, values in field_values.items():
            if all(isinstance(v, (int, float)) for _, v in values):
                # Numeric weighted average
                total_weight = sum(w for w, v in values)
                weighted_sum = sum(w * v for w, v in values)
                avg_value = weighted_sum / total_weight if total_weight > 0 else 0
                self._set_nested_value(synthesized, field_path, avg_value)
            else:
                # For non-numeric, use highest weighted value
                max_weight_value = max(values, key=lambda x: x[0])
                self._set_nested_value(synthesized, field_path, max_weight_value[1])

        return synthesized

    async def _confidence_based_synthesis(self,
                                         results: List[AgentResult],
                                         resolved_conflicts: List[Conflict],
                                         context: Optional[Dict]) -> Dict[str, Any]:
        """Synthesize based on agent confidence"""
        # Sort results by confidence
        sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)

        # Start with highest confidence result as base
        synthesized = self._deep_copy_dict(sorted_results[0].result_data)

        # Merge other results where they add value
        for result in sorted_results[1:]:
            if result.confidence > 0.7:  # Only merge high-confidence results
                synthesized = self._merge_structured_data(synthesized, result.result_data)

        return synthesized

    async def _expert_consensus_synthesis(self,
                                          results: List[AgentResult],
                                          resolved_conflicts: List[Conflict],
                                          context: Optional[Dict]) -> Dict[str, Any]:
        """Synthesize based on expert agent consensus"""
        # Identify expert agents (high reliability)
        expert_results = [r for r in results if r.reliability_score > 0.8]

        if expert_results:
            # Use only expert results for synthesis
            return await self._weighted_average_synthesis(expert_results, resolved_conflicts, context)
        else:
            # Fall back to weighted average of all results
            return await self._weighted_average_synthesis(results, resolved_conflicts, context)

    async def _hierarchical_merge_synthesis(self,
                                          results: List[AgentResult],
                                          resolved_conflicts: List[Conflict],
                                          context: Optional[Dict]) -> Dict[str, Any]:
        """Synthesize using hierarchical merging"""
        # Build a hierarchy based on result complexity
        result_complexities = []
        for result in results:
            complexity = self._calculate_result_complexity(result.result_data)
            result_complexities.append((complexity, result))

        # Sort by complexity (simple to complex)
        result_complexities.sort(key=lambda x: x[0])

        # Start with simplest result and build up
        synthesized = {}
        for _, result in result_complexities:
            synthesized = self._merge_structured_data(synthesized, result.result_data)

        return synthesized

    async def _conflict_resolution_synthesis(self,
                                           results: List[AgentResult],
                                           resolved_conflicts: List[Conflict],
                                           context: Optional[Dict]) -> Dict[str, Any]:
        """Synthesize prioritizing conflict resolution"""
        # Start with base synthesis
        base_synthesis = await self._weighted_average_synthesis(results, resolved_conflicts, context)

        # Apply conflict resolutions
        for conflict in resolved_conflicts:
            if conflict.resolution_result and conflict.field_path:
                self._set_nested_value(
                    base_synthesis,
                    conflict.field_path,
                    conflict.resolution_result.get('resolved_value')
                )

        return base_synthesis

    async def _temporal_sequence_synthesis(self,
                                          results: List[AgentResult],
                                          resolved_conflicts: List[Conflict],
                                          context: Optional[Dict]) -> Dict[str, Any]:
        """Synthesize maintaining temporal sequence"""
        # Sort results by timestamp
        sorted_results = sorted(results, key=lambda r: r.timestamp)

        # Build chronological synthesis
        synthesized = {
            'temporal_sequence': [],
            'latest_results': {},
            'evolution_summary': {}
        }

        for result in sorted_results:
            synthesized['temporal_sequence'].append({
                'timestamp': result.timestamp.isoformat(),
                'agent_id': result.agent_id,
                'confidence': result.confidence,
                'data': result.result_data
            })

        # Latest result takes precedence
        if sorted_results:
            synthesized['latest_results'] = sorted_results[-1].result_data

        return synthesized

    def _calculate_synthesis_confidence(self,
                                       results: List[AgentResult],
                                       synthesized_data: Dict,
                                       resolved_conflicts: List[Conflict]) -> float:
        """Calculate confidence score for synthesis"""
        if not results:
            return 0.0

        # Base confidence from contributing agents
        base_confidence = sum(r.confidence for r in results) / len(results)

        # Adjust based on conflict resolution
        total_conflicts = len(resolved_conflicts) + len([c for c in resolved_conflicts if not c.resolved])
        conflict_penalty = total_conflicts * 0.1

        # Adjust based on agent agreement
        agreement_score = self._calculate_agent_agreement(results)
        agreement_bonus = agreement_score * 0.2

        final_confidence = base_confidence - conflict_penalty + agreement_bonus
        return max(0.0, min(1.0, final_confidence))

    def _calculate_quality_score(self,
                                results: List[AgentResult],
                                synthesized_data: Dict,
                                conflicts: List[Conflict],
                                resolved_conflicts: List[Conflict]) -> float:
        """Calculate quality score for synthesis"""
        if not results:
            return 0.0

        quality_factors = {
            'completeness': self._calculate_completeness(results, synthesized_data),
            'consistency': self._calculate_consistency(results, conflicts),
            'coherence': self._calculate_coherence(synthesized_data),
            'reliability': self._calculate_reliability(results)
        }

        # Weighted average of quality factors
        weights = {'completeness': 0.3, 'consistency': 0.3, 'coherence': 0.2, 'reliability': 0.2}

        quality_score = sum(
            quality_factors[factor] * weights[factor]
            for factor in weights
        )

        return max(0.0, min(1.0, quality_score))

    def _calculate_completeness(self, results: List[AgentResult], synthesized_data: Dict) -> float:
        """Calculate completeness of synthesis"""
        # Count unique fields from all results
        all_fields = set()
        for result in results:
            all_fields.update(self._flatten_dict(result.result_data).keys())

        # Count fields in synthesized data
        synthesized_fields = set(self._flatten_dict(synthesized_data).keys())

        if not all_fields:
            return 1.0

        return len(synthesized_fields) / len(all_fields)

    def _calculate_consistency(self, results: List[AgentResult], conflicts: List[Conflict]) -> float:
        """Calculate consistency of synthesis"""
        if not conflicts:
            return 1.0

        # Penalty based on number and severity of conflicts
        total_severity = sum(c.severity for c in conflicts)
        max_possible_severity = len(conflicts)

        if max_possible_severity == 0:
            return 1.0

        consistency = 1.0 - (total_severity / max_possible_severity)
        return max(0.0, consistency)

    def _calculate_coherence(self, synthesized_data: Dict) -> float:
        """Calculate coherence of synthesized data"""
        # Simple coherence check based on data structure
        try:
            # Check for valid JSON structure
            json.dumps(synthesized_data)

            # Check for reasonable depth and complexity
            depth = self._calculate_dict_depth(synthesized_data)

            if depth > 10:
                return 0.5  # Too deeply nested
            elif depth < 1:
                return 0.8  # Too shallow

            return 1.0  # Good structure

        except:
            return 0.0  # Invalid structure

    def _calculate_reliability(self, results: List[AgentResult]) -> float:
        """Calculate reliability based on agent reliability scores"""
        if not results:
            return 0.0

        return sum(r.reliability_score for r in results) / len(results)

    def _calculate_agent_agreement(self, results: List[AgentResult]) -> float:
        """Calculate how much agents agree on results"""
        if len(results) < 2:
            return 1.0

        # Simple agreement based on field similarity
        agreements = []
        all_fields = defaultdict(list)

        for result in results:
            for field_path, value in self._flatten_dict(result.result_data).items():
                all_fields[field_path].append(str(value))

        for field_path, values in all_fields.items():
            if len(values) > 1:
                # Calculate similarity of values
                similarity = self._calculate_string_similarity(values)
                agreements.append(similarity)

        if not agreements:
            return 1.0

        return sum(agreements) / len(agreements)

    def _calculate_string_similarity(self, strings: List[str]) -> float:
        """Calculate similarity between strings"""
        if len(strings) < 2:
            return 1.0

        similarities = []
        for i in range(len(strings)):
            for j in range(i + 1, len(strings)):
                similarity = difflib.SequenceMatcher(None, strings[i], strings[j]).ratio()
                similarities.append(similarity)

        return sum(similarities) / len(similarities)

    def _calculate_result_complexity(self, data: Dict) -> int:
        """Calculate complexity of result data"""
        complexity = 0

        def count_complexity(obj):
            nonlocal complexity
            if isinstance(obj, dict):
                complexity += len(obj)
                for value in obj.values():
                    count_complexity(value)
            elif isinstance(obj, list):
                complexity += len(obj)
                for item in obj:
                    count_complexity(item)
            else:
                complexity += 1

        count_complexity(data)
        return complexity

    def _calculate_dict_depth(self, d: Dict, current_depth: int = 0) -> int:
        """Calculate maximum depth of nested dictionary"""
        if not isinstance(d, dict):
            return current_depth

        max_depth = current_depth
        for value in d.values():
            if isinstance(value, dict):
                depth = self._calculate_dict_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)

        return max_depth

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _set_nested_value(self, data: Dict, field_path: str, value: Any):
        """Set value in nested dictionary using dot notation"""
        keys = field_path.split('.')
        current = data

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _get_nested_value(self, data: Dict, field_path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = field_path.split('.')
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    def _deep_copy_dict(self, d: Dict) -> Dict:
        """Deep copy dictionary"""
        import copy
        return copy.deepcopy(d)

    def _merge_structured_data(self, base: Dict, new: Dict) -> Dict:
        """Merge two structured dictionaries"""
        result = self._deep_copy_dict(base)

        for key, value in new.items():
            if key in result:
                if isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._merge_structured_data(result[key], value)
                elif isinstance(result[key], list) and isinstance(value, list):
                    result[key].extend(value)
                else:
                    # Keep the value from the more reliable source
                    # This could be enhanced with more sophisticated logic
                    result[key] = value
            else:
                result[key] = value

        return result

    def _get_base_synthesis(self, results: List[AgentResult]) -> Dict:
        """Get base synthesis from results"""
        if not results:
            return {}
        return self._deep_copy_dict(results[0].result_data)

# Example usage and testing
if __name__ == "__main__":
    async def test_result_synthesis():
        """Test the result synthesizer"""
        from multi_agent_orchestrator import AgentResult

        # Create sample agent results
        result1 = AgentResult(
            agent_id="agent_1",
            task_id="task_1",
            result_data={
                "analysis": {
                    "sentiment": "positive",
                    "confidence": 0.8,
                    "key_points": ["good quality", "reliable"]
                },
                "score": 0.85
            },
            confidence=0.9,
            reliability_score=0.8
        )

        result2 = AgentResult(
            agent_id="agent_2",
            task_id="task_1",
            result_data={
                "analysis": {
                    "sentiment": "neutral",
                    "confidence": 0.6,
                    "key_points": ["adequate quality", "moderate reliability"]
                },
                "score": 0.75
            },
            confidence=0.7,
            reliability_score=0.9
        )

        # Create synthesizer
        synthesizer = ResultSynthesizer()

        # Test synthesis
        synthesis_result = await synthesizer.synthesize_results(
            [result1, result2],
            strategy=SynthesisStrategy.WEIGHTED_AVERAGE
        )

        print(f"Synthesis Result:")
        print(f"Confidence: {synthesis_result.confidence:.2f}")
        print(f"Quality Score: {synthesis_result.quality_score:.2f}")
        print(f"Conflicts Detected: {len(synthesis_result.conflicts_detected)}")
        print(f"Conflicts Resolved: {len(synthesis_result.conflicts_resolved)}")
        print(f"Synthesized Data: {json.dumps(synthesis_result.synthesized_data, indent=2)}")

    # Run test
    asyncio.run(test_result_synthesis())