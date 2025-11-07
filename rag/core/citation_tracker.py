"""
Citation Tracking and Source Verification System for Advanced RAG

Provides comprehensive citation management, source verification, and trust scoring
for manufacturing knowledge base responses.
"""

import hashlib
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from pathlib import Path

# Citation imports
from document_chunker import DocumentChunk, ContentType


class SourceType(Enum):
    """Types of sources in the manufacturing knowledge base"""
    TECHNICAL_MANUAL = "technical_manual"
    SAFETY_GUIDELINE = "safety_guideline"
    QUALITY_STANDARD = "quality_standard"
    MAINTENANCE_RECORD = "maintenance_record"
    EQUIPMENT_SPEC = "equipment_specification"
    PROCEDURE = "procedure"
    TRAINING_MATERIAL = "training_material"
    REGULATION = "regulation"
    EXPERT_KNOWLEDGE = "expert_knowledge"
    CASE_STUDY = "case_study"
    EXTERNAL_REFERENCE = "external_reference"


class CitationType(Enum):
    """Types of citations"""
    DIRECT_QUOTE = "direct_quote"
    PARAPHRASE = "paraphrase"
    STATISTICAL_DATA = "statistical_data"
    TECHNICAL_SPECIFICATION = "technical_specification"
    PROCEDURE_STEP = "procedure_step"
    SAFETY_WARNING = "safety_warning"
    REGULATORY_REQUIREMENT = "regulatory_requirement"


class TrustLevel(Enum):
    """Trust levels for sources"""
    HIGH = "high"          # Official documentation, standards
    MEDIUM = "medium"      # Internal procedures, expert knowledge
    LOW = "low"           # External references, unverified content
    UNVERIFIED = "unverified"  # New or unchecked content


@dataclass
class Source:
    """Represents a source document or reference"""
    source_id: str
    title: str
    source_type: SourceType
    trust_level: TrustLevel
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    last_verified: Optional[datetime] = None
    verification_status: str = "pending"
    url: Optional[str] = None
    doi: Optional[str] = None
    version: Optional[str] = None
    department: Optional[str] = None
    compliance_standards: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    """Represents a citation within a response"""
    citation_id: str
    source_id: str
    citation_type: CitationType
    content_snippet: str
    page_number: Optional[int] = None
    section_reference: Optional[str] = None
    confidence_score: float = 0.0
    relevance_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    verification_status: str = "pending"
    cross_references: List[str] = field(default_factory=list)
    context_before: Optional[str] = None
    context_after: Optional[str] = None


@dataclass
class SourceClaim:
    """Represents a claim or statement that needs citation"""
    claim_id: str
    claim_text: str
    claim_type: str
    confidence_score: float
    supporting_citations: List[str] = field(default_factory=list)
    conflicting_citations: List[str] = field(default_factory=list)
    verification_status: str = "pending"
    fact_check_result: Optional[Dict[str, Any]] = None


@dataclass
class CitationNetwork:
    """Represents connections between citations and sources"""
    citation_id: str
    related_citations: List[str] = field(default_factory=list)
    supporting_sources: List[str] = field(default_factory=list)
    contradictory_sources: List[str] = field(default_factory=list)
    temporal_context: Optional[Dict[str, Any]] = None


class CitationTracker:
    """Advanced citation tracking and source verification system"""

    def __init__(self, db_path: str = "knowledge_base.db"):
        self.db_path = Path(db_path)
        self.sources: Dict[str, Source] = {}
        self.citations: Dict[str, Citation] = {}
        self.claims: Dict[str, SourceClaim] = {}
        self.citation_networks: Dict[str, CitationNetwork] = {}
        self.verification_rules = self._initialize_verification_rules()
        self._initialize_database()

    def _initialize_verification_rules(self) -> Dict[str, Any]:
        """Initialize source verification rules"""
        return {
            'trust_weights': {
                TrustLevel.HIGH: 1.0,
                TrustLevel.MEDIUM: 0.7,
                TrustLevel.LOW: 0.4,
                TrustLevel.UNVERIFIED: 0.1
            },
            'source_type_weights': {
                SourceType.TECHNICAL_MANUAL: 0.95,
                SourceType.SAFETY_GUIDELINE: 1.0,
                SourceType.QUALITY_STANDARD: 0.95,
                SourceType.MAINTENANCE_RECORD: 0.8,
                SourceType.EQUIPMENT_SPEC: 0.9,
                SourceType.PROCEDURE: 0.85,
                SourceType.TRAINING_MATERIAL: 0.7,
                SourceType.REGULATION: 1.0,
                SourceType.EXPERT_KNOWLEDGE: 0.6,
                SourceType.CASE_STUDY: 0.5,
                SourceType.EXTERNAL_REFERENCE: 0.3
            },
            'recency_decay': {
                'high_importance': 365,  # days
                'medium_importance': 730,
                'low_importance': 1825
            },
            'compliance_bonus': 0.15,
            'verification_required': [
                SourceType.EXTERNAL_REFERENCE,
                SourceType.EXPERT_KNOWLEDGE
            ]
        }

    def _initialize_database(self):
        """Initialize SQLite database for citation tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Sources table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS citation_sources (
                source_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source_type TEXT NOT NULL,
                trust_level TEXT NOT NULL,
                author TEXT,
                publication_date TEXT,
                last_verified TEXT,
                verification_status TEXT DEFAULT 'pending',
                url TEXT,
                doi TEXT,
                version TEXT,
                department TEXT,
                compliance_standards TEXT,
                keywords TEXT,
                metadata TEXT
            )
        ''')

        # Citations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS citations (
                citation_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                citation_type TEXT NOT NULL,
                content_snippet TEXT NOT NULL,
                page_number INTEGER,
                section_reference TEXT,
                confidence_score REAL DEFAULT 0.0,
                relevance_score REAL DEFAULT 0.0,
                timestamp TEXT NOT NULL,
                verification_status TEXT DEFAULT 'pending',
                cross_references TEXT,
                context_before TEXT,
                context_after TEXT,
                FOREIGN KEY (source_id) REFERENCES citation_sources (source_id)
            )
        ''')

        # Claims table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS source_claims (
                claim_id TEXT PRIMARY KEY,
                claim_text TEXT NOT NULL,
                claim_type TEXT NOT NULL,
                confidence_score REAL DEFAULT 0.0,
                supporting_citations TEXT,
                conflicting_citations TEXT,
                verification_status TEXT DEFAULT 'pending',
                fact_check_result TEXT
            )
        ''')

        # Citation networks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS citation_networks (
                citation_id TEXT PRIMARY KEY,
                related_citations TEXT,
                supporting_sources TEXT,
                contradictory_sources TEXT,
                temporal_context TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def add_source(self, source: Source) -> str:
        """Add a new source to the citation system"""
        self.sources[source.source_id] = source

        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO citation_sources
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            source.source_id, source.title, source.source_type.value,
            source.trust_level.value, source.author,
            source.publication_date.isoformat() if source.publication_date else None,
            source.last_verified.isoformat() if source.last_verified else None,
            source.verification_status, source.url, source.doi, source.version,
            source.department,
            json.dumps(source.compliance_standards),
            json.dumps(source.keywords),
            json.dumps(source.metadata)
        ))
        conn.commit()
        conn.close()

        return source.source_id

    def add_citation(self, citation: Citation) -> str:
        """Add a new citation to the system"""
        self.citations[citation.citation_id] = citation

        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO citations
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            citation.citation_id, citation.source_id, citation.citation_type.value,
            citation.content_snippet, citation.page_number, citation.section_reference,
            citation.confidence_score, citation.relevance_score,
            citation.timestamp.isoformat(), citation.verification_status,
            json.dumps(citation.cross_references),
            citation.context_before, citation.context_after
        ))
        conn.commit()
        conn.close()

        # Update citation network
        self._update_citation_network(citation.citation_id)

        return citation.citation_id

    def create_citation_for_content(self, content: str, source_id: str,
                                  document_chunk: Optional[DocumentChunk] = None,
                                  citation_type: CitationType = CitationType.PARAPHRASE) -> List[Citation]:
        """Create citations for a piece of content"""
        citations = []

        # Split content into citable segments
        segments = self._extract_citable_segments(content)

        for i, segment in enumerate(segments):
            citation_id = f"cite_{source_id}_{int(datetime.now().timestamp())}_{i}"

            # Calculate confidence score based on various factors
            confidence_score = self._calculate_citation_confidence(
                segment, source_id, document_chunk
            )

            # Extract context if chunk is available
            context_before, context_after = self._extract_context(
                segment, document_chunk
            ) if document_chunk else (None, None)

            citation = Citation(
                citation_id=citation_id,
                source_id=source_id,
                citation_type=citation_type,
                content_snippet=segment,
                confidence_score=confidence_score,
                relevance_score=self._calculate_relevance_score(segment),
                context_before=context_before,
                context_after=context_after,
                page_number=document_chunk.metadata.get('page_number') if document_chunk else None,
                section_reference=document_chunk.metadata.get('section_title') if document_chunk else None
            )

            citations.append(citation)
            self.add_citation(citation)

        return citations

    def verify_source(self, source_id: str) -> Dict[str, Any]:
        """Verify a source and return verification results"""
        source = self.sources.get(source_id)
        if not source:
            return {'status': 'error', 'message': 'Source not found'}

        verification_result = {
            'source_id': source_id,
            'verification_timestamp': datetime.now().isoformat(),
            'trust_score': 0.0,
            'verification_checks': {},
            'recommendations': [],
            'status': 'pending'
        }

        # Perform verification checks
        checks_passed = 0
        total_checks = 0

        # Check 1: Source type trustworthiness
        type_weight = self.verification_rules['source_type_weights'].get(
            source.source_type, 0.5
        )
        verification_result['verification_checks']['source_type'] = {
            'score': type_weight,
            'status': 'passed' if type_weight >= 0.7 else 'warning'
        }
        checks_passed += 1 if type_weight >= 0.7 else 0
        total_checks += 1

        # Check 2: Publication recency
        if source.publication_date:
            days_old = (datetime.now() - source.publication_date).days
            recency_score = self._calculate_recency_score(source.source_type, days_old)
            verification_result['verification_checks']['publication_recency'] = {
                'score': recency_score,
                'status': 'passed' if recency_score >= 0.6 else 'warning',
                'days_old': days_old
            }
            checks_passed += 1 if recency_score >= 0.6 else 0
            total_checks += 1

        # Check 3: Compliance standards
        if source.compliance_standards:
            compliance_score = min(len(source.compliance_standards) * 0.2, 1.0)
            verification_result['verification_checks']['compliance_standards'] = {
                'score': compliance_score,
                'standards': source.compliance_standards,
                'status': 'passed' if compliance_score >= 0.5 else 'warning'
            }
            checks_passed += 1 if compliance_score >= 0.5 else 0
            total_checks += 1

        # Check 4: Author verification (if available)
        if source.author:
            author_score = self._verify_author(source.author, source.department)
            verification_result['verification_checks']['author_verification'] = {
                'score': author_score,
                'author': source.author,
                'status': 'passed' if author_score >= 0.6 else 'warning'
            }
            checks_passed += 1 if author_score >= 0.6 else 0
            total_checks += 1

        # Calculate overall trust score
        verification_result['trust_score'] = checks_passed / total_checks if total_checks > 0 else 0.0
        verification_result['status'] = 'verified' if checks_passed / total_checks >= 0.7 else 'needs_review'

        # Update source verification status
        source.verification_status = verification_result['status']
        source.last_verified = datetime.now()

        # Save updated source
        self.add_source(source)

        return verification_result

    def cross_verify_citations(self, citation_ids: List[str]) -> Dict[str, Any]:
        """Cross-verify multiple citations for consistency"""
        verification_result = {
            'citation_count': len(citation_ids),
            'consistency_score': 0.0,
            'conflicts': [],
            'correlations': [],
            'recommendations': []
        }

        if len(citation_ids) < 2:
            verification_result['status'] = 'insufficient_citations'
            return verification_result

        citations = [self.citations[cid] for cid in citation_ids if cid in self.citations]

        # Check for conflicts and correlations
        conflicts = self._detect_citation_conflicts(citations)
        correlations = self._detect_citation_correlations(citations)

        verification_result['conflicts'] = conflicts
        verification_result['correlations'] = correlations

        # Calculate consistency score
        consistency_score = max(0.0, 1.0 - (len(conflicts) / len(citations)))
        verification_result['consistency_score'] = consistency_score

        # Generate recommendations
        if conflicts:
            verification_result['recommendations'].append(
                "Conflicting citations detected. Review sources for accuracy."
            )

        if consistency_score < 0.5:
            verification_result['recommendations'].append(
                "Low consistency score. Consider adding more authoritative sources."
            )

        verification_result['status'] = 'completed'

        return verification_result

    def _extract_citable_segments(self, content: str) -> List[str]:
        """Extract segments from content that should be cited"""
        segments = []

        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', content)

        current_segment = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if sentence contains technical information that should be cited
            if self._should_cite_sentence(sentence):
                if current_segment:
                    segments.append(current_segment.strip())
                    current_segment = ""
                segments.append(sentence)
            else:
                current_segment += " " + sentence if current_segment else sentence

        if current_segment:
            segments.append(current_segment.strip())

        return [seg for seg in segments if seg and len(seg.split()) > 3]

    def _should_cite_sentence(self, sentence: str) -> bool:
        """Determine if a sentence should be cited"""
        # Patterns that indicate need for citation
        citation_patterns = [
            r'\d+(?:\.\d+)?\s*(?:%|mg|kg|psi|bar|°c|°f|v|a|w)',  # Technical measurements
            r'according to', r'studies show', r'research indicates',
            r'standard\s+\w+', r'iso\s+\d+', r'ansi\s+\d+',
            r'specification\s+\w+', r'model\s+\w+', r'version\s+\d+',
            r'risk\s+level', r'safety\s+factor', r'tolerance\s+\w+',
            r'procedure\s+step\s+\d+', r'critical\s+parameter'
        ]

        sentence_lower = sentence.lower()
        for pattern in citation_patterns:
            if re.search(pattern, sentence_lower):
                return True

        return False

    def _calculate_citation_confidence(self, segment: str, source_id: str,
                                     document_chunk: Optional[DocumentChunk] = None) -> float:
        """Calculate confidence score for a citation"""
        confidence = 0.5  # Base confidence

        # Factor in source trustworthiness
        source = self.sources.get(source_id)
        if source:
            trust_weight = self.verification_rules['trust_weights'].get(
                source.trust_level, 0.5
            )
            confidence += trust_weight * 0.3

        # Factor in content specificity
        if self._contains_specific_data(segment):
            confidence += 0.2

        # Factor in document chunk relevance
        if document_chunk:
            confidence += document_chunk.relevance_score * 0.2

        # Factor in length and completeness
        word_count = len(segment.split())
        if word_count >= 10:
            confidence += 0.1
        elif word_count < 5:
            confidence -= 0.1

        return min(max(confidence, 0.0), 1.0)

    def _contains_specific_data(self, text: str) -> bool:
        """Check if text contains specific, citable data"""
        specific_patterns = [
            r'\d+(?:\.\d+)?%',  # Percentages
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # Monetary values
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # Dates
            r'\b(?:kg|g|mg|lb|oz)\b',  # Weights
            r'\b(?:m|cm|mm|in|ft)\b',  # Lengths
            r'\b(?:psi|bar|kpa|mpa)\b',  # Pressures
            r'\b(?:°c|°f|k)\b',  # Temperatures
        ]

        for pattern in specific_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _calculate_relevance_score(self, segment: str) -> float:
        """Calculate relevance score for a content segment"""
        relevance = 0.5

        # Manufacturing-specific keywords that increase relevance
        relevant_keywords = [
            'safety', 'quality', 'procedure', 'standard', 'specification',
            'maintenance', 'inspection', 'compliance', 'regulation',
            'risk', 'hazard', 'control', 'process', 'equipment'
        ]

        segment_lower = segment.lower()
        keyword_count = sum(1 for keyword in relevant_keywords if keyword in segment_lower)
        relevance += min(keyword_count * 0.1, 0.3)

        return min(max(relevance, 0.0), 1.0)

    def _extract_context(self, segment: str, document_chunk: DocumentChunk) -> Tuple[Optional[str], Optional[str]]:
        """Extract context around a segment within a document chunk"""
        chunk_content = document_chunk.content
        segment_index = chunk_content.find(segment)

        if segment_index == -1:
            return None, None

        # Extract 50 characters before and after
        start = max(0, segment_index - 50)
        end = min(len(chunk_content), segment_index + len(segment) + 50)

        context_before = chunk_content[start:segment_index].strip()
        context_after = chunk_content[segment_index + len(segment):end].strip()

        return context_before or None, context_after or None

    def _calculate_recency_score(self, source_type: SourceType, days_old: int) -> float:
        """Calculate recency score based on source type and age"""
        importance = 'high_importance' if source_type in [
            SourceType.SAFETY_GUIDELINE, SourceType.QUALITY_STANDARD, SourceType.REGULATION
        ] else 'medium_importance' if source_type in [
            SourceType.TECHNICAL_MANUAL, SourceType.EQUIPMENT_SPEC, SourceType.PROCEDURE
        ] else 'low_importance'

        threshold = self.verification_rules['recency_decay'][importance]

        if days_old <= threshold:
            return 1.0
        else:
            # Linear decay after threshold
            return max(0.0, 1.0 - ((days_old - threshold) / threshold))

    def _verify_author(self, author: str, department: Optional[str] = None) -> float:
        """Verify author credibility"""
        # Simple heuristic based on name patterns and department
        credibility = 0.5

        # Check for academic or professional indicators
        if any(title in author.lower() for title in ['dr', 'prof', 'eng', 'pe', 'phd']):
            credibility += 0.3

        # Department-based credibility
        if department:
            high_credibility_depts = ['quality', 'engineering', 'safety', 'compliance']
            if any(dept in department.lower() for dept in high_credibility_depts):
                credibility += 0.2

        return min(max(credibility, 0.0), 1.0)

    def _detect_citation_conflicts(self, citations: List[Citation]) -> List[Dict[str, Any]]:
        """Detect conflicts between citations"""
        conflicts = []

        for i, cite1 in enumerate(citations):
            for cite2 in citations[i+1:]:
                # Simple conflict detection based on content similarity and different sources
                if cite1.source_id != cite2.source_id:
                    similarity = self._calculate_content_similarity(
                        cite1.content_snippet, cite2.content_snippet
                    )

                    if 0.3 < similarity < 0.7:  # Partial similarity might indicate conflict
                        conflicts.append({
                            'citation_1': cite1.citation_id,
                            'citation_2': cite2.citation_id,
                            'similarity': similarity,
                            'conflict_type': 'partial_overlap'
                        })

        return conflicts

    def _detect_citation_correlations(self, citations: List[Citation]) -> List[Dict[str, Any]]:
        """Detect correlations between citations"""
        correlations = []

        for i, cite1 in enumerate(citations):
            for cite2 in citations[i+1:]:
                # Check for supporting correlations
                if cite1.source_id != cite2.source_id:
                    similarity = self._calculate_content_similarity(
                        cite1.content_snippet, cite2.content_snippet
                    )

                    if similarity >= 0.8:  # High similarity indicates support
                        correlations.append({
                            'citation_1': cite1.citation_id,
                            'citation_2': cite2.citation_id,
                            'similarity': similarity,
                            'correlation_type': 'supporting'
                        })

        return correlations

    def _calculate_content_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text segments"""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _update_citation_network(self, citation_id: str):
        """Update the citation network for a citation"""
        if citation_id not in self.citation_networks:
            self.citation_networks[citation_id] = CitationNetwork(citation_id=citation_id)

        network = self.citation_networks[citation_id]

        # Find related citations based on content similarity and source relationships
        for other_id, other_citation in self.citations.items():
            if other_id == citation_id:
                continue

            # Check for content similarity
            similarity = self._calculate_content_similarity(
                self.citations[citation_id].content_snippet,
                other_citation.content_snippet
            )

            if similarity > 0.5:
                if similarity > 0.8:
                    network.related_citations.append(other_id)
                else:
                    network.contradictory_sources.append(other_citation.source_id)

        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO citation_networks
            VALUES (?, ?, ?, ?, ?)
        ''', (
            citation_id,
            json.dumps(network.related_citations),
            json.dumps(network.supporting_sources),
            json.dumps(network.contradictory_sources),
            json.dumps(network.temporal_context) if network.temporal_context else None
        ))
        conn.commit()
        conn.close()

    def get_citation_report(self, citation_ids: List[str]) -> Dict[str, Any]:
        """Generate a comprehensive citation report"""
        report = {
            'total_citations': len(citation_ids),
            'sources_covered': set(),
            'citation_types': {},
            'trust_level_distribution': {},
            'verification_status': {},
            'average_confidence': 0.0,
            'recommendations': []
        }

        confidence_scores = []

        for citation_id in citation_ids:
            citation = self.citations.get(citation_id)
            if not citation:
                continue

            source = self.sources.get(citation.source_id)
            if not source:
                continue

            report['sources_covered'].add(citation.source_id)

            # Count citation types
            cite_type = citation.citation_type.value
            report['citation_types'][cite_type] = report['citation_types'].get(cite_type, 0) + 1

            # Count trust levels
            trust_level = source.trust_level.value
            report['trust_level_distribution'][trust_level] = \
                report['trust_level_distribution'].get(trust_level, 0) + 1

            # Count verification status
            verify_status = citation.verification_status
            report['verification_status'][verify_status] = \
                report['verification_status'].get(verify_status, 0) + 1

            confidence_scores.append(citation.confidence_score)

        report['sources_covered'] = list(report['sources_covered'])
        report['average_confidence'] = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        # Generate recommendations
        if report['average_confidence'] < 0.6:
            report['recommendations'].append(
                "Consider adding more authoritative sources to improve confidence scores."
            )

        if report['verification_status'].get('pending', 0) > len(citation_ids) * 0.3:
            report['recommendations'].append(
                "Many citations need verification. Complete the verification process."
            )

        return report

    def export_citations(self, format_type: str = 'json') -> str:
        """Export citations in specified format"""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'sources': {sid: {
                'title': source.title,
                'type': source.source_type.value,
                'trust_level': source.trust_level.value,
                'verification_status': source.verification_status
            } for sid, source in self.sources.items()},
            'citations': {cid: {
                'source_id': citation.source_id,
                'type': citation.citation_type.value,
                'content': citation.content_snippet,
                'confidence': citation.confidence_score,
                'verification_status': citation.verification_status
            } for cid, citation in self.citations.items()}
        }

        if format_type.lower() == 'json':
            return json.dumps(export_data, indent=2)
        elif format_type.lower() == 'bibliography':
            return self._generate_bibliography(export_data)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def _generate_bibliography(self, export_data: Dict[str, Any]) -> str:
        """Generate bibliography from export data"""
        bibliography = []

        for source_id, source_info in export_data['sources'].items():
            entry = f"{source_info['title']}. "
            if source_info['type']:
                entry += f"Type: {source_info['type']}. "
            entry += f"Trust Level: {source_info['trust_level'].upper()}. "
            entry += f"Verification: {source_info['verification_status']}."
            bibliography.append(entry)

        return "\n\n".join(bibliography)


# Factory function
def create_citation_tracker(db_path: str = "knowledge_base.db") -> CitationTracker:
    """Create and initialize a citation tracker"""
    return CitationTracker(db_path)


# Usage example
if __name__ == "__main__":
    # Create citation tracker
    tracker = create_citation_tracker()

    # Add a sample source
    source = Source(
        source_id="iso_9001_2015",
        title="ISO 9001:2015 Quality Management Systems",
        source_type=SourceType.QUALITY_STANDARD,
        trust_level=TrustLevel.HIGH,
        publication_date=datetime(2015, 9, 15),
        compliance_standards=["ISO 9001"],
        keywords=["quality", "management", "standards", "compliance"]
    )

    tracker.add_source(source)

    # Verify the source
    verification = tracker.verify_source("iso_9001_2015")
    print(f"Verification result: {verification}")

    print("Citation tracking and source verification system initialized successfully!")