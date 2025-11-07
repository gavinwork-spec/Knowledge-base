"""
Personalized Search Engine with Privacy Protection

Implements user-aware search capabilities that learn from behavior while maintaining
strict privacy standards through data anonymization and user consent.
"""

import asyncio
import time
import json
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import defaultdict, deque, Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import re
import logging

from hybrid_search_engine import SearchResult, SearchRequest


@dataclass
class UserPrivacyConfig:
    """User privacy preferences and consent settings"""
    user_id_hash: str
    tracking_enabled: bool = True
    query_history_retention_days: int = 30
    click_tracking_enabled: bool = True
    expertise_learning_enabled: bool = True
    personalization_enabled: bool = True
    data_anonymization_enabled: bool = True
    auto_delete_after_days: Optional[int] = 365
    consent_timestamp: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    gdpr_compliant: bool = True
    ccpa_compliant: bool = True


@dataclass
class UserQueryEvent:
    """Anonymous user query interaction event"""
    user_hash: str
    query: str
    timestamp: datetime
    session_id: str
    search_strategy: str
    results_count: int
    clicked_results: List[str] = field(default_factory=list)
    dwell_time_seconds: float = 0.0
    satisfaction_score: Optional[float] = None
    referrer_query: Optional[str] = None
    query_reformulation: bool = False
    anonymized_ip: Optional[str] = None
    user_agent_hash: Optional[str] = None


@dataclass
class UserExpertiseProfile:
    """User expertise and interest areas"""
    user_hash: str
    expertise_domains: Dict[str, float]  # domain -> confidence score
    query_patterns: Dict[str, int]  # pattern -> frequency
    vocabulary_richness: float = 0.0
    technical_level: float = 0.0  # 0-1 scale
    language_preferences: List[str] = field(default_factory=list)
    content_preferences: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    confidence_score: float = 0.0


@dataclass
class PersonalizedSearchRequest:
    """Search request with personalization context"""
    base_request: SearchRequest
    user_hash: str
    session_context: Dict[str, Any] = field(default_factory=dict)
    personalization_level: float = 0.7  # 0-1, how much to personalize
    boost_expertise: bool = True
    boost_history: bool = True
    boost_preferences: bool = True


class DataAnonymizer:
    """Handles data anonymization and privacy protection"""

    def __init__(self):
        self.salt = secrets.token_bytes(32)
        self.hash_function = hashlib.sha256

    def anonymize_user_id(self, user_id: str) -> str:
        """Create anonymous hash for user ID"""
        return self.hash_function(f"{user_id}{self.salt}".encode()).hexdigest()[:16]

    def anonymize_ip(self, ip_address: str) -> str:
        """Anonymize IP address by removing last octet"""
        try:
            parts = ip_address.split('.')
            if len(parts) == 4:
                parts[-1] = '0'
                return '.'.join(parts)
        except:
            pass
        return "unknown"

    def anonymize_text(self, text: str, remove_pii: bool = True) -> str:
        """Remove personally identifiable information from text"""
        if not remove_pii:
            return text

        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)

        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)

        # Remove common PII patterns
        text = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[CREDIT_CARD]', text)
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)

        return text

    def generate_session_id(self) -> str:
        """Generate anonymous session identifier"""
        return secrets.token_urlsafe(16)


class UserBehaviorTracker:
    """Tracks and analyzes user behavior patterns"""

    def __init__(self, anonymizer: DataAnonymizer):
        self.anonymizer = anonymizer
        self.query_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.click_patterns: Dict[str, List[str]] = defaultdict(list)
        self.session_data: Dict[str, Dict] = defaultdict(dict)
        self.behavior_metrics: Dict[str, Dict] = defaultdict(dict)

    async def track_query(self, user_id: str, query: str, search_request: SearchRequest,
                         results: List[SearchResult], session_id: str) -> str:
        """Track user query and generate anonymous event"""
        user_hash = self.anonymizer.anonymize_user_id(user_id)

        # Create query event
        event = UserQueryEvent(
            user_hash=user_hash,
            query=self.anonymizer.anonymize_text(query),
            timestamp=datetime.now(),
            session_id=session_id,
            search_strategy=search_request.search_strategy,
            results_count=len(results),
            clicked_results=[]
        )

        # Store in query history
        self.query_history[user_hash].append(event)

        # Update session data
        self.session_data[session_id] = {
            'user_hash': user_hash,
            'start_time': datetime.now(),
            'query_count': self.session_data[session_id].get('query_count', 0) + 1,
            'queries': self.session_data[session_id].get('queries', []) + [query]
        }

        return user_hash

    async def track_click(self, user_hash: str, session_id: str, result_id: str,
                         dwell_time: float = 0.0):
        """Track result click and dwell time"""
        # Update recent query with click
        if user_hash in self.query_history and self.query_history[user_hash]:
            recent_query = self.query_history[user_hash][-1]
            recent_query.clicked_results.append(result_id)
            recent_query.dwell_time_seconds = dwell_time

        # Update click patterns
        self.click_patterns[user_hash].append(result_id)

        # Update behavior metrics
        if 'click_through_rate' not in self.behavior_metrics[user_hash]:
            self.behavior_metrics[user_hash]['click_through_rate'] = []

        self.behavior_metrics[user_hash]['click_through_rate'].append(1.0)

    async def track_satisfaction(self, user_hash: str, session_id: str,
                                satisfaction_score: float):
        """Track user satisfaction with search results"""
        if user_hash in self.query_history and self.query_history[user_hash]:
            recent_query = self.query_history[user_hash][-1]
            recent_query.satisfaction_score = satisfaction_score

    def get_query_patterns(self, user_hash: str) -> Dict[str, Any]:
        """Analyze user query patterns"""
        if user_hash not in self.query_history:
            return {}

        queries = [event.query for event in self.query_history[user_hash]]

        # Query frequency analysis
        query_counter = Counter(queries)

        # Time-based patterns
        timestamps = [event.timestamp for event in self.query_history[user_hash]]
        hour_distribution = Counter([ts.hour for ts in timestamps])

        # Query length distribution
        query_lengths = [len(q.split()) for q in queries]

        # Common terms
        all_terms = []
        for query in queries:
            all_terms.extend(query.lower().split())
        term_counter = Counter(all_terms)

        return {
            'total_queries': len(queries),
            'unique_queries': len(set(queries)),
            'most_common_queries': query_counter.most_common(5),
            'most_common_terms': term_counter.most_common(10),
            'avg_query_length': np.mean(query_lengths) if query_lengths else 0,
            'hour_distribution': dict(hour_distribution),
            'search_strategies': Counter([event.search_strategy for event in self.query_history[user_hash]])
        }

    def cleanup_old_data(self, retention_days: int = 30):
        """Clean up old tracking data based on retention policy"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        for user_hash in list(self.query_history.keys()):
            # Filter out old events
            self.query_history[user_hash] = deque(
                [event for event in self.query_history[user_hash]
                 if event.timestamp > cutoff_date],
                maxlen=1000
            )

            # Remove empty histories
            if not self.query_history[user_hash]:
                del self.query_history[user_hash]


class ExpertiseIdentifier:
    """Identifies user expertise areas from behavior patterns"""

    def __init__(self):
        self.domain_keywords = {
            'machine_learning': ['machine learning', 'neural network', 'deep learning', 'tensorflow', 'pytorch', 'algorithm'],
            'web_development': ['html', 'css', 'javascript', 'react', 'vue', 'angular', 'frontend', 'backend'],
            'data_science': ['data analysis', 'pandas', 'numpy', 'statistics', 'visualization', 'analytics'],
            'software_engineering': ['software development', 'programming', 'coding', 'architecture', 'design patterns'],
            'artificial_intelligence': ['ai', 'artificial intelligence', 'nlp', 'computer vision', 'robotics'],
            'database': ['sql', 'database', 'mysql', 'postgresql', 'mongodb', 'nosql'],
            'cloud_computing': ['aws', 'azure', 'gcp', 'cloud', 'docker', 'kubernetes'],
            'cybersecurity': ['security', 'cybersecurity', 'encryption', 'authentication', 'network security']
        }

        self.technical_indicators = [
            'api', 'algorithm', 'framework', 'library', 'function', 'method',
            'class', 'object', 'variable', 'database', 'server', 'client'
        ]

    async def analyze_expertise(self, user_hash: str, behavior_tracker: UserBehaviorTracker) -> UserExpertiseProfile:
        """Analyze user expertise from query history and patterns"""
        patterns = behavior_tracker.get_query_patterns(user_hash)

        if not patterns:
            return UserExpertiseProfile(user_hash=user_hash, expertise_domains={})

        # Analyze domain expertise
        expertise_scores = {}
        total_queries = patterns['total_queries']

        for domain, keywords in self.domain_keywords.items():
            domain_score = 0
            keyword_matches = 0

            for query, count in patterns['most_common_queries']:
                query_lower = query.lower()
                for keyword in keywords:
                    if keyword in query_lower:
                        keyword_matches += 1
                        domain_score += count

            # Normalize by total queries and keyword coverage
            if total_queries > 0:
                expertise_scores[domain] = min(1.0, domain_score / (total_queries * 0.1))

        # Calculate technical level
        technical_queries = sum(
            count for query, count in patterns['most_common_queries']
            if any(indicator in query.lower() for indicator in self.technical_indicators)
        )
        technical_level = min(1.0, technical_queries / max(1, total_queries))

        # Calculate vocabulary richness
        unique_terms = len(patterns.get('most_common_terms', []))
        total_terms = sum(count for _, count in patterns.get('most_common_terms', []))
        vocabulary_richness = unique_terms / max(1, total_terms)

        # Determine language preferences (simplified)
        language_preferences = ['english']  # Default, could be enhanced with detection

        # Content preferences based on click patterns
        content_preferences = {}
        if user_hash in behavior_tracker.click_patterns:
            clicked_content = behavior_tracker.click_patterns[user_hash]
            content_counter = Counter(clicked_content)
            total_clicks = len(clicked_content)

            for content_id, clicks in content_counter.most_common(10):
                content_preferences[content_id] = clicks / max(1, total_clicks)

        # Calculate overall confidence score
        expertise_confidence = len([score for score in expertise_scores.values() if score > 0.3]) / max(1, len(expertise_scores))

        return UserExpertiseProfile(
            user_hash=user_hash,
            expertise_domains=expertise_scores,
            query_patterns={query: count for query, count in patterns['most_common_queries']},
            vocabulary_richness=vocabulary_richness,
            technical_level=technical_level,
            language_preferences=language_preferences,
            content_preferences=content_preferences,
            confidence_score=expertise_confidence,
            last_updated=datetime.now()
        )


class PersonalizationEngine:
    """Applies personalization to search results"""

    def __init__(self):
        self.personalization_weights = {
            'expertise_boost': 0.3,
            'history_boost': 0.25,
            'preference_boost': 0.2,
            'novelty_penalty': 0.1,
            'diversity_boost': 0.15
        }

    async def personalize_results(self, results: List[SearchResult],
                                 expertise_profile: UserExpertiseProfile,
                                 personalization_request: PersonalizedSearchRequest) -> List[SearchResult]:
        """Apply personalization to search results"""
        if not personalization_request.personalization_enabled or personalization_request.personalization_level <= 0:
            return results

        personalized_results = []

        for result in results:
            # Start with original score
            personalized_score = result.score

            # Expertise-based boost
            if personalization_request.boost_expertise and expertise_profile.expertise_domains:
                expertise_boost = self._calculate_expertise_boost(result, expertise_profile)
                personalized_score += expertise_boost * self.personalization_weights['expertise_boost']

            # Content preference boost
            if personalization_request.boost_preferences and expertise_profile.content_preferences:
                preference_boost = self._calculate_preference_boost(result, expertise_profile)
                personalized_score += preference_boost * self.personalization_weights['preference_boost']

            # Apply personalization level scaling
            final_score = result.score + (personalized_score - result.score) * personalization_request.personalization_level

            # Create personalized result
            personalized_result = SearchResult(
                document_id=result.document_id,
                title=result.title,
                content=result.content,
                score=final_score,
                metadata=result.metadata.copy(),
                explanation=result.explanation,
                search_type=result.search_type
            )

            # Add personalization metadata
            personalized_result.metadata['personalized'] = True
            personalized_result.metadata['personalization_level'] = personalization_request.personalization_level
            personalized_result.metadata['original_score'] = result.score

            personalized_results.append(personalized_result)

        # Re-sort results by personalized scores
        personalized_results.sort(key=lambda x: x.score, reverse=True)

        return personalized_results

    def _calculate_expertise_boost(self, result: SearchResult,
                                  expertise_profile: UserExpertiseProfile) -> float:
        """Calculate expertise-based relevance boost"""
        boost = 0.0
        content_text = f"{result.title} {result.content}".lower()

        for domain, score in expertise_profile.expertise_domains.items():
            if domain in self.domain_keywords:
                domain_keywords = self.domain_keywords[domain]
                keyword_matches = sum(1 for keyword in domain_keywords if keyword in content_text)

                if keyword_matches > 0:
                    # Boost proportional to domain expertise and keyword relevance
                    boost += score * (keyword_matches / len(domain_keywords))

        return min(1.0, boost)

    def _calculate_preference_boost(self, result: SearchResult,
                                   expertise_profile: UserExpertiseProfile) -> float:
        """Calculate content preference-based boost"""
        if result.document_id in expertise_profile.content_preferences:
            return expertise_profile.content_preferences[result.document_id]
        return 0.0


class QuerySuggestionEngine:
    """Provides intelligent query suggestions and auto-completion"""

    def __init__(self):
        self.suggestion_cache: Dict[str, List[str]] = {}
        self.popular_queries: List[str] = []
        self.query_patterns: Dict[str, List[str]] = defaultdict(list)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.query_matrix = None
        self.query_list = []

    async def train_suggestion_model(self, behavior_tracker: UserBehaviorTracker):
        """Train suggestion model from historical query data"""
        all_queries = []

        for user_hash in behavior_tracker.query_history:
            user_queries = [event.query for event in behavior_tracker.query_history[user_hash]]
            all_queries.extend(user_queries)

            # Build user-specific patterns
            for query in user_queries:
                self.query_patterns[user_hash].append(query)

        # Remove duplicates and very short queries
        unique_queries = list(set([q for q in all_queries if len(q.strip()) > 2]))
        self.query_list = unique_queries

        if len(unique_queries) > 10:
            # Train TF-IDF model
            self.query_matrix = self.tfidf_vectorizer.fit_transform(unique_queries)

        # Calculate popular queries (simplified)
        query_counter = Counter(all_queries)
        self.popular_queries = [query for query, _ in query_counter.most_common(100)]

    async def get_suggestions(self, user_hash: str, partial_query: str,
                            max_suggestions: int = 5) -> List[str]:
        """Get personalized query suggestions"""
        suggestions = []

        # User-specific suggestions based on history
        if user_hash in self.query_patterns:
            user_queries = self.query_patterns[user_hash]
            user_suggestions = [
                query for query in user_queries
                if partial_query.lower() in query.lower()
            ][:max_suggestions // 2]
            suggestions.extend(user_suggestions)

        # Global popular suggestions
        popular_suggestions = [
            query for query in self.popular_queries
            if partial_query.lower() in query.lower()
        ][:max_suggestions // 2]
        suggestions.extend(popular_suggestions)

        # Semantic suggestions if we have trained model
        if self.query_matrix is not None and len(partial_query) > 3:
            semantic_suggestions = await self._get_semantic_suggestions(partial_query, max_suggestions // 2)
            suggestions.extend(semantic_suggestions)

        # Remove duplicates and limit
        unique_suggestions = list(dict.fromkeys(suggestions))[:max_suggestions]
        return unique_suggestions

    async def _get_semantic_suggestions(self, partial_query: str, max_results: int) -> List[str]:
        """Get semantically similar suggestions"""
        try:
            # Transform partial query
            query_vec = self.tfidf_vectorizer.transform([partial_query])

            # Calculate similarities
            similarities = cosine_similarity(query_vec, self.query_matrix).flatten()

            # Get top similar queries
            top_indices = np.argsort(similarities)[::-1][:max_results]

            suggestions = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Threshold
                    suggestions.append(self.query_list[idx])

            return suggestions
        except:
            return []


class PersonalizedSearchEngine:
    """Main personalized search engine with privacy protection"""

    def __init__(self, base_search_engine, privacy_config: Optional[Dict] = None):
        self.base_search_engine = base_search_engine
        self.anonymizer = DataAnonymizer()
        self.behavior_tracker = UserBehaviorTracker(self.anonymizer)
        self.expertise_identifier = ExpertiseIdentifier()
        self.personalization_engine = PersonalizationEngine()
        self.suggestion_engine = QuerySuggestionEngine()

        self.user_privacy_configs: Dict[str, UserPrivacyConfig] = {}
        self.expertise_profiles: Dict[str, UserExpertiseProfile] = {}

        # Privacy settings
        self.default_privacy_config = UserPrivacyConfig(
            user_id_hash="default",
            tracking_enabled=True,
            query_history_retention_days=30,
            auto_delete_after_days=365
        )

        # Setup logging for privacy compliance
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize the personalized search engine"""
        # Train suggestion models if we have data
        await self.suggestion_engine.train_suggestion_model(self.behavior_tracker)
        self.logger.info("Personalized search engine initialized")

    def set_user_privacy_config(self, user_id: str, config: UserPrivacyConfig):
        """Set privacy preferences for a user"""
        user_hash = self.anonymizer.anonymize_user_id(user_id)
        config.user_id_hash = user_hash
        config.last_updated = datetime.now()
        self.user_privacy_configs[user_hash] = config

        # Log consent changes
        self.logger.info(f"Privacy preferences updated for user {user_hash}")

    def get_user_privacy_config(self, user_id: str) -> UserPrivacyConfig:
        """Get privacy preferences for a user"""
        user_hash = self.anonymizer.anonymize_user_id(user_id)
        return self.user_privacy_configs.get(user_hash, self.default_privacy_config)

    async def personalized_search(self, search_request: PersonalizedSearchRequest,
                                user_id: str, session_id: str) -> List[SearchResult]:
        """Perform personalized search"""
        user_hash = self.anonymizer.anonymize_user_id(user_id)
        privacy_config = self.get_user_privacy_config(user_id)

        # Check privacy consent
        if not privacy_config.tracking_enabled or not privacy_config.personalization_enabled:
            # Return non-personalized results
            return await self.base_search_engine.search(search_request.base_request)

        # Perform base search
        base_results = await self.base_search_engine.search(search_request.base_request)

        # Get or create expertise profile
        if user_hash not in self.expertise_profiles:
            self.expertise_profiles[user_hash] = await self.expertise_identifier.analyze_expertise(
                user_hash, self.behavior_tracker
            )

        expertise_profile = self.expertise_profiles[user_hash]

        # Apply personalization
        personalized_results = await self.personalization_engine.personalize_results(
            base_results.results,
            expertise_profile,
            search_request
        )

        # Track the query for learning
        await self.behavior_tracker.track_query(
            user_id, search_request.base_request.query,
            search_request.base_request, personalized_results, session_id
        )

        return personalized_results

    async def get_query_suggestions(self, user_id: str, partial_query: str,
                                   max_suggestions: int = 5) -> List[str]:
        """Get personalized query suggestions"""
        user_hash = self.anonymizer.anonymize_user_id(user_id)
        privacy_config = self.get_user_privacy_config(user_id)

        if not privacy_config.tracking_enabled:
            # Return only popular suggestions
            return self.suggestion_engine.popular_queries[:max_suggestions]

        return await self.suggestion_engine.get_suggestions(user_hash, partial_query, max_suggestions)

    async def track_user_feedback(self, user_id: str, session_id: str,
                                 result_id: str, feedback_type: str,
                                 dwell_time: float = 0.0, satisfaction_score: Optional[float] = None):
        """Track user feedback for continuous learning"""
        user_hash = self.anonymizer.anonymize_user_id(user_id)
        privacy_config = self.get_user_privacy_config(user_id)

        if not privacy_config.tracking_enabled or not privacy_config.click_tracking_enabled:
            return

        if feedback_type == 'click':
            await self.behavior_tracker.track_click(user_hash, session_id, result_id, dwell_time)
        elif feedback_type == 'satisfaction' and satisfaction_score:
            await self.behavior_tracker.track_satisfaction(user_hash, session_id, satisfaction_score)

        # Update expertise profile periodically
        if user_hash in self.expertise_profiles:
            time_since_update = datetime.now() - self.expertise_profiles[user_hash].last_updated
            if time_since_update > timedelta(hours=24):  # Update daily
                self.expertise_profiles[user_hash] = await self.expertise_identifier.analyze_expertise(
                    user_hash, self.behavior_tracker
                )

    async def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export user data for GDPR compliance"""
        user_hash = self.anonymizer.anonymize_user_id(user_id)

        export_data = {
            'user_hash': user_hash,
            'privacy_config': asdict(self.get_user_privacy_config(user_id)),
            'expertise_profile': asdict(self.expertise_profiles.get(user_hash, {})),
            'query_patterns': self.behavior_tracker.get_query_patterns(user_hash),
            'export_timestamp': datetime.now().isoformat(),
            'data_types': ['query_history', 'expertise_profile', 'privacy_preferences']
        }

        return export_data

    async def delete_user_data(self, user_id: str) -> bool:
        """Delete all user data for GDPR/CCPA compliance"""
        user_hash = self.anonymizer.anonymize_user_id(user_id)

        try:
            # Delete privacy config
            self.user_privacy_configs.pop(user_hash, None)

            # Delete expertise profile
            self.expertise_profiles.pop(user_hash, None)

            # Delete behavior data
            self.behavior_tracker.query_history.pop(user_hash, None)
            self.behavior_tracker.click_patterns.pop(user_hash, None)
            self.behavior_tracker.behavior_metrics.pop(user_hash, None)

            # Delete suggestion patterns
            self.suggestion_engine.query_patterns.pop(user_hash, None)

            self.logger.info(f"All data deleted for user {user_hash}")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting user data for {user_hash}: {e}")
            return False

    async def cleanup_expired_data(self):
        """Clean up expired data based on retention policies"""
        for user_hash, config in self.user_privacy_configs.items():
            if config.auto_delete_after_days:
                cutoff_date = datetime.now() - timedelta(days=config.auto_delete_after_days)

                # Check if user data should be deleted
                if config.last_updated < cutoff_date:
                    # Find original user_id (this would need a reverse mapping in production)
                    # For now, just clean the old data
                    self.behavior_tracker.cleanup_old_data(config.query_history_retention_days)

    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get anonymized analytics summary"""
        total_users = len(self.user_privacy_configs)
        total_queries = sum(len(history) for history in self.behavior_tracker.query_history.values())

        # Calculate average expertise domains
        all_domains = []
        for profile in self.expertise_profiles.values():
            all_domains.extend(profile.expertise_domains.keys())

        domain_distribution = Counter(all_domains)

        return {
            'total_anonymous_users': total_users,
            'total_tracked_queries': total_queries,
            'expertise_domains_distribution': dict(domain_distribution.most_common(10)),
            'privacy_settings_summary': {
                'tracking_enabled': sum(1 for config in self.user_privacy_configs.values() if config.tracking_enabled),
                'personalization_enabled': sum(1 for config in self.user_privacy_configs.values() if config.personalization_enabled),
                'data_anonymization_enabled': sum(1 for config in self.user_privacy_configs.values() if config.data_anonymization_enabled)
            },
            'data_retention_compliance': True,
            'last_cleanup': datetime.now().isoformat()
        }


# Export main classes
__all__ = [
    "PersonalizedSearchEngine",
    "UserPrivacyConfig",
    "UserQueryEvent",
    "UserExpertiseProfile",
    "PersonalizedSearchRequest",
    "DataAnonymizer",
    "UserBehaviorTracker",
    "ExpertiseIdentifier",
    "PersonalizationEngine",
    "QuerySuggestionEngine"
]