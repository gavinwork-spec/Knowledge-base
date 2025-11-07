"""
User behavior analytics system for observability.

Tracks user sessions, behavior patterns, engagement metrics,
and provides insights for user experience optimization.
"""

import time
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from collections import defaultdict, deque
import threading
import json

from pydantic import BaseModel, Field

from ..core.metrics import get_metrics_collector
from ..core.logging import get_logger, log_user_action


class UserSegment(str, Enum):
    """User segmentation categories"""
    NEW_USER = "new_user"
    ACTIVE_USER = "active_user"
    POWER_USER = "power_user"
    CHURNED_USER = "churned_user"
    PREMIUM_USER = "premium_user"
    TRIAL_USER = "trial_user"


class SessionStatus(str, Enum):
    """Session status"""
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    TIMEOUT = "timeout"


class ActionType(str, Enum):
    """Types of user actions"""
    SEARCH_QUERY = "search_query"
    DOCUMENT_VIEW = "document_view"
    DOCUMENT_DOWNLOAD = "document_download"
    PERSONALIZED_SEARCH = "personalized_search"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    FEEDBACK_SUBMITTED = "feedback_submitted"
    SETTINGS_UPDATED = "settings_updated"
    EXPORT_DATA = "export_data"
    SHARE_CONTENT = "share_content"
    BOOKMARK_ADDED = "bookmark_added"


class UserBehavior(BaseModel):
    """Single user behavior/action"""
    user_id: str
    session_id: str
    action_type: ActionType
    resource_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


class UserSession(BaseModel):
    """User session information"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: SessionStatus = SessionStatus.ACTIVE
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    referrer: Optional[str] = None
    location: Optional[Dict[str, str]] = None
    device_info: Optional[Dict[str, str]] = None
    behaviors: List[UserBehavior] = Field(default_factory=list)
    total_actions: int = 0
    total_duration_ms: float = 0.0
    conversion_events: List[str] = Field(default_factory=list)
    satisfaction_score: Optional[float] = None
    is_first_session: bool = False

    @property
    def duration_minutes(self) -> float:
        """Get session duration in minutes"""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds() / 60

    @property
    def bounce_rate(self) -> float:
        """Calculate bounce rate (1 action = bounce)"""
        return 1.0 if self.total_actions <= 1 else 0.0

    @property
    def engagement_score(self) -> float:
        """Calculate engagement score (0-100)"""
        score = 0.0

        # Time-based score (max 30 points)
        duration_score = min(30, self.duration_minutes * 2)
        score += duration_score

        # Action-based score (max 40 points)
        action_score = min(40, self.total_actions * 4)
        score += action_score

        # Conversion-based score (max 30 points)
        conversion_score = min(30, len(self.conversion_events) * 10)
        score += conversion_score

        return min(100, score)


class UserProfile(BaseModel):
    """Comprehensive user profile"""
    user_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    last_seen: datetime = Field(default_factory=datetime.now)
    total_sessions: int = 0
    total_actions: int = 0
    total_duration_minutes: float = 0.0
    average_session_duration: float = 0.0
    average_actions_per_session: float = 0.0
    preferred_search_types: List[str] = Field(default_factory=list)
    expertise_domains: Dict[str, float] = Field(default_factory=dict)
    user_segment: UserSegment = UserSegment.NEW_USER
    satisfaction_scores: List[float] = Field(default_factory=list)
    average_satisfaction: float = 0.0
    conversion_events: Set[str] = Field(default_factory=set)
    retention_days: int = 0
    churn_probability: float = 0.0
    ltv_estimate: float = 0.0
    preferences: Dict[str, Any] = Field(default_factory=dict)
    custom_properties: Dict[str, Any] = Field(default_factory=dict)

    @property
    def days_since_last_seen(self) -> int:
        """Days since user was last active"""
        return (datetime.now() - self.last_seen).days

    @property
    def days_since_creation(self) -> int:
        """Days since user account was created"""
        return (datetime.now() - self.created_at).days

    @property
    def activity_frequency(self) -> float:
        """Average sessions per day"""
        days = max(1, self.days_since_creation)
        return self.total_sessions / days


class UserAnalyticsManager:
    """
    Manages user analytics, behavior tracking, and insights generation.
    """

    def __init__(self, session_timeout_minutes: int = 30):
        self.logger = get_logger()
        self.metrics = get_metrics_collector()
        self.session_timeout_minutes = session_timeout_minutes

        # Data storage
        self.active_sessions: Dict[str, UserSession] = {}
        self.user_profiles: Dict[str, UserProfile] = {}
        self.behavior_history: List[UserBehavior] = []
        self.cohort_analysis: Dict[str, Any] = {}

        # Thread safety
        self.lock = threading.RLock()

        # Start background tasks
        self._start_background_tasks()

    def start_session(
        self,
        user_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        referrer: Optional[str] = None,
        location: Optional[Dict[str, str]] = None,
        device_info: Optional[Dict[str, str]] = None
    ) -> UserSession:
        """Start a new user session"""
        with self.lock:
            # Check if user exists
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = UserProfile(
                    user_id=user_id,
                    is_first_session=True
                )
                self.logger.info(f"New user created: {user_id}")

            user_profile = self.user_profiles[user_id]

            # Create new session
            session = UserSession(
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                referrer=referrer,
                location=location,
                device_info=device_info,
                is_first_session=user_profile.total_sessions == 0
            )

            self.active_sessions[session.session_id] = session

            # Update user profile
            user_profile.last_seen = datetime.now()
            user_profile.total_sessions += 1

            # Log session start
            log_user_action(
                action_type="session_start",
                user_id=user_id,
                session_id=session.session_id,
                metadata={
                    "ip_address": ip_address,
                    "user_agent": user_agent,
                    "referrer": referrer,
                    "is_first_session": session.is_first_session
                }
            )

            # Update metrics
            self.metrics.increment_counter(
                "user_sessions_started",
                labels={
                    "is_first_session": str(session.is_first_session).lower(),
                    "user_segment": user_profile.user_segment.value
                }
            )

            self.logger.info(f"Session started: {session.session_id} for user {user_id}")
            return session

    def end_session(
        self,
        session_id: str,
        status: SessionStatus = SessionStatus.COMPLETED,
        satisfaction_score: Optional[float] = None
    ) -> Optional[UserSession]:
        """End a user session"""
        with self.lock:
            if session_id not in self.active_sessions:
                self.logger.warning(f"Session not found: {session_id}")
                return None

            session = self.active_sessions[session_id]
            session.end_time = datetime.now()
            session.status = status
            session.satisfaction_score = satisfaction_score

            # Update user profile
            user_profile = self.user_profiles[session.user_id]
            user_profile.total_duration_minutes += session.duration_minutes
            user_profile.average_session_duration = (
                user_profile.total_duration_minutes / user_profile.total_sessions
            )
            user_profile.total_actions += session.total_actions
            user_profile.average_actions_per_session = (
                user_profile.total_actions / user_profile.total_sessions
            )

            if satisfaction_score is not None:
                user_profile.satisfaction_scores.append(satisfaction_score)
                user_profile.average_satisfaction = sum(user_profile.satisfaction_scores) / len(user_profile.satisfaction_scores)

            # Update user segment
            self._update_user_segment(user_profile)

            # Remove from active sessions
            del self.active_sessions[session_id]

            # Add to behavior history
            for behavior in session.behaviors:
                self.behavior_history.append(behavior)

            # Keep history manageable
            if len(self.behavior_history) > 100000:
                self.behavior_history = self.behavior_history[-50000]

            # Log session end
            log_user_action(
                action_type="session_end",
                user_id=session.user_id,
                session_id=session_id,
                metadata={
                    "status": status.value,
                    "duration_minutes": session.duration_minutes,
                    "total_actions": session.total_actions,
                    "engagement_score": session.engagement_score,
                    "satisfaction_score": satisfaction_score
                }
            )

            # Update metrics
            self.metrics.increment_counter(
                "user_sessions_completed",
                labels={
                    "status": status.value,
                    "user_segment": user_profile.user_segment.value
                }
            )

            self.metrics.observe_histogram(
                "session_duration_minutes",
                session.duration_minutes,
                labels={
                    "status": status.value,
                    "user_segment": user_profile.user_segment.value
                }
            )

            self.logger.info(f"Session ended: {session_id} ({status.value})")
            return session

    def track_behavior(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        action_type: ActionType,
        resource_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Track user behavior/action"""
        with self.lock:
            # Find or create session
            if session_id and session_id in self.active_sessions:
                session = self.active_sessions[session_id]
            else:
                # Create implicit session if none exists
                session = self.start_session(user_id)
                session_id = session.session_id

            # Create behavior record
            behavior = UserBehavior(
                user_id=user_id,
                session_id=session_id,
                action_type=action_type,
                resource_id=resource_id,
                metadata=metadata or {},
                duration_ms=duration_ms,
                success=success,
                error_message=error_message
            )

            # Update session
            session.behaviors.append(behavior)
            session.total_actions += 1
            if duration_ms:
                session.total_duration_ms += duration_ms

            # Track conversion events
            if self._is_conversion_event(action_type):
                session.conversion_events.append(action_type.value)
                user_profile = self.user_profiles[user_id]
                user_profile.conversion_events.add(action_type.value)

            # Update expertise domains for search actions
            if action_type == ActionType.SEARCH_QUERY and metadata:
                self._update_expertise_domains(user_id, metadata)

            # Update preferences
            if metadata:
                self._update_user_preferences(user_id, metadata)

            # Log the behavior
            log_user_action(
                action_type=action_type.value,
                user_id=user_id,
                session_id=session_id,
                resource_id=resource_id,
                metadata={
                    "duration_ms": duration_ms,
                    "success": success,
                    "error_message": error_message,
                    **(metadata or {})
                }
            )

            # Update metrics
            self.metrics.increment_counter(
                "user_actions",
                labels={
                    "action_type": action_type.value,
                    "success": str(success).lower()
                }
            )

            if duration_ms:
                self.metrics.observe_histogram(
                    "action_duration_ms",
                    duration_ms,
                    labels={"action_type": action_type.value}
                )

    def _is_conversion_event(self, action_type: ActionType) -> bool:
        """Check if action type is a conversion event"""
        conversion_events = {
            ActionType.DOCUMENT_DOWNLOAD,
            ActionType.FEEDBACK_SUBMITTED,
            ActionType.EXPORT_DATA,
            ActionType.SHARE_CONTENT,
            ActionType.BOOKMARK_ADDED
        }
        return action_type in conversion_events

    def _update_expertise_domains(self, user_id: str, metadata: Dict[str, Any]):
        """Update user expertise domains based on search behavior"""
        user_profile = self.user_profiles[user_id]

        # Extract domain information from search query or results
        query = metadata.get("query", "").lower()
        categories = metadata.get("categories", [])
        tags = metadata.get("tags", [])

        # Simple domain extraction (in practice, this would be more sophisticated)
        domains = set()
        if "machine learning" in query or "ml" in query:
            domains.add("machine_learning")
        if "api" in query or "rest" in query:
            domains.add("api_development")
        if "database" in query or "sql" in query:
            domains.add("database")
        if categories:
            domains.update(categories)
        if tags:
            domains.update(tags)

        # Update expertise scores
        for domain in domains:
            current_score = user_profile.expertise_domains.get(domain, 0.0)
            user_profile.expertise_domains[domain] = min(1.0, current_score + 0.05)

    def _update_user_preferences(self, user_id: str, metadata: Dict[str, Any]):
        """Update user preferences based on behavior"""
        user_profile = self.user_profiles[user_id]

        # Track preferred search types
        if "search_strategy" in metadata:
            strategy = metadata["search_strategy"]
            if strategy not in user_profile.preferred_search_types:
                user_profile.preferred_search_types.append(strategy)
                # Keep only recent preferences
                user_profile.preferred_search_types = user_profile.preferred_search_types[-10:]

    def _update_user_segment(self, user_profile: UserProfile):
        """Update user segment based on behavior patterns"""
        days_since_creation = user_profile.days_since_creation
        days_since_last_seen = user_profile.days_since_last_seen
        activity_frequency = user_profile.activity_frequency
        avg_session_duration = user_profile.average_session_duration

        # Churned user
        if days_since_last_seen > 30:
            user_profile.user_segment = UserSegment.CHURNED_USER
        # Power user
        elif activity_frequency > 1.0 and avg_session_duration > 10:
            user_profile.user_segment = UserSegment.POWER_USER
        # Active user
        elif days_since_creation > 7 and days_since_last_seen < 7:
            user_profile.user_segment = UserSegment.ACTIVE_USER
        # Trial user (first 7 days)
        elif days_since_creation <= 7:
            user_profile.user_segment = UserSegment.TRIAL_USER
        # New user (7-30 days with low activity)
        elif days_since_creation <= 30 and activity_frequency < 0.5:
            user_profile.user_segment = UserSegment.NEW_USER

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID"""
        with self.lock:
            return self.user_profiles.get(user_id)

    def get_active_session(self, session_id: str) -> Optional[UserSession]:
        """Get active session by ID"""
        with self.lock:
            return self.active_sessions.get(session_id)

    def get_user_sessions(
        self,
        user_id: str,
        limit: int = 100,
        include_active: bool = True
    ) -> List[UserSession]:
        """Get user's session history"""
        sessions = []

        # Get active sessions
        if include_active:
            for session in self.active_sessions.values():
                if session.user_id == user_id:
                    sessions.append(session)

        # Get historical sessions from behavior history
        user_sessions = defaultdict(list)
        for behavior in self.behavior_history:
            if behavior.user_id == user_id:
                user_sessions[behavior.session_id].append(behavior)

        # Convert behavior groups to sessions (simplified)
        for session_id, behaviors in user_sessions.items():
            if len(behaviors) > 0:
                session = UserSession(
                    session_id=session_id,
                    user_id=user_id,
                    start_time=min(b.timestamp for b in behaviors),
                    end_time=max(b.timestamp for b in behaviors),
                    behaviors=behaviors,
                    total_actions=len(behaviors),
                    status=SessionStatus.COMPLETED
                )
                sessions.append(session)

        # Sort by start time and limit
        sessions.sort(key=lambda s: s.start_time, reverse=True)
        return sessions[:limit]

    def get_behavior_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
        action_type: Optional[ActionType] = None
    ) -> Dict[str, Any]:
        """Get behavior analytics"""
        end_date = end_date or datetime.now()
        start_date = start_date or end_date - timedelta(days=30)

        with self.lock:
            # Filter behaviors
            filtered_behaviors = [
                b for b in self.behavior_history
                if start_date <= b.timestamp <= end_date
                and (user_id is None or b.user_id == user_id)
                and (action_type is None or b.action_type == action_type)
            ]

        if not filtered_behaviors:
            return {
                "period": {"start": start_date, "end": end_date},
                "total_behaviors": 0,
                "unique_users": 0,
                "behaviors_by_type": {},
                "behaviors_by_user": {},
                "success_rate": 0.0,
                "average_duration_ms": 0.0,
                "hourly_distribution": {},
                "daily_distribution": {}
            }

        # Calculate aggregates
        total_behaviors = len(filtered_behaviors)
        unique_users = len(set(b.user_id for b in filtered_behaviors))
        behaviors_by_type = defaultdict(int)
        behaviors_by_user = defaultdict(int)
        successful_behaviors = 0
        total_duration = 0
        hourly_distribution = defaultdict(int)
        daily_distribution = defaultdict(int)

        for behavior in filtered_behaviors:
            behaviors_by_type[behavior.action_type.value] += 1
            behaviors_by_user[behavior.user_id] += 1

            if behavior.success:
                successful_behaviors += 1

            if behavior.duration_ms:
                total_duration += behavior.duration_ms

            hourly_distribution[behavior.timestamp.hour] += 1
            daily_distribution[behavior.timestamp.strftime("%Y-%m-%d")] += 1

        return {
            "period": {"start": start_date, "end": end_date},
            "total_behaviors": total_behaviors,
            "unique_users": unique_users,
            "behaviors_by_type": dict(behaviors_by_type),
            "behaviors_by_user": dict(sorted(behaviors_by_user.items(), key=lambda x: x[1], reverse=True)[:20]),
            "success_rate": successful_behaviors / total_behaviors,
            "average_duration_ms": total_duration / total_behaviors if total_behaviors > 0 else 0,
            "hourly_distribution": dict(hourly_distribution),
            "daily_distribution": dict(daily_distribution)
        }

    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive insights for a specific user"""
        user_profile = self.get_user_profile(user_id)
        if not user_profile:
            return {"error": "User not found"}

        sessions = self.get_user_sessions(user_id, limit=50)
        recent_behaviors = [
            b for b in self.behavior_history
            if b.user_id == user_id and b.timestamp > datetime.now() - timedelta(days=30)
        ]

        # Calculate insights
        insights = {
            "user_profile": user_profile.dict(),
            "session_summary": {
                "total_sessions": len(sessions),
                "average_duration": user_profile.average_session_duration,
                "average_actions": user_profile.average_actions_per_session,
                "bounce_rate": sum(s.bounce_rate for s in sessions) / len(sessions) if sessions else 0
            },
            "behavior_patterns": {
                "most_common_actions": self._get_most_common_actions(recent_behaviors),
                "peak_activity_hours": self._get_peak_activity_hours(recent_behaviors),
                "preferred_search_types": user_profile.preferred_search_types,
                "expertise_domains": sorted(
                    user_profile.expertise_domains.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            },
            "engagement_metrics": {
                "engagement_trend": self._calculate_engagement_trend(sessions),
                "retention_days": user_profile.days_since_last_seen,
                "churn_probability": user_profile.churn_probability,
                "ltv_estimate": user_profile.ltv_estimate
            },
            "recommendations": self._generate_user_recommendations(user_profile, sessions, recent_behaviors)
        }

        return insights

    def _get_most_common_actions(self, behaviors: List[UserBehavior]) -> List[Dict[str, Any]]:
        """Get most common actions for user"""
        action_counts = defaultdict(int)
        action_durations = defaultdict(list)

        for behavior in behaviors:
            action_counts[behavior.action_type.value] += 1
            if behavior.duration_ms:
                action_durations[behavior.action_type.value].append(behavior.duration_ms)

        result = []
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            durations = action_durations[action]
            avg_duration = sum(durations) / len(durations) if durations else 0

            result.append({
                "action": action,
                "count": count,
                "average_duration_ms": avg_duration
            })

        return result[:10]

    def _get_peak_activity_hours(self, behaviors: List[UserBehavior]) -> List[int]:
        """Get peak activity hours for user"""
        hour_counts = defaultdict(int)
        for behavior in behaviors:
            hour_counts[behavior.timestamp.hour] += 1

        return sorted(hour_counts.keys(), key=lambda h: hour_counts[h], reverse=True)[:3]

    def _calculate_engagement_trend(self, sessions: List[UserSession]) -> str:
        """Calculate engagement trend over time"""
        if len(sessions) < 2:
            return "insufficient_data"

        # Split sessions into two halves
        mid_point = len(sessions) // 2
        old_sessions = sessions[mid_point:]
        new_sessions = sessions[:mid_point]

        old_avg_engagement = sum(s.engagement_score for s in old_sessions) / len(old_sessions)
        new_avg_engagement = sum(s.engagement_score for s in new_sessions) / len(new_sessions)

        if new_avg_engagement > old_avg_engagement * 1.1:
            return "improving"
        elif new_avg_engagement < old_avg_engagement * 0.9:
            return "declining"
        else:
            return "stable"

    def _generate_user_recommendations(
        self,
        user_profile: UserProfile,
        sessions: List[UserSession],
        behaviors: List[UserBehavior]
    ) -> List[str]:
        """Generate personalized recommendations for user"""
        recommendations = []

        # Low engagement recommendations
        if user_profile.average_session_duration < 2:
            recommendations.append("Try exploring different search features to get more value from the platform")

        if user_profile.average_actions_per_session < 3:
            recommendations.append("Consider using advanced search filters to find more relevant results")

        # Expertise-based recommendations
        if len(user_profile.expertise_domains) == 0:
            recommendations.append("Explore different knowledge domains to build your expertise profile")
        else:
            top_domain = max(user_profile.expertise_domains.items(), key=lambda x: x[1])
            recommendations.append(f"You show expertise in {top_domain[0]} - explore more content in this area")

        # Session-based recommendations
        if user_profile.days_since_last_seen > 7:
            recommendations.append("Welcome back! You have new content waiting in your areas of interest")

        # Satisfaction-based recommendations
        if user_profile.average_satisfaction < 3.0 and user_profile.satisfaction_scores:
            recommendations.append("We notice you've had some challenges - try our guided search feature")

        return recommendations[:5]

    def _start_background_tasks(self):
        """Start background tasks for session cleanup and analytics"""
        def cleanup_tasks():
            while True:
                try:
                    time.sleep(300)  # Run every 5 minutes
                    self._cleanup_expired_sessions()
                    self._update_analytics()
                except Exception as e:
                    self.logger.error(f"Error in analytics cleanup: {e}")

        thread = threading.Thread(target=cleanup_tasks, daemon=True)
        thread.start()

    def _cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        with self.lock:
            expired_sessions = []
            current_time = datetime.now()

            for session_id, session in self.active_sessions.items():
                session_duration = (current_time - session.start_time).total_seconds() / 60
                if session_duration > self.session_timeout_minutes:
                    expired_sessions.append(session_id)

            for session_id in expired_sessions:
                self.end_session(session_id, SessionStatus.TIMEOUT)

            if expired_sessions:
                self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    def _update_analytics(self):
        """Update analytical calculations"""
        with self.lock:
            # Update churn probability and LTV for all users
            for user_id, profile in self.user_profiles.items():
                # Simple churn probability based on recent activity
                days_since_last_seen = profile.days_since_last_seen
                if days_since_last_seen > 30:
                    profile.churn_probability = min(1.0, days_since_last_seen / 90)
                elif days_since_last_seen > 14:
                    profile.churn_probability = 0.3
                else:
                    profile.churn_probability = 0.1

                # Simple LTV estimate based on activity and satisfaction
                avg_monthly_value = profile.total_sessions * 0.1  # Assumed value per session
                satisfaction_multiplier = profile.average_satisfaction / 5.0 if profile.average_satisfaction > 0 else 1.0
                profile.ltv_estimate = avg_monthly_value * satisfaction_multiplier * 12  # Annual estimate

                # Update retention days
                profile.retention_days = profile.days_since_creation


# Global user analytics manager instance
_global_user_analytics: Optional[UserAnalyticsManager] = None


def get_user_analytics() -> UserAnalyticsManager:
    """Get global user analytics manager instance"""
    global _global_user_analytics
    if _global_user_analytics is None:
        _global_user_analytics = UserAnalyticsManager()
    return _global_user_analytics


def configure_user_analytics(**kwargs):
    """Configure global user analytics manager"""
    global _global_user_analytics
    _global_user_analytics = UserAnalyticsManager(**kwargs)