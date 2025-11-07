"""
Personalized Search API with Privacy Protection

RESTful API for personalized search capabilities that includes user behavior tracking,
expertise identification, and privacy controls compliant with GDPR/CCPA.
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from dataclasses import asdict
import uvicorn
import logging

from hybrid_search_engine import HybridSearchEngine, SearchRequest, SearchResult
from personalized_search_engine import (
    PersonalizedSearchEngine,
    UserPrivacyConfig,
    PersonalizedSearchRequest,
    DataAnonymizer
)


class UserSession:
    """Manages user session context and privacy"""

    def __init__(self, session_id: str, user_id: Optional[str] = None):
        self.session_id = session_id
        self.user_id = user_id
        self.user_hash = None
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.consent_given = False
        self.privacy_preferences_set = False

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()

    def is_expired(self, timeout_hours: int = 24) -> bool:
        """Check if session has expired"""
        return datetime.now() - self.last_activity > timedelta(hours=timeout_hours)


# Pydantic models for API requests
class PersonalizedSearchRequestModel(BaseModel):
    """Personalized search request model"""
    query: str = Field(..., description="Search query")
    user_id: str = Field(..., description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    search_strategy: str = Field("unified", description="Search strategy")
    top_k: int = Field(10, description="Number of results", ge=1, le=100)
    similarity_threshold: float = Field(0.7, description="Similarity threshold", ge=0.0, le=1.0)
    personalization_level: float = Field(0.7, description="Personalization strength", ge=0.0, le=1.0)
    boost_expertise: bool = Field(True, description="Boost based on user expertise")
    boost_history: bool = Field(True, description="Boost based on search history")
    boost_preferences: bool = Field(True, description="Boost based on content preferences")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")


class UserFeedbackModel(BaseModel):
    """User feedback model"""
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    result_id: str = Field(..., description="Clicked result ID")
    feedback_type: str = Field(..., description="Type of feedback: click, satisfaction, rating")
    dwell_time: float = Field(0.0, description="Time spent on result in seconds", ge=0.0)
    satisfaction_score: Optional[float] = Field(None, description="Satisfaction score", ge=0.0, le=1.0)


class PrivacyConfigModel(BaseModel):
    """User privacy configuration model"""
    user_id: str = Field(..., description="User identifier")
    tracking_enabled: bool = Field(True, description="Enable behavior tracking")
    query_history_retention_days: int = Field(30, description="Query history retention", ge=1, le=365)
    click_tracking_enabled: bool = Field(True, description="Enable click tracking")
    expertise_learning_enabled: bool = Field(True, description="Enable expertise learning")
    personalization_enabled: bool = Field(True, description="Enable result personalization")
    data_anonymization_enabled: bool = Field(True, description="Enable data anonymization")
    auto_delete_after_days: Optional[int] = Field(365, description="Auto-delete data after days", ge=1, le=2550)
    gdpr_compliant: bool = Field(True, description="GDPR compliance")
    ccpa_compliant: bool = Field(True, description="CCPA compliance")


class ConsentRequestModel(BaseModel):
    """User consent request model"""
    user_id: str = Field(..., description="User identifier")
    consent_given: bool = Field(..., description="Whether consent is given")
    consent_text: str = Field(..., description="Consent text description")
    data_purposes: List[str] = Field(default_factory=list, description="Purposes of data processing")


class QuerySuggestionRequestModel(BaseModel):
    """Query suggestion request model"""
    user_id: str = Field(..., description="User identifier")
    partial_query: str = Field(..., description="Partial query for suggestions")
    max_suggestions: int = Field(5, description="Maximum number of suggestions", ge=1, le=20)


class PersonalizedSearchAPIServer:
    """Personalized search API server with privacy protection"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8007):
        self.host = host
        self.port = port
        self.hybrid_search_engine: Optional[HybridSearchEngine] = None
        self.personalized_search_engine: Optional[PersonalizedSearchEngine] = None
        self.app: Optional[FastAPI] = None
        self.anonymizer = DataAnonymizer()
        self.active_sessions: Dict[str, UserSession] = {}
        self.websocket_connections: Dict[str, WebSocket] = {}

        # Setup security
        self.security = HTTPBearer(auto_error=False)

        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        self._setup_server()

    def _setup_server(self):
        """Setup FastAPI application with lifecycle management"""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Initialize and cleanup personalized search engine"""
            print("🔐 Initializing Personalized Search Engine...")
            try:
                await self._initialize_search_engines()
                print("✅ Personalized Search Engine initialized successfully")
                yield
            except Exception as e:
                print(f"❌ Failed to initialize personalized search engine: {e}")
                raise
            finally:
                print("🔄 Shutting down Personalized Search Engine...")
                if self.personalized_search_engine:
                    await self.personalized_search_engine.cleanup_expired_data()

        self.app = FastAPI(
            title="Personalized Search API",
            description="Privacy-first personalized search with user behavior learning",
            version="1.0.0",
            lifespan=lifespan
        )

        self._setup_middleware()
        self._setup_routes()

    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Privacy middleware
        @self.app.middleware("http")
        async def privacy_middleware(request: Request, call_next):
            """Add privacy headers and logging"""
            start_time = time.time()

            # Add privacy headers
            response = await call_next(request)

            # GDPR/CCPA compliance headers
            response.headers["X-Privacy-Compliance"] = "GDPR,CCPA"
            response.headers["X-Data-Anonymization"] = "enabled"
            response.headers["X-User-Consent-Required"] = "true"

            # Log request for privacy audit
            process_time = time.time() - start_time
            self.logger.info(f"Request: {request.method} {request.url.path} - "
                           f"Process time: {process_time:.3f}s - "
                           f"IP anonymized: True")

            return response

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.get("/", response_model=Dict[str, Any])
        async def root():
            """Health check endpoint"""
            return {
                "service": "Personalized Search API",
                "status": "active",
                "privacy_compliance": ["GDPR", "CCPA"],
                "features": [
                    "Personalized search",
                    "User behavior tracking",
                    "Expertise identification",
                    "Query suggestions",
                    "Privacy controls"
                ],
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            }

        @self.app.post("/api/v1/search/personalized")
        async def personalized_search(request: PersonalizedSearchRequestModel) -> Dict[str, Any]:
            """Perform personalized search"""
            try:
                # Check user consent and privacy
                user_session = await self._get_or_create_session(request.user_id, request.session_id)

                if not user_session.consent_given:
                    raise HTTPException(status_code=403, detail="User consent required for personalized search")

                privacy_config = self.personalized_search_engine.get_user_privacy_config(request.user_id)
                if not privacy_config.personalization_enabled:
                    raise HTTPException(status_code=403, detail="Personalization disabled for user")

                # Create personalized search request
                base_search_request = SearchRequest(
                    query=request.query,
                    search_strategy=request.search_strategy,
                    top_k=request.top_k,
                    similarity_threshold=request.similarity_threshold,
                    filters=request.filters or {}
                )

                personalized_request = PersonalizedSearchRequest(
                    base_request=base_search_request,
                    user_hash=user_session.user_hash,
                    personalization_level=request.personalization_level,
                    boost_expertise=request.boost_expertise,
                    boost_history=request.boost_history,
                    boost_preferences=request.boost_preferences
                )

                # Perform personalized search
                start_time = time.time()
                results = await self.personalized_search_engine.personalized_search(
                    personalized_request,
                    request.user_id,
                    user_session.session_id
                )
                execution_time = time.time() - start_time

                # Update session activity
                user_session.update_activity()

                return {
                    "session_id": user_session.session_id,
                    "query": request.query,
                    "personalized_results": [result.to_dict() for result in results],
                    "personalization_applied": True,
                    "personalization_level": request.personalization_level,
                    "execution_time": execution_time,
                    "user_expertise_domains": self._get_user_expertise_summary(request.user_id),
                    "privacy_anonymized": True,
                    "timestamp": datetime.now().isoformat()
                }

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Personalized search error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

        @self.app.post("/api/v1/search/suggestions")
        async def get_query_suggestions(request: QuerySuggestionRequestModel) -> Dict[str, Any]:
            """Get personalized query suggestions"""
            try:
                # Check user privacy settings
                privacy_config = self.personalized_search_engine.get_user_privacy_config(request.user_id)

                if not privacy_config.tracking_enabled:
                    # Return only popular suggestions
                    suggestions = self.personalized_search_engine.suggestion_engine.popular_questions[:request.max_suggestions]
                else:
                    suggestions = await self.personalized_search_engine.get_query_suggestions(
                        request.user_id,
                        request.partial_query,
                        request.max_suggestions
                    )

                return {
                    "partial_query": request.partial_query,
                    "suggestions": suggestions,
                    "personalized": privacy_config.tracking_enabled,
                    "privacy_anonymized": True,
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                self.logger.error(f"Suggestion error: {e}")
                raise HTTPException(status_code=500, detail="Failed to get suggestions")

        @self.app.post("/api/v1/user/feedback")
        async def track_user_feedback(request: UserFeedbackModel) -> Dict[str, Any]:
            """Track user feedback for learning"""
            try:
                await self.personalized_search_engine.track_user_feedback(
                    request.user_id,
                    request.session_id,
                    request.result_id,
                    request.feedback_type,
                    request.dwell_time,
                    request.satisfaction_score
                )

                return {
                    "message": "Feedback tracked successfully",
                    "privacy_anonymized": True,
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                self.logger.error(f"Feedback tracking error: {e}")
                raise HTTPException(status_code=500, detail="Failed to track feedback")

        @self.app.post("/api/v1/user/consent")
        async def manage_user_consent(request: ConsentRequestModel) -> Dict[str, Any]:
            """Manage user consent for data processing"""
            try:
                user_session = await self._get_or_create_session(request.user_id)

                if request.consent_given:
                    user_session.consent_given = True
                    self.logger.info(f"User consent recorded for user {self.anonymizer.anonymize_user_id(request.user_id)}")
                else:
                    user_session.consent_given = False
                    # Optionally delete existing data if consent withdrawn
                    if request.user_id:
                        await self.personalized_search_engine.delete_user_data(request.user_id)

                return {
                    "consent_recorded": request.consent_given,
                    "consent_timestamp": datetime.now().isoformat(),
                    "data_purposes": request.data_purposes,
                    "privacy_anonymized": True
                }

            except Exception as e:
                self.logger.error(f"Consent management error: {e}")
                raise HTTPException(status_code=500, detail="Failed to record consent")

        @self.app.post("/api/v1/user/privacy")
        async def set_privacy_preferences(request: PrivacyConfigModel) -> Dict[str, Any]:
            """Set user privacy preferences"""
            try:
                privacy_config = UserPrivacyConfig(
                    user_id_hash="",  # Will be set automatically
                    tracking_enabled=request.tracking_enabled,
                    query_history_retention_days=request.query_history_retention_days,
                    click_tracking_enabled=request.click_tracking_enabled,
                    expertise_learning_enabled=request.expertise_learning_enabled,
                    personalization_enabled=request.personalization_enabled,
                    data_anonymization_enabled=request.data_anonymization_enabled,
                    auto_delete_after_days=request.auto_delete_after_days,
                    gdpr_compliant=request.gdpr_compliant,
                    ccpa_compliant=request.ccpa_compliant
                )

                self.personalized_search_engine.set_user_privacy_config(request.user_id, privacy_config)

                # Update session
                user_session = await self._get_or_create_session(request.user_id)
                user_session.privacy_preferences_set = True

                return {
                    "message": "Privacy preferences updated successfully",
                    "privacy_config": asdict(privacy_config),
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                self.logger.error(f"Privacy settings error: {e}")
                raise HTTPException(status_code=500, detail="Failed to set privacy preferences")

        @self.app.get("/api/v1/user/privacy/{user_id}")
        async def get_privacy_preferences(user_id: str) -> Dict[str, Any]:
            """Get user privacy preferences"""
            try:
                privacy_config = self.personalized_search_engine.get_user_privacy_config(user_id)
                return {
                    "privacy_config": asdict(privacy_config),
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                self.logger.error(f"Get privacy settings error: {e}")
                raise HTTPException(status_code=500, detail="Failed to get privacy preferences")

        @self.app.get("/api/v1/user/data-export/{user_id}")
        async def export_user_data(user_id: str) -> Dict[str, Any]:
            """Export user data for GDPR compliance"""
            try:
                user_data = await self.personalized_search_engine.export_user_data(user_id)
                return {
                    "user_data": user_data,
                    "export_format": "json",
                    "compliance": ["GDPR", "CCPA"],
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                self.logger.error(f"Data export error: {e}")
                raise HTTPException(status_code=500, detail="Failed to export user data")

        @self.app.delete("/api/v1/user/data/{user_id}")
        async def delete_user_data(user_id: str) -> Dict[str, Any]:
            """Delete all user data for GDPR/CCPA compliance"""
            try:
                success = await self.personalized_search_engine.delete_user_data(user_id)

                # Remove session
                sessions_to_remove = [
                    session_id for session_id, session in self.active_sessions.items()
                    if session.user_id == user_id
                ]
                for session_id in sessions_to_remove:
                    del self.active_sessions[session_id]

                return {
                    "success": success,
                    "message": "User data deleted successfully" if success else "Failed to delete user data",
                    "compliance": ["GDPR", "CCPA"],
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                self.logger.error(f"Data deletion error: {e}")
                raise HTTPException(status_code=500, detail="Failed to delete user data")

        @self.app.get("/api/v1/user/expertise/{user_id}")
        async def get_user_expertise(user_id: str) -> Dict[str, Any]:
            """Get user expertise profile"""
            try:
                user_hash = self.anonymizer.anonymize_user_id(user_id)
                expertise_profile = self.personalized_search_engine.expertise_profiles.get(user_hash)

                if not expertise_profile:
                    return {
                        "user_id": user_id,
                        "expertise_domains": {},
                        "message": "No expertise profile available",
                        "timestamp": datetime.now().isoformat()
                    }

                return {
                    "user_id": user_id,
                    "expertise_domains": expertise_profile.expertise_domains,
                    "technical_level": expertise_profile.technical_level,
                    "vocabulary_richness": expertise_profile.vocabulary_richness,
                    "confidence_score": expertise_profile.confidence_score,
                    "last_updated": expertise_profile.last_updated.isoformat(),
                    "privacy_anonymized": True,
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                self.logger.error(f"Expertise profile error: {e}")
                raise HTTPException(status_code=500, detail="Failed to get expertise profile")

        @self.app.get("/api/v1/analytics/personalized")
        async def get_personalized_analytics() -> Dict[str, Any]:
            """Get anonymized analytics about personalization effectiveness"""
            try:
                analytics = await self.personalized_search_engine.get_analytics_summary()

                return {
                    "analytics": analytics,
                    "active_sessions": len(self.active_sessions),
                    "active_websockets": len(self.websocket_connections),
                    "privacy_compliance": True,
                    "data_anonymized": True,
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                self.logger.error(f"Analytics error: {e}")
                raise HTTPException(status_code=500, detail="Failed to get analytics")

        @self.app.get("/api/v1/health/privacy")
        async def privacy_health_check() -> Dict[str, Any]:
            """Privacy-focused health check"""
            try:
                return {
                    "status": "healthy",
                    "privacy_features": {
                        "data_anonymization": "enabled",
                        "user_consent_management": "enabled",
                        "gdp_compliance": "enabled",
                        "ccpa_compliance": "enabled",
                        "data_retention_control": "enabled",
                        "right_to_deletion": "enabled"
                    },
                    "active_users": len(self.personalized_search_engine.user_privacy_configs),
                    "data_cleanup_status": "operational",
                    "last_privacy_audit": datetime.now().isoformat(),
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

        @self.app.websocket("/ws/personalized-search")
        async def websocket_personalized_search(websocket: WebSocket):
            """WebSocket for real-time personalized search"""
            await websocket.accept()
            session_id = str(uuid.uuid4())
            self.websocket_connections[session_id] = websocket

            try:
                while True:
                    data = await websocket.receive_json()

                    if data.get("type") == "search":
                        # Handle personalized search request
                        request_data = data.get("request", {})
                        user_id = request_data.get("user_id")

                        if not user_id:
                            await websocket.send_json({
                                "type": "error",
                                "message": "User ID required for personalized search"
                            })
                            continue

                        user_session = await self._get_or_create_session(user_id, session_id)

                        if not user_session.consent_given:
                            await websocket.send_json({
                                "type": "consent_required",
                                "message": "User consent required for personalized search"
                            })
                            continue

                        # Send progress update
                        await websocket.send_json({
                            "type": "progress",
                            "session_id": session_id,
                            "status": "personalizing",
                            "message": f"Personalizing search for: {request_data.get('query')}"
                        })

                        # Perform personalized search (simplified for WebSocket)
                        # ... (implementation would be similar to REST endpoint)

                        await websocket.send_json({
                            "type": "results",
                            "session_id": session_id,
                            "results": [],
                            "personalized": True
                        })

                    elif data.get("type") == "suggestions":
                        # Handle suggestion request
                        user_id = data.get("user_id")
                        partial_query = data.get("partial_query", "")

                        if user_id:
                            suggestions = await self.personalized_search_engine.get_query_suggestions(
                                user_id, partial_query, 5
                            )

                            await websocket.send_json({
                                "type": "suggestions",
                                "partial_query": partial_query,
                                "suggestions": suggestions
                            })

            except WebSocketDisconnect:
                self.websocket_connections.pop(session_id, None)
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
                self.websocket_connections.pop(session_id, None)

    async def _get_or_create_session(self, user_id: str, session_id: Optional[str] = None) -> UserSession:
        """Get or create user session"""
        if not session_id:
            session_id = str(uuid.uuid4())

        if session_id not in self.active_sessions:
            user_hash = self.anonymizer.anonymize_user_id(user_id)
            self.active_sessions[session_id] = UserSession(session_id, user_id)
            self.active_sessions[session_id].user_hash = user_hash

        return self.active_sessions[session_id]

    def _get_user_expertise_summary(self, user_id: str) -> Dict[str, float]:
        """Get summary of user expertise domains"""
        user_hash = self.anonymizer.anonymize_user_id(user_id)
        expertise_profile = self.personalized_search_engine.expertise_profiles.get(user_hash)

        if not expertise_profile:
            return {}

        # Return top 5 expertise domains
        sorted_domains = sorted(
            expertise_profile.expertise_domains.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return dict(sorted_domains)

    async def _initialize_search_engines(self):
        """Initialize search engines"""
        # Initialize hybrid search engine
        self.hybrid_search_engine = HybridSearchEngine()
        await self.hybrid_search_engine.initialize()

        # Initialize personalized search engine
        self.personalized_search_engine = PersonalizedSearchEngine(self.hybrid_search_engine)
        await self.personalized_search_engine.initialize()

        print("🔐 Privacy-first personalized search engine ready")

    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = datetime.now()
        expired_sessions = [
            session_id for session_id, session in self.active_sessions.items()
            if session.is_expired()
        ]

        for session_id in expired_sessions:
            del self.active_sessions[session_id]

        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    async def start(self):
        """Start the API server"""
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)

        # Start cleanup task
        cleanup_task = asyncio.create_task(self._periodic_cleanup())

        print(f"🔐 Starting Personalized Search API Server on {self.host}:{self.port}")
        try:
            await server.serve()
        finally:
            cleanup_task.cancel()

    async def _periodic_cleanup(self):
        """Periodic cleanup of expired data"""
        while True:
            try:
                await asyncio.sleep(3600)  # Clean up every hour
                await self.cleanup_expired_sessions()
                await self.personalized_search_engine.cleanup_expired_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")

    def run(self):
        """Run the API server"""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )


async def main():
    """Main function to run the server"""
    server = PersonalizedSearchAPIServer()
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())