"""
LobeChat Integration Implementation
Manufacturing Knowledge Base - Modern Chat Interface Integration

This module provides the main LobeChat integration implementation with manufacturing-specific
enhancements and optimizations.
"""

import asyncio
import json
import logging
import websockets
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from ..shared.base import IntegrationBase, ManufacturingContext, IntegrationStatus
from ..shared.errors import IntegrationError

logger = logging.getLogger(__name__)


class ChatMessageType(Enum):
    """Chat message types"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    SAFETY_ALERT = "safety_alert"
    QUALITY_NOTIFICATION = "quality_notification"
    MAINTENANCE_ALERT = "maintenance_alert"


@dataclass
class ChatMessage:
    """Chat message structure"""
    id: str
    type: ChatMessageType
    content: str
    timestamp: datetime
    user_id: str
    session_id: str
    metadata: Dict[str, Any] = None
    attachments: List[Dict[str, Any]] = None


@dataclass
class ChatSession:
    """Chat session information"""
    session_id: str
    user_id: str
    user_role: str
    equipment_type: Optional[str] = None
    process_type: Optional[str] = None
    context: Dict[str, Any] = None
    messages: List[ChatMessage] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self.context is None:
            self.context = {}
        if self.created_at is None:
            self.created_at = datetime.now()


class LobeChatIntegration(IntegrationBase):
    """
    LobeChat integration for manufacturing knowledge base.
    Provides modern chat interface with manufacturing-specific enhancements.
    """

    def __init__(self, name: str, config):
        super().__init__(name, config)

        self.websocket_manager = None
        self.session_manager = None
        self.chat_interface = None
        self.active_sessions: Dict[str, ChatSession] = {}

        # WebSocket configuration
        self.websocket_url = self.config.get("websocket.url", "ws://localhost:8765")
        self.reconnect_attempts = self.config.get("websocket.reconnect_attempts", 5)
        self.reconnect_delay = self.config.get("websocket.reconnect_delay", 1000)
        self.heartbeat_interval = self.config.get("websocket.heartbeat_interval", 30)

        # Chat interface configuration
        self.theme = self.config.get("frontend.theme", "manufacturing")
        self.language = self.config.get("frontend.language", "en")
        self.custom_components = self.config.get("frontend.custom_components", True)
        self.manufacturing_theme = self.config.get("frontend.manufacturing_theme", True)

        # Manufacturing-specific features
        self.quick_actions = self.config.get("features.quick_actions", True)
        self.file_upload = self.config.get("features.file_upload", True)
        self.image_recognition = self.config.get("features.image_recognition", False)
        self.voice_input = self.config.get("features.voice_input", False)
        self.suggested_responses = self.config.get("features.suggested_responses", True)
        self.manufacturing_templates = self.config.get("features.manufacturing_templates", True)

        logger.info(f"Initialized LobeChat integration with {self.theme} theme")

    async def initialize(self) -> bool:
        """Initialize LobeChat components"""
        try:
            logger.info("Initializing LobeChat integration components")

            # Initialize WebSocket manager
            await self._initialize_websocket_manager()

            # Initialize session manager
            await self._initialize_session_manager()

            # Initialize chat interface
            await self._initialize_chat_interface()

            # Load manufacturing templates
            await self._load_manufacturing_templates()

            # Setup quick actions
            await self._setup_quick_actions()

            self.status = IntegrationStatus.READY
            self.start_time = datetime.now().timestamp()

            logger.info("LobeChat integration initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize LobeChat integration: {e}")
            self.status = IntegrationStatus.ERROR
            return False

    async def _initialize_websocket_manager(self):
        """Initialize WebSocket manager for real-time communication"""
        try:
            self.websocket_manager = WebSocketManager(
                url=self.websocket_url,
                reconnect_attempts=self.reconnect_attempts,
                reconnect_delay=self.reconnect_delay,
                heartbeat_interval=self.heartbeat_interval,
                integration=self
            )

            await self.websocket_manager.start()

        except Exception as e:
            logger.error(f"Failed to initialize WebSocket manager: {e}")
            raise

    async def _initialize_session_manager(self):
        """Initialize chat session manager"""
        try:
            self.session_manager = ChatSessionManager(
                max_sessions_per_user=5,
                session_timeout=3600,  # 1 hour
                persistence_enabled=True
            )

        except Exception as e:
            logger.error(f"Failed to initialize session manager: {e}")
            raise

    async def _initialize_chat_interface(self):
        """Initialize chat interface components"""
        try:
            from .ui import ChatComponents
            from .themes import ManufacturingTheme

            # Initialize chat components with manufacturing theme
            if self.manufacturing_theme:
                theme = ManufacturingTheme()
            else:
                theme = ManufacturingTheme()  # Default to manufacturing theme

            self.chat_interface = ChatComponents(
                theme=theme,
                language=self.language,
                custom_components=self.custom_components,
                features={
                    'file_upload': self.file_upload,
                    'image_recognition': self.image_recognition,
                    'voice_input': self.voice_input,
                    'suggested_responses': self.suggested_responses
                }
            )

        except Exception as e:
            logger.error(f"Failed to initialize chat interface: {e}")
            raise

    async def _load_manufacturing_templates(self):
        """Load manufacturing-specific chat templates"""
        try:
            from .templates import ManufacturingTemplates

            templates = ManufacturingTemplates()
            self.manufacturing_templates = templates.get_templates()

            logger.info(f"Loaded {len(self.manufacturing_templates)} manufacturing templates")

        except Exception as e:
            logger.error(f"Failed to load manufacturing templates: {e}")
            # Continue without templates
            self.manufacturing_templates = {}

    async def _setup_quick_actions(self):
        """Setup manufacturing-specific quick actions"""
        try:
            if not self.quick_actions:
                return

            self.quick_actions_config = [
                {
                    "id": "safety_check",
                    "title": "Safety Check",
                    "description": "Check safety procedures for equipment",
                    "icon": "shield",
                    "color": "#dc2626",  # Safety red
                    "template": "safety_procedure",
                    "params": {"equipment_type": "auto"}
                },
                {
                    "id": "quality_inspection",
                    "title": "Quality Inspection",
                    "description": "Get quality inspection procedures",
                    "icon": "check-circle",
                    "color": "#059669",  # Quality green
                    "template": "quality_control",
                    "params": {"inspection_type": "general"}
                },
                {
                    "id": "maintenance_request",
                    "title": "Maintenance Request",
                    "description": "Create maintenance request",
                    "icon": "tool",
                    "color": "#7c3aed",  # Maintenance purple
                    "template": "maintenance",
                    "params": {"priority": "medium"}
                },
                {
                    "id": "technical_manual",
                    "title": "Technical Manual",
                    "description": "Access technical documentation",
                    "icon": "book",
                    "color": "#2563eb",  # Documentation blue
                    "template": "technical_spec",
                    "params": {"document_type": "manual"}
                },
                {
                    "id": "procedure_lookup",
                    "title": "Procedure Lookup",
                    "description": "Find standard operating procedures",
                    "icon": "list",
                    "color": "#0891b2",  # Procedure cyan
                    "template": "procedure",
                    "params": {"procedure_type": "standard"}
                }
            ]

            logger.info(f"Setup {len(self.quick_actions_config)} quick actions")

        except Exception as e:
            logger.error(f"Failed to setup quick actions: {e}")
            self.quick_actions_config = []

    async def shutdown(self) -> bool:
        """Shutdown LobeChat integration gracefully"""
        try:
            logger.info("Shutting down LobeChat integration")

            # Close all active sessions
            for session_id in list(self.active_sessions.keys()):
                await self.close_session(session_id)

            # Shutdown WebSocket manager
            if self.websocket_manager:
                await self.websocket_manager.stop()

            # Shutdown session manager
            if self.session_manager:
                await self.session_manager.shutdown()

            # Clear components
            self.active_sessions.clear()
            self.chat_interface = None

            self.status = IntegrationStatus.SHUTDOWN
            logger.info("LobeChat integration shutdown successfully")
            return True

        except Exception as e:
            logger.error(f"Error during LobeChat shutdown: {e}")
            return False

    async def _integration_health_check(self) -> Dict[str, Any]:
        """Perform LobeChat-specific health check"""
        try:
            health_status = {
                "websocket_manager_available": self.websocket_manager is not None,
                "session_manager_available": self.session_manager is not None,
                "chat_interface_available": self.chat_interface is not None,
                "active_sessions": len(self.active_sessions),
                "quick_actions_configured": len(getattr(self, 'quick_actions_config', [])),
                "templates_loaded": len(getattr(self, 'manufacturing_templates', {})),
                "theme": self.theme,
                "language": self.language,
            }

            # Test WebSocket connection
            if self.websocket_manager:
                try:
                    ws_health = await self.websocket_manager.health_check()
                    health_status["websocket_test"] = ws_health.get("status", "unknown")
                except Exception as e:
                    health_status["websocket_test"] = f"failed: {str(e)}"

            # Test session manager
            if self.session_manager:
                try:
                    session_health = await self.session_manager.health_check()
                    health_status["session_manager_test"] = session_health.get("status", "unknown")
                except Exception as e:
                    health_status["session_manager_test"] = f"failed: {str(e)}"

            return health_status

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def _process_with_context(
        self,
        request_data: Any,
        context: ManufacturingContext
    ) -> Any:
        """
        Process chat request with LobeChat and manufacturing context
        """
        try:
            # Extract request parameters
            if isinstance(request_data, str):
                # Simple message
                return await self._process_chat_message(
                    message=request_data,
                    context=context,
                    session_id=None
                )
            elif isinstance(request_data, dict):
                # Structured chat request
                message_type = request_data.get("type", "message")
                session_id = request_data.get("session_id")
                message = request_data.get("message", "")
                user_id = request_data.get("user_id", context.user_role)

                if message_type == "message":
                    return await self._process_chat_message(
                        message=message,
                        context=context,
                        session_id=session_id,
                        user_id=user_id
                    )
                elif message_type == "quick_action":
                    return await self._process_quick_action(
                        action_id=request_data.get("action_id"),
                        params=request_data.get("params", {}),
                        context=context,
                        session_id=session_id
                    )
                elif message_type == "file_upload":
                    return await self._process_file_upload(
                        file_data=request_data.get("file_data"),
                        context=context,
                        session_id=session_id
                    )
                else:
                    raise ValueError(f"Unknown message type: {message_type}")

            else:
                raise ValueError("Invalid request data format")

        except Exception as e:
            logger.error(f"Error processing request with LobeChat: {e}")
            raise IntegrationError(f"Request processing failed: {e}")

    async def _process_chat_message(
        self,
        message: str,
        context: ManufacturingContext,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process chat message"""
        try:
            # Create or get session
            if session_id is None:
                session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"

            session = await self._get_or_create_session(session_id, user_id or "anonymous", context)

            # Create user message
            user_message = ChatMessage(
                id=f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                type=ChatMessageType.USER,
                content=message,
                timestamp=datetime.now(),
                user_id=user_id or "anonymous",
                session_id=session_id,
                metadata={
                    "equipment_type": context.equipment_type,
                    "process_type": context.process_type,
                    "user_role": context.user_role
                }
            )

            session.messages.append(user_message)

            # Process message with context-aware response
            response = await self._generate_contextual_response(message, context, session)

            # Create assistant message
            assistant_message = ChatMessage(
                id=f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                type=ChatMessageType.ASSISTANT,
                content=response["content"],
                timestamp=datetime.now(),
                user_id="assistant",
                session_id=session_id,
                metadata=response.get("metadata", {})
            )

            session.messages.append(assistant_message)

            # Broadcast message to WebSocket
            if self.websocket_manager:
                await self.websocket_manager.broadcast_message(session_id, assistant_message)

            return {
                "response": response,
                "session_id": session_id,
                "message_id": assistant_message.id,
                "timestamp": assistant_message.timestamp.isoformat(),
                "session_history": len(session.messages),
                "suggested_responses": response.get("suggested_responses", []),
                "quick_actions": response.get("quick_actions", [])
            }

        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            raise

    async def _process_quick_action(
        self,
        action_id: str,
        params: Dict[str, Any],
        context: ManufacturingContext,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process quick action"""
        try:
            # Find quick action configuration
            action_config = None
            for action in getattr(self, 'quick_actions_config', []):
                if action["id"] == action_id:
                    action_config = action
                    break

            if not action_config:
                raise ValueError(f"Unknown quick action: {action_id}")

            # Merge params with action defaults
            action_params = {**action_config.get("params", {}), **params}

            # Generate contextual response for quick action
            template = self.manufacturing_templates.get(action_config["template"])
            if template:
                response = await self._execute_template(template, context, action_params)
            else:
                response = await self._generate_quick_action_response(action_config, context, action_params)

            return {
                "response": response,
                "action_id": action_id,
                "action_title": action_config["title"],
                "action_type": "quick_action",
                "timestamp": datetime.now().isoformat(),
                "suggested_responses": response.get("suggested_responses", [])
            }

        except Exception as e:
            logger.error(f"Error processing quick action {action_id}: {e}")
            raise

    async def _process_file_upload(
        self,
        file_data: Dict[str, Any],
        context: ManufacturingContext,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process file upload"""
        try:
            filename = file_data.get("filename", "unknown_file")
            file_content = file_data.get("content", "")
            file_type = file_data.get("type", "unknown")

            # Process file based on type
            if file_type.startswith("image/"):
                response = await self._process_image_file(file_content, filename, context)
            elif file_type == "application/pdf":
                response = await self._process_pdf_file(file_content, filename, context)
            else:
                response = await self._process_text_file(file_content, filename, context)

            return {
                "response": response,
                "filename": filename,
                "file_type": file_type,
                "processing_status": "completed",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error processing file upload: {e}")
            raise

    async def _get_or_create_session(
        self,
        session_id: str,
        user_id: str,
        context: ManufacturingContext
    ) -> ChatSession:
        """Get existing session or create new one"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            # Update context
            session.context.update(context.get_context_dict())
            return session

        # Create new session
        session = ChatSession(
            session_id=session_id,
            user_id=user_id,
            user_role=context.user_role,
            equipment_type=context.equipment_type,
            process_type=context.process_type,
            context=context.get_context_dict()
        )

        self.active_sessions[session_id] = session
        return session

    async def _generate_contextual_response(
        self,
        message: str,
        context: ManufacturingContext,
        session: ChatSession
    ) -> Dict[str, Any]:
        """Generate contextual response using AI backend"""
        try:
            # Get LangChain integration for AI processing
            from ..shared.base import IntegrationManager
            manager = IntegrationManager()

            # This would integrate with LangChain for AI processing
            # For now, return a contextual response
            contextual_info = []

            if context.equipment_type:
                contextual_info.append(f"Equipment: {context.equipment_type}")
            if context.process_type:
                contextual_info.append(f"Process: {context.process_type}")
            if context.user_role:
                contextual_info.append(f"User Role: {context.user_role}")

            # Generate manufacturing-specific response
            response_content = self._generate_manufacturing_response(
                message,
                contextual_info,
                session
            )

            return {
                "content": response_content,
                "metadata": {
                    "context_used": contextual_info,
                    "message_type": "manufacturing_query",
                    "user_role": context.user_role
                },
                "suggested_responses": self._generate_suggested_responses(message, context),
                "quick_actions": self._suggest_quick_actions(message, context)
            }

        except Exception as e:
            logger.error(f"Error generating contextual response: {e}")
            return {
                "content": "I apologize, but I'm having trouble processing your request right now. Please try again.",
                "metadata": {"error": str(e)},
                "suggested_responses": ["Try rephrasing your question"],
                "quick_actions": []
            }

    def _generate_manufacturing_response(
        self,
        message: str,
        contextual_info: List[str],
        session: ChatSession
    ) -> str:
        """Generate manufacturing-specific response"""
        # This would integrate with LangChain for actual AI processing
        # For now, provide contextual responses

        message_lower = message.lower()

        # Safety-related queries
        if any(keyword in message_lower for keyword in ["safety", "danger", "hazard", "protective"]):
            if contextual_info:
                return f"Based on your role as {contextual_info[-1]} and the {contextual_info[0] if contextual_info else 'equipment'}, here are the safety procedures:\n\n1. Always wear appropriate PPE including safety glasses and steel-toed boots\n2. Ensure all machine guards are in place before operation\n3. Follow proper lockout/tagout procedures during maintenance\n4. Keep the work area clean and organized\n5. Know the location of emergency stop buttons\n\nWould you like more specific safety procedures for a particular operation?"
            else:
                return "For safety procedures, I need to know what type of equipment or process you're working with. Could you please specify the equipment type or operation you need safety information for?"

        # Quality-related queries
        elif any(keyword in message_lower for keyword in ["quality", "inspection", "defect", "tolerance"]):
            return f"Here are the quality control procedures:\n\n1. Verify all specifications before starting\n2. Use calibrated measuring equipment\n3. Follow the inspection checklist\n4. Document all findings and measurements\n5. Report any deviations or concerns immediately\n\nWould you like specific inspection procedures for a particular part or process?"

        # Technical specification queries
        elif any(keyword in message_lower for keyword in ["specification", "manual", "blueprint", "drawing"]):
            return f"I can help you with technical specifications and documentation. Please provide:\n\n• The equipment model or part number\n• Specific specification you need\n• The process or application\n\nThis will help me provide you with the most relevant technical information and procedures."

        # Maintenance queries
        elif any(keyword in message_lower for keyword in ["maintenance", "repair", "service", "lubrication"]):
            return f"For maintenance procedures:\n\n1. Follow the manufacturer's maintenance schedule\n2. Use proper lubricants and consumables\n3. Keep detailed maintenance records\n4. Inspect for wear and tear regularly\n5. Address issues promptly to prevent downtime\n\nWhat specific maintenance procedure do you need help with?"

        # General manufacturing queries
        else:
            return f"I'm here to help with manufacturing questions! I can assist with:\n\n• Safety procedures and OSHA compliance\n• Quality control and inspection methods\n• Technical specifications and blueprints\n• Equipment maintenance and troubleshooting\n• Process optimization and best practices\n\nPlease let me know what specific manufacturing topic you'd like to discuss."

    def _generate_suggested_responses(self, message: str, context: ManufacturingContext) -> List[str]:
        """Generate suggested follow-up responses"""
        suggestions = []

        message_lower = message.lower()

        if "safety" in message_lower:
            suggestions.extend([
                "What are the emergency procedures?",
                "Show me the lockout/tagout process",
                "What PPE is required?"
            ])
        elif "quality" in message_lower:
            suggestions.extend([
                "What are the inspection criteria?",
                "How do I document quality issues?",
                "What measuring tools do I need?"
            ])
        elif "maintenance" in message_lower:
            suggestions.extend([
                "What's the maintenance schedule?",
                "How do I troubleshoot this issue?",
                "What lubricants should I use?"
            ])
        elif "specification" in message_lower:
            suggestions.extend([
                "What are the operating parameters?",
                "What are the material specifications?",
                "Where can I find the manual?"
            ])

        # Context-aware suggestions
        if context.equipment_type:
            suggestions.append(f"Procedures for {context.equipment_type}")

        if context.process_type:
            suggestions.append(f"Best practices for {context.process_type}")

        return suggestions[:3]  # Limit to 3 suggestions

    def _suggest_quick_actions(self, message: str, context: ManufacturingContext) -> List[str]:
        """Suggest relevant quick actions"""
        actions = []

        # Always suggest quick actions if enabled
        if self.quick_actions:
            actions = [action["id"] for action in getattr(self, 'quick_actions_config', [])]

        # Context-specific actions
        if context.equipment_type:
            actions.append("equipment_procedures")

        if context.user_role == "safety_officer":
            actions.append("safety_checklist")

        return actions[:3]  # Limit to 3 actions

    async def _execute_template(
        self,
        template: Dict[str, Any],
        context: ManufacturingContext,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a predefined template"""
        template_content = template.get("content", "")

        # Replace template variables
        replacements = {
            "{equipment_type}": params.get("equipment_type", context.equipment_type or "equipment"),
            "{process_type}": params.get("process_type", context.process_type or "process"),
            "{user_role}": params.get("user_role", context.user_role or "user"),
            "{inspection_type}": params.get("inspection_type", "general"),
            "{maintenance_type}": params.get("maintenance_type", "routine"),
        }

        for placeholder, value in replacements.items():
            template_content = template_content.replace(placeholder, str(value))

        return {
            "content": template_content,
            "template_used": template.get("name", "unknown"),
            "suggested_responses": template.get("suggested_responses", [])
        }

    async def _process_image_file(self, file_content: bytes, filename: str, context: ManufacturingContext) -> Dict[str, Any]:
        """Process uploaded image file"""
        # This would implement image recognition and analysis
        return {
            "content": f"I've received the image '{filename}'. For image analysis, I can help you identify equipment, processes, or potential safety issues. Please let me know what specific information you need from this image.",
            "analysis": "Image received and ready for analysis",
            "supported_formats": ["jpg", "png", "bmp", "tiff"]
        }

    async def _process_pdf_file(self, file_content: bytes, filename: str, context: ManufacturingContext) -> Dict[str, Any]:
        """Process uploaded PDF file"""
        return {
            "content": f"I've received the PDF document '{filename}'. I can help you extract technical specifications, procedures, or compliance information from this document. What specific information would you like me to help you with?",
            "analysis": "PDF document received and ready for processing",
            "supported_formats": ["pdf", "text", "manuals", "specifications"]
        }

    async def _process_text_file(self, file_content: bytes, filename: str, context: ManufacturingContext) -> Dict[str, Any]:
        """Process uploaded text file"""
        return {
            "content": f"I've received the text file '{filename}'. I can help you analyze the content, extract key information, or assist with understanding the procedures or specifications within this document.",
            "analysis": "Text file received and ready for processing",
            "supported_formats": ["txt", "csv", "log", "data"]
        }

    async def close_session(self, session_id: str) -> bool:
        """Close a chat session"""
        try:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]

                # Save session if persistence is enabled
                if self.session_manager and hasattr(self.session_manager, 'save_session'):
                    await self.session_manager.save_session(session)

                del self.active_sessions[session_id]
                logger.info(f"Closed session: {session_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error closing session {session_id}: {e}")
            return False

    # Manufacturing-specific convenience methods
    async def start_safety_consultation(
        self,
        user_id: str,
        equipment_type: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Start safety consultation session"""
        context = ManufacturingContext(
            domain=self.manufacturing_context.domain,
            user_role="safety_officer",
            equipment_type=equipment_type
        )

        initial_message = f"I'm starting a safety consultation for {equipment_type}. What specific safety procedures or concerns do you have?"

        return await self._process_chat_message(
            message=initial_message,
            context=context,
            session_id=session_id,
            user_id=user_id
        )

    async def start_quality_inspection(
        self,
        user_id: str,
        product_type: str,
        inspection_type: str = "general",
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Start quality inspection session"""
        context = ManufacturingContext(
            domain=self.manufacturing_context.domain,
            user_role="quality_inspector",
            process_type="quality_inspection"
        )

        initial_message = f"I'm starting a quality inspection for {product_type} ({inspection_type} inspection). What inspection procedures or criteria would you like to focus on?"

        return await self._process_chat_message(
            message=initial_message,
            context=context,
            session_id=session_id,
            user_id=user_id
        )

    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active sessions"""
        return {
            session_id: {
                "user_id": session.user_id,
                "user_role": session.user_role,
                "equipment_type": session.equipment_type,
                "process_type": session.process_type,
                "message_count": len(session.messages),
                "created_at": session.created_at.isoformat(),
                "last_activity": max(msg.timestamp for msg in session.messages).isoformat() if session.messages else None
            }
            for session_id, session in self.active_sessions.items()
        }