#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Chat Interface with LangChain RAG and LangFuse Observability
基于LangChain RAG和LangFuse可观测性的增强聊天接口

This provides an advanced conversational AI interface with RAG capabilities,
manufacturing context awareness, and comprehensive observability.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import sqlite3
import json
import logging
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
import sys

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.callbacks.base import AsyncCallbackHandler

# LangFuse imports
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler('data/processed/enhanced_chat_interface.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'enhanced-chat-secret-key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Configuration
DB_PATH = "knowledge_base.db"
VECTOR_STORE_PATH = "data/processed/vector_store"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LangFuse configuration
try:
    langfuse = Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-..."),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-..."),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )
    langfuse_handler = CallbackHandler()
    logger.info("✅ LangFuse observability enabled for chat")
except Exception as e:
    logger.warning(f"⚠️ LangFuse initialization failed: {e}")
    langfuse = None
    langfuse_handler = None

class ManufacturingContextHandler:
    """Handles manufacturing-specific context and expertise"""

    def __init__(self):
        self.context_templates = {
            "safety": {
                "system_prompt": """You are a manufacturing safety expert with deep knowledge of:
- OSHA regulations and compliance requirements
- Machine safety procedures and lockout/tagout (LOTO) protocols
- Personal protective equipment (PPE) requirements
- Emergency response procedures
- Risk assessment and mitigation strategies

Always prioritize safety and provide clear, actionable guidance.""",
                "quick_responses": {
                    "emergency_stop": "EMERGENCY: Immediately press the emergency stop button. Check for hazards before restarting.",
                    "ppe_requirements": "Required PPE includes: safety glasses, steel-toed boots, hearing protection, and appropriate gloves for the task.",
                    "lockout_procedure": "LOTO Procedure: 1) Notify affected employees 2) Shut down equipment 3) Isolate energy sources 4) Apply lockout devices 5) Verify zero energy state"
                }
            },
            "quality": {
                "system_prompt": """You are a quality control specialist with expertise in:
- ISO 9001 and AS9100 quality standards
- First article inspection procedures
- Statistical process control (SPC)
- Non-conformance reporting and corrective actions
- Measurement and calibration requirements

Provide precise, standards-compliant guidance for quality assurance.""",
                "quick_responses": {
                    "first_article": "First Article Inspection requires: 1) Complete dimension verification 2) Material certification 3) Process capability study 4) Documentation approval",
                    "spc_chart": "SPC charts help monitor: Upper Control Limit (UCL), Center Line, Lower Control Limit (LCL), and process variation trends."
                }
            },
            "maintenance": {
                "system_prompt": """You are a maintenance specialist with knowledge of:
- Preventive maintenance schedules and procedures
- Troubleshooting common equipment issues
- CMMS (Computerized Maintenance Management System)
- Reliability-centered maintenance principles
- Equipment performance optimization

Provide systematic, practical maintenance guidance.""",
                "quick_responses": {
                    "pm_schedule": "Preventive maintenance schedule includes: daily inspections, weekly lubrication, monthly alignments, and annual overhauls.",
                    "troubleshooting": "Systematic troubleshooting: 1) Identify symptoms 2) Check basics 3) Isolate the problem 4) Implement solution 5) Verify fix"
                }
            }
        }

    def get_context_prompt(self, context_type: str, user_role: str = "operator") -> str:
        """Get context-specific system prompt"""
        if context_type in self.context_templates:
            base_prompt = self.context_templates[context_type]["system_prompt"]
            role_specific = f"\n\nYou are currently assisting a {user_role}. Adjust your response complexity and terminology accordingly."
            return base_prompt + role_specific
        return "You are a helpful manufacturing expert assistant."

    def get_quick_response(self, context_type: str, query: str) -> Optional[str]:
        """Get quick response for common queries"""
        if context_type in self.context_templates:
            quick_responses = self.context_templates[context_type]["quick_responses"]
            for key, response in quick_responses.items():
                if key.lower() in query.lower():
                    return response
        return None

class LangFuseCallbackHandler(AsyncCallbackHandler):
    """Custom callback handler for LangFuse observability"""

    def __init__(self, session_id: str, user_id: str = None):
        super().__init__()
        self.session_id = session_id
        self.user_id = user_id
        self.trace = None

    async def on_llm_start(self, serialized: Dict, prompts: List[str], **kwargs) -> None:
        """Track LLM start"""
        if langfuse:
            self.trace = langfuse.trace(
                name="chat_llm_call",
                session_id=self.session_id,
                user_id=self.user_id,
                input={"prompts": prompts}
            )

    async def on_llm_end(self, response, **kwargs) -> None:
        """Track LLM completion"""
        if self.trace and langfuse:
            self.trace.update(
                output={
                    "response": response.generations[0][0].text if response.generations else ""
                }
            )
            self.trace = None

    async def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Track LLM errors"""
        if self.trace and langfuse:
            self.trace.update(
                output={"error": str(error)},
                level="ERROR"
            )
            self.trace = None

class EnhancedChatInterface:
    """Enhanced chat interface with RAG and manufacturing context"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = None
        self.embeddings = None
        self.vector_store = None
        self.conversation_chains = {}  # Per-session conversation chains
        self.context_handler = ManufacturingContextHandler()

    def connect(self):
        """Connect to database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.execute("PRAGMA foreign_keys = ON")
            logger.info(f"✅ Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"❌ Database connection failed: {e}")

    def disconnect(self):
        """Disconnect from database"""
        if self.conn:
            self.conn.close()
            logger.info("✅ Disconnected from database")

    def initialize_rag(self):
        """Initialize RAG components"""
        try:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDINGS_MODEL,
                cache_folder="data/processed/embeddings_cache"
            )
            logger.info("✅ Initialized embeddings for chat")

            # Load vector store
            if os.path.exists(VECTOR_STORE_PATH):
                self.vector_store = FAISS.load_local(
                    VECTOR_STORE_PATH,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("✅ Loaded vector store for chat")
                return True
            else:
                logger.warning("⚠️ No vector store found for chat")
                return False
        except Exception as e:
            logger.error(f"❌ RAG initialization failed: {e}")
            return False

    def create_conversation_chain(
        self,
        session_id: str,
        user_id: str = None,
        context_type: str = "general",
        user_role: str = "operator"
    ) -> ConversationalRetrievalChain:
        """Create conversation chain with manufacturing context"""
        try:
            # Initialize LLM
            if os.getenv("OPENAI_API_KEY"):
                llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=0.1,
                    callbacks=[langfuse_handler] if langfuse_handler else None
                )
            else:
                # Fallback to HuggingFace model
                llm = HuggingFaceHub(
                    repo_id="google/flan-t5-large",
                    model_kwargs={"temperature": 0.1}
                )

            # Create memory with LangFuse callback
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                callbacks=[LangFuseCallbackHandler(session_id, user_id)] if langfuse else None
            )

            # Create manufacturing-specific prompt
            system_prompt = self.context_handler.get_context_prompt(context_type, user_role)

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
                ("context", "Relevant manufacturing knowledge: {context}")
            ])

            # Create conversation chain
            if self.vector_store:
                chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=self.vector_store.as_retriever(
                        search_kwargs={"k": 5}
                    ),
                    memory=memory,
                    combine_docs_chain_kwargs={"prompt": prompt},
                    callbacks=[langfuse_handler] if langfuse_handler else None,
                    verbose=False
                )
                logger.info(f"✅ Created conversation chain for session {session_id}")
                return chain
            else:
                logger.error("❌ Vector store not available for conversation chain")
                return None

        except Exception as e:
            logger.error(f"❌ Conversation chain creation failed: {e}")
            return None

    async def get_response(
        self,
        message: str,
        session_id: str,
        user_id: str = None,
        context_type: str = "general",
        user_role: str = "operator",
        manufacturing_context: Dict = None
    ) -> Dict:
        """Get AI response with RAG and manufacturing context"""
        try:
            # Create trace for LangFuse
            if langfuse:
                trace = langfuse.trace(
                    name="chat_message",
                    session_id=session_id,
                    user_id=user_id,
                    input={
                        "message": message,
                        "context_type": context_type,
                        "user_role": user_role,
                        "manufacturing_context": manufacturing_context
                    }
                )

            # Check for quick responses
            quick_response = self.context_handler.get_quick_response(context_type, message)
            if quick_response:
                response_data = {
                    "response": quick_response,
                    "source": "quick_response",
                    "context_type": context_type,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                }

                if langfuse:
                    trace.update(
                        output=response_data,
                        metadata={"response_type": "quick_response"}
                    )
                return response_data

            # Get or create conversation chain
            if session_id not in self.conversation_chains:
                self.conversation_chains[session_id] = self.create_conversation_chain(
                    session_id, user_id, context_type, user_role
                )

            chain = self.conversation_chains[session_id]

            if chain:
                # Get response from RAG chain
                result = await chain.acall({"question": message})

                response_data = {
                    "response": result.get("answer", ""),
                    "source_documents": [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        } for doc in result.get("source_documents", [])
                    ],
                    "context_type": context_type,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                }

                # Log with LangFuse
                if langfuse:
                    trace.update(
                        output={
                            "response": response_data["response"],
                            "sources_count": len(response_data["source_documents"])
                        },
                        metadata={"response_type": "rag_conversation"}
                    )

                return response_data
            else:
                # Fallback response
                fallback_response = f"I understand you're asking about {message}. As a {user_role} in the {context_type} area, I'd be happy to help, but I need access to our knowledge base to provide specific guidance."

                response_data = {
                    "response": fallback_response,
                    "source": "fallback",
                    "context_type": context_type,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                }

                if langfuse:
                    trace.update(
                        output=response_data,
                        metadata={"response_type": "fallback"}
                    )

                return response_data

        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            error_response = {
                "error": str(e),
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }

            if langfuse:
                trace.update(
                    output=error_response,
                    level="ERROR"
                )

            return error_response

    def clear_session(self, session_id: str):
        """Clear conversation session"""
        if session_id in self.conversation_chains:
            del self.conversation_chains[session_id]
            logger.info(f"✅ Cleared conversation session {session_id}")

    def get_session_summary(self, session_id: str) -> Dict:
        """Get session summary for analytics"""
        # This would query stored conversation history
        return {
            "session_id": session_id,
            "message_count": 0,  # Would be calculated from stored messages
            "duration": 0,  # Would be calculated from timestamps
            "topics_discussed": [],  # Would be extracted from messages
        }

# Initialize chat interface
chat = EnhancedChatInterface()
chat.connect()
chat.initialize_rag()

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"🔗 Client connected: {request.sid}")
    emit('status', {'message': 'Connected to enhanced chat server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"🔌 Client disconnected: {request.sid}")

@socketio.on('join_session')
def handle_join_session(data):
    """Handle joining a chat session"""
    session_id = data.get('session_id', str(uuid.uuid4()))
    user_id = data.get('user_id', 'anonymous')
    context_type = data.get('context_type', 'general')
    user_role = data.get('user_role', 'operator')

    join_room(session_id)
    logger.info(f"👤 User {user_id} joined session {session_id}")

    emit('session_joined', {
        'session_id': session_id,
        'user_id': user_id,
        'context_type': context_type,
        'user_role': user_role
    })

@socketio.on('chat_message')
async def handle_chat_message(data):
    """Handle chat message with RAG"""
    try:
        message = data.get('message', '').strip()
        session_id = data.get('session_id')
        user_id = data.get('user_id', 'anonymous')
        context_type = data.get('context_type', 'general')
        user_role = data.get('user_role', 'operator')
        manufacturing_context = data.get('manufacturing_context', {})

        if not message:
            emit('error', {'message': 'Message cannot be empty'})
            return

        # Get response
        response = await chat.get_response(
            message=message,
            session_id=session_id,
            user_id=user_id,
            context_type=context_type,
            user_role=user_role,
            manufacturing_context=manufacturing_context
        )

        # Broadcast response to session room
        emit('chat_response', {
            'message': message,
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id
        }, room=session_id)

    except Exception as e:
        logger.error(f"❌ Chat message handling failed: {e}")
        emit('error', {'message': str(e)})

# REST API endpoints
@app.route('/api/chat/message', methods=['POST'])
async def send_message():
    """REST endpoint for sending messages"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        session_id = data.get('session_id', str(uuid.uuid4()))
        user_id = data.get('user_id', 'anonymous')
        context_type = data.get('context_type', 'general')
        user_role = data.get('user_role', 'operator')
        manufacturing_context = data.get('manufacturing_context', {})

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        response = await chat.get_response(
            message=message,
            session_id=session_id,
            user_id=user_id,
            context_type=context_type,
            user_role=user_role,
            manufacturing_context=manufacturing_context
        )

        return jsonify({
            'message': message,
            'response': response,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"❌ REST message handling failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/session/<session_id>', methods=['DELETE'])
def clear_session(session_id):
    """Clear conversation session"""
    chat.clear_session(session_id)
    return jsonify({'message': f'Session {session_id} cleared'})

@app.route('/api/chat/session/<session_id>/summary', methods=['GET'])
def get_session_summary(session_id):
    """Get session summary"""
    summary = chat.get_session_summary(session_id)
    return jsonify(summary)

@app.route('/api/chat/health', methods=['GET'])
def chat_health_check():
    """Health check for chat service"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0-enhanced',
        'rag_enabled': chat.vector_store is not None,
        'langfuse_enabled': langfuse is not None,
        'active_sessions': len(chat.conversation_chains)
    })

if __name__ == '__main__':
    logger.info("🚀 Starting Enhanced Chat Interface with LangChain RAG and LangFuse observability...")

    # Print system status
    logger.info("📊 Chat System Status:")
    logger.info(f"   - Vector Store: {'✅ Ready' if chat.vector_store else '❌ Not initialized'}")
    logger.info(f"   - LangFuse: {'✅ Enabled' if langfuse else '❌ Disabled'}")
    logger.info(f"   - Database: {'✅ Connected' if chat.conn else '❌ Not connected'}")
    logger.info(f"   - WebSocket: ✅ Enabled")

    socketio.run(
        app,
        host='0.0.0.0',
        port=8002,
        debug=False,
        allow_unsafe_werkzeug=True
    )