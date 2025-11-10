"""
LangChain Integration Implementation
Manufacturing Knowledge Base - Advanced AI/LLM Framework Integration

This module provides the main LangChain integration implementation with manufacturing-specific
enhancements and optimizations.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

try:
    from langchain.chains import (
        LLMChain,
        ConversationalRetrievalChain,
        RetrievalQA,
        SequentialChain
    )
    from langchain.chat_models import ChatOpenAI
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma, FAISS
    from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.schema import HumanMessage, AIMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from ..shared.base import IntegrationBase, ManufacturingContext, IntegrationStatus
from ..shared.errors import IntegrationError
from .chains import (
    ManufacturingQueryChain,
    SafetyProcedureChain,
    QualityControlChain
)
from .prompts import ManufacturingPromptTemplate
from .memory import ManufacturingMemory
from .agents import ManufacturingAgent
from .retrievers import ManufacturingRetriever

logger = logging.getLogger(__name__)


class LangChainIntegration(IntegrationBase):
    """
    LangChain integration for manufacturing knowledge base.
    Provides advanced AI/LLM capabilities with manufacturing-specific enhancements.
    """

    def __init__(self, name: str, config):
        super().__init__(name, config)

        if not LANGCHAIN_AVAILABLE:
            raise IntegrationError("LangChain is not installed. Install with: pip install langchain")

        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.memory = None
        self.chains = {}
        self.agents = {}
        self.retrievers = {}

        # Manufacturing-specific configurations
        self.manufacturing_prompts = self._initialize_manufacturing_prompts()
        self.safety_procedures = self._load_safety_procedures()
        self.quality_standards = self._load_quality_standards()
        self.compliance_frameworks = self._load_compliance_frameworks()

        logger.info(f"Initialized LangChain integration for manufacturing domain")

    def _initialize_manufacturing_prompts(self) -> Dict[str, Any]:
        """Initialize manufacturing-specific prompt templates"""
        prompts = {
            "system_prompt": self.config.get("prompts.system_prompt",
                "You are a knowledgeable manufacturing assistant specializing in industrial processes, safety procedures, and quality control."),
            "manufacturing_query": self.config.get("prompts.manufacturing_query",
                "Process the following manufacturing query with attention to safety, quality standards, and compliance: {query}"),
            "safety_procedure": self.config.get("prompts.safety_procedure",
                "Provide detailed safety procedures for manufacturing operations: {equipment_type}"),
            "quality_check": self.config.get("prompts.quality_check",
                "Analyze quality requirements and provide inspection procedures: {product_spec}"),
        }
        return prompts

    def _load_safety_procedures(self) -> Dict[str, Any]:
        """Load safety procedure templates and regulations"""
        return {
            "ansi_standards": self.config.get("safety.ansi_standards", ["ANSI_Z535", "ANSI_B11"]),
            "osha_regulations": self.config.get("safety.osha_regulations", ["29 CFR 1910", "29 CFR 1926"]),
            "machine_safety": self.config.get("safety.machine_safety", ["Lockout/Tagout", "Machine Guarding"]),
            "ppe_requirements": self.config.get("safety.ppe_requirements",
                ["Safety Glasses", "Steel Toe Boots", "Hearing Protection"]),
        }

    def _load_quality_standards(self) -> Dict[str, Any]:
        """Load quality standards and inspection procedures"""
        return {
            "iso_standards": self.config.get("quality.iso_standards", ["ISO 9001", "ISO 13485"]),
            "inspection_methods": self.config.get("quality.inspection_methods",
                ["Visual Inspection", "Dimensional Analysis", "Non-destructive Testing"]),
            "spc_tools": self.config.get("quality.spc_tools",
                ["Control Charts", "Process Capability", "Measurement System Analysis"]),
            "documentation": self.config.get("quality.documentation",
                ["Inspection Reports", "Quality Records", "Calibration Certificates"]),
        }

    def _load_compliance_frameworks(self) -> Dict[str, Any]:
        """Load compliance frameworks and regulatory requirements"""
        return {
            "industrial_standards": self.config.get("compliance.industrial_standards",
                ["ASME", "ASTM", "AWS"]),
            "environmental": self.config.get("compliance.environmental",
                ["EPA Regulations", "Waste Management", "Environmental Impact"]),
            "export_controls": self.config.get("compliance.export_controls",
                ["ITAR", "EAR", "Dual-Use Regulations"]),
        }

    async def initialize(self) -> bool:
        """Initialize LangChain components"""
        try:
            logger.info("Initializing LangChain integration components")

            # Initialize LLM
            await self._initialize_llm()

            # Initialize embeddings
            await self._initialize_embeddings()

            # Initialize vector store
            await self._initialize_vectorstore()

            # Initialize memory systems
            await self._initialize_memory()

            # Initialize manufacturing-specific chains
            await self._initialize_chains()

            # Initialize agents
            await self._initialize_agents()

            # Initialize retrievers
            await self._initialize_retrievers()

            self.status = IntegrationStatus.READY
            self.start_time = datetime.now().timestamp()

            logger.info("LangChain integration initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize LangChain integration: {e}")
            self.status = IntegrationStatus.ERROR
            return False

    async def _initialize_llm(self):
        """Initialize the main LLM"""
        llm_config = self.config.get("llm", {})

        model_name = llm_config.get("model", "gpt-4")
        temperature = llm_config.get("temperature", 0.1)
        max_tokens = llm_config.get("max_tokens", 2000)

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=self.config.get("openai_api_key")
        )

    async def _initialize_embeddings(self):
        """Initialize embeddings model"""
        embedding_config = self.config.get("embeddings", {})
        model_name = embedding_config.get("model", "text-embedding-ada-002")

        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=self.config.get("openai_api_key")
        )

    async def _initialize_vectorstore(self):
        """Initialize vector store for retrieval"""
        vectorstore_config = self.config.get("vectorstore", {})

        # For now, initialize with Chroma - can be extended to support other stores
        collection_name = vectorstore_config.get("collection_name", "manufacturing_kb")
        persist_directory = vectorstore_config.get("persist_directory", "./chroma_db")

        # Try to load existing vectorstore or create new one
        try:
            self.vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
        except Exception as e:
            logger.warning(f"Could not load existing vectorstore, creating new one: {e}")
            self.vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )

    async def _initialize_memory(self):
        """Initialize memory systems"""
        memory_config = self.config.get("memory", {})
        memory_type = memory_config.get("type", "buffer")

        if memory_type == "buffer":
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        elif memory_type == "summary":
            self.memory = ConversationSummaryMemory(
                llm=self.llm,
                memory_key="chat_history",
                return_messages=True
            )
        else:
            # Use manufacturing-specific memory
            self.memory = ManufacturingMemory(
                llm=self.llm,
                manufacturing_context=self.manufacturing_context
            )

    async def _initialize_chains(self):
        """Initialize manufacturing-specific chains"""
        try:
            # Manufacturing query chain
            self.chains["manufacturing_query"] = ManufacturingQueryChain(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(),
                memory=self.memory,
                manufacturing_context=self.manufacturing_context
            )

            # Safety procedure chain
            self.chains["safety_procedure"] = SafetyProcedureChain(
                llm=self.llm,
                safety_procedures=self.safety_procedures,
                memory=self.memory
            )

            # Quality control chain
            self.chains["quality_control"] = QualityControlChain(
                llm=self.llm,
                quality_standards=self.quality_standards,
                memory=self.memory
            )

        except Exception as e:
            logger.error(f"Failed to initialize chains: {e}")
            raise

    async def _initialize_agents(self):
        """Initialize manufacturing-specific agents"""
        try:
            # Main manufacturing agent
            self.agents["manufacturing"] = ManufacturingAgent(
                llm=self.llm,
                tools=[],
                manufacturing_context=self.manufacturing_context
            )

            # Safety inspector agent
            self.agents["safety_inspector"] = ManufacturingAgent(
                llm=self.llm,
                role="Safety Inspector",
                tools=[],
                focus_area="safety_procedures",
                manufacturing_context=self.manufacturing_context
            )

            # Quality control agent
            self.agents["quality_control"] = ManufacturingAgent(
                llm=self.llm,
                role="Quality Inspector",
                tools=[],
                focus_area="quality_standards",
                manufacturing_context=self.manufacturing_context
            )

        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            raise

    async def _initialize_retrievers(self):
        """Initialize manufacturing-specific retrievers"""
        try:
            # General manufacturing retriever
            self.retrievers["manufacturing"] = ManufacturingRetriever(
                vectorstore=self.vectorstore,
                search_kwargs={"k": 5},
                manufacturing_context=self.manufacturing_context
            )

            # Safety documents retriever
            self.retrievers["safety"] = ManufacturingRetriever(
                vectorstore=self.vectorstore,
                search_kwargs={"k": 3, "filter": {"document_type": "safety"}},
                manufacturing_context=self.manufacturing_context
            )

            # Technical specifications retriever
            self.retrievers["technical"] = ManufacturingRetriever(
                vectorstore=self.vectorstore,
                search_kwargs={"k": 3, "filter": {"document_type": "technical"}},
                manufacturing_context=self.manufacturing_context
            )

        except Exception as e:
            logger.error(f"Failed to initialize retrievers: {e}")
            raise

    async def shutdown(self) -> bool:
        """Shutdown LangChain integration gracefully"""
        try:
            logger.info("Shutting down LangChain integration")

            # Clear memory
            if self.memory:
                self.memory.clear()

            # Persist vectorstore
            if hasattr(self.vectorstore, 'persist'):
                self.vectorstore.persist()

            # Clear components
            self.chains.clear()
            self.agents.clear()
            self.retrievers.clear()

            self.status = IntegrationStatus.SHUTDOWN
            logger.info("LangChain integration shutdown successfully")
            return True

        except Exception as e:
            logger.error(f"Error during LangChain shutdown: {e}")
            return False

    async def _integration_health_check(self) -> Dict[str, Any]:
        """Perform LangChain-specific health check"""
        try:
            health_status = {
                "llm_available": self.llm is not None,
                "embeddings_available": self.embeddings is not None,
                "vectorstore_available": self.vectorstore is not None,
                "chains_loaded": len(self.chains),
                "agents_loaded": len(self.agents),
                "retrievers_loaded": len(self.retrievers),
                "memory_available": self.memory is not None,
            }

            # Test LLM connectivity
            if self.llm:
                try:
                    test_result = await self.llm.ainvoke([HumanMessage(content="Test")])
                    health_status["llm_test"] = "passed"
                    health_status["llm_response_time"] = 0.1  # Placeholder
                except Exception as e:
                    health_status["llm_test"] = f"failed: {str(e)}"

            # Test embeddings
            if self.embeddings:
                try:
                    test_embedding = await self.embeddings.aembed_query("test")
                    health_status["embeddings_test"] = "passed"
                    health_status["embedding_dimension"] = len(test_embedding)
                except Exception as e:
                    health_status["embeddings_test"] = f"failed: {str(e)}"

            # Test vectorstore
            if self.vectorstore:
                try:
                    test_docs = await self.vectorstore.asimilarity_search("test", k=1)
                    health_status["vectorstore_test"] = "passed"
                    health_status["vectorstore_count"] = len(test_docs)
                except Exception as e:
                    health_status["vectorstore_test"] = f"failed: {str(e)}"

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
        Process request with LangChain and manufacturing context
        """
        try:
            # Extract query and parameters from request
            if isinstance(request_data, str):
                query = request_data
                request_type = "general"
                parameters = {}
            elif isinstance(request_data, dict):
                query = request_data.get("query", "")
                request_type = request_data.get("type", "general")
                parameters = request_data.get("parameters", {})
            else:
                raise ValueError("Invalid request data format")

            # Route to appropriate chain or agent based on request type
            if request_type == "safety_procedure":
                return await self._process_safety_procedure(query, context, parameters)
            elif request_type == "quality_control":
                return await self._process_quality_control(query, context, parameters)
            elif request_type == "technical_specification":
                return await self._process_technical_specification(query, context, parameters)
            elif request_type == "manufacturing_query":
                return await self._process_manufacturing_query(query, context, parameters)
            else:
                return await self._process_general_query(query, context, parameters)

        except Exception as e:
            logger.error(f"Error processing request with LangChain: {e}")
            raise IntegrationError(f"Request processing failed: {e}")

    async def _process_manufacturing_query(
        self,
        query: str,
        context: ManufacturingContext,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process general manufacturing query"""
        try:
            # Use manufacturing query chain
            if "manufacturing_query" in self.chains:
                result = await self.chains["manufacturing_query"].arun({
                    "query": query,
                    "context": context.get_context_dict(),
                    **parameters
                })
            else:
                # Fallback to basic LLM query
                prompt = self.manufacturing_prompts["manufacturing_query"].format(
                    query=query,
                    context=context.get_context_dict(),
                    **parameters
                )
                result = await self.llm.ainvoke([HumanMessage(content=prompt)])

            return {
                "response": result,
                "type": "manufacturing_query",
                "context": context.get_context_dict(),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error processing manufacturing query: {e}")
            raise

    async def _process_safety_procedure(
        self,
        query: str,
        context: ManufacturingContext,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process safety procedure request"""
        try:
            equipment_type = parameters.get("equipment_type", context.equipment_type)

            if "safety_procedure" in self.chains:
                result = await self.chains["safety_procedure"].arun({
                    "equipment_type": equipment_type,
                    "query": query,
                    "context": context.get_context_dict(),
                    **parameters
                })
            else:
                # Fallback to safety prompt
                prompt = self.manufacturing_prompts["safety_procedure"].format(
                    equipment_type=equipment_type,
                    query=query,
                    context=context.get_context_dict()
                )
                result = await self.llm.ainvoke([HumanMessage(content=prompt)])

            return {
                "response": result,
                "type": "safety_procedure",
                "equipment_type": equipment_type,
                "safety_standards": self.safety_procedures,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error processing safety procedure: {e}")
            raise

    async def _process_quality_control(
        self,
        query: str,
        context: ManufacturingContext,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process quality control request"""
        try:
            if "quality_control" in self.chains:
                result = await self.chains["quality_control"].arun({
                    "query": query,
                    "context": context.get_context_dict(),
                    **parameters
                })
            else:
                # Fallback to quality prompt
                prompt = self.manufacturing_prompts["quality_check"].format(
                    product_spec=parameters.get("product_spec", query),
                    context=context.get_context_dict()
                )
                result = await self.llm.ainvoke([HumanMessage(content=prompt)])

            return {
                "response": result,
                "type": "quality_control",
                "quality_standards": self.quality_standards,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error processing quality control: {e}")
            raise

    async def _process_technical_specification(
        self,
        query: str,
        context: ManufacturingContext,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process technical specification request"""
        try:
            # Use technical retriever for enhanced results
            relevant_docs = []
            if "technical" in self.retrievers:
                relevant_docs = await self.retrievers["technical"].aget_relevant_documents(query)

            # Generate response with context
            context_prompt = f"Query: {query}\n\nRelevant technical specifications: {relevant_docs}"
            result = await self.llm.ainvoke([HumanMessage(content=context_prompt)])

            return {
                "response": result,
                "type": "technical_specification",
                "relevant_documents": len(relevant_docs),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error processing technical specification: {e}")
            raise

    async def _process_general_query(
        self,
        query: str,
        context: ManufacturingContext,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process general query"""
        try:
            # Use general manufacturing retriever
            relevant_docs = []
            if "manufacturing" in self.retrievers:
                relevant_docs = await self.retrievers["manufacturing"].aget_relevant_documents(query)

            # Generate contextual response
            system_prompt = self.manufacturing_prompts["system_prompt"]
            context_info = f"Manufacturing Context: {context.get_context_dict()}"

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"{context_info}\n\nQuery: {query}")
            ]

            result = await self.llm.ainvoke(messages)

            return {
                "response": result,
                "type": "general_query",
                "relevant_documents": len(relevant_docs),
                "context_used": context.get_context_dict(),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error processing general query: {e}")
            raise

    # Manufacturing-specific convenience methods
    async def get_safety_procedure(self, equipment_type: str, operation: str = "standard") -> Dict[str, Any]:
        """Get safety procedures for specific equipment"""
        return await self._process_safety_procedure(
            f"Safety procedures for {equipment_type}",
            ManufacturingContext(
                domain=self.manufacturing_context.domain,
                equipment_type=equipment_type,
                user_role="safety_officer"
            ),
            {"operation": operation}
        )

    async def get_quality_inspection_procedure(self, product_type: str, inspection_type: str = "incoming") -> Dict[str, Any]:
        """Get quality inspection procedures"""
        return await self._process_quality_control(
            f"Quality inspection procedures for {product_type}",
            ManufacturingContext(
                domain=self.manufacturing_context.domain,
                process_type="quality_inspection",
                user_role="quality_inspector"
            ),
            {"product_type": product_type, "inspection_type": inspection_type}
        )

    async def search_technical_specifications(self, equipment_model: str, specification_type: str = "general") -> Dict[str, Any]:
        """Search technical specifications"""
        return await self._process_technical_specification(
            f"Technical specifications for {equipment_model}",
            ManufacturingContext(
                domain=self.manufacturing_context.domain,
                equipment_type=equipment_model,
                user_role="engineer"
            ),
            {"specification_type": specification_type}
        )