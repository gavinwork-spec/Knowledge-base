"""
Coordinator Agent
XAgent-inspired coordinator agent for complex task decomposition,
agent orchestration, and result aggregation with intelligent planning.
"""

import asyncio
import json
import logging
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import time
import statistics

# Import agent framework
from multi_agent_orchestrator import BaseAgent, AgentTask, AgentCapability, AgentStatus
from multi_agent_system.protocols.agent_communication import (
    MessageRouter, TaskDelegator, AgentMessage, TaskRequest, TaskResponse,
    MessageType, Priority
)
from multi_agent_system.agents.specialized_agents import EnhancedBaseAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class ExecutionStrategy(Enum):
    """Task execution strategies"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    COLLABORATIVE = "collaborative"
    ADAPTIVE = "adaptive"


@dataclass
class SubTask:
    """Represents a sub-task in decomposition"""
    task_id: str
    parent_task_id: str
    task_type: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_capabilities: Set[str] = field(default_factory=set)
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: float = 0.0
    priority: Priority = Priority.NORMAL
    assigned_agent: Optional[str] = None
    status: str = "pending"
    result: Optional[Any] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Execution plan for complex tasks"""
    plan_id: str
    root_task_id: str
    complexity: TaskComplexity
    strategy: ExecutionStrategy
    sub_tasks: List[SubTask] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    execution_graph: Optional[nx.DiGraph] = None
    estimated_total_duration: float = 0.0
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)
    rollback_plan: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)


class CoordinatorAgent(EnhancedBaseAgent):
    """Advanced coordinator agent for complex task orchestration"""

    def __init__(self, orchestrator):
        super().__init__("coordinator", "Task Coordinator", orchestrator)
        self.capabilities.update({
            "task_decomposition",
            "agent_orchestration",
            "result_aggregation",
            "complex_planning",
            "dependency_resolution",
            "workflow_management",
            "adaptive_execution",
            "failure_recovery",
            "performance_optimization",
            "resource_allocation"
        })
        self.active_plans: Dict[str, ExecutionPlan] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        self.decomposition_patterns: Dict[str, Dict[str, Any]] = {}
        self.performance_metrics = defaultdict(list)
        self.collaboration_cache: Dict[str, List[str]] = {}
        self.load_balancer = TaskLoadBalancer()

    async def initialize(self):
        """Initialize coordinator agent"""
        await super().initialize()

        # Load decomposition patterns
        await self._load_decomposition_patterns()

        # Register message handlers
        self.message_router.register_handler(MessageType.RESULT_AGGREGATION, self._handle_result_aggregation)
        self.message_router.register_handler(MessageType.STATUS_UPDATE, self._handle_status_update)

        # Initialize agent registry
        await self._initialize_agent_registry()

        logger.info("Coordinator Agent initialized with advanced planning capabilities")

    async def _execute_task_logic(self, task: TaskRequest) -> Dict[str, Any]:
        """Execute coordination task"""
        task_type = task.task_type
        parameters = task.parameters

        if task_type == "coordinate_complex_task":
            return await self._coordinate_complex_task(parameters)
        elif task_type == "decompose_task":
            return await self._decompose_task(parameters)
        elif task_type == "create_execution_plan":
            return await self._create_execution_plan(parameters)
        elif task_type == "orchestrate_workflow":
            return await self._orchestrate_workflow(parameters)
        elif task_type == "aggregate_results":
            return await self._aggregate_results(parameters)
        elif task_type == "optimize_execution":
            return await self._optimize_execution(parameters)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def _coordinate_complex_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate a complex multi-agent task"""
        root_task = parameters.get('root_task')
        if not root_task:
            raise ValueError("Root task is required")

        try:
            # Step 1: Analyze task complexity
            complexity = await self._analyze_task_complexity(root_task)

            # Step 2: Decompose task
            execution_plan = await self._decompose_and_plan(root_task, complexity)

            # Step 3: Optimize execution strategy
            optimized_plan = await self._optimize_execution_plan(execution_plan)

            # Step 4: Execute plan
            execution_result = await self._execute_plan(optimized_plan)

            # Step 5: Aggregate results
            final_result = await self._aggregate_final_results(execution_result)

            # Step 6: Update performance metrics
            await self._update_performance_metrics(execution_plan, execution_result)

            return {
                'task_id': root_task.get('task_id'),
                'coordination_result': final_result,
                'execution_plan_id': optimized_plan.plan_id,
                'complexity': complexity.value,
                'strategy': optimized_plan.strategy.value,
                'sub_tasks_completed': len([st for st in optimized_plan.sub_tasks if st.status == 'completed']),
                'total_sub_tasks': len(optimized_plan.sub_tasks),
                'execution_time': execution_result.get('total_execution_time', 0),
                'success_rate': execution_result.get('success_rate', 0)
            }

        except Exception as e:
            logger.error(f"Error coordinating complex task: {e}")
            raise

    async def _analyze_task_complexity(self, task: Dict[str, Any]) -> TaskComplexity:
        """Analyze task complexity"""
        complexity_score = 0

        # Task type complexity
        task_type = task.get('task_type', '')
        if task_type in ['comprehensive_analysis', 'multi_modal_processing', 'cross_domain_research']:
            complexity_score += 30
        elif task_type in ['data_analysis', 'document_processing', 'price_analysis']:
            complexity_score += 20
        else:
            complexity_score += 10

        # Parameter complexity
        parameters = task.get('parameters', {})
        if len(parameters) > 10:
            complexity_score += 25
        elif len(parameters) > 5:
            complexity_score += 15
        else:
            complexity_score += 5

        # Data source complexity
        data_sources = parameters.get('data_sources', [])
        if len(data_sources) > 5:
            complexity_score += 20
        elif len(data_sources) > 2:
            complexity_score += 10

        # Requirement complexity
        requirements = task.get('requirements', {})
        if requirements.get('real_time_processing'):
            complexity_score += 15
        if requirements.get('high_accuracy'):
            complexity_score += 10
        if requirements.get('collaboration_needed'):
            complexity_score += 20

        # Output complexity
        expected_outputs = task.get('expected_outputs', [])
        if len(expected_outputs) > 5:
            complexity_score += 15
        elif len(expected_outputs) > 2:
            complexity_score += 8

        # Determine complexity level
        if complexity_score >= 80:
            return TaskComplexity.VERY_COMPLEX
        elif complexity_score >= 60:
            return TaskComplexity.COMPLEX
        elif complexity_score >= 35:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE

    async def _decompose_and_plan(self, task: Dict[str, Any], complexity: TaskComplexity) -> ExecutionPlan:
        """Decompose task and create execution plan"""
        plan_id = str(uuid.uuid4())
        task_id = task.get('task_id', str(uuid.uuid4()))

        # Determine execution strategy
        strategy = self._determine_execution_strategy(complexity, task)

        # Decompose task based on patterns
        sub_tasks = await self._decompose_using_patterns(task, complexity)

        # Create execution graph
        execution_graph = await self._create_execution_graph(sub_tasks)

        # Estimate durations and resources
        await self._estimate_task_requirements(sub_tasks)

        plan = ExecutionPlan(
            plan_id=plan_id,
            root_task_id=task_id,
            complexity=complexity,
            strategy=strategy,
            sub_tasks=sub_tasks,
            execution_graph=execution_graph,
            estimated_total_duration=sum(st.estimated_duration for st in sub_tasks)
        )

        # Store active plan
        self.active_plans[plan_id] = plan

        return plan

    def _determine_execution_strategy(self, complexity: TaskComplexity, task: Dict[str, Any]) -> ExecutionStrategy:
        """Determine optimal execution strategy"""
        requirements = task.get('requirements', {})

        # Real-time requirements favor parallel processing
        if requirements.get('real_time_processing', False):
            return ExecutionStrategy.PARALLEL

        # Collaboration needed favors collaborative approach
        if requirements.get('collaboration_needed', False):
            return ExecutionStrategy.COLLABORATIVE

        # High complexity favors adaptive approach
        if complexity in [TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX]:
            return ExecutionStrategy.ADAPTIVE

        # Check task type for pipeline suitability
        task_type = task.get('task_type', '')
        if task_type in ['data_processing_pipeline', 'document_processing_chain']:
            return ExecutionStrategy.PIPELINE

        # Default strategies based on complexity
        if complexity == TaskComplexity.SIMPLE:
            return ExecutionStrategy.SEQUENTIAL
        elif complexity == TaskComplexity.MODERATE:
            return ExecutionStrategy.PARALLEL
        else:
            return ExecutionStrategy.ADAPTIVE

    async def _decompose_using_patterns(self, task: Dict[str, Any], complexity: TaskComplexity) -> List[SubTask]:
        """Decompose task using known patterns"""
        task_type = task.get('task_type', '')
        parameters = task.get('parameters', {})

        # Check for existing patterns
        if task_type in self.decomposition_patterns:
            pattern = self.decomposition_patterns[task_type]
            return await self._apply_decomposition_pattern(task, pattern)

        # Use default decomposition strategies
        if task_type == 'comprehensive_analysis':
            return await self._decompose_comprehensive_analysis(task)
        elif task_type == 'multi_modal_processing':
            return await self._decompose_multimodal_processing(task)
        elif task_type == 'market_intelligence':
            return await self._decompose_market_intelligence(task)
        elif task_type == 'customer_insights':
            return await self._decompose_customer_insights(task)
        else:
            return await self._decompose_generic_task(task)

    async def _decompose_comprehensive_analysis(self, task: Dict[str, Any]) -> List[SubTask]:
        """Decompose comprehensive analysis task"""
        task_id = task.get('task_id')
        parameters = task.get('parameters', {})
        data_sources = parameters.get('data_sources', [])

        sub_tasks = []

        # Data collection and preprocessing
        sub_tasks.append(SubTask(
            task_id=f"{task_id}_collect_data",
            parent_task_id=task_id,
            task_type="data_collection",
            description="Collect and preprocess data from multiple sources",
            parameters={'data_sources': data_sources},
            required_capabilities={"data_processing", "file_system_access"},
            estimated_duration=300  # 5 minutes
        ))

        # Document processing (if documents provided)
        if any(source.get('type') == 'document' for source in data_sources):
            sub_tasks.append(SubTask(
                task_id=f"{task_id}_process_documents",
                parent_task_id=task_id,
                task_type="document_processing",
                description="Process and extract information from documents",
                parameters={'documents': [s for s in data_sources if s.get('type') == 'document']},
                required_capabilities={"document_processing", "text_extraction"},
                dependencies=[f"{task_id}_collect_data"],
                estimated_duration=600  # 10 minutes
            ))

        # Data analysis
        sub_tasks.append(SubTask(
            task_id=f"{task_id}_analyze_data",
            parent_task_id=task_id,
            task_type="data_analysis",
            description="Perform statistical and trend analysis",
            parameters={'analysis_type': parameters.get('analysis_type', 'comprehensive')},
            required_capabilities={"data_analysis", "statistical_modeling"},
            dependencies=[f"{task_id}_collect_data"],
            estimated_duration=900  # 15 minutes
        ))

        # Price analysis (if relevant)
        if parameters.get('include_price_analysis', False):
            sub_tasks.append(SubTask(
                task_id=f"{task_id}_price_analysis",
                parent_task_id=task_id,
                task_type="price_analysis",
                description="Analyze pricing trends and market positioning",
                parameters={'market_data': parameters.get('market_data', {})},
                required_capabilities={"price_analysis", "market_intelligence"},
                dependencies=[f"{task_id}_analyze_data"],
                estimated_duration=600  # 10 minutes
            ))

        # Generate insights and recommendations
        sub_tasks.append(SubTask(
            task_id=f"{task_id}_generate_insights",
            parent_task_id=task_id,
            task_type="insight_generation",
            description="Generate actionable insights and recommendations",
            parameters={'output_format': parameters.get('output_format', 'detailed_report')},
            required_capabilities={"insight_generation", "report_generation"},
            dependencies=[
                f"{task_id}_analyze_data",
                f"{task_id}_price_analysis" if parameters.get('include_price_analysis') else None
            ],
            estimated_duration=450  # 7.5 minutes
        ))

        # Filter out None dependencies
        for sub_task in sub_tasks:
            sub_task.dependencies = [dep for dep in sub_task.dependencies if dep is not None]

        return sub_tasks

    async def _decompose_multimodal_processing(self, task: Dict[str, Any]) -> List[SubTask]:
        """Decompose multi-modal processing task"""
        task_id = task.get('task_id')
        parameters = task.get('parameters', {})
        files = parameters.get('files', [])

        sub_tasks = []

        # Categorize files by type
        text_files = [f for f in files if f.get('type') in ['txt', 'doc', 'docx']]
        image_files = [f for f in files if f.get('type') in ['jpg', 'jpeg', 'png', 'tiff']]
        tabular_files = [f for f in files if f.get('type') in ['xlsx', 'xls', 'csv']]

        # Process text files
        if text_files:
            sub_tasks.append(SubTask(
                task_id=f"{task_id}_process_text",
                parent_task_id=task_id,
                task_type="text_processing",
                description="Process text documents",
                parameters={'files': text_files},
                required_capabilities={"text_processing", "nlp"},
                estimated_duration=len(text_files) * 120  # 2 minutes per file
            ))

        # Process images (OCR)
        if image_files:
            sub_tasks.append(SubTask(
                task_id=f"{task_id}_process_images",
                parent_task_id=task_id,
                task_type="image_processing",
                description="Process images with OCR",
                parameters={'files': image_files},
                required_capabilities={"image_processing", "ocr"},
                estimated_duration=len(image_files) * 180  # 3 minutes per image
            ))

        # Process tabular data
        if tabular_files:
            sub_tasks.append(SubTask(
                task_id=f"{task_id}_process_tables",
                parent_task_id=task_id,
                task_type="tabular_processing",
                description="Process spreadsheet and CSV data",
                parameters={'files': tabular_files},
                required_capabilities={"tabular_processing", "data_analysis"},
                estimated_duration=len(tabular_files) * 90  # 1.5 minutes per file
            ))

        # Integrate multi-modal results
        sub_tasks.append(SubTask(
            task_id=f"{task_id}_integrate_results",
            parent_task_id=task_id,
            task_type="result_integration",
            description="Integrate results from different modalities",
            parameters={},
            required_capabilities={"data_integration", "synthesis"},
            dependencies=[
                f"{task_id}_process_text" if text_files else None,
                f"{task_id}_process_images" if image_files else None,
                f"{task_id}_process_tables" if tabular_files else None
            ],
            estimated_duration=300  # 5 minutes
        ))

        # Filter out None dependencies
        for sub_task in sub_tasks:
            sub_task.dependencies = [dep for dep in sub_task.dependencies if dep is not None]

        return sub_tasks

    async def _create_execution_graph(self, sub_tasks: List[SubTask]) -> nx.DiGraph:
        """Create execution graph from sub-tasks"""
        G = nx.DiGraph()

        # Add nodes
        for sub_task in sub_tasks:
            G.add_node(sub_task.task_id, task=sub_task)

        # Add edges based on dependencies
        for sub_task in sub_tasks:
            for dep_id in sub_task.dependencies:
                if G.has_node(dep_id):
                    G.add_edge(dep_id, sub_task.task_id)

        # Validate graph for cycles
        if not nx.is_directed_acyclic_graph(G):
            # Remove cycles by breaking the latest edge
            cycles = list(nx.simple_cycles(G))
            for cycle in cycles:
                # Remove the edge that creates the cycle
                for i in range(len(cycle)):
                    src = cycle[i]
                    dst = cycle[(i + 1) % len(cycle)]
                    if G.has_edge(src, dst):
                        G.remove_edge(src, dst)
                        logger.warning(f"Removed edge {src} -> {dst} to break cycle")

        return G

    async def _estimate_task_requirements(self, sub_tasks: List[SubTask]):
        """Estimate duration and resource requirements for sub-tasks"""
        for sub_task in sub_tasks:
            # Base duration estimation
            if sub_task.estimated_duration == 0:
                sub_task.estimated_duration = self._estimate_default_duration(sub_task)

            # Resource requirements based on capabilities
            if "document_processing" in sub_task.required_capabilities:
                sub_task.metadata['memory_mb'] = 512
                sub_task.metadata['cpu_cores'] = 2
            elif "image_processing" in sub_task.required_capabilities:
                sub_task.metadata['memory_mb'] = 1024
                sub_task.metadata['cpu_cores'] = 4
            elif "data_analysis" in sub_task.required_capabilities:
                sub_task.metadata['memory_mb'] = 2048
                sub_task.metadata['cpu_cores'] = 2
            else:
                sub_task.metadata['memory_mb'] = 256
                sub_task.metadata['cpu_cores'] = 1

    def _estimate_default_duration(self, sub_task: SubTask) -> float:
        """Estimate default duration for a sub-task"""
        base_durations = {
            "data_collection": 300,
            "document_processing": 600,
            "data_analysis": 900,
            "price_analysis": 600,
            "text_processing": 120,
            "image_processing": 180,
            "tabular_processing": 90,
            "result_integration": 300
        }

        return base_durations.get(sub_task.task_type, 300)

    async def _optimize_execution_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Optimize execution plan for better performance"""
        # Analyze execution graph for parallelization opportunities
        if plan.execution_graph:
            # Find tasks that can be executed in parallel
            parallel_groups = self._find_parallelizable_groups(plan.execution_graph)

            # Reorganize tasks for optimal parallel execution
            plan.metadata['parallel_groups'] = parallel_groups

        # Optimize resource allocation
        await self._optimize_resource_allocation(plan)

        # Add fallback strategies
        plan.rollback_plan = self._create_rollback_plan(plan)

        return plan

    def _find_parallelizable_groups(self, G: nx.DiGraph) -> List[List[str]]:
        """Find groups of tasks that can be executed in parallel"""
        groups = []
        remaining_nodes = set(G.nodes())

        while remaining_nodes:
            # Find nodes with no unmet dependencies
            ready_nodes = []
            for node in remaining_nodes:
                predecessors = set(G.predecessors(node))
                if not predecessors or not any(p in remaining_nodes for p in predecessors):
                    ready_nodes.append(node)

            if ready_nodes:
                groups.append(ready_nodes)
                remaining_nodes -= set(ready_nodes)
            else:
                # Break dependency cycle if stuck
                if remaining_nodes:
                    groups.append(list(remaining_nodes))
                    break

        return groups

    async def _execute_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute the coordinated plan"""
        start_time = time.time()
        execution_results = {}
        completed_tasks = 0
        failed_tasks = 0

        try:
            # Execute based on strategy
            if plan.strategy == ExecutionStrategy.SEQUENTIAL:
                execution_results = await self._execute_sequential(plan)
            elif plan.strategy == ExecutionStrategy.PARALLEL:
                execution_results = await self._execute_parallel(plan)
            elif plan.strategy == ExecutionStrategy.PIPELINE:
                execution_results = await self._execute_pipeline(plan)
            elif plan.strategy == ExecutionStrategy.COLLABORATIVE:
                execution_results = await self._execute_collaborative(plan)
            elif plan.strategy == ExecutionStrategy.ADAPTIVE:
                execution_results = await self._execute_adaptive(plan)

            # Calculate statistics
            completed_tasks = len([st for st in plan.sub_tasks if st.status == 'completed'])
            failed_tasks = len([st for st in plan.sub_tasks if st.status == 'failed'])
            total_execution_time = time.time() - start_time

            return {
                'plan_id': plan.plan_id,
                'execution_results': execution_results,
                'completed_tasks': completed_tasks,
                'failed_tasks': failed_tasks,
                'total_tasks': len(plan.sub_tasks),
                'success_rate': completed_tasks / len(plan.sub_tasks) if plan.sub_tasks else 0,
                'total_execution_time': total_execution_time,
                'strategy_used': plan.strategy.value
            }

        except Exception as e:
            logger.error(f"Error executing plan {plan.plan_id}: {e}")
            return {
                'plan_id': plan.plan_id,
                'error': str(e),
                'success_rate': 0,
                'total_execution_time': time.time() - start_time
            }

    async def _execute_parallel(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute tasks in parallel where possible"""
        results = {}

        if not plan.execution_graph:
            return results

        # Get parallel execution groups
        parallel_groups = plan.metadata.get('parallel_groups', [])
        if not parallel_groups:
            parallel_groups = self._find_parallelizable_groups(plan.execution_graph)

        for group in parallel_groups:
            # Execute tasks in this group in parallel
            group_tasks = [plan.sub_tasks[i] for i, st in enumerate(plan.sub_tasks) if st.task_id in group]

            if len(group_tasks) == 1:
                # Single task, execute sequentially
                result = await self._execute_single_task(group_tasks[0], plan)
                results[group_tasks[0].task_id] = result
            else:
                # Multiple tasks, execute in parallel
                task_coroutines = [self._execute_single_task(task, plan) for task in group_tasks]
                group_results = await asyncio.gather(*task_coroutines, return_exceptions=True)

                for i, result in enumerate(group_results):
                    task_id = group_tasks[i].task_id
                    if isinstance(result, Exception):
                        results[task_id] = {'error': str(result), 'success': False}
                    else:
                        results[task_id] = result

        return results

    async def _execute_single_task(self, sub_task: SubTask, plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute a single sub-task"""
        sub_task.status = "running"
        sub_task.start_time = datetime.now()

        try:
            # Find suitable agent
            assigned_agent = await self._find_agent_for_task(sub_task)
            sub_task.assigned_agent = assigned_agent

            # Create task request
            task_request = TaskRequest(
                task_id=sub_task.task_id,
                task_type=sub_task.task_type,
                task_description=sub_task.description,
                parameters=sub_task.parameters,
                requirements={'capabilities': list(sub_task.required_capabilities)},
                priority=sub_task.priority
            )

            # Delegate task
            delegation_result = await self.task_delegator.delegate_task(task_request)

            # Wait for result (with timeout)
            timeout = sub_task.estimated_duration * 2  # Double the estimated time
            result = await self._wait_for_task_result(sub_task.task_id, timeout)

            sub_task.status = "completed"
            sub_task.end_time = datetime.now()
            sub_task.result = result

            return {
                'task_id': sub_task.task_id,
                'agent': assigned_agent,
                'success': True,
                'result': result,
                'execution_time': (sub_task.end_time - sub_task.start_time).total_seconds()
            }

        except Exception as e:
            sub_task.status = "failed"
            sub_task.end_time = datetime.now()
            logger.error(f"Error executing sub-task {sub_task.task_id}: {e}")

            return {
                'task_id': sub_task.task_id,
                'success': False,
                'error': str(e),
                'execution_time': (sub_task.end_time - sub_task.start_time).total_seconds()
            }

    async def _find_agent_for_task(self, sub_task: SubTask) -> str:
        """Find the best agent for a sub-task"""
        suitable_agents = []

        for agent_id, agent_info in self.agent_registry.items():
            agent_capabilities = set(agent_info.get('capabilities', []))

            # Check if agent has required capabilities
            if sub_task.required_capabilities.issubset(agent_capabilities):
                suitable_agents.append(agent_id)

        if not suitable_agents:
            raise ValueError(f"No suitable agent found for task {sub_task.task_id}")

        # Select best agent based on performance and load
        best_agent = self.load_balancer.select_agent(suitable_agents, sub_task)
        return best_agent

    async def _wait_for_task_result(self, task_id: str, timeout: float) -> Any:
        """Wait for task result with timeout"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if result is available (implement result tracking)
            await asyncio.sleep(1)  # Poll every second

        raise TimeoutError(f"Task {task_id} timed out")

    async def _aggregate_final_results(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate final results from all sub-tasks"""
        execution_results = execution_result.get('execution_results', {})

        # Collect successful results
        successful_results = {
            task_id: result for task_id, result in execution_results.items()
            if result.get('success', False)
        }

        # Collect errors
        errors = {
            task_id: result.get('error', 'Unknown error') for task_id, result in execution_results.items()
            if not result.get('success', False)
        }

        # Create aggregated result
        aggregated_result = {
            'summary': {
                'total_tasks': len(execution_results),
                'successful_tasks': len(successful_results),
                'failed_tasks': len(errors),
                'success_rate': len(successful_results) / len(execution_results) if execution_results else 0
            },
            'results': successful_results,
            'errors': errors,
            'aggregated_data': self._combine_task_data(successful_results),
            'recommendations': self._generate_execution_recommendations(execution_result)
        }

        return aggregated_result

    def _combine_task_data(self, successful_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine data from successful task results"""
        combined_data = {}

        for task_id, result in successful_results.items():
            task_data = result.get('result', {})

            # Merge data intelligently based on task type
            if isinstance(task_data, dict):
                for key, value in task_data.items():
                    if key in combined_data:
                        # Handle conflicts/merges
                        if isinstance(combined_data[key], list):
                            combined_data[key].append(value)
                        else:
                            combined_data[key] = [combined_data[key], value]
                    else:
                        combined_data[key] = value

        return combined_data

    def _generate_execution_recommendations(self, execution_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on execution results"""
        recommendations = []

        success_rate = execution_result.get('success_rate', 0)
        total_execution_time = execution_result.get('total_execution_time', 0)

        if success_rate < 0.8:
            recommendations.append("Consider improving task reliability or adding error handling")

        if total_execution_time > 3600:  # More than 1 hour
            recommendations.append("Consider optimizing parallel execution or breaking down large tasks")

        failed_tasks = execution_result.get('failed_tasks', 0)
        if failed_tasks > 0:
            recommendations.append(f"Address {failed_tasks} failed tasks in future executions")

        return recommendations


class TaskLoadBalancer:
    """Load balancer for agent task distribution"""

    def __init__(self):
        self.agent_loads: Dict[str, float] = defaultdict(float)
        self.agent_performance: Dict[str, Dict[str, float]] = defaultdict(dict)

    def select_agent(self, suitable_agents: List[str], sub_task: SubTask) -> str:
        """Select best agent based on load and performance"""
        if not suitable_agents:
            raise ValueError("No suitable agents available")

        # Calculate scores for each agent
        agent_scores = {}
        for agent_id in suitable_agents:
            score = 100.0

            # Subtract load penalty
            load_penalty = self.agent_loads[agent_id] * 10
            score -= load_penalty

            # Add performance bonus
            performance_score = self.agent_performance[agent_id].get('success_rate', 0.5) * 20
            score += performance_score

            # Speed bonus
            speed_score = (1.0 - self.agent_performance[agent_id].get('avg_response_time', 1.0)) * 15
            score += speed_score

            agent_scores[agent_id] = score

        # Select agent with highest score
        best_agent = max(agent_scores.keys(), key=lambda x: agent_scores[x])
        return best_agent

    def update_agent_load(self, agent_id: str, load_delta: float):
        """Update agent load"""
        self.agent_loads[agent_id] += load_delta

    def update_agent_performance(self, agent_id: str, metrics: Dict[str, float]):
        """Update agent performance metrics"""
        self.agent_performance[agent_id].update(metrics)


# Factory function
def create_coordinator_agent(orchestrator) -> CoordinatorAgent:
    """Create a coordinator agent"""
    return CoordinatorAgent(orchestrator)


# Usage example
if __name__ == "__main__":
    from multi_agent_orchestrator import MultiAgentOrchestrator

    async def test_coordinator():
        # Create orchestrator
        orchestrator = MultiAgentOrchestrator()
        await orchestrator.initialize()

        # Create coordinator
        coordinator = create_coordinator_agent(orchestrator)
        await coordinator.initialize()

        # Test complex task coordination
        complex_task = {
            'task_id': 'complex_analysis_001',
            'task_type': 'comprehensive_analysis',
            'description': 'Analyze market data and generate insights',
            'parameters': {
                'data_sources': [
                    {'type': 'document', 'path': '/path/to/market_report.pdf'},
                    {'type': 'database', 'table': 'factory_quotes'}
                ],
                'include_price_analysis': True,
                'output_format': 'detailed_report'
            },
            'requirements': {
                'high_accuracy': True,
                'collaboration_needed': True
            }
        }

        result = await coordinator._coordinate_complex_task(complex_task)
        print(f"Coordination result: {result}")

    asyncio.run(test_coordinator())