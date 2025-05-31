from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from src.aot.dataclasses import LLMCallStats # Import LLMCallStats
from src.aot.enums import AssessmentDecision # Import AssessmentDecision
from src.llm_config import LLMConfig # Make sure this import is added
from .enums import L2TTriggerMode # Import L2TTriggerMode

# Import constants for L2TModelConfigs.from_l2t_config default values
from .constants import (
    DEFAULT_L2T_CLASSIFICATION_MODEL_NAMES,
    # DEFAULT_L2T_CLASSIFICATION_TEMPERATURE, # Removed, will be in LLMConfig
    DEFAULT_L2T_INITIAL_PROMPT_MODEL_NAMES,
    # DEFAULT_L2T_INITIAL_PROMPT_TEMPERATURE, # Removed, will be in LLMConfig
    DEFAULT_L2T_MAX_STEPS,
    DEFAULT_L2T_MAX_TIME_SECONDS,
    DEFAULT_L2T_MAX_TOTAL_NODES,
    DEFAULT_L2T_THOUGHT_GENERATION_MODEL_NAMES,
    # DEFAULT_L2T_THOUGHT_GENERATION_TEMPERATURE, # Removed, will be in LLMConfig
    DEFAULT_L2T_X_EVA_DEFAULT,
    DEFAULT_L2T_X_FMT_DEFAULT,
    # Need to ensure these are available if from_l2t_config uses them directly
    # For the from_l2t_config method, we'll use the actual default temperature values from constants.py
    DEFAULT_L2T_INITIAL_PROMPT_TEMPERATURE,
    DEFAULT_L2T_CLASSIFICATION_TEMPERATURE,
    DEFAULT_L2T_THOUGHT_GENERATION_TEMPERATURE,
)


class L2TNodeCategory(Enum):
    CONTINUE = "CONTINUE"
    TERMINATE_BRANCH = "TERMINATE_BRANCH"
    FINAL_ANSWER = "FINAL_ANSWER"
    BACKTRACK = "BACKTRACK"


@dataclass
class L2TNode:
    id: str
    content: str
    parent_id: Optional[str]
    generation_step: int
    children_ids: List[str] = field(default_factory=list)
    category: Optional[L2TNodeCategory] = None


@dataclass
class L2TGraph:
    nodes: Dict[str, L2TNode] = field(default_factory=dict)
    v_pres: List[str] = field(default_factory=list)
    v_hist: List[str] = field(default_factory=list)
    root_node_id: Optional[str] = None

    def add_node(self, node: L2TNode, is_root: bool = False) -> None:
        if node.id in self.nodes:
            raise ValueError(f"Node with id {node.id} already exists.")
        self.nodes[node.id] = node
        self.v_pres.append(node.id)
        if is_root:
            if self.root_node_id is not None:
                raise ValueError(
                    f"Root node already set to {self.root_node_id}. Cannot set {node.id} as root."
                )
            self.root_node_id = node.id
        if node.parent_id and node.parent_id in self.nodes:
            parent_node = self.nodes[node.parent_id]
            if node.id not in parent_node.children_ids:
                parent_node.children_ids.append(node.id)

    def classify_node(self, node_id: str, category: L2TNodeCategory) -> None:
        if node_id not in self.nodes:
            raise ValueError(f"Node with id {node_id} not found.")
        self.nodes[node_id].category = category

    def move_to_hist(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise ValueError(f"Node with id {node_id} not found.")
        if node_id not in self.v_pres:
            if node_id not in self.v_hist:
                 raise ValueError(f"Node {node_id} not in v_pres to move to v_hist.")
            return
        self.v_pres.remove(node_id)
        if node_id not in self.v_hist:
            self.v_hist.append(node_id)

    def re_add_to_v_pres(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise ValueError(f"Node with id {node_id} not found in graph.")
        
        if node_id in self.v_hist:
            self.v_hist.remove(node_id)
        
        if node_id not in self.v_pres:
            self.v_pres.append(node_id)
        
        self.nodes[node_id].category = None

    def get_node(self, node_id: str) -> Optional[L2TNode]:
        return self.nodes.get(node_id)

    def get_parent(self, node_id: str) -> Optional[L2TNode]:
        node = self.get_node(node_id)
        if node and node.parent_id:
            return self.get_node(node.parent_id)
        return None

    def get_children(self, node_id: str) -> List[L2TNode]:
        node = self.get_node(node_id)
        if node:
            return [self.nodes[child_id] for child_id in node.children_ids if child_id in self.nodes]
        return []


@dataclass
class L2TConfig: # Modified: Removed temperature fields
    classification_model_names: List[str] = field(
        default_factory=lambda: DEFAULT_L2T_CLASSIFICATION_MODEL_NAMES
    )
    thought_generation_model_names: List[str] = field(
        default_factory=lambda: DEFAULT_L2T_THOUGHT_GENERATION_MODEL_NAMES
    )
    initial_prompt_model_names: List[str] = field(
        default_factory=lambda: DEFAULT_L2T_INITIAL_PROMPT_MODEL_NAMES
    )
    # classification_temperature: float = DEFAULT_L2T_CLASSIFICATION_TEMPERATURE # REMOVED
    # thought_generation_temperature: float = DEFAULT_L2T_THOUGHT_GENERATION_TEMPERATURE # REMOVED
    # initial_prompt_temperature: float = DEFAULT_L2T_INITIAL_PROMPT_TEMPERATURE # REMOVED
    max_steps: int = DEFAULT_L2T_MAX_STEPS
    max_total_nodes: int = DEFAULT_L2T_MAX_TOTAL_NODES
    max_time_seconds: int = DEFAULT_L2T_MAX_TIME_SECONDS
    x_fmt_default: str = DEFAULT_L2T_X_FMT_DEFAULT
    x_eva_default: str = DEFAULT_L2T_X_EVA_DEFAULT
    pass_remaining_steps_pct: Optional[float] = None


@dataclass
class L2TModelConfigs:
    initial_thought_config: LLMConfig = field(default_factory=LLMConfig)
    node_classification_config: LLMConfig = field(default_factory=LLMConfig)
    node_thought_generation_config: LLMConfig = field(default_factory=LLMConfig)
    orchestrator_oneshot_config: LLMConfig = field(default_factory=LLMConfig)
    summary_config: LLMConfig = field(default_factory=LLMConfig)

    @classmethod
    def from_l2t_config(cls, l2t_config: L2TConfig) -> 'L2TModelConfigs':
        # Uses the default temperatures imported from .constants
        # Note: l2t_config no longer holds these temperatures directly.
        # This method might need rethinking if it's meant to transfer old L2TConfig temps
        # to new L2TModelConfigs. For now, it uses the global defaults for temperatures.
        # If the goal was to use temperatures from an *old* L2TConfig instance, that instance
        # would need to be passed before its temperature fields are removed.
        # Assuming for now it populates with default temperatures for newly created configs.
        return cls(
            initial_thought_config=LLMConfig(temperature=DEFAULT_L2T_INITIAL_PROMPT_TEMPERATURE),
            node_classification_config=LLMConfig(temperature=DEFAULT_L2T_CLASSIFICATION_TEMPERATURE),
            node_thought_generation_config=LLMConfig(temperature=DEFAULT_L2T_THOUGHT_GENERATION_TEMPERATURE),
            orchestrator_oneshot_config=LLMConfig(temperature=DEFAULT_L2T_INITIAL_PROMPT_TEMPERATURE),
            summary_config=LLMConfig(temperature=DEFAULT_L2T_INITIAL_PROMPT_TEMPERATURE)
        )


@dataclass
class L2TResult:
    final_answer: Optional[str] = None
    reasoning_graph: Optional[L2TGraph] = None
    total_llm_calls: int = 0
    total_completion_tokens: int = 0
    total_prompt_tokens: int = 0
    total_llm_interaction_time_seconds: float = 0.0
    total_process_wall_clock_time_seconds: float = 0.0
    succeeded: bool = False
    error_message: Optional[str] = None


@dataclass
class L2TSolution:
    final_answer: Optional[str] = None
    total_wall_clock_time_seconds: float = 0.0
    assessment_stats: Optional[LLMCallStats] = None
    assessment_decision: Optional[AssessmentDecision] = None
    main_call_stats: Optional[LLMCallStats] = None
    l2t_result: Optional[L2TResult] = None
    l2t_summary_output: Optional[str] = None
    l2t_failed_and_fell_back: bool = False
    fallback_call_stats: Optional[LLMCallStats] = None

    @property
    def succeeded(self) -> bool:
        return self.final_answer is not None

    @property
    def total_completion_tokens(self) -> int:
        total = 0
        if self.assessment_stats:
            total += self.assessment_stats.completion_tokens
        if self.main_call_stats:
            total += self.main_call_stats.completion_tokens
        if self.l2t_result:
            total += self.l2t_result.total_completion_tokens
        if self.fallback_call_stats:
            total += self.fallback_call_stats.completion_tokens
        return total

    @property
    def total_prompt_tokens(self) -> int:
        total = 0
        if self.assessment_stats:
            total += self.assessment_stats.prompt_tokens
        if self.main_call_stats:
            total += self.main_call_stats.prompt_tokens
        if self.l2t_result:
            total += self.l2t_result.total_prompt_tokens
        if self.fallback_call_stats:
            total += self.fallback_call_stats.prompt_tokens
        return total

    @property
    def grand_total_tokens(self) -> int:
        return self.total_completion_tokens + self.total_prompt_tokens

    @property
    def total_llm_interaction_time_seconds(self) -> float:
        total = 0.0
        if self.assessment_stats:
            total += self.assessment_stats.call_duration_seconds
        if self.main_call_stats:
            total += self.main_call_stats.call_duration_seconds
        if self.l2t_result:
            total += self.l2t_result.total_llm_interaction_time_seconds
        if self.fallback_call_stats:
            total += self.fallback_call_stats.call_duration_seconds
        return total
