from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set

from src.llm_config import LLMConfig
from src.aot.dataclasses import LLMCallStats
from src.aot.enums import AssessmentDecision

# Enum for Thought Status
class GoTThoughtStatus(Enum):
    ACTIVE = "ACTIVE"
    REFINED = "REFINED"
    AGGREGATED = "AGGREGATED"
    PRUNED = "PRUNED"
    SOLUTION_CANDIDATE = "SOLUTION_CANDIDATE"

@dataclass
class GoTThought:
    id: str
    content: str
    parent_ids: Set[str] = field(default_factory=set)
    children_ids: Set[str] = field(default_factory=set)
    generation_step: int = 0
    score: float = 0.0
    status: GoTThoughtStatus = GoTThoughtStatus.ACTIVE
    # Potentially store different versions if a thought is refined
    history: List[str] = field(default_factory=list)

@dataclass
class GoTGraph:
    thoughts: Dict[str, GoTThought] = field(default_factory=dict)
    # Edges could be represented implicitly by parent_ids and children_ids in GoTThought
    # or explicitly if edges need their own attributes (e.g., edge_type, weight)
    # For now, implicit representation is simpler.

    def add_thought(self, thought: GoTThought) -> None:
        if thought.id in self.thoughts:
            # Potentially update existing thought or raise error
            # For now, let's overwrite, assuming new version is preferred
            # Or, better, only add if not present. Updates should be explicit.
            raise ValueError(f"Thought with id {thought.id} already exists.")
        self.thoughts[thought.id] = thought
        for parent_id in thought.parent_ids:
            if parent_id in self.thoughts:
                self.thoughts[parent_id].children_ids.add(thought.id)
            # Else: handle dangling parent_id if necessary, or assume valid graph construction

    def get_thought(self, thought_id: str) -> Optional[GoTThought]:
        return self.thoughts.get(thought_id)

    def update_thought_status(self, thought_id: str, status: GoTThoughtStatus) -> None:
        thought = self.get_thought(thought_id)
        if thought:
            thought.status = status
        else:
            raise ValueError(f"Thought with id {thought_id} not found for status update.")

    def update_thought_score(self, thought_id: str, score: float) -> None:
        thought = self.get_thought(thought_id)
        if thought:
            thought.score = score
        else:
            raise ValueError(f"Thought with id {thought_id} not found for score update.")

    def add_edge(self, parent_id: str, child_id: str) -> None:
        parent_thought = self.get_thought(parent_id)
        child_thought = self.get_thought(child_id)
        if not parent_thought:
            raise ValueError(f"Parent thought with id {parent_id} not found.")
        if not child_thought:
            raise ValueError(f"Child thought with id {child_id} not found.")

        parent_thought.children_ids.add(child_id)
        child_thought.parent_ids.add(parent_id)

    def get_parents(self, thought_id: str) -> List[GoTThought]:
        thought = self.get_thought(thought_id)
        if thought:
            return [self.thoughts[pid] for pid in thought.parent_ids if pid in self.thoughts]
        return []

    def get_children(self, thought_id: str) -> List[GoTThought]:
        thought = self.get_thought(thought_id)
        if thought:
            return [self.thoughts[cid] for cid in thought.children_ids if cid in self.thoughts]
        return []

@dataclass
class GoTConfig:
    # LLM Model names for various operations
    thought_generation_model_names: List[str] = field(default_factory=lambda: ["openai/gpt-3.5-turbo"])
    scoring_model_names: List[str] = field(default_factory=lambda: ["openai/gpt-3.5-turbo"]) # Can be a smaller model
    aggregation_model_names: List[str] = field(default_factory=lambda: ["openai/gpt-3.5-turbo"])
    refinement_model_names: List[str] = field(default_factory=lambda: ["openai/gpt-3.5-turbo"])

    # Operational parameters
    max_thoughts: int = 50 # Max total thoughts in the graph
    max_iterations: int = 10 # Max iterations of generation/transformation
    min_score_for_expansion: float = 0.5 # Minimum score to consider a thought for expansion
    pruning_threshold_score: Optional[float] = 0.2 # Thoughts below this score might be pruned
    max_children_per_thought: int = 3 # Max new thoughts to generate from one parent
    max_parents_for_aggregation: int = 5 # Max parents to consider for aggregation

    # Control flags for transformations
    enable_aggregation: bool = True
    enable_refinement: bool = True
    enable_pruning: bool = True

    # Termination conditions
    solution_found_score_threshold: float = 0.9 # If a thought reaches this score, it might be a solution
    max_time_seconds: int = 300 # Max time for the GoT process

@dataclass
class GoTModelConfigs:
    thought_generation_config: LLMConfig = field(default_factory=LLMConfig)
    scoring_config: LLMConfig = field(default_factory=lambda: LLMConfig(temperature=0.2)) # Often lower temp for scoring
    aggregation_config: LLMConfig = field(default_factory=LLMConfig)
    refinement_config: LLMConfig = field(default_factory=LLMConfig)
    # Config for one-shot fallback if GoT itself fails (used by GoTProcess/Orchestrator)
    orchestrator_oneshot_config: LLMConfig = field(default_factory=LLMConfig)

@dataclass
class GoTResult:
    final_answer: Optional[str] = None
    final_graph: Optional[GoTGraph] = None
    total_llm_calls: int = 0
    total_completion_tokens: int = 0
    total_prompt_tokens: int = 0
    total_llm_interaction_time_seconds: float = 0.0
    total_process_wall_clock_time_seconds: float = 0.0
    succeeded: bool = False
    error_message: Optional[str] = None
    # Optional: Store a list of candidate solutions if multiple emerge
    solution_candidates: List[GoTThought] = field(default_factory=list)

@dataclass
class GoTSolution:
    final_answer: Optional[str] = None
    total_wall_clock_time_seconds: float = 0.0 # Overall time for orchestrator

    # Stats from different phases
    assessment_stats: Optional[LLMCallStats] = None
    assessment_decision: Optional[AssessmentDecision] = None
    # main_call_stats would be part of got_result if GoT runs
    # fallback_call_stats if GoT fails and orchestrator falls back
    fallback_call_stats: Optional[LLMCallStats] = None

    got_result: Optional[GoTResult] = None
    got_summary_output: Optional[str] = None # Summary from GoTProcessor
    got_failed_and_fell_back: bool = False

    @property
    def succeeded(self) -> bool:
        # Considered succeeded if GoT itself succeeded and produced an answer,
        # or if GoT failed but fallback produced an answer.
        if self.got_result and self.got_result.succeeded and self.final_answer is not None:
            return True
        if self.got_failed_and_fell_back and self.final_answer is not None:
            return True
        # If it's a "NEVER_GOT" mode, success depends on the main_call_stats (direct oneshot)
        # This part might need adjustment based on how NEVER_GOT is handled (i.e. if main_call_stats is stored directly on GoTSolution)
        # For now, let's assume direct oneshot in NEVER_GOT also populates final_answer.
        if self.final_answer is not None and not self.got_result and not self.got_failed_and_fell_back: # Implies direct oneshot
             # This case needs to be handled carefully based on Orchestrator logic for NEVER_GOT
             # Let's assume for now that if final_answer is present, it's a success.
            return True
        return False

    @property
    def total_completion_tokens(self) -> int:
        total = 0
        if self.assessment_stats:
            total += self.assessment_stats.completion_tokens
        if self.got_result:
            total += self.got_result.total_completion_tokens
        if self.fallback_call_stats:
            total += self.fallback_call_stats.completion_tokens
        return total

    @property
    def total_prompt_tokens(self) -> int:
        total = 0
        if self.assessment_stats:
            total += self.assessment_stats.prompt_tokens
        if self.got_result:
            total += self.got_result.total_prompt_tokens
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
        if self.got_result:
            total += self.got_result.total_llm_interaction_time_seconds
        if self.fallback_call_stats:
            total += self.fallback_call_stats.call_duration_seconds
        return total
