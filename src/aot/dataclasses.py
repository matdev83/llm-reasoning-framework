from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from .enums import AssessmentDecision

from .constants import (
    DEFAULT_MAIN_MODEL_NAMES, DEFAULT_MAX_STEPS, DEFAULT_MAX_TIME_SECONDS,
    DEFAULT_NO_PROGRESS_LIMIT, DEFAULT_MAIN_TEMPERATURE # DEFAULT_MAIN_TEMPERATURE will be unused here after removal
)

@dataclass
class LLMCallStats:
    completion_tokens: int = 0
    prompt_tokens: int = 0
    call_duration_seconds: float = 0.0
    model_name: Optional[str] = None

@dataclass
class ParsedLLMOutput:
    valid_steps_for_trace: List[str] = field(default_factory=list)
    all_lines_from_model_for_context: List[str] = field(default_factory=list)
    last_current_answer: Optional[str] = None
    ran_out_of_steps_signal: bool = False
    final_answer_text: Optional[str] = None
    is_final_answer_marked_done: bool = False
    # New fields for proper AoT
    initial_answer: Optional[str] = None
    reflection_text: Optional[str] = None
    refined_answer: Optional[str] = None
    is_initial_answer_provided: bool = False
    is_reflection_provided: bool = False
    is_refined_answer_provided: bool = False

@dataclass
class AoTRunnerConfig:
    main_model_names: List[str] = field(default_factory=lambda: list(DEFAULT_MAIN_MODEL_NAMES))
    # temperature: float = DEFAULT_MAIN_TEMPERATURE # REMOVED
    max_steps: int = DEFAULT_MAX_STEPS
    max_reasoning_tokens: Optional[int] = None
    max_time_seconds: int = DEFAULT_MAX_TIME_SECONDS
    no_progress_limit: int = DEFAULT_NO_PROGRESS_LIMIT
    pass_remaining_steps_pct: Optional[float] = None

@dataclass
class AoTResult:
    final_answer: Optional[str] = None
    reasoning_trace: List[str] = field(default_factory=list)
    full_history_for_context: List[str] = field(default_factory=list)
    total_completion_tokens: int = 0
    total_prompt_tokens: int = 0
    total_llm_interaction_time_seconds: float = 0.0
    total_process_wall_clock_time_seconds: float = 0.0
    reasoning_completion_tokens: int = 0
    succeeded: bool = False
    # New fields for proper AoT tracking
    initial_answer: Optional[str] = None
    reflections: List[str] = field(default_factory=list)
    refined_answers: List[str] = field(default_factory=list)
    iterations_completed: int = 0

@dataclass
class Solution:
    final_answer: Optional[str] = None
    reasoning_trace: List[str] = field(default_factory=list)
    assessment_stats: Optional[LLMCallStats] = None
    assessment_decision: Optional[AssessmentDecision] = None # Add this line
    main_call_stats: Optional[LLMCallStats] = None
    aot_result: Optional[AoTResult] = None
    fallback_call_stats: Optional[LLMCallStats] = None
    total_wall_clock_time_seconds: float = 0.0
    aot_failed_and_fell_back: bool = False
    aot_summary_output: Optional[str] = None

    @property
    def total_completion_tokens(self) -> int:
        tokens = 0
        if self.assessment_stats: tokens += self.assessment_stats.completion_tokens
        if self.main_call_stats: tokens += self.main_call_stats.completion_tokens
        if self.aot_result: tokens += self.aot_result.total_completion_tokens
        if self.fallback_call_stats: tokens += self.fallback_call_stats.completion_tokens
        return tokens

    @property
    def total_prompt_tokens(self) -> int:
        tokens = 0
        if self.assessment_stats: tokens += self.assessment_stats.prompt_tokens
        if self.main_call_stats: tokens += self.main_call_stats.prompt_tokens
        if self.aot_result: tokens += self.aot_result.total_prompt_tokens
        if self.fallback_call_stats: tokens += self.fallback_call_stats.prompt_tokens
        return tokens

    @property
    def grand_total_tokens(self) -> int:
        return self.total_completion_tokens + self.total_prompt_tokens

    @property
    def total_llm_interaction_time_seconds(self) -> float:
        time_sum = 0.0
        if self.assessment_stats: time_sum += self.assessment_stats.call_duration_seconds
        if self.main_call_stats: time_sum += self.main_call_stats.call_duration_seconds
        if self.aot_result: time_sum += self.aot_result.total_llm_interaction_time_seconds
        if self.fallback_call_stats: time_sum += self.fallback_call_stats.call_duration_seconds
        return time_sum
