from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

from src.aot.dataclasses import LLMCallStats # Reusing this for now
from src.aot.enums import AssessmentDecision # Import for the new field

@dataclass
class HybridConfig:
    reasoning_model_name: str
    response_model_name: str
    reasoning_model_temperature: float = 0.1
    reasoning_prompt_template: str = "Problem: {problem_description}\n\nThink step-by-step to reach the solution. After you have finished your reasoning, output the token sequence: {reasoning_complete_token}\n\nReasoning:"
    reasoning_complete_token: str = "<REASONING_COMPLETE>"
    response_model_temperature: float = 0.7
    response_prompt_template: str = "Original Problem: {problem_description}\n\nExtracted Reasoning:\n<thinking>{extracted_reasoning}</thinking>\n\nBased on the original problem and the extracted reasoning, provide the final solution."
    max_reasoning_tokens: int = 1500
    max_response_tokens: int = 1500

@dataclass
class HybridResult:
    succeeded: bool = False
    final_answer: Optional[str] = None
    extracted_reasoning: Optional[str] = None
    reasoning_call_stats: Optional[LLMCallStats] = None
    response_call_stats: Optional[LLMCallStats] = None
    error_message: Optional[str] = None

@dataclass
class HybridSolution:
    hybrid_result: Optional[HybridResult] = None
    final_answer: Optional[str] = None
    reasoning_trace: List[str] = field(default_factory=list) # For compatibility, might not be directly used by hybrid initially

    # Stats
    main_call_stats: Optional[LLMCallStats] = None # For direct one-shot if hybrid is skipped
    fallback_call_stats: Optional[LLMCallStats] = None # If hybrid fails and falls back
    assessment_stats: Optional[LLMCallStats] = None # If complexity assessment is used
    assessment_decision: Optional[AssessmentDecision] = None # Added this field

    hybrid_failed_and_fell_back: bool = False

    # Summary outputs
    hybrid_summary_output: Optional[str] = None # Summary from HybridProcessor

    # Timing
    total_wall_clock_time_seconds: Optional[float] = None

    @property
    def total_completion_tokens(self) -> int:
        tokens = 0
        if self.hybrid_result and self.hybrid_result.reasoning_call_stats:
            tokens += self.hybrid_result.reasoning_call_stats.completion_tokens
        if self.hybrid_result and self.hybrid_result.response_call_stats:
            tokens += self.hybrid_result.response_call_stats.completion_tokens
        if self.main_call_stats:
            tokens += self.main_call_stats.completion_tokens
        if self.fallback_call_stats:
            tokens += self.fallback_call_stats.completion_tokens
        if self.assessment_stats:
            tokens += self.assessment_stats.completion_tokens
        return tokens

    @property
    def total_prompt_tokens(self) -> int:
        tokens = 0
        if self.hybrid_result and self.hybrid_result.reasoning_call_stats:
            tokens += self.hybrid_result.reasoning_call_stats.prompt_tokens
        if self.hybrid_result and self.hybrid_result.response_call_stats:
            tokens += self.hybrid_result.response_call_stats.prompt_tokens
        if self.main_call_stats:
            tokens += self.main_call_stats.prompt_tokens
        if self.fallback_call_stats:
            tokens += self.fallback_call_stats.prompt_tokens
        if self.assessment_stats:
            tokens += self.assessment_stats.prompt_tokens
        return tokens

    @property
    def grand_total_tokens(self) -> int:
        return self.total_completion_tokens + self.total_prompt_tokens

    @property
    def total_llm_interaction_time_seconds(self) -> float:
        time_sum = 0.0
        if self.hybrid_result and self.hybrid_result.reasoning_call_stats:
            time_sum += self.hybrid_result.reasoning_call_stats.call_duration_seconds
        if self.hybrid_result and self.hybrid_result.response_call_stats:
            time_sum += self.hybrid_result.response_call_stats.call_duration_seconds
        if self.main_call_stats:
            time_sum += self.main_call_stats.call_duration_seconds
        if self.fallback_call_stats:
            time_sum += self.fallback_call_stats.call_duration_seconds
        if self.assessment_stats:
            time_sum += self.assessment_stats.call_duration_seconds
        return time_sum
