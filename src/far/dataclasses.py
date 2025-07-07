from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# Attempt to import LLMCallStats from a common location first, then fallback to aot
try:
    from src.common.dataclasses import LLMCallStats
except ImportError:
    from src.aot.dataclasses import LLMCallStats # Fallback if not in common

@dataclass
class FaRConfig:
    """Configuration for the Fact-and-Reflection (FaR) process."""
    fact_model_names: List[str] = field(default_factory=lambda: ["perplexity/sonar-small-online"]) # Default as per request
    main_model_names: List[str] = field(default_factory=lambda: ["deepseek/deepseek-chat"]) # Default as per request (using deepseek-chat as r1-0528 not available on OpenRouter)

    fact_model_temperature: float = 0.3
    main_model_temperature: float = 0.7

    max_fact_tokens: int = 1000
    max_main_tokens: int = 2000

    # Resource management - similar to L2T and AoT
    max_time_seconds: int = 180 # Max time for the entire FaR process
    max_reasoning_tokens: Optional[int] = None # Max cumulative completion tokens for both fact and main calls
    max_steps: int = 1  # FaR is typically a 2-step process (fact + reflection), but could be extended
    no_progress_limit: int = 1  # Not directly applicable to FaR but included for consistency
    
    # Add other relevant configurations, similar to AoTRunnerConfig or HybridConfig


@dataclass
class FaRResult:
    """Stores the results of a single Fact-and-Reflection (FaR) process run."""
    succeeded: bool = False
    problem_description: str = ""
    elicited_facts: Optional[str] = None
    final_answer: Optional[str] = None
    error_message: Optional[str] = None

    fact_call_stats: Optional[LLMCallStats] = None
    main_call_stats: Optional[LLMCallStats] = None
    reasoning_completion_tokens: int = 0  # Track tokens used specifically for reasoning operations

    total_process_wall_clock_time_seconds: float = 0.0

    @property
    def total_completion_tokens(self) -> int:
        return (self.fact_call_stats.completion_tokens if self.fact_call_stats else 0) + \
               (self.main_call_stats.completion_tokens if self.main_call_stats else 0)

    @property
    def total_prompt_tokens(self) -> int:
        return (self.fact_call_stats.prompt_tokens if self.fact_call_stats else 0) + \
               (self.main_call_stats.prompt_tokens if self.main_call_stats else 0)

    @property
    def grand_total_tokens(self) -> int:
        return self.total_completion_tokens + self.total_prompt_tokens

    @property
    def total_llm_interaction_time_seconds(self) -> float:
        return (self.fact_call_stats.call_duration_seconds if self.fact_call_stats else 0.0) + \
               (self.main_call_stats.call_duration_seconds if self.main_call_stats else 0.0)


@dataclass
class FaRSolution:
    """Overall solution object for a problem solved using FaR, including potential fallbacks."""
    far_result: Optional[FaRResult] = None
    final_answer: Optional[str] = None
    reasoning_trace: List[str] = field(default_factory=list) # For compatibility, might store facts + reflection steps

    # For orchestrator-level stats and decisions
    assessment_stats: Optional[LLMCallStats] = None
    # from src.aot.enums import AssessmentDecision # Would need this if using assessment
    # assessment_decision: Optional[AssessmentDecision] = None # Example

    # If FaR process itself fails and orchestrator falls back to one-shot
    far_failed_and_fell_back: bool = False
    fallback_call_stats: Optional[LLMCallStats] = None # Stats for the orchestrator's fallback one-shot call

    # If orchestrator decides to use one-shot directly (e.g., NEVER_FAR or after assessment)
    main_call_stats: Optional[LLMCallStats] = None # Stats for the orchestrator's direct one-shot call

    total_wall_clock_time_seconds: Optional[float] = None # Overall wall clock time for the orchestrator's solve method

    # Summary output from the FaRProcess execution, if available
    far_summary_output: Optional[str] = None


    # Aggregate token counts and times, considering all potential calls
    @property
    def total_completion_tokens(self) -> int:
        tokens = 0
        if self.far_result:
            tokens += self.far_result.total_completion_tokens
        if self.assessment_stats:
            tokens += self.assessment_stats.completion_tokens
        if self.fallback_call_stats:
            tokens += self.fallback_call_stats.completion_tokens
        if self.main_call_stats: # Orchestrator's direct one-shot
            tokens += self.main_call_stats.completion_tokens
        return tokens

    @property
    def total_prompt_tokens(self) -> int:
        tokens = 0
        if self.far_result:
            tokens += self.far_result.total_prompt_tokens
        if self.assessment_stats:
            tokens += self.assessment_stats.prompt_tokens
        if self.fallback_call_stats:
            tokens += self.fallback_call_stats.prompt_tokens
        if self.main_call_stats:
            tokens += self.main_call_stats.prompt_tokens
        return tokens

    @property
    def grand_total_tokens(self) -> int:
        return self.total_completion_tokens + self.total_prompt_tokens

    @property
    def total_llm_interaction_time_seconds(self) -> float:
        time_sum = 0.0
        if self.far_result:
            time_sum += self.far_result.total_llm_interaction_time_seconds
        if self.assessment_stats and self.assessment_stats.call_duration_seconds is not None:
            time_sum += self.assessment_stats.call_duration_seconds
        if self.fallback_call_stats and self.fallback_call_stats.call_duration_seconds is not None:
            time_sum += self.fallback_call_stats.call_duration_seconds
        if self.main_call_stats and self.main_call_stats.call_duration_seconds is not None:
            time_sum += self.main_call_stats.call_duration_seconds
        return time_sum

# Example of Enums that might be useful (can be moved to enums.py later)
# from enum import Enum
# class FaRTriggerMode(Enum):
#     ALWAYS_FAR = "always_far"
#     ASSESS_FIRST_FAR = "assess_first_far"
#     NEVER_FAR = "never_far"

# class FaRAssessmentDecision(Enum): # If specific decisions are needed for FaR
#     USE_FAR = "use_far"
#     USE_ONESHOT = "use_oneshot"
#     ERROR = "error"
