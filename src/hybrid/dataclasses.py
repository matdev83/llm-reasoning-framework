from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Literal

from src.aot.dataclasses import LLMCallStats # Reusing this for now
from src.aot.enums import AssessmentDecision # Import for the new field

@dataclass
class ReasoningConfig:
    """Configuration for OpenRouter reasoning tokens"""
    enabled: bool = True
    effort: Optional[Literal["low", "medium", "high"]] = None
    max_tokens: Optional[int] = None
    exclude: bool = False  # Whether to exclude reasoning tokens from response
    
    def to_openrouter_dict(self) -> Dict[str, Any]:
        """Convert to OpenRouter API format"""
        config = {"enabled": self.enabled}
        
        if self.effort is not None:
            config["effort"] = self.effort
        
        if self.max_tokens is not None:
            config["max_tokens"] = self.max_tokens
        
        # Always include exclude parameter (it's universally supported)
        config["exclude"] = self.exclude
        
        return config

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
    
    # New OpenRouter reasoning configuration
    reasoning_config: Optional[ReasoningConfig] = None
    use_streaming: bool = True  # Enable streaming for token optimization
    
    # Model-specific configurations
    model_specific_headers: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    def _is_reasoning_model(self, model_name: str) -> bool:
        """Check if a model supports reasoning tokens"""
        reasoning_models = [
            "deepseek", "r1", "gemini", "thinking", "claude", "o1", "o3", "o4", "qwq", "grok"
        ]
        model_lower = model_name.lower()
        return any(keyword in model_lower for keyword in reasoning_models)
    
    def _get_model_reasoning_support(self, model_name: str) -> Dict[str, bool]:
        """
        Determine which reasoning parameters a model supports.
        
        Returns:
            Dict with 'effort', 'max_tokens', and 'basic' support flags
        """
        model_lower = model_name.lower()
        
        # OpenAI o-series models support effort levels
        if any(pattern in model_lower for pattern in ["openai/o", "/o1", "/o3", "/o4"]):
            return {"effort": True, "max_tokens": False, "basic": True}
        
        # Grok models support effort levels
        if "grok" in model_lower:
            return {"effort": True, "max_tokens": False, "basic": True}
        
        # Anthropic models support max_tokens (including claude-3.7-sonnet)
        if any(pattern in model_lower for pattern in ["anthropic/", "claude"]):
            return {"effort": False, "max_tokens": True, "basic": True}
        
        # Gemini thinking models support max_tokens
        if "gemini" in model_lower and "thinking" in model_lower:
            return {"effort": False, "max_tokens": True, "basic": True}
        
        # DeepSeek and other reasoning models support basic reasoning but no effort/max_tokens control
        if any(pattern in model_lower for pattern in ["deepseek", "r1", "qwq"]):
            return {"effort": False, "max_tokens": False, "basic": True}
        
        # Non-reasoning models
        return {"effort": False, "max_tokens": False, "basic": False}
    
    def get_model_default_reasoning_config(self, model_name: str) -> Optional[ReasoningConfig]:
        """
        Get model-specific default reasoning configuration.
        
        Based on OpenRouter documentation and model capabilities:
        - OpenAI/Grok: effort="high" (best reasoning quality)
        - Gemini Thinking: max_tokens=32000 (confirmed valid, no specific limit mentioned)
        - Anthropic Claude: max_tokens=8000 (equivalent to "high" effort, within 32K limit)
        - DeepSeek-R1: enabled=True (basic reasoning only)
        - Other models: None (no reasoning support)
        """
        support = self._get_model_reasoning_support(model_name)
        
        if not support["basic"]:
            return None
        
        model_lower = model_name.lower()
        
        # OpenAI o-series models: Use high effort
        if any(pattern in model_lower for pattern in ["openai/o", "/o1", "/o3", "/o4"]):
            return ReasoningConfig(
                enabled=True,
                effort="high",
                exclude=False
            )
        
        # Grok models: Use high effort
        elif "grok" in model_lower:
            return ReasoningConfig(
                enabled=True,
                effort="high",
                exclude=False
            )
        
        # Gemini Thinking models: Use 32K tokens (confirmed valid)
        elif "gemini" in model_lower and "thinking" in model_lower:
            return ReasoningConfig(
                enabled=True,
                max_tokens=32000,  # Maximum reasoning capability
                exclude=False
            )
        
        # Anthropic Claude models: Use 8K tokens (equivalent to high effort)
        elif any(pattern in model_lower for pattern in ["anthropic/", "claude"]):
            return ReasoningConfig(
                enabled=True,
                max_tokens=8000,  # High reasoning effort equivalent
                exclude=False
            )
        
        # DeepSeek-R1 and other basic reasoning models: Just enable
        else:
            return ReasoningConfig(
                enabled=True,
                exclude=False
            )
    
    def get_effective_reasoning_config(self, model_name: str) -> Optional[ReasoningConfig]:
        """
        Get the effective reasoning configuration for a specific model,
        filtering out unsupported parameters and applying defaults if needed.
        """
        support = self._get_model_reasoning_support(model_name)
        
        if not support["basic"]:
            # Model doesn't support reasoning at all
            return None
        
        # Use provided config or fall back to model defaults
        base_config = self.reasoning_config or self.get_model_default_reasoning_config(model_name)
        
        if not base_config:
            return None
        
        # Create a copy of the config with only supported parameters
        effective_config = ReasoningConfig(
            enabled=base_config.enabled,
            exclude=base_config.exclude
        )
        
        # Add effort only if supported
        if support["effort"] and base_config.effort:
            effective_config.effort = base_config.effort
        
        # Add max_tokens only if supported
        if support["max_tokens"] and base_config.max_tokens:
            effective_config.max_tokens = base_config.max_tokens
        
        return effective_config
    
    def get_model_headers(self, model_name: str) -> Dict[str, str]:
        """Get model-specific headers"""
        # Check for exact match first
        if model_name in self.model_specific_headers:
            return self.model_specific_headers[model_name]
        
        # Check for partial matches (e.g., "deepseek" matches "deepseek/deepseek-r1")
        for pattern, headers in self.model_specific_headers.items():
            if pattern.lower() in model_name.lower():
                return headers
        
        return {}

    def get_model_specific_token_limits(self, model_name: str) -> Dict[str, int]:
        """
        Get model-specific token limits based on reasoning capabilities.
        
        For models with high reasoning token usage, we need to ensure adequate
        space for both reasoning and response tokens.
        
        Returns:
            Dict with 'max_reasoning_tokens' and 'max_response_tokens'
        """
        model_lower = model_name.lower()
        
        # Gemini Thinking models: Large context (1M tokens), high reasoning usage
        if "gemini" in model_lower and "thinking" in model_lower:
            return {
                "max_reasoning_tokens": 40000,  # Increased to accommodate 32K reasoning + overhead
                "max_response_tokens": 8000     # Generous space for detailed responses
            }
        
        # Anthropic Claude models: Large context (200K tokens), moderate reasoning usage
        elif any(pattern in model_lower for pattern in ["anthropic/", "claude"]):
            return {
                "max_reasoning_tokens": 12000,  # Increased to accommodate 8K reasoning + overhead
                "max_response_tokens": 4000     # Good space for responses
            }
        
        # OpenAI o-series models: Variable context, effort-based reasoning
        elif any(pattern in model_lower for pattern in ["openai/o", "/o1", "/o3", "/o4"]):
            return {
                "max_reasoning_tokens": 8000,   # Effort-based, usually moderate usage
                "max_response_tokens": 3000     # Standard response space
            }
        
        # Grok models: Similar to OpenAI, effort-based
        elif "grok" in model_lower:
            return {
                "max_reasoning_tokens": 8000,   # Effort-based reasoning
                "max_response_tokens": 3000     # Standard response space
            }
        
        # DeepSeek-R1 and other models: Moderate usage
        else:
            return {
                "max_reasoning_tokens": 2000,   # Conservative for basic reasoning
                "max_response_tokens": 1500     # Standard response space
            }
    
    def get_effective_token_limits(self) -> Dict[str, int]:
        """
        Get the effective token limits for the current configuration.
        
        Uses model-specific limits if available, otherwise falls back to
        configured values.
        """
        model_limits = self.get_model_specific_token_limits(self.reasoning_model_name)
        
        # Use model-specific limits if they differ from defaults, otherwise use configured values
        default_reasoning = 1500  # Default from dataclass
        default_response = 1500   # Default from dataclass
        
        # If user hasn't explicitly set custom values, use model-specific limits
        if self.max_reasoning_tokens == default_reasoning:
            max_reasoning = model_limits["max_reasoning_tokens"]
        else:
            max_reasoning = self.max_reasoning_tokens
            
        if self.max_response_tokens == default_response:
            max_response = model_limits["max_response_tokens"]
        else:
            max_response = self.max_response_tokens
        
        return {
            "max_reasoning_tokens": max_reasoning,
            "max_response_tokens": max_response
        }

@dataclass
class HybridResult:
    succeeded: bool = False
    final_answer: Optional[str] = None
    extracted_reasoning: Optional[str] = None
    reasoning_call_stats: Optional[LLMCallStats] = None
    response_call_stats: Optional[LLMCallStats] = None
    error_message: Optional[str] = None
    detected_reasoning_format: Optional[str] = None  # Track which format was detected

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
