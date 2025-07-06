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
    
    def get_model_reasoning_support(self, model_name: str) -> Dict[str, bool]:
        """
        Get model-specific reasoning support capabilities.
        
        Returns:
            Dict with boolean flags for supported reasoning features
        """
        model_lower = model_name.lower()
        
        # OpenAI o-series models: Use effort-based reasoning
        if any(model in model_lower for model in ["o1", "o3", "o4"]):
            return {
                "supports_effort": True,
                "supports_max_tokens": False,
                "supports_exclude": True,
                "uses_prompt_activation": False,  # Uses API headers
                "reasoning_enabled_by_default": True
            }
        
        # Grok models: Use effort-based reasoning  
        if "grok" in model_lower:
            return {
                "supports_effort": True,
                "supports_max_tokens": False,
                "supports_exclude": True,
                "uses_prompt_activation": False,  # Uses API headers
                "reasoning_enabled_by_default": True
            }
        
        # Gemini Thinking models: Use max_tokens-based reasoning
        if "gemini" in model_lower and "thinking" in model_lower:
            return {
                "supports_effort": False,
                "supports_max_tokens": True,
                "supports_exclude": True,
                "uses_prompt_activation": False,  # Uses API headers
                "reasoning_enabled_by_default": True
            }
        
        # Anthropic Claude models: Use max_tokens-based reasoning
        if "anthropic" in model_lower or "claude" in model_lower:
            return {
                "supports_effort": False,
                "supports_max_tokens": True,
                "supports_exclude": True,
                "uses_prompt_activation": False,  # Uses API headers
                "reasoning_enabled_by_default": True
            }
        
        # Qwen models: Use prompt-based activation with slash commands
        if "qwen" in model_lower or "qwq" in model_lower:
            return {
                "supports_effort": False,
                "supports_max_tokens": False,
                "supports_exclude": False,
                "uses_prompt_activation": True,   # Uses /think and /no_think commands
                "reasoning_enabled_by_default": True
            }
        
        # DeepSeek-R1: Basic reasoning support
        if "deepseek" in model_lower and ("r1" in model_lower or "reasoning" in model_lower):
            return {
                "supports_effort": False,
                "supports_max_tokens": False,
                "supports_exclude": False,
                "uses_prompt_activation": False,  # Uses API headers
                "reasoning_enabled_by_default": True
            }
        
        # Default: No reasoning support
        return {
            "supports_effort": False,
            "supports_max_tokens": False,
            "supports_exclude": False,
            "uses_prompt_activation": False,
            "reasoning_enabled_by_default": False
        }
    
    def get_model_default_reasoning_config(self, model_name: str) -> ReasoningConfig:
        """
        Get model-specific default reasoning configuration.
        
        These defaults are based on actual OpenRouter API constraints and
        optimal settings for each model family.
        
        Returns:
            ReasoningConfig with appropriate defaults for the model
        """
        model_lower = model_name.lower()
        
        # OpenAI o-series models: Use effort-based reasoning (high output limits)
        if any(model in model_lower for model in ["o1", "o3", "o4"]):
            return ReasoningConfig(
                enabled=True,
                effort="high",  # o-series models work best with high effort
                max_tokens=None,
                exclude=False
            )
        
        # Grok models: Use effort-based reasoning
        elif "grok" in model_lower:
            return ReasoningConfig(
                enabled=True,
                effort="high",  # Grok models support effort levels
                max_tokens=None,
                exclude=False
            )
        
        # OpenAI GPT-4o models: Use effort-based reasoning (16K output limit)
        elif "gpt-4o" in model_lower:
            return ReasoningConfig(
                enabled=True,
                effort="high",  # GPT-4o supports effort levels
                max_tokens=None,
                exclude=False
            )
        
        # OpenAI GPT-4 models: Use effort-based reasoning (lower output limits)
        elif "gpt-4" in model_lower:
            return ReasoningConfig(
                enabled=True,
                effort="medium",  # More conservative for lower limits
                max_tokens=None,
                exclude=False
            )
        
        # OpenAI GPT-3.5 models: Use effort-based reasoning (low output limits)
        elif "gpt-3.5" in model_lower:
            return ReasoningConfig(
                enabled=True,
                effort="low",  # Very conservative for low limits
                max_tokens=None,
                exclude=False
            )
        
        # Anthropic Claude models: Use max_tokens (estimated ~8K output limit)
        elif "claude" in model_lower:
            return ReasoningConfig(
                enabled=True,
                effort=None,
                max_tokens=4000,  # Conservative reasoning token allocation
                exclude=False
            )
        
        # Gemini models: Use max_tokens (estimated ~8-16K output limit)
        elif "gemini" in model_lower:
            if "thinking" in model_lower:
                return ReasoningConfig(
                    enabled=True,
                    effort=None,
                    max_tokens=8000,  # Higher for thinking models
                    exclude=False
                )
            else:
                return ReasoningConfig(
                    enabled=True,
                    effort=None,
                    max_tokens=4000,  # Conservative for regular models
                    exclude=False
                )
        
        # Qwen models: Use prompt-based activation (no API parameters)
        elif "qwen" in model_lower or "qwq" in model_lower:
            return ReasoningConfig(
                enabled=True,
                effort=None,       # Not supported - uses /think command
                max_tokens=None,   # Not supported - uses /think command
                exclude=False
            )
        
        # DeepSeek and other reasoning models: Use max_tokens (estimated ~8K output limit)
        elif any(model in model_lower for model in ["deepseek", "minimax"]):
            return ReasoningConfig(
                enabled=True,
                effort=None,
                max_tokens=4000,  # Conservative reasoning allocation
                exclude=False
            )
        
        # Default for unknown models: Conservative settings
        return ReasoningConfig(
            enabled=True,
            effort="low",  # Safe default
            max_tokens=None,
            exclude=False
        )
    
    def get_effective_reasoning_config(self, model_name: str) -> Optional[ReasoningConfig]:
        """
        Get the effective reasoning configuration for a specific model,
        filtering out unsupported parameters and applying defaults if needed.
        """
        support = self.get_model_reasoning_support(model_name)
        
        if not support["reasoning_enabled_by_default"]:
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
        if support["supports_effort"] and base_config.effort:
            effective_config.effort = base_config.effort
        
        # Add max_tokens only if supported
        if support["supports_max_tokens"] and base_config.max_tokens:
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
        Get model-specific token limits based on actual OpenRouter API constraints.
        
        These limits are based on the real output token limits for each model family
        on OpenRouter, not their context windows. Output tokens include both 
        reasoning and response tokens.
        
        Returns:
            Dict with 'max_reasoning_tokens' and 'max_response_tokens'
        """
        model_lower = model_name.lower()
        
        # OpenAI o-series models: Very high output limits (100K tokens)
        if any(model in model_lower for model in ["o1", "o3", "o4"]):
            return {
                "max_reasoning_tokens": 32000,  # High reasoning capacity
                "max_response_tokens": 8000,    # Adequate response space
            }
        
        # OpenAI GPT-4o models: 16,384 max output tokens
        elif "gpt-4o" in model_lower:
            return {
                "max_reasoning_tokens": 12000,  # Leave room for response
                "max_response_tokens": 4000,    # Adequate response space
            }
        
        # OpenAI GPT-4 Turbo models: 4,096 max output tokens
        elif "gpt-4" in model_lower and ("turbo" in model_lower or "preview" in model_lower):
            return {
                "max_reasoning_tokens": 3000,   # Conservative limit
                "max_response_tokens": 1000,    # Adequate response space
            }
        
        # OpenAI GPT-4 base models: 8,192 max output tokens
        elif "gpt-4" in model_lower:
            return {
                "max_reasoning_tokens": 6000,   # Conservative limit
                "max_response_tokens": 2000,    # Adequate response space
            }
        
        # OpenAI GPT-3.5 models: 4,096 max output tokens
        elif "gpt-3.5" in model_lower:
            return {
                "max_reasoning_tokens": 3000,   # Conservative limit
                "max_response_tokens": 1000,    # Adequate response space
            }
        
        # Anthropic Claude models: ~8,000 estimated max output tokens
        elif "claude" in model_lower:
            return {
                "max_reasoning_tokens": 6000,   # Conservative reasoning limit
                "max_response_tokens": 2000,    # Adequate response space
            }
        
        # Gemini models: ~8,000-16,000 estimated max output tokens
        elif "gemini" in model_lower:
            # Gemini Thinking models may have higher limits
            if "thinking" in model_lower:
                return {
                    "max_reasoning_tokens": 12000,  # Higher reasoning capacity
                    "max_response_tokens": 4000,    # Adequate response space
                }
            else:
                return {
                    "max_reasoning_tokens": 6000,   # Conservative reasoning limit
                    "max_response_tokens": 2000,    # Adequate response space
                }
        
        # Qwen models: Moderate reasoning capabilities (various output limits)
        elif "qwen" in model_lower or "qwq" in model_lower:
            # QwQ-32B and larger Qwen models have higher output limits
            if "qwq" in model_lower or any(size in model_lower for size in ["32b", "235b", "30b"]):
                return {
                    "max_reasoning_tokens": 12000,  # Higher reasoning allocation for larger models
                    "max_response_tokens": 4000     # Adequate response space
                }
            else:
                # Smaller Qwen models (4B, 8B, 14B)
                return {
                    "max_reasoning_tokens": 6000,   # Moderate reasoning allocation
                    "max_response_tokens": 2000     # Adequate response space
                }
        
        # DeepSeek and other reasoning models: ~8,000 estimated max output tokens
        elif any(model in model_lower for model in ["deepseek", "minimax"]):
            return {
                "max_reasoning_tokens": 6000,   # Conservative reasoning limit
                "max_response_tokens": 2000,    # Adequate response space
            }
        
        # Default for unknown models: Conservative limits
        return {
            "max_reasoning_tokens": 3000,   # Safe default
            "max_response_tokens": 1000,    # Safe default
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

    def apply_prompt_based_reasoning(self, prompt: str, model_name: str, reasoning_config: ReasoningConfig = None) -> str:
        """
        Apply prompt-based reasoning activation for models that use slash commands.
        
        For Qwen models, this adds /think or /no_think commands to activate/deactivate
        reasoning mode through the prompt rather than API headers.
        
        Args:
            prompt: The original user prompt
            model_name: Name of the model being used
            reasoning_config: Reasoning configuration (if any)
            
        Returns:
            Modified prompt with reasoning activation commands
        """
        support = self.get_model_reasoning_support(model_name)
        
        # Only apply for models that use prompt-based activation
        if not support["uses_prompt_activation"]:
            return prompt
        
        # Determine if reasoning should be enabled
        reasoning_enabled = True  # Default for Qwen models
        
        if reasoning_config:
            reasoning_enabled = reasoning_config.enabled and not reasoning_config.exclude
        
        # Add the appropriate slash command for Qwen models
        if "qwen" in model_name.lower() or "qwq" in model_name.lower():
            if reasoning_enabled:
                # Add /think command to activate reasoning mode
                if "/think" not in prompt and "/no_think" not in prompt:
                    return f"{prompt} /think"
            else:
                # Add /no_think command to disable reasoning mode
                if "/think" not in prompt and "/no_think" not in prompt:
                    return f"{prompt} /no_think"
        
        return prompt

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
