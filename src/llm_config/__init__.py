from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class LLMConfig:
    """
    Configuration for Large Language Model calls.

    Attributes:
        temperature: Controls randomness. Lower is more deterministic. (0.0-2.0, default: 1.0)
        top_p: Nucleus sampling. Considers tokens with top_p probability mass. (0.0-1.0, default: 1.0)
        top_k: Considers top_k most likely tokens. (0 or above, default: 0 - disabled)
        frequency_penalty: Penalizes new tokens based on their existing frequency. (-2.0-2.0, default: 0.0)
        presence_penalty: Penalizes new tokens based on whether they appear in the text so far. (-2.0-2.0, default: 0.0)
        repetition_penalty: Penalizes new tokens based on their repetition in the window. (0.0-2.0, default: 1.0)
        seed: Seed for deterministic sampling (if supported by the model).
        max_tokens: Maximum number of tokens to generate.
        stop: List of sequences where the API will stop generating further tokens.
        logit_bias: Modify the likelihood of specified tokens appearing.
        response_format: Specify the output format (e.g., JSON). Example: {"type": "json_object"}

        # OpenRouter specific, but good to have as a standard field if we use "reasoning" features
        reasoning_effort: Controls the amount of reasoning effort. (e.g., "low", "medium", "high", "auto")

        provider_specific_params: Dictionary for any other provider-specific parameters.
    """
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 0 # OpenRouter defaults to 0 (disabled) if not present
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0 # OpenRouter defaults to 1.0
    seed: Optional[int] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    logit_bias: Optional[Dict[str, float]] = None # Maps token IDs (as strings) to bias
    response_format: Optional[Dict[str, str]] = None # e.g. {"type": "json_object"}

    reasoning_effort: Optional[str] = "high" # Defaulting based on current LLMClient usage

    # For any parameters not explicitly listed or for provider-specific ones
    provider_specific_params: Dict[str, Any] = field(default_factory=dict)

    def to_payload_dict(self) -> Dict[str, Any]:
        """Converts the config to a dictionary suitable for the API payload, excluding None values."""
        payload = {}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None and self.top_k > 0: # Only include if explicitly set to a value > 0
            payload["top_k"] = self.top_k
        if self.frequency_penalty is not None:
            payload["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            payload["presence_penalty"] = self.presence_penalty
        if self.repetition_penalty is not None: # Default is 1.0, include if different or explicitly set
            payload["repetition_penalty"] = self.repetition_penalty
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.stop is not None:
            payload["stop"] = self.stop
        if self.logit_bias is not None:
            payload["logit_bias"] = self.logit_bias
        if self.response_format is not None:
            payload["response_format"] = self.response_format

        if self.reasoning_effort is not None:
            # OpenRouter's "reasoning" parameter is a nested object
            payload["reasoning"] = {"effort": self.reasoning_effort}

        if self.provider_specific_params:
            payload.update(self.provider_specific_params)

        return payload
