import logging
import time
from typing import Tuple, Optional

from src.llm_client import LLMClient
from src.llm_config import LLMConfig
from src.hybrid.dataclasses import HybridConfig, HybridResult, LLMCallStats
from src.hybrid.reasoning_extractor import ReasoningExtractor, ReasoningFormat

logger = logging.getLogger(__name__)

class HybridProcessor:
    def __init__(self, llm_client: LLMClient, config: HybridConfig):
        self.llm_client = llm_client
        self.config = config
        self.reasoning_extractor = ReasoningExtractor()

    def _extract_reasoning(self, full_output: str) -> Tuple[str, ReasoningFormat]:
        """
        Extract reasoning from model output using various formats.
        
        Args:
            full_output: The complete model output text
            
        Returns:
            Tuple of (extracted_reasoning, detected_format)
        """
        logger.debug(f"Raw LLM output for reasoning extraction:\n{repr(full_output)}")
        logger.debug(f"Raw LLM output length: {len(full_output)} characters")
        
        # Try to extract reasoning using the new flexible extractor
        reasoning, remaining_text, detected_format = self.reasoning_extractor.extract_reasoning(
            text=full_output,
            format_hint=self._get_format_hint(),
            custom_token=self.config.reasoning_complete_token
        )
        
        if reasoning:
            logger.info(f"Successfully extracted reasoning using format: {detected_format.value}")
            logger.debug(f"Extracted reasoning length: {len(reasoning)} characters")
            logger.debug(f"Extracted reasoning:\n{repr(reasoning)}")
            return reasoning, detected_format
        
        # Fallback to original method if no reasoning found
        logger.warning("No reasoning extracted using flexible extractor, falling back to original method")
        logger.debug(f"Trying to split on custom token: {self.config.reasoning_complete_token}")
        parts = full_output.split(self.config.reasoning_complete_token, 1)
        logger.debug(f"Split result: {len(parts)} parts")
        if len(parts) > 1:
            logger.debug(f"Parts[0] (reasoning): {repr(parts[0])}")
            logger.debug(f"Parts[1] (remainder): {repr(parts[1])}")
        
        fallback_reasoning = parts[0].strip()
        logger.debug(f"Fallback reasoning: {repr(fallback_reasoning)}")
        
        return fallback_reasoning, ReasoningFormat.CUSTOM_TOKEN
    
    def _get_format_hint(self) -> Optional[ReasoningFormat]:
        """
        Get format hint based on the reasoning model name.
        
        Returns:
            Suggested format based on model name, or None if unknown
        """
        model_name = self.config.reasoning_model_name.lower()
        
        if 'deepseek' in model_name and 'r1' in model_name:
            return ReasoningFormat.DEEPSEEK_R1
        elif 'o1' in model_name or 'openai' in model_name:
            return ReasoningFormat.OPENAI_O1
        elif 'gemini' in model_name:
            return ReasoningFormat.GEMINI_THINKING
        elif 'claude' in model_name:
            return ReasoningFormat.CLAUDE_THINKING
        elif 'qwq' in model_name:
            return ReasoningFormat.QWQ_THINKING
        else:
            return None

    def _should_use_stop_token(self) -> bool:
        """
        Determine if we should use stop tokens based on the model.
        
        DeepSeek-R1 models output the completion token first, then reasoning,
        so using stop tokens causes empty responses.
        """
        model_name = self.config.reasoning_model_name.lower()
        
        # DeepSeek-R1 models should not use stop tokens
        if 'deepseek' in model_name and 'r1' in model_name:
            return False
        
        return True

    def run(self, problem_description: str) -> HybridResult:
        result = HybridResult()

        # Stage 1: Call Reasoning Model
        reasoning_prompt = self.config.reasoning_prompt_template.format(
            problem_description=problem_description,
            reasoning_complete_token=self.config.reasoning_complete_token
        )

        logger.info(f"Calling reasoning model ({self.config.reasoning_model_name}) for Hybrid process.")
        logger.debug(f"Reasoning prompt:\n{reasoning_prompt}")

        try:
            # Prepare reasoning configuration with model-specific filtering
            reasoning_config = None
            effective_reasoning_config = self.config.get_effective_reasoning_config(self.config.reasoning_model_name)
            
            if effective_reasoning_config:
                reasoning_config = effective_reasoning_config.to_openrouter_dict()
                logger.debug(f"Using model-specific reasoning config for {self.config.reasoning_model_name}: {reasoning_config}")
            else:
                logger.debug(f"Model {self.config.reasoning_model_name} does not support reasoning configuration")
            
            # Get model-specific headers
            model_headers = self.config.get_model_headers(self.config.reasoning_model_name)
            
            # Get model-specific token limits
            token_limits = self.config.get_effective_token_limits()
            
            # Create LLMConfig for reasoning model
            reasoning_llm_config = LLMConfig(
                temperature=self.config.reasoning_model_temperature,
                max_tokens=token_limits["max_reasoning_tokens"],
                stop=None  # Don't use stop tokens - handle in reasoning extraction
            )
            
            logger.debug(f"Using streaming: {self.config.use_streaming}")
            logger.debug(f"Model headers: {model_headers}")
            
            # Call the reasoning model with new enhanced method
            raw_reasoning_output, extracted_reasoning_from_api, reasoning_stats = self.llm_client.call_with_reasoning(
                prompt=reasoning_prompt,
                models=[self.config.reasoning_model_name],
                config=reasoning_llm_config,
                reasoning_config=reasoning_config,
                use_streaming=self.config.use_streaming,
                model_headers=model_headers
            )
            
            result.reasoning_call_stats = reasoning_stats
            logger.info(f"Reasoning model ({reasoning_stats.model_name}) call successful. Duration: {reasoning_stats.call_duration_seconds:.2f}s")
            logger.debug(f"Raw reasoning output:\n{raw_reasoning_output}")
            logger.debug(f"API extracted reasoning:\n{extracted_reasoning_from_api}")

            # Determine which reasoning to use
            if extracted_reasoning_from_api:
                # Use reasoning extracted by OpenRouter API
                result.extracted_reasoning = extracted_reasoning_from_api
                result.detected_reasoning_format = "openrouter_api"
                logger.info("Using reasoning extracted by OpenRouter API")
            elif raw_reasoning_output:
                # Fallback to manual extraction
                logger.info("No API reasoning found, attempting manual extraction")
                result.extracted_reasoning, detected_format = self._extract_reasoning(raw_reasoning_output)
                result.detected_reasoning_format = detected_format.value
            else:
                logger.warning("No reasoning output received from model")
                result.extracted_reasoning = ""
                result.detected_reasoning_format = "none"

        except Exception as e:
            logger.error(f"Error during reasoning model call: {e}")
            result.succeeded = False
            result.error_message = f"Reasoning model call failed: {str(e)}"
            # Populate with placeholder stats if the call failed before stats were created
            if not result.reasoning_call_stats:
                 result.reasoning_call_stats = LLMCallStats(model_name=self.config.reasoning_model_name, completion_tokens=0, prompt_tokens=0, call_duration_seconds=0)
            return result
        
        if not result.extracted_reasoning:
            logger.warning("No reasoning output was extracted. This might be due to the model not following instructions or an empty output before the completion token.")
            # Potentially treat as failure or continue, depending on desired strictness.
            # For now, we'll continue, but the response stage might suffer.
        else:
            logger.info(f"Extracted reasoning using format: {result.detected_reasoning_format}")

        logger.debug(f"Extracted reasoning:\n{result.extracted_reasoning}")

        # Stage 2: Call Response Model
        response_prompt = self.config.response_prompt_template.format(
            problem_description=problem_description,
            extracted_reasoning=result.extracted_reasoning
        )

        logger.info(f"Calling response model ({self.config.response_model_name}) for Hybrid process.")
        logger.debug(f"Response prompt:\n{response_prompt}")

        try:
            # Get model-specific headers for response model
            response_model_headers = self.config.get_model_headers(self.config.response_model_name)
            
            # Create LLMConfig for response model
            response_config = LLMConfig(
                temperature=self.config.response_model_temperature,
                max_tokens=token_limits["max_response_tokens"]
            )
            
            # Use legacy call method for response model (no reasoning needed)
            final_answer_output, response_stats = self.llm_client.call(
                prompt=response_prompt,
                models=[self.config.response_model_name],
                config=response_config
            )
            result.response_call_stats = response_stats
            logger.info(f"Response model ({response_stats.model_name}) call successful. Duration: {response_stats.call_duration_seconds:.2f}s")
            logger.debug(f"Raw response output:\n{final_answer_output}")

        except Exception as e:
            logger.error(f"Error during response model call: {e}")
            result.succeeded = False
            result.error_message = f"Response model call failed: {str(e)}"
            if not result.response_call_stats: # Populate with placeholder stats
                 result.response_call_stats = LLMCallStats(model_name=self.config.response_model_name, completion_tokens=0, prompt_tokens=0, call_duration_seconds=0)
            return result

        result.final_answer = final_answer_output.strip()
        result.succeeded = True

        return result
