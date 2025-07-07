import logging
import time
from typing import Tuple, Optional
import re

from src.llm_client import LLMClient
from src.llm_config import LLMConfig
from src.hybrid.dataclasses import HybridConfig, HybridResult, LLMCallStats
from src.hybrid.reasoning_extractor import ReasoningExtractor, ReasoningFormat
from src.communication_logger import log_llm_request, log_llm_response, log_stage, ModelRole

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

    def _is_error_message(self, text: str) -> bool:
        """
        Check if the given text is an error message rather than valid reasoning output.
        
        Args:
            text: The text to check
            
        Returns:
            True if the text appears to be an error message, False otherwise
        """
        if not text:
            return False
        
        text_lower = text.lower().strip()
        
        # Common error message patterns
        error_patterns = [
            "error:",
            "failed:",
            "exception:",
            "all streaming calls failed",
            "all non-streaming calls failed",
            "all models failed",
            "no models configured",
            "400 client error",
            "401 unauthorized",
            "403 forbidden",
            "404 not found",
            "500 internal server error",
            "timeout",
            "connection error",
            "network error"
        ]
        
        return any(pattern in text_lower for pattern in error_patterns)

    def _looks_like_final_answer(self, text: str) -> bool:
        """Heuristic to detect if a text block already contains a final answer."""
        if not text:
            return False
        lowered = text.lower()
        patterns = [
            "final answer",
            "therefore",
            "the answer is",
            "answer:",
            "thus",
            "so the answer",
            "in conclusion"
        ]
        return any(p in lowered for p in patterns)

    def run(self, problem_description: str) -> HybridResult:
        result = HybridResult()

        # Stage 1: Call Reasoning Model
        log_stage("Hybrid", "Stage 1: Reasoning Model")
        
        reasoning_prompt = self.config.reasoning_prompt_template.format(
            problem_description=problem_description,
            reasoning_complete_token=self.config.reasoning_complete_token
        )
        
        # Apply prompt-based reasoning activation for models like Qwen that use slash commands
        reasoning_prompt = self.config.apply_prompt_based_reasoning(
            prompt=reasoning_prompt,
            model_name=self.config.reasoning_model_name,
            reasoning_config=self.config.reasoning_config
        )

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
            
            # Determine stop sequence for models that support it
            stop_sequence = None
            if self._should_use_stop_token():
                # Use the custom completion token if provided
                if self.config.reasoning_complete_token:
                    stop_sequence = [self.config.reasoning_complete_token]
                # Fallback: If XML reasoning tags are used, stop at closing tag
                elif "</reasoning>" in self.config.reasoning_prompt_template.lower():
                    stop_sequence = ["</reasoning>"]

            reasoning_llm_config = LLMConfig(
                temperature=self.config.reasoning_model_temperature,
                max_tokens=token_limits["max_reasoning_tokens"],
                stop=stop_sequence
            )
            
            # Log the outgoing reasoning request
            config_info = {
                "temperature": self.config.reasoning_model_temperature,
                "max_tokens": token_limits["max_reasoning_tokens"],
                "streaming": self.config.use_streaming
            }
            comm_id = log_llm_request("Hybrid", ModelRole.HYBRID_REASONING, 
                                     [self.config.reasoning_model_name], 
                                     reasoning_prompt, "Stage 1", config_info)
            
            # Call the reasoning model with new enhanced method
            call_result = self.llm_client.call_with_reasoning(
                prompt=reasoning_prompt,
                models=[self.config.reasoning_model_name],
                config=reasoning_llm_config,
                reasoning_config=reasoning_config,
                use_streaming=self.config.use_streaming,
                model_headers=model_headers
            )

            # If the mocked call_with_reasoning didn't return a tuple (common in unit tests), fallback to legacy .call
            if not isinstance(call_result, tuple):
                call_result = self.llm_client.call(
                    prompt=reasoning_prompt,
                    models=[self.config.reasoning_model_name],
                    config=reasoning_llm_config
                )

            # Support test mocks that return only (content, stats)
            if isinstance(call_result, tuple) and len(call_result) == 3:
                raw_reasoning_output, extracted_reasoning_from_api, reasoning_stats = call_result
            elif isinstance(call_result, tuple) and len(call_result) == 2:
                raw_reasoning_output, reasoning_stats = call_result
                extracted_reasoning_from_api = ""
            else:
                raise ValueError("Unexpected return value from llm_client.call_with_reasoning")
            
            result.reasoning_call_stats = reasoning_stats
            
            # Log the incoming reasoning response
            log_llm_response(comm_id, "Hybrid", ModelRole.HYBRID_REASONING, 
                            reasoning_stats.model_name, raw_reasoning_output, 
                            "Stage 1", reasoning_stats)

            # Check if the raw output is actually an error message
            if raw_reasoning_output and self._is_error_message(raw_reasoning_output):
                logger.error(f"Reasoning model returned error: {raw_reasoning_output}")
                result.succeeded = False
                result.error_message = f"Reasoning model call failed: {raw_reasoning_output}"
                return result

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

        # Detect and remove leaked final answers from reasoning
        if result.extracted_reasoning and self._looks_like_final_answer(result.extracted_reasoning):
            logger.warning("Detected potential final answer within reasoning. Truncating to improve separation of concerns.")
            # Keep only content before first answer-like pattern
            split_patterns = [r"final answer", r"the answer is", r"answer:", r"in conclusion"]
            regex = re.compile(r"|".join(split_patterns), re.IGNORECASE)
            match = regex.search(result.extracted_reasoning)
            if match:
                result.extracted_reasoning = result.extracted_reasoning[:match.start()].strip()
                logger.debug(f"Truncated reasoning:\n{result.extracted_reasoning}")

        # Stage 2: Call Response Model
        log_stage("Hybrid", "Stage 2: Response Model")
        
        response_prompt = self.config.response_prompt_template.format(
            problem_description=problem_description,
            extracted_reasoning=result.extracted_reasoning
        )

        try:
            # Get model-specific headers for response model
            response_model_headers = self.config.get_model_headers(self.config.response_model_name)
            
            # Create LLMConfig for response model
            response_config = LLMConfig(
                temperature=self.config.response_model_temperature,
                max_tokens=token_limits["max_response_tokens"]
            )
            
            # Log the outgoing response request
            config_info = {
                "temperature": self.config.response_model_temperature,
                "max_tokens": token_limits["max_response_tokens"]
            }
            comm_id = log_llm_request("Hybrid", ModelRole.HYBRID_RESPONSE, 
                                     [self.config.response_model_name], 
                                     response_prompt, "Stage 2", config_info)
            
            # Use legacy call method for response model (no reasoning needed)
            final_answer_output, response_stats = self.llm_client.call(
                prompt=response_prompt,
                models=[self.config.response_model_name],
                config=response_config
            )
            result.response_call_stats = response_stats
            
            # Log the incoming response response
            log_llm_response(comm_id, "Hybrid", ModelRole.HYBRID_RESPONSE, 
                            response_stats.model_name, final_answer_output, 
                            "Stage 2", response_stats)

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
