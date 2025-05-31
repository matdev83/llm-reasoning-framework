import logging
import time
from typing import Tuple

from src.llm_client import LLMClient
from src.hybrid.dataclasses import HybridConfig, HybridResult, LLMCallStats

logger = logging.getLogger(__name__)

class HybridProcessor:
    def __init__(self, llm_client: LLMClient, config: HybridConfig):
        self.llm_client = llm_client
        self.config = config

    def _extract_reasoning(self, full_output: str) -> str:
        # Extracts text before the reasoning_complete_token
        parts = full_output.split(self.config.reasoning_complete_token, 1)
        return parts[0].strip()

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
            reasoning_model_output, reasoning_stats = self.llm_client.call(
                prompt=reasoning_prompt,
                models=[self.config.reasoning_model_name], # llm_client expects a list of models
                temperature=self.config.reasoning_model_temperature,
                max_tokens=self.config.max_reasoning_tokens,
                # TODO: Add stop sequence if llm_client supports it, to stop at reasoning_complete_token
            )
            result.reasoning_call_stats = reasoning_stats
            logger.info(f"Reasoning model ({reasoning_stats.model_name}) call successful. Duration: {reasoning_stats.call_duration_seconds:.2f}s")
            logger.debug(f"Raw reasoning output:\n{reasoning_model_output}")

        except Exception as e:
            logger.error(f"Error during reasoning model call: {e}")
            result.succeeded = False
            result.error_message = f"Reasoning model call failed: {str(e)}"
            # Populate with placeholder stats if the call failed before stats were created
            if not result.reasoning_call_stats:
                 result.reasoning_call_stats = LLMCallStats(model_name=self.config.reasoning_model_name, completion_tokens=0, prompt_tokens=0, call_duration_seconds=0)
            return result

        result.extracted_reasoning = self._extract_reasoning(reasoning_model_output)
        if not result.extracted_reasoning:
            logger.warning("No reasoning output was extracted. This might be due to the model not following instructions or an empty output before the completion token.")
            # Potentially treat as failure or continue, depending on desired strictness.
            # For now, we'll continue, but the response stage might suffer.

        logger.debug(f"Extracted reasoning:\n{result.extracted_reasoning}")

        # Stage 2: Call Response Model
        response_prompt = self.config.response_prompt_template.format(
            problem_description=problem_description,
            extracted_reasoning=result.extracted_reasoning
        )

        logger.info(f"Calling response model ({self.config.response_model_name}) for Hybrid process.")
        logger.debug(f"Response prompt:\n{response_prompt}")

        try:
            final_answer_output, response_stats = self.llm_client.call(
                prompt=response_prompt,
                models=[self.config.response_model_name], # llm_client expects a list of models
                temperature=self.config.response_model_temperature,
                max_tokens=self.config.max_response_tokens
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
