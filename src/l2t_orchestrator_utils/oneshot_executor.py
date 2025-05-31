import logging
import time
from typing import List, Tuple
from src.llm_client import LLMClient
from src.llm_config import LLMConfig # Added
from src.aot.dataclasses import LLMCallStats

logger = logging.getLogger(__name__)

class OneShotExecutor:
    def __init__(self, llm_client: LLMClient, direct_oneshot_model_names: List[str], llm_config: LLMConfig): # Modified
        self.llm_client = llm_client
        self.direct_oneshot_model_names = direct_oneshot_model_names
        self.llm_config = llm_config # Added

    def run_direct_oneshot(self, problem_text: str, is_fallback: bool = False) -> Tuple[str, LLMCallStats]:
        mode = "FALLBACK ONESHOT" if is_fallback else "ONESHOT"
        logger.info(f"--- Proceeding with {mode} Answer ---")
        logger.info(f"Using models: {', '.join(self.direct_oneshot_model_names)}, Config: {self.llm_config}") # Modified

        response_content, stats = self.llm_client.call(
            prompt=problem_text, models=self.direct_oneshot_model_names, config=self.llm_config # Modified
        )
        logger.debug(f"Direct {mode} response from {stats.model_name}:\n{response_content}")
        logger.info(f"LLM call ({stats.model_name}) for {mode}: Duration: {stats.call_duration_seconds:.2f}s, Tokens (C:{stats.completion_tokens}, P:{stats.prompt_tokens})")
        return response_content, stats
