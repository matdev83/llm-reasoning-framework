import logging
from typing import List, Tuple

from src.aot_dataclasses import LLMCallStats
from src.aot_enums import AssessmentDecision
from src.llm_client import LLMClient
from src.prompt_generator import PromptGenerator
from src.heuristic_detector import HeuristicDetector # Import the new class

logger = logging.getLogger(__name__)

class ComplexityAssessor:
    def __init__(self, llm_client: LLMClient, small_model_names: List[str], temperature: float, use_heuristic_shortcut: bool = True):
        self.llm_client = llm_client
        self.small_model_names = small_model_names
        self.temperature = temperature
        self.use_heuristic_shortcut = use_heuristic_shortcut

    def assess(self, problem_text: str) -> Tuple[AssessmentDecision, LLMCallStats]:
        # Check heuristic shortcut first
        if self.use_heuristic_shortcut:
            if HeuristicDetector.should_trigger_complex_process_heuristically(problem_text):
                logging.info("Heuristic shortcut triggered: Problem classified as AOT.")
                # Return a dummy LLMCallStats as no LLM call was made
                dummy_stats = LLMCallStats(
                    model_name="heuristic_shortcut",
                    prompt_tokens=0,
                    completion_tokens=0,
                    call_duration_seconds=0.0
                )
                return AssessmentDecision.AOT, dummy_stats

        logging.info(f"--- Initial Complexity Assessment using models: {', '.join(self.small_model_names)} ---")
        assessment_prompt = PromptGenerator.construct_assessment_prompt(problem_text)
        response_content, stats = self.llm_client.call(
            prompt=assessment_prompt, models=self.small_model_names, temperature=self.temperature
        )
        logging.debug(f"Assessment model ({stats.model_name}) raw response: '{response_content.strip()}'")
        logging.info(f"Assessment call: {stats.model_name}, Duration: {stats.call_duration_seconds:.2f}s, Tokens (C:{stats.completion_tokens}, P:{stats.prompt_tokens})")
        
        decision = AssessmentDecision.AOT # Default decision
        if response_content.startswith("Error:"):
            logging.warning(f"Assessment model call failed. Defaulting to AOT. Error: {response_content}")
            decision = AssessmentDecision.ERROR # Specific error state
        else:
            cleaned_response = response_content.strip().upper()
            if cleaned_response == AssessmentDecision.ONESHOT.value:
                decision = AssessmentDecision.ONESHOT
                logging.info("Assessment: Problem classified as ONESHOT.")
            elif cleaned_response == AssessmentDecision.AOT.value:
                decision = AssessmentDecision.AOT
                logging.info("Assessment: Problem classified as AOT.")
            else:
                logging.warning(f"Assessment model output ('{cleaned_response}') was not '{AssessmentDecision.ONESHOT.value}' or '{AssessmentDecision.AOT.value}'. Defaulting to AOT.")
                # decision remains AOT (the default)
        return decision, stats
