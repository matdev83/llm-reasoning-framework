import time
import logging
from typing import Tuple

from src.llm_client import LLMClient
from src.llm_config import LLMConfig
from src.far.dataclasses import FaRConfig, FaRResult, LLMCallStats
# Assuming prompt templates will be managed by a PromptGenerator or directly here for simplicity first
# from src.prompt_generator import PromptGenerator # Or a specific FaRPromptGenerator

# For communication logging
from src.communication_logger import log_llm_request, log_llm_response, log_stage, ModelRole


logger = logging.getLogger(__name__)

class FaRProcessor:
    def __init__(self, llm_client: LLMClient, config: FaRConfig):
        self.llm_client = llm_client
        self.config = config
        # Prompts can be loaded from files or defined here.
        # For now, we'll assume they are constructed directly or passed in if complex.
        # self.fact_prompt_template = "Extract key facts relevant to the following problem: {problem_description}"
        # self.reflection_prompt_template = "Based on the problem: {problem_description}\nAnd the following facts: {elicited_facts}\nProvide a comprehensive answer."
        # Actual prompts will be loaded from conf/prompts/ via a PromptGenerator or similar mechanism in the orchestrator/main CLI flow.
        # For the processor itself, it expects the fully formed prompt text, or constructs it from templates if those are part of its responsibility.
        # Let's assume prompts are handled by a yet-to-be-created FaRPromptGenerator or passed in.
        # For now, we'll use hardcoded basic templates for structure.

    def _load_prompt_template(self, template_name: str) -> str:
        # This is a placeholder for actual prompt loading logic
        # In a real scenario, this would read from conf/prompts/far_*.txt
        # This is a placeholder for actual prompt loading logic
        # In a real scenario, this would read from conf/prompts/far_*.txt
        try:
            # Construct path relative to a known root or pass full path
            # For now, assume `conf` is accessible from where the script runs
            # A more robust solution would use `pathlib` and relative paths from project root.
            with open(f"conf/prompts/{template_name}", "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt template file not found: conf/prompts/{template_name}")
            # Potentially raise an error or return a default/empty string
            # Raising an error might be better to catch configuration issues early.
            raise FileNotFoundError(f"Prompt template file not found: conf/prompts/{template_name}")
        except Exception as e:
            logger.error(f"Error loading prompt template {template_name}: {e}")
            raise  # Re-raise the exception after logging


    def run(self, problem_description: str) -> FaRResult:
        result = FaRResult(problem_description=problem_description)
        process_start_time = time.monotonic()

        logger.info(f"Starting FaR process for problem: '{problem_description[:100].strip()}...'")
        logger.info(f"Fact Models: {', '.join(self.config.fact_model_names)}, Main Models: {', '.join(self.config.main_model_names)}")

        # === Step 1: Fact Elicitation ===
        log_stage("FaR", "Phase 1: Fact Elicitation")
        try:
            fact_prompt_template_content = self._load_prompt_template("far_fact_elicitation.txt")
            if not fact_prompt_template_content:
                raise ValueError("Fact elicitation prompt template is empty or not found.")

            fact_prompt = fact_prompt_template_content.format(problem_description=problem_description)

            fact_llm_config = LLMConfig(
                temperature=self.config.fact_model_temperature,
                max_tokens=self.config.max_fact_tokens
            )

            logger.info(f"Calling fact model(s) {self.config.fact_model_names} with temp {self.config.fact_model_temperature}, max_tokens {self.config.max_fact_tokens}")

            # Log outgoing request for fact elicitation
            fact_config_info = {"temperature": fact_llm_config.temperature, "max_tokens": fact_llm_config.max_tokens}
            fact_comm_id = log_llm_request("FaR", ModelRole.FAR_FACT_EXTRACTION, self.config.fact_model_names,
                                           fact_prompt, "Fact Elicitation", fact_config_info)

            elicited_facts_raw, fact_stats = self.llm_client.call(
                prompt=fact_prompt,
                models=self.config.fact_model_names,
                config=fact_llm_config
            )
            result.fact_call_stats = fact_stats
            
            # Track reasoning tokens
            if fact_stats:
                result.reasoning_completion_tokens += fact_stats.completion_tokens

            # Log incoming response for fact elicitation
            log_llm_response(fact_comm_id, "FaR", ModelRole.FAR_FACT_EXTRACTION, fact_stats.model_name,
                             elicited_facts_raw, "Fact Elicitation", fact_stats)

            if elicited_facts_raw.startswith("Error:"):
                logger.error(f"Fact elicitation failed: {elicited_facts_raw}")
                result.succeeded = False
                result.error_message = f"Fact elicitation LLM call failed: {elicited_facts_raw}"
                result.total_process_wall_clock_time_seconds = time.monotonic() - process_start_time
                return result

            result.elicited_facts = elicited_facts_raw.strip()
            logger.info(f"Elicited facts: {result.elicited_facts[:200].strip()}...")

        except Exception as e:
            logger.error(f"Exception during fact elicitation: {e}", exc_info=True)
            result.succeeded = False
            result.error_message = f"Exception in fact elicitation phase: {str(e)}"
            result.total_process_wall_clock_time_seconds = time.monotonic() - process_start_time
            # Ensure stats object exists even if call failed partway
            if not result.fact_call_stats:
                result.fact_call_stats = LLMCallStats(model_name=",".join(self.config.fact_model_names), completion_tokens=0, prompt_tokens=0, call_duration_seconds=0)
            return result

        # === Resource Constraint Checks Before Phase 2 ===
        elapsed_time = time.monotonic() - process_start_time
        
        # Check time limit
        if self.config.max_time_seconds > 0 and elapsed_time >= self.config.max_time_seconds:
            logger.info(f"Time limit ({self.config.max_time_seconds}s) reached after fact elicitation. Skipping reflection phase.")
            result.succeeded = False
            result.error_message = f"Time limit ({self.config.max_time_seconds}s) reached after fact elicitation."
            result.total_process_wall_clock_time_seconds = time.monotonic() - process_start_time
            return result
        
        # Check token budget limit
        if (self.config.max_reasoning_tokens and 
            result.reasoning_completion_tokens >= self.config.max_reasoning_tokens):
            logger.info(f"Token limit ({self.config.max_reasoning_tokens}) reached after fact elicitation. Skipping reflection phase.")
            result.succeeded = False
            result.error_message = f"Token limit ({self.config.max_reasoning_tokens}) reached after fact elicitation."
            result.total_process_wall_clock_time_seconds = time.monotonic() - process_start_time
            return result

        # === Step 2: Reflection and Answer Generation ===
        log_stage("FaR", "Phase 2: Reflection and Answer Generation")
        if not result.elicited_facts: # Should not happen if first step succeeded without error, but good check.
            logger.warning("No facts were elicited in the first step. Proceeding to reflection with empty facts.")
            result.elicited_facts = "No specific facts were elicited."


        try:
            reflection_prompt_template_content = self._load_prompt_template("far_reflection_answer.txt")
            if not reflection_prompt_template_content:
                raise ValueError("Reflection and answer prompt template is empty or not found.")

            reflection_prompt = reflection_prompt_template_content.format(
                problem_description=problem_description,
                elicited_facts=result.elicited_facts
            )

            main_llm_config = LLMConfig(
                temperature=self.config.main_model_temperature,
                max_tokens=self.config.max_main_tokens
            )

            logger.info(f"Calling main model(s) {self.config.main_model_names} with temp {self.config.main_model_temperature}, max_tokens {self.config.max_main_tokens}")

            # Log outgoing request for reflection/answer
            main_config_info = {"temperature": main_llm_config.temperature, "max_tokens": main_llm_config.max_tokens}
            main_comm_id = log_llm_request("FaR", ModelRole.FAR_REFLECTION_ANSWER, self.config.main_model_names,
                                           reflection_prompt, "Reflection & Answer", main_config_info)

            final_answer_raw, main_stats = self.llm_client.call(
                prompt=reflection_prompt,
                models=self.config.main_model_names,
                config=main_llm_config
            )
            result.main_call_stats = main_stats
            
            # Track reasoning tokens
            if main_stats:
                result.reasoning_completion_tokens += main_stats.completion_tokens

            # Log incoming response for reflection/answer
            log_llm_response(main_comm_id, "FaR", ModelRole.FAR_REFLECTION_ANSWER, main_stats.model_name,
                             final_answer_raw, "Reflection & Answer", main_stats)


            if final_answer_raw.startswith("Error:"):
                logger.error(f"Reflection and answer generation failed: {final_answer_raw}")
                result.succeeded = False
                result.error_message = f"Reflection/Answer LLM call failed: {final_answer_raw}"
                result.total_process_wall_clock_time_seconds = time.monotonic() - process_start_time
                return result

            result.final_answer = final_answer_raw.strip()
            logger.info(f"Final answer generated: {result.final_answer[:200].strip()}...")
            result.succeeded = True

        except Exception as e:
            logger.error(f"Exception during reflection and answer generation: {e}", exc_info=True)
            result.succeeded = False
            result.error_message = f"Exception in reflection/answer phase: {str(e)}"
            # Ensure stats object exists
            if not result.main_call_stats:
                result.main_call_stats = LLMCallStats(model_name=",".join(self.config.main_model_names), completion_tokens=0, prompt_tokens=0, call_duration_seconds=0)
            # result.final_answer remains None or previous value

        result.total_process_wall_clock_time_seconds = time.monotonic() - process_start_time
        return result

    # Placeholder for a more robust prompt loading mechanism if needed within processor
    # Usually, prompt generation is handled by a dedicated PromptGenerator class or by the orchestrator.
    # For FaR, the prompts are relatively static, so direct formatting might be acceptable here or
    # in a small helper method.
    def _get_prompt_content(self, prompt_key: str) -> str:
        # This method would ideally load from `conf/prompts/far_{prompt_key}.txt`
        # This is a simplified version.
        # Example:
        # prompt_path = f"conf/prompts/far_{prompt_key}.txt"
        # try:
        #     with open(prompt_path, "r") as f:
        #         return f.read()
        # except FileNotFoundError:
        #     logger.error(f"Prompt file not found: {prompt_path}")
        #     return ""
        # For now, this is not used as templates are hardcoded in run() for simplicity.
        # The _load_prompt_template is a more direct placeholder.
        pass
