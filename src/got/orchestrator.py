import time
import logging
from typing import Tuple, Optional, List

from src.reasoning_process import ReasoningProcess
from src.llm_client import LLMClient
from src.llm_config import LLMConfig
from src.aot.enums import AssessmentDecision # Using existing enum
from src.complexity_assessor import ComplexityAssessor
from src.heuristic_detector import HeuristicDetector
# Assuming a similar OneShotExecutor can be used or adapted if needed for GoT's fallback
# For simplicity, GoTProcess will handle its own fallback for now.
# from src.l2t_orchestrator_utils.oneshot_executor import OneShotExecutor # Example

from .dataclasses import (
    GoTConfig,
    GoTModelConfigs,
    GoTSolution,
    GoTResult
)
from .enums import GoTTriggerMode
from .processor import GoTProcessor
from .summary_generator import GoTSummaryGenerator

# Configure logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class GoTProcess(ReasoningProcess):
    def __init__(self,
                 llm_client: LLMClient,
                 got_config: GoTConfig,
                 got_model_configs: GoTModelConfigs,
                 # direct_oneshot_llm_config is used for fallback if GoTProcessor fails
                 direct_oneshot_llm_config: LLMConfig,
                 direct_oneshot_model_names: List[str]):

        self.llm_client = llm_client
        self.got_config = got_config
        self.got_model_configs = got_model_configs
        self.direct_oneshot_llm_config = direct_oneshot_llm_config
        self.direct_oneshot_model_names = direct_oneshot_model_names

        self.got_processor = GoTProcessor(
            llm_client=self.llm_client,
            config=self.got_config,
            model_configs=self.got_model_configs
        )
        self.summary_generator = GoTSummaryGenerator(trigger_mode=GoTTriggerMode.ALWAYS_GOT) # Default for process summary

        self._solution: Optional[GoTSolution] = None
        self._process_summary: Optional[str] = None

    def _run_direct_oneshot_fallback(self, problem_text: str) -> Tuple[str, Optional[object]]: # Opt object is LLMCallStats
        logger.info("--- GoTProcess: Falling back to Direct One-Shot ---")
        logger.info(f"Using models: {', '.join(self.direct_oneshot_model_names)}, LLMConfig: {self.direct_oneshot_llm_config}")

        # Re-using the llm_client instance from GoTProcess
        response_content, stats = self.llm_client.call(
            prompt=problem_text,
            models=self.direct_oneshot_model_names,
            config=self.direct_oneshot_llm_config
        )
        if stats:
            logger.info(f"LLM call ({stats.model_name}) for Fallback One-Shot: "
                        f"Duration: {stats.call_duration_seconds:.2f}s, "
                        f"Tokens (C:{stats.completion_tokens}, P:{stats.prompt_tokens})")
        return response_content, stats

    def execute(self, problem_description: str, model_name: str, *args, **kwargs) -> None:
        # model_name param might be for orchestrator to pass to processor if needed, currently GoTConfig has models
        overall_start_time = time.monotonic()
        self._solution = GoTSolution() # Initialize solution object
        logger.info(f"GoTProcess executing for problem: {problem_description[:100]}...")

        if not self.got_processor: # Should not happen if constructor ran
            logger.critical("GoTProcessor not initialized within GoTProcess.")
            self._solution.final_answer = "Error: GoTProcessor not initialized."
            # Create a minimal GoTResult for consistency
            self._solution.got_result = GoTResult(succeeded=False, error_message=self._solution.final_answer)
        else:
            got_result_data = self.got_processor.run(problem_description)
            self._solution.got_result = got_result_data

            if got_result_data.succeeded and got_result_data.final_answer is not None:
                self._solution.final_answer = got_result_data.final_answer
            else:
                logger.warning(f"GoT process failed or yielded no answer (Reason: {got_result_data.error_message}). Falling back to one-shot.")
                self._solution.got_failed_and_fell_back = True
                fallback_answer, fallback_stats = self._run_direct_oneshot_fallback(problem_description)
                self._solution.final_answer = fallback_answer
                self._solution.fallback_call_stats = fallback_stats

        self._solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time
        self._process_summary = self.summary_generator.generate_got_process_summary(self._solution)


    def get_result(self) -> Tuple[Optional[GoTSolution], Optional[str]]:
        return self._solution, self._process_summary


class GoTOrchestrator:
    def __init__(self,
                 llm_client: LLMClient, # Shared client
                 trigger_mode: GoTTriggerMode,
                 got_config: GoTConfig, # Specific to GoT processor
                 got_model_configs: GoTModelConfigs, # Specific to GoT processor
                 # For direct one-shot by orchestrator (NEVER_GOT or fallback from assessment)
                 direct_oneshot_llm_config: LLMConfig,
                 direct_oneshot_model_names: List[str],
                 # For assessment phase
                 assessment_llm_config: LLMConfig,
                 assessment_model_names: List[str],
                 use_heuristic_shortcut: bool = True,
                 heuristic_detector: Optional[HeuristicDetector] = None):

        self.llm_client = llm_client
        self.trigger_mode = trigger_mode
        self.got_config = got_config
        self.got_model_configs = got_model_configs

        self.direct_oneshot_llm_config = direct_oneshot_llm_config
        self.direct_oneshot_model_names = direct_oneshot_model_names

        self.assessment_llm_config = assessment_llm_config
        self.assessment_model_names = assessment_model_names

        self.use_heuristic_shortcut = use_heuristic_shortcut
        self.heuristic_detector = heuristic_detector

        self.summary_generator = GoTSummaryGenerator(
            trigger_mode=self.trigger_mode,
            use_heuristic_shortcut=self.use_heuristic_shortcut
        )

        self.complexity_assessor: Optional[ComplexityAssessor] = None
        if self.trigger_mode == GoTTriggerMode.ASSESS_FIRST_GOT:
            self.complexity_assessor = ComplexityAssessor(
                llm_client=self.llm_client,
                small_model_names=self.assessment_model_names,
                llm_config=self.assessment_llm_config,
                use_heuristic_shortcut=self.use_heuristic_shortcut,
                heuristic_detector=self.heuristic_detector
            )

        self.got_process_instance: Optional[GoTProcess] = None
        # GoTProcess instance will be created on demand in solve() for ALWAYS_GOT or if assessment leads to ADVANCED_REASONING.
        # This change is to prevent unnecessary instantiation when assessment might lead to ONE_SHOT.
        # Storing configs needed for on-demand instantiation.
        # if self.trigger_mode == GoTTriggerMode.ALWAYS_GOT or self.trigger_mode == GoTTriggerMode.ASSESS_FIRST_GOT:
        #     self.got_process_instance = GoTProcess(
        #         llm_client=self.llm_client, # Pass shared client
        #         got_config=self.got_config,
        #         got_model_configs=self.got_model_configs, # This line had incorrect indentation
        #         direct_oneshot_llm_config=self.direct_oneshot_llm_config, # For GoTProcess's internal fallback
        #         direct_oneshot_model_names=self.direct_oneshot_model_names
        #     )

    def _get_or_create_got_process_instance(self) -> GoTProcess:
        if not self.got_process_instance:
            logger.info("Creating GoTProcess instance on demand.")
            self.got_process_instance = GoTProcess(
                llm_client=self.llm_client,
                got_config=self.got_config,
                got_model_configs=self.got_model_configs,
                direct_oneshot_llm_config=self.direct_oneshot_llm_config, # For GoTProcess's internal fallback
                direct_oneshot_model_names=self.direct_oneshot_model_names
            )
        return self.got_process_instance

    def _run_orchestrator_direct_oneshot(self, problem_text: str, is_fallback_path:bool = False) -> Tuple[str, Optional[object]]: # Opt object is LLMCallStats
        mode_desc = "FALLBACK ONESHOT (Orchestrator)" if is_fallback_path else "DIRECT ONESHOT (Orchestrator)"
        logger.info(f"--- Proceeding with {mode_desc} ---")
        logger.info(f"Using models: {', '.join(self.direct_oneshot_model_names)}, LLMConfig: {self.direct_oneshot_llm_config}")

        response_content, stats = self.llm_client.call(
            prompt=problem_text,
            models=self.direct_oneshot_model_names,
            config=self.direct_oneshot_llm_config
        )
        if stats:
            logger.info(f"LLM call ({stats.model_name}) for {mode_desc}: "
                        f"Duration: {stats.call_duration_seconds:.2f}s, "
                        f"Tokens (C:{stats.completion_tokens}, P:{stats.prompt_tokens})")
        return response_content, stats

    def solve(self, problem_text: str, model_name_for_got: str = "default_got_model") -> Tuple[GoTSolution, str]:
        # model_name_for_got is for compatibility, actual models from GoTConfig
        overall_start_time = time.monotonic()
        orchestrator_solution = GoTSolution()
        got_process_execution_summary: Optional[str] = None

        if self.trigger_mode == GoTTriggerMode.NEVER_GOT:
            logger.info("Trigger mode: NEVER_GOT. Orchestrator performing direct one-shot call.")
            final_answer, oneshot_stats = self._run_orchestrator_direct_oneshot(problem_text)
            orchestrator_solution.final_answer = final_answer
            orchestrator_solution.fallback_call_stats = oneshot_stats

        elif self.trigger_mode == GoTTriggerMode.ALWAYS_GOT:
            logger.info("Trigger mode: ALWAYS_GOT. Orchestrator delegating to GoTProcess.")
            current_got_process = self._get_or_create_got_process_instance()
            current_got_process.execute(problem_description=problem_text, model_name=model_name_for_got)
            got_solution_obj, got_process_execution_summary = current_got_process.get_result()

            if got_solution_obj:
                # Transfer relevant fields from GoTProcess's solution to orchestrator's solution
                orchestrator_solution.final_answer = got_solution_obj.final_answer
                orchestrator_solution.got_result = got_solution_obj.got_result
                orchestrator_solution.got_summary_output = got_solution_obj.got_summary_output
                orchestrator_solution.got_failed_and_fell_back = got_solution_obj.got_failed_and_fell_back
                if got_solution_obj.fallback_call_stats:
                    orchestrator_solution.fallback_call_stats = got_solution_obj.fallback_call_stats
            else: # Corrected indentation for this else block
                er_msg = "Error: GoTProcess executed but returned no solution object."
                logger.error(er_msg)
                orchestrator_solution.final_answer = er_msg
                orchestrator_solution.got_result = GoTResult(succeeded=False, error_message=er_msg) # Corrected indentation

        elif self.trigger_mode == GoTTriggerMode.ASSESS_FIRST_GOT:
            logger.info("Trigger mode: ASSESS_FIRST_GOT. Orchestrator performing complexity assessment.")
            if not self.complexity_assessor:
                logger.critical("ComplexityAssessor not initialized for ASSESS_FIRST_GOT mode.")
                orchestrator_solution.final_answer = "Error: ComplexityAssessor not initialized."
                orchestrator_solution.got_result = GoTResult(succeeded=False, error_message=orchestrator_solution.final_answer)
            else:
                assessment_decision, assessment_stats = self.complexity_assessor.assess(problem_text)
                orchestrator_solution.assessment_stats = assessment_stats
                orchestrator_solution.assessment_decision = assessment_decision

                if assessment_decision == AssessmentDecision.ONE_SHOT:
                    logger.info("Assessment: ONE_SHOT. Orchestrator performing direct one-shot call.")
                    final_answer, oneshot_stats = self._run_orchestrator_direct_oneshot(problem_text)
                    orchestrator_solution.final_answer = final_answer
                    orchestrator_solution.fallback_call_stats = oneshot_stats

                elif assessment_decision == AssessmentDecision.ADVANCED_REASONING:
                    logger.info("Assessment: ADVANCED_REASONING. Orchestrator delegating to GoTProcess.")
                    current_got_process = self._get_or_create_got_process_instance()
                    current_got_process.execute(problem_description=problem_text, model_name=model_name_for_got)
                    got_solution_obj, got_process_execution_summary = current_got_process.get_result()
                    if got_solution_obj:
                        orchestrator_solution.final_answer = got_solution_obj.final_answer
                        orchestrator_solution.got_result = got_solution_obj.got_result
                        orchestrator_solution.got_summary_output = got_solution_obj.got_summary_output
                        orchestrator_solution.got_failed_and_fell_back = got_solution_obj.got_failed_and_fell_back
                        if got_solution_obj.fallback_call_stats:
                            orchestrator_solution.fallback_call_stats = got_solution_obj.fallback_call_stats
                    else: # Corrected indentation for this block
                        er_msg = "Error: GoTProcess (post-assessment) returned no solution object."
                        logger.error(er_msg) # Corrected indentation
                        orchestrator_solution.final_answer = er_msg # Corrected indentation
                        orchestrator_solution.got_result = GoTResult(succeeded=False, error_message=er_msg) # Corrected indentation
                else: # Assessment failed or other unexpected decision
                    logger.error(f"Complexity assessment yielded unexpected decision '{assessment_decision}' or failed. Orchestrator attempting one-shot call as a last resort.")
                    final_answer, oneshot_stats = self._run_orchestrator_direct_oneshot(problem_text, is_fallback_path=True)
                    orchestrator_solution.final_answer = final_answer
                    orchestrator_solution.fallback_call_stats = oneshot_stats # Store as fallback
                    # Mark that GoT effectively "failed" if assessment led to this path without explicit ONE_SHOT decision
                    if assessment_decision != AssessmentDecision.ONE_SHOT:
                         orchestrator_solution.got_failed_and_fell_back = True # To indicate main process was bypassed due to error

        orchestrator_solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time

        # Use the GoTProcess's summary if GoTProcess was run, otherwise it's None
        final_summary_output = self.summary_generator.generate_overall_summary(
            orchestrator_solution,
            got_process_specific_summary=got_process_execution_summary
        )
        return orchestrator_solution, final_summary_output
