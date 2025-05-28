import time
import logging
import io # Required for summary generation in L2TProcess
from typing import List, Tuple, Optional, Any # Import Any

from src.reasoning_process import ReasoningProcess # Import the base class
from src.aot_dataclasses import LLMCallStats # Kept for OneShotExecutor if used by L2TProcess
from src.aot_enums import AssessmentDecision 
from src.complexity_assessor import ComplexityAssessor 
from src.llm_client import LLMClient
from src.l2t_dataclasses import L2TConfig, L2TResult, L2TSolution 
from src.l2t_enums import L2TTriggerMode 
from src.l2t_processor import L2TProcessor
from src.heuristic_detector import HeuristicDetector 
# Utilities that might be used by L2TProcess or remain in L2TOrchestrator
from src.l2t_orchestrator_utils.summary_generator import L2TSummaryGenerator
from src.l2t_orchestrator_utils.oneshot_executor import OneShotExecutor

# Configure basic logging if no handlers are present
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Definition of L2TProcess class starts here
class L2TProcess(ReasoningProcess):
    """
    Implements the Learn to Think (L2T) reasoning process.
    This class encapsulates the core L2T logic, including running the L2TProcessor
    and handling fallbacks to a direct one-shot call if L2T fails.
    """
    def __init__(self,
                 l2t_config: L2TConfig,
                 direct_oneshot_model_names: List[str], # For fallback
                 direct_oneshot_temperature: float,    # For fallback
                 api_key: str,
                 enable_rate_limiting: bool = True,
                 enable_audit_logging: bool = True):

        self.l2t_config = l2t_config
        # LLMClient for L2TProcess's own operations (e.g. fallback via OneShotExecutor)
        self.llm_client = LLMClient(
            api_key=api_key,
            enable_rate_limiting=enable_rate_limiting,
            enable_audit_logging=enable_audit_logging
        )
        self.l2t_processor = L2TProcessor(
            api_key=api_key, # L2TProcessor creates its own client or uses one passed
            config=self.l2t_config,
            enable_rate_limiting=enable_rate_limiting,
            enable_audit_logging=enable_audit_logging
        )
        self.oneshot_executor = OneShotExecutor( # For fallback mechanism
            llm_client=self.llm_client, # Use L2TProcess's client
            direct_oneshot_model_names=direct_oneshot_model_names,
            direct_oneshot_temperature=direct_oneshot_temperature
        )
        # Summary generator specific to L2TProcess's output
        self.summary_generator = L2TSummaryGenerator(trigger_mode=L2TTriggerMode.ALWAYS_L2T, use_heuristic_shortcut=False)


        self._solution: Optional[L2TSolution] = None
        self._process_summary: Optional[str] = None

    def execute(self, problem_description: str, model_name: str, *args, **kwargs) -> None:
        """
        Executes the L2T reasoning process.
        'model_name' is for interface compatibility; L2T models are per l2t_config.
        """
        overall_start_time = time.monotonic()
        self._solution = L2TSolution()
        logger.info(f"L2TProcess executing for problem: {problem_description[:100]}... (model_name param: {model_name})")

        if not self.l2t_processor: # Should be initialized in __init__
            logger.critical("L2TProcessor not initialized within L2TProcess.")
            self._solution.final_answer = "Error: L2TProcessor not initialized in L2TProcess."
            self._solution.l2t_result = L2TResult(succeeded=False, error_message=self._solution.final_answer)
            if self._solution.total_wall_clock_time_seconds is None:
                 self._solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time
            self._process_summary = self.summary_generator.generate_l2t_process_summary(self._solution)
            return

        l2t_result_data = self.l2t_processor.run(problem_description)
        self._solution.l2t_result = l2t_result_data
        
        # The L2TProcessor's own summary (if any detailed one exists) could be stored here
        # self._solution.l2t_processor_internal_summary = l2t_result_data.get("processor_summary_str", "") 

        if l2t_result_data.succeeded:
            self._solution.final_answer = l2t_result_data.final_answer
            # Potentially add other L2T specific results to _solution from l2t_result_data
        else:
            logger.warning(f"L2T process failed (Reason: {l2t_result_data.error_message}). Falling back to one-shot within L2TProcess.")
            self._solution.l2t_failed_and_fell_back = True # Mark fallback
            fallback_answer, fallback_stats = self.oneshot_executor.run_direct_oneshot(problem_description, is_fallback=True)
            self._solution.final_answer = fallback_answer
            self._solution.fallback_call_stats = fallback_stats
            # Keep L2T result even if failed, for diagnostics
        
        if self._solution.total_wall_clock_time_seconds is None:
            self._solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time
        
        # Generate a summary for this L2TProcess execution
        self._process_summary = self.summary_generator.generate_l2t_process_summary(self._solution)


    def get_result(self) -> Tuple[Optional[L2TSolution], Optional[str]]:
        return self._solution, self._process_summary

# End of L2TProcess class definition


class L2TOrchestrator:
    def __init__(self,
                 trigger_mode: L2TTriggerMode,
                 l2t_config: L2TConfig,
                 direct_oneshot_model_names: List[str],
                 direct_oneshot_temperature: float,
                 assessment_model_names: List[str],
                 assessment_temperature: float,
                 api_key: str,
                 use_heuristic_shortcut: bool = True,
                 heuristic_detector: Optional[HeuristicDetector] = None,
                 enable_rate_limiting: bool = True,
                 enable_audit_logging: bool = True):

        self.trigger_mode = trigger_mode
        self.use_heuristic_shortcut = use_heuristic_shortcut
        # self.l2t_config = l2t_config # No longer stored directly if passed to L2TProcess
        
        # LLMClient for orchestrator's direct one-shot calls and for ComplexityAssessor
        self.llm_client = LLMClient(
            api_key=api_key,
            enable_rate_limiting=enable_rate_limiting,
            enable_audit_logging=enable_audit_logging
        )
        self.heuristic_detector = heuristic_detector
        
        # Orchestrator's summary generator for overall summary
        self.summary_generator = L2TSummaryGenerator(
            trigger_mode=trigger_mode,
            use_heuristic_shortcut=use_heuristic_shortcut
        )
        # Orchestrator's one-shot executor for non-L2T paths
        self.oneshot_executor = OneShotExecutor(
            llm_client=self.llm_client,
            direct_oneshot_model_names=direct_oneshot_model_names,
            direct_oneshot_temperature=direct_oneshot_temperature
        )

        self.complexity_assessor = None
        if self.trigger_mode == L2TTriggerMode.ASSESS_FIRST:
            self.complexity_assessor = ComplexityAssessor(
                llm_client=self.llm_client,
                small_model_names=assessment_model_names,
                temperature=assessment_temperature,
                use_heuristic_shortcut=self.use_heuristic_shortcut,
                heuristic_detector=self.heuristic_detector
            )
        
        self.l2t_process_instance: Optional[L2TProcess] = None
        # Instantiate L2TProcess if L2T might be used
        if self.trigger_mode == L2TTriggerMode.ALWAYS_L2T or self.trigger_mode == L2TTriggerMode.ASSESS_FIRST:
             self.l2t_process_instance = L2TProcess(
                 l2t_config=l2t_config,
                 direct_oneshot_model_names=direct_oneshot_model_names, # For L2TProcess's own fallback
                 direct_oneshot_temperature=direct_oneshot_temperature, # For L2TProcess's own fallback
                 api_key=api_key, # L2TProcess will manage its client for L2TProcessor
                 enable_rate_limiting=enable_rate_limiting,
                 enable_audit_logging=enable_audit_logging
             )
        # self.l2t_processor is removed; L2TProcess handles L2TProcessor.

    def solve(self, problem_text: str, model_name_for_l2t: str = "default_l2t_model") -> Tuple[L2TSolution, str]:
        overall_start_time = time.monotonic()
        orchestrator_solution = L2TSolution() # Use L2TSolution
        l2t_process_specific_summary: Optional[str] = None

        if self.trigger_mode == L2TTriggerMode.NEVER_L2T:
            logger.info("Trigger mode: NEVER_L2T. Orchestrator performing direct one-shot call.")
            # Use orchestrator's oneshot_executor
            final_answer, oneshot_stats = self.oneshot_executor.run_direct_oneshot(problem_text)
            orchestrator_solution.final_answer = final_answer
            orchestrator_solution.main_call_stats = oneshot_stats

        elif self.trigger_mode == L2TTriggerMode.ALWAYS_L2T:
            logger.info("Trigger mode: ALWAYS_L2T. Orchestrator delegating to L2TProcess.")
            if not self.l2t_process_instance: 
                logger.critical("L2TProcess not initialized for ALWAYS_L2T mode.")
                orchestrator_solution.final_answer = "Error: L2TProcess not initialized for ALWAYS_L2T mode."
                orchestrator_solution.l2t_result = L2TResult(succeeded=False, error_message=orchestrator_solution.final_answer)
            else:
                self.l2t_process_instance.execute(problem_description=problem_text, model_name=model_name_for_l2t)
                l2t_solution_obj, l2t_process_specific_summary = self.l2t_process_instance.get_result()
                
                if l2t_solution_obj:
                    # Integrate results from L2TProcess's solution
                    orchestrator_solution.final_answer = l2t_solution_obj.final_answer
                    orchestrator_solution.l2t_result = l2t_solution_obj.l2t_result
                    orchestrator_solution.l2t_summary_output = l2t_solution_obj.l2t_summary_output # This is summary from L2TProcessor via L2TProcess
                    orchestrator_solution.l2t_failed_and_fell_back = l2t_solution_obj.l2t_failed_and_fell_back
                    if l2t_solution_obj.fallback_call_stats:
                        orchestrator_solution.fallback_call_stats = l2t_solution_obj.fallback_call_stats
                    if l2t_solution_obj.main_call_stats: # If L2TProcess had a primary call stat
                         orchestrator_solution.main_call_stats = l2t_solution_obj.main_call_stats
                    # orchestrator_solution.add_llm_call_stats_from_solution(l2t_solution_obj) # Removed this line

                else: # Should not happen if L2TProcess always returns a solution
                    orchestrator_solution.final_answer = "Error: L2TProcess executed but returned no solution object."
                    orchestrator_solution.l2t_result = L2TResult(succeeded=False, error_message=orchestrator_solution.final_answer)
                    logger.error("L2TProcess returned None for L2TSolution in ALWAYS_L2T mode.")


        elif self.trigger_mode == L2TTriggerMode.ASSESS_FIRST:
            logger.info("Trigger mode: ASSESS_FIRST. Orchestrator performing complexity assessment.")
            if not self.complexity_assessor:
                logger.critical("ComplexityAssessor not initialized for ASSESS_FIRST mode.")
                orchestrator_solution.final_answer = "Error: ComplexityAssessor not initialized."
                orchestrator_solution.l2t_result = L2TResult(succeeded=False, error_message=orchestrator_solution.final_answer)
            else:
                assessment_decision, assessment_stats = self.complexity_assessor.assess(problem_text)
                orchestrator_solution.assessment_stats = assessment_stats
                orchestrator_solution.assessment_decision = assessment_decision

                if assessment_decision == AssessmentDecision.ONESHOT:
                    logger.info("Assessment: ONESHOT. Orchestrator performing direct one-shot call.")
                    final_answer, oneshot_stats = self.oneshot_executor.run_direct_oneshot(problem_text)
                    orchestrator_solution.final_answer = final_answer
                    orchestrator_solution.main_call_stats = oneshot_stats
                elif assessment_decision == AssessmentDecision.AOT: # AOT here means L2T path
                    logger.info("Assessment: L2T. Orchestrator delegating to L2TProcess.")
                    if not self.l2t_process_instance:
                        logger.critical("L2TProcess not initialized for ASSESS_FIRST mode (L2T path).")
                        orchestrator_solution.final_answer = "Error: L2TProcess not initialized for L2T path."
                        orchestrator_solution.l2t_result = L2TResult(succeeded=False, error_message=orchestrator_solution.final_answer)
                    else:
                        self.l2t_process_instance.execute(problem_description=problem_text, model_name=model_name_for_l2t)
                        l2t_solution_obj, l2t_process_specific_summary = self.l2t_process_instance.get_result()

                        if l2t_solution_obj:
                            orchestrator_solution.final_answer = l2t_solution_obj.final_answer
                            orchestrator_solution.l2t_result = l2t_solution_obj.l2t_result
                            orchestrator_solution.l2t_summary_output = l2t_solution_obj.l2t_summary_output
                            orchestrator_solution.l2t_failed_and_fell_back = l2t_solution_obj.l2t_failed_and_fell_back
                            if l2t_solution_obj.fallback_call_stats:
                                orchestrator_solution.fallback_call_stats = l2t_solution_obj.fallback_call_stats
                            if l2t_solution_obj.main_call_stats:
                                orchestrator_solution.main_call_stats = l2t_solution_obj.main_call_stats
                            # orchestrator_solution.add_llm_call_stats_from_solution(l2t_solution_obj) # Removed this line
                        else: # Should not happen
                             orchestrator_solution.final_answer = "Error: L2TProcess (post-assessment) returned no solution object."
                             orchestrator_solution.l2t_result = L2TResult(succeeded=False, error_message=orchestrator_solution.final_answer)
                             logger.error("L2TProcess returned None for L2TSolution in ASSESS_FIRST (L2T path) mode.")

                else: # AssessmentDecision.ERROR
                    logger.error("Complexity assessment failed. Orchestrator attempting one-shot call as a last resort.")
                    # Store this specific outcome for summary
                    if orchestrator_solution.assessment_stats:
                        orchestrator_solution.assessment_stats.assessment_decision_for_summary = "ERROR_FALLBACK" # Custom field
                    orchestrator_solution.l2t_failed_and_fell_back = True # Indicate fallback due to assessment error
                    fallback_answer, fallback_stats = self.oneshot_executor.run_direct_oneshot(problem_text, is_fallback=True)
                    orchestrator_solution.final_answer = fallback_answer
                    orchestrator_solution.fallback_call_stats = fallback_stats

        if orchestrator_solution.total_wall_clock_time_seconds is None:
            orchestrator_solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time
        
        # Use the orchestrator's summary_generator for the final overall summary
        # It can internally decide how to use l2t_process_specific_summary
        final_summary_output = self.summary_generator.generate_overall_summary(
            orchestrator_solution, 
            l2t_process_execution_summary=l2t_process_specific_summary
        )
        return orchestrator_solution, final_summary_output
