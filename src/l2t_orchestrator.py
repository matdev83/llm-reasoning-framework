import time
import logging
from typing import List, Tuple, Optional # Import Optional

from src.aot_dataclasses import LLMCallStats
from src.aot_enums import AssessmentDecision # Import AssessmentDecision
from src.complexity_assessor import ComplexityAssessor # Import ComplexityAssessor
from src.llm_client import LLMClient
from src.l2t_dataclasses import L2TConfig, L2TResult, L2TSolution # Import L2TSolution
from src.l2t_enums import L2TTriggerMode # Import L2TTriggerMode
from src.l2t_processor import L2TProcessor
from src.heuristic_detector import HeuristicDetector # Import HeuristicDetector
from src.l2t_orchestrator_utils.summary_generator import L2TSummaryGenerator
from src.l2t_orchestrator_utils.oneshot_executor import OneShotExecutor

# Configure basic logging if no handlers are present
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        self.l2t_config = l2t_config
        self.direct_oneshot_model_names = direct_oneshot_model_names
        self.direct_oneshot_temperature = direct_oneshot_temperature
        self.llm_client = LLMClient(
            api_key=api_key,
            enable_rate_limiting=enable_rate_limiting,
            enable_audit_logging=enable_audit_logging
        )
        self.heuristic_detector = heuristic_detector
        self.summary_generator = L2TSummaryGenerator(
            trigger_mode=trigger_mode,
            use_heuristic_shortcut=use_heuristic_shortcut
        )
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
        
        self.l2t_processor = None
        if self.trigger_mode != L2TTriggerMode.NEVER_L2T:
             self.l2t_processor = L2TProcessor(
                 api_key=api_key, # Pass api_key instead of llm_client
                 config=self.l2t_config,
                 enable_rate_limiting=enable_rate_limiting,
                 enable_audit_logging=enable_audit_logging
             )

    def solve(self, problem_text: str) -> Tuple[L2TSolution, str]: # Modified return type
        overall_start_time = time.monotonic()
        solution = L2TSolution() # Use L2TSolution

        if self.trigger_mode == L2TTriggerMode.NEVER_L2T:
            logger.info("Trigger mode: NEVER_L2T. Direct one-shot call.")
            final_answer, oneshot_stats = self.oneshot_executor.run_direct_oneshot(problem_text)
            solution.final_answer = final_answer
            solution.main_call_stats = oneshot_stats

        elif self.trigger_mode == L2TTriggerMode.ALWAYS_L2T:
            logger.info("Trigger mode: ALWAYS_L2T. Direct L2T process.")
            if not self.l2t_processor: 
                logger.critical("L2TProcessor not initialized for ALWAYS_L2T mode.")
                raise Exception("L2TProcessor not initialized for ALWAYS_L2T mode.")
            l2t_result_data = self.l2t_processor.run(problem_text)
            solution.l2t_result = l2t_result_data
            solution.l2t_summary_output = self.summary_generator.generate_l2t_summary_from_result(l2t_result_data) # Generate summary from result
            if l2t_result_data.succeeded:
                solution.final_answer = l2t_result_data.final_answer
            else:
                logger.warning(f"L2T process (ALWAYS_L2T mode) failed (Reason: {l2t_result_data.error_message}). Falling back to one-shot.")
                solution.l2t_failed_and_fell_back = True
                fallback_answer, fallback_stats = self.oneshot_executor.run_direct_oneshot(problem_text, is_fallback=True)
                solution.final_answer = fallback_answer
                solution.fallback_call_stats = fallback_stats

        elif self.trigger_mode == L2TTriggerMode.ASSESS_FIRST:
            if not self.complexity_assessor:
                logger.critical("ComplexityAssessor not initialized for ASSESS_FIRST mode.")
                raise Exception("ComplexityAssessor not initialized for ASSESS_FIRST mode.")
            
            assessment_decision, assessment_stats = self.complexity_assessor.assess(problem_text)
            solution.assessment_stats = assessment_stats
            solution.assessment_decision = assessment_decision # Store the decision

            if assessment_decision == AssessmentDecision.ONESHOT:
                logger.info("Assessment: ONESHOT. Direct one-shot call.")
                final_answer, oneshot_stats = self.oneshot_executor.run_direct_oneshot(problem_text)
                solution.final_answer = final_answer
                solution.main_call_stats = oneshot_stats
            elif assessment_decision == AssessmentDecision.AOT: # AOT here means L2T for this orchestrator
                logger.info("Assessment: L2T. Proceeding with L2T process.")
                if not self.l2t_processor:
                    logger.critical("L2TProcessor not initialized for ASSESS_FIRST mode (L2T path).")
                    raise Exception("L2TProcessor not initialized for ASSESS_FIRST mode (L2T path).")
                
                l2t_result_data = self.l2t_processor.run(problem_text)
                solution.l2t_result = l2t_result_data
                solution.l2t_summary_output = self.summary_generator.generate_l2t_summary_from_result(l2t_result_data) # Generate summary from result
                if l2t_result_data.succeeded:
                    solution.final_answer = l2t_result_data.final_answer
                else:
                    logger.warning(f"L2T process (after ASSESS_FIRST) failed (Reason: {l2t_result_data.error_message}). Falling back to one-shot.")
                    solution.l2t_failed_and_fell_back = True
                    fallback_answer, fallback_stats = self.oneshot_executor.run_direct_oneshot(problem_text, is_fallback=True)
                    solution.final_answer = fallback_answer
                    solution.fallback_call_stats = fallback_stats
            else: # AssessmentDecision.ERROR
                logger.error("Complexity assessment failed. Attempting one-shot call as a last resort.")
                solution.l2t_failed_and_fell_back = True # Mark as a form of fallback due to assessment error
                fallback_answer, fallback_stats = self.oneshot_executor.run_direct_oneshot(problem_text, is_fallback=True)
                solution.final_answer = fallback_answer
                solution.fallback_call_stats = fallback_stats

        solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time
        summary_output = self.summary_generator.generate_overall_summary(solution) # Call the new method
        return solution, summary_output # Return both
