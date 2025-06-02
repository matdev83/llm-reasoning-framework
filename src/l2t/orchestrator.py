import time
import logging
import io
from typing import List, Tuple, Optional, Any

from src.reasoning_process import ReasoningProcess
from src.aot.dataclasses import LLMCallStats
from src.aot.enums import AssessmentDecision
from src.complexity_assessor import ComplexityAssessor 
from src.llm_client import LLMClient
from src.llm_config import LLMConfig # Added
from .dataclasses import L2TConfig, L2TResult, L2TSolution, L2TModelConfigs # Added L2TModelConfigs
from .enums import L2TTriggerMode
from .processor import L2TProcessor
from src.heuristic_detector import HeuristicDetector 
from src.l2t_orchestrator_utils.summary_generator import L2TSummaryGenerator # Moved back to module level
from src.l2t_orchestrator_utils.oneshot_executor import OneShotExecutor

# Removed: if not logging.getLogger().hasHandlers():
# Removed:     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class L2TProcess(ReasoningProcess):
    def __init__(self,
                 l2t_config: L2TConfig, # L2TConfig for model names, max_steps etc.
                 model_configs: L2TModelConfigs, # Contains all LLMConfigs
                 api_key: str,
                 enable_rate_limiting: bool = True,
                 enable_audit_logging: bool = True):

        self.l2t_config = l2t_config
        self.model_configs = model_configs

        self.llm_client = LLMClient(
            api_key=api_key,
            enable_rate_limiting=enable_rate_limiting,
            enable_audit_logging=enable_audit_logging
        )
        self.l2t_processor = L2TProcessor(
            api_key=api_key,
            l2t_config=self.l2t_config,
            initial_thought_llm_config=self.model_configs.initial_thought_config,
            node_processor_llm_config=self.model_configs.node_thought_generation_config, # Or a more specific one if NodeProcessor differentiates
            enable_rate_limiting=enable_rate_limiting,
            enable_audit_logging=enable_audit_logging
        )
        self.oneshot_executor = OneShotExecutor(
            llm_client=self.llm_client,
            direct_oneshot_model_names=self.l2t_config.initial_prompt_model_names, # Fallback uses initial prompt models
            llm_config=self.model_configs.orchestrator_oneshot_config
        )
        self.summary_generator = L2TSummaryGenerator(trigger_mode=L2TTriggerMode.ALWAYS_L2T, use_heuristic_shortcut=False)

        self._solution: Optional[L2TSolution] = None
        self._process_summary: Optional[str] = None

    def execute(self, problem_description: str, model_name: str, *args, **kwargs) -> None:
        overall_start_time = time.monotonic()
        self._solution = L2TSolution()
        logger.info(f"L2TProcess executing for problem: {problem_description[:100]}... (model_name param: {model_name})")

        if not self.l2t_processor:
            logger.critical("L2TProcessor not initialized within L2TProcess.")
            self._solution.final_answer = "Error: L2TProcessor not initialized in L2TProcess."
            self._solution.l2t_result = L2TResult(succeeded=False, error_message=self._solution.final_answer)
            if self._solution.total_wall_clock_time_seconds is None:
                 self._solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time
            self._process_summary = self.summary_generator.generate_l2t_process_summary(self._solution)
            return

        l2t_result_data = self.l2t_processor.run(problem_description)
        self._solution.l2t_result = l2t_result_data
        
        if l2t_result_data.succeeded:
            self._solution.final_answer = l2t_result_data.final_answer
        else:
            logger.warning(f"L2T process failed (Reason: {l2t_result_data.error_message}). Falling back to one-shot within L2TProcess.")
            self._solution.l2t_failed_and_fell_back = True
            # Fallback uses orchestrator_oneshot_config via self.oneshot_executor
            fallback_answer, fallback_stats = self.oneshot_executor.run_direct_oneshot(problem_description, is_fallback=True)
            self._solution.final_answer = fallback_answer
            self._solution.fallback_call_stats = fallback_stats
        
        if self._solution.total_wall_clock_time_seconds is None:
            self._solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time
        
        self._process_summary = self.summary_generator.generate_l2t_process_summary(self._solution)

    def get_result(self) -> Tuple[Optional[L2TSolution], Optional[str]]:
        return self._solution, self._process_summary

class L2TOrchestrator:
    def __init__(self,
                 trigger_mode: L2TTriggerMode,
                 l2t_config: L2TConfig, # For model names, max_steps etc.
                 model_configs: L2TModelConfigs, # Contains all LLMConfigs
                 api_key: str,
                 use_heuristic_shortcut: bool = True,
                 heuristic_detector: Optional[HeuristicDetector] = None,
                 enable_rate_limiting: bool = True,
                 enable_audit_logging: bool = True):
        # from src.l2t_orchestrator_utils.summary_generator import L2TSummaryGenerator # No longer here

        self.trigger_mode = trigger_mode
        self.use_heuristic_shortcut = use_heuristic_shortcut
        self.l2t_config = l2t_config # Store l2t_config for model names etc.
        self.model_configs = model_configs # Store model_configs
        
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
            direct_oneshot_model_names=self.l2t_config.initial_prompt_model_names, # Orchestrator one-shot uses initial prompt models
            llm_config=self.model_configs.orchestrator_oneshot_config
        )

        self.complexity_assessor = None
        if self.trigger_mode == L2TTriggerMode.ASSESS_FIRST:
            self.complexity_assessor = ComplexityAssessor(
                llm_client=self.llm_client,
                small_model_names=self.l2t_config.classification_model_names, # Assessor uses classification models
                llm_config=self.model_configs.node_classification_config, # Or a dedicated assessor_config
                use_heuristic_shortcut=self.use_heuristic_shortcut,
                heuristic_detector=self.heuristic_detector
            )
        
        self.l2t_process_instance: Optional[L2TProcess] = None
        if self.trigger_mode == L2TTriggerMode.ALWAYS_L2T or self.trigger_mode == L2TTriggerMode.ASSESS_FIRST:
             self.l2t_process_instance = L2TProcess(
                 l2t_config=self.l2t_config,
                 model_configs=self.model_configs, # Pass the whole bundle
                 api_key=api_key,
                 enable_rate_limiting=enable_rate_limiting,
                 enable_audit_logging=enable_audit_logging
             )

    def solve(self, problem_text: str, model_name_for_l2t: str = "default_l2t_model") -> Tuple[L2TSolution, str]:
        overall_start_time = time.monotonic()
        orchestrator_solution = L2TSolution()
        l2t_process_specific_summary: Optional[str] = None

        if self.trigger_mode == L2TTriggerMode.NEVER_L2T:
            logger.info("Trigger mode: NEVER_L2T. Orchestrator performing direct one-shot call.")
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
                    orchestrator_solution.final_answer = l2t_solution_obj.final_answer
                    orchestrator_solution.l2t_result = l2t_solution_obj.l2t_result
                    orchestrator_solution.l2t_summary_output = l2t_solution_obj.l2t_summary_output
                    orchestrator_solution.l2t_failed_and_fell_back = l2t_solution_obj.l2t_failed_and_fell_back
                    if l2t_solution_obj.fallback_call_stats:
                        orchestrator_solution.fallback_call_stats = l2t_solution_obj.fallback_call_stats
                    if l2t_solution_obj.main_call_stats:
                         orchestrator_solution.main_call_stats = l2t_solution_obj.main_call_stats
                else:
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

                if assessment_decision == AssessmentDecision.ONE_SHOT:
                    logger.info("Assessment: ONE_SHOT. Orchestrator performing direct one-shot call.")
                    final_answer, oneshot_stats = self.oneshot_executor.run_direct_oneshot(problem_text)
                    orchestrator_solution.final_answer = final_answer
                    orchestrator_solution.main_call_stats = oneshot_stats
                elif assessment_decision == AssessmentDecision.ADVANCED_REASONING:
                    logger.info("Assessment: ADVANCED_REASONING. Orchestrator delegating to L2TProcess.")
                    if not self.l2t_process_instance:
                        logger.critical("L2TProcess not initialized for ASSESS_FIRST mode (ADVANCED_REASONING path).")
                        orchestrator_solution.final_answer = "Error: L2TProcess not initialized for ADVANCED_REASONING path."
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
                        else:
                             orchestrator_solution.final_answer = "Error: L2TProcess (post-assessment) returned no solution object."
                             orchestrator_solution.l2t_result = L2TResult(succeeded=False, error_message=orchestrator_solution.final_answer)
                             logger.error("L2TProcess returned None for L2TSolution in ASSESS_FIRST (ADVANCED_REASONING path) mode.")
                else:
                    logger.error("Complexity assessment failed. Orchestrator attempting one-shot call as a last resort.")
                    orchestrator_solution.l2t_failed_and_fell_back = True
                    fallback_answer, fallback_stats = self.oneshot_executor.run_direct_oneshot(problem_text, is_fallback=True)
                    orchestrator_solution.final_answer = fallback_answer
                    orchestrator_solution.fallback_call_stats = fallback_stats

        if orchestrator_solution.total_wall_clock_time_seconds is None:
            orchestrator_solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time
        
        final_summary_output = self.summary_generator.generate_overall_summary(
            orchestrator_solution, 
            l2t_process_execution_summary=l2t_process_specific_summary
        )
        return orchestrator_solution, final_summary_output
