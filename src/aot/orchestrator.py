import time
import logging
import io
from typing import List, Tuple, Optional, Any # Import Optional and Any

from src.reasoning_process import ReasoningProcess # Import the base class
from .enums import AotTriggerMode, AssessmentDecision
from .dataclasses import LLMCallStats, AoTRunnerConfig, Solution
from src.llm_client import LLMClient # Moved here
from src.llm_config import LLMConfig # Added
from src.complexity_assessor import ComplexityAssessor
from .processor import AoTProcessor
from src.heuristic_detector import HeuristicDetector


# Definition of AoTProcess class starts here
class AoTProcess(ReasoningProcess):
    """
    Implements the Algorithm of Thoughts (AoT) reasoning process.
    This class is responsible for executing the AoT chain, including
    running the AoTProcessor and handling fallbacks to a direct one-shot call if AoT fails.
    """
    def __init__(self,
                 llm_client: LLMClient, # Passed in
                 aot_config: AoTRunnerConfig,
                 aot_main_llm_config: LLMConfig, # For AoTProcessor
                 direct_oneshot_llm_config: LLMConfig): # For fallback

        self.llm_client = llm_client # Use passed-in client
        self.aot_config = aot_config
        self.aot_main_llm_config = aot_main_llm_config
        self.direct_oneshot_llm_config = direct_oneshot_llm_config

        self.aot_processor = AoTProcessor(llm_client=self.llm_client, runner_config=self.aot_config, llm_config=self.aot_main_llm_config)

        self._solution: Optional[Solution] = None
        self._process_summary: Optional[str] = None

    def _run_direct_oneshot(self, problem_text: str, is_fallback:bool = False) -> Tuple[str, LLMCallStats]:
        # This method is for AoTProcess's internal use (e.g., fallback)
        mode = "FALLBACK ONESHOT (AoTProcess)" if is_fallback else "ONESHOT (AoTProcess)"
        logging.info(f"--- Proceeding with {mode} Answer ---")
        logging.info(f"Using models: {', '.join(self.aot_config.main_model_names)}, LLMConfig: {self.direct_oneshot_llm_config}")

        response_content, stats = self.llm_client.call(
            prompt=problem_text, models=self.aot_config.main_model_names, config=self.direct_oneshot_llm_config
        )
        logging.debug(f"Direct {mode} response from {stats.model_name}:\n{response_content}")
        logging.info(f"LLM call ({stats.model_name}) for {mode}: Duration: {stats.call_duration_seconds:.2f}s, Tokens (C:{stats.completion_tokens}, P:{stats.prompt_tokens})")
        return response_content, stats

    def execute(self, problem_description: str, model_name: str, *args, **kwargs) -> None:
        overall_start_time = time.monotonic()
        self._solution = Solution()
        logging.info(f"AoTProcess executing for problem: {problem_description[:100]}... (model_name param: {model_name})")

        if not self.aot_processor:
            logging.critical("AoTProcessor not initialized within AoTProcess.")
            self._solution.final_answer = "Error: AoTProcessor not initialized in AoTProcess."
            # Ensure solution object tracks stats even for this failure path
            if self._solution.total_wall_clock_time_seconds is None:
                 self._solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time
            self._process_summary = self._generate_process_summary(self._solution)
            return

        aot_result_data, aot_processor_summary_str = self.aot_processor.run(problem_description)
        self._solution.aot_result = aot_result_data
        self._solution.aot_summary_output = aot_processor_summary_str # Summary from AoTProcessor's run

        if aot_result_data.succeeded:
            self._solution.final_answer = aot_result_data.final_answer
            self._solution.reasoning_trace = aot_result_data.reasoning_trace
        else:
            logging.warning(f"AoT process failed (Reason: {aot_result_data.final_answer}). Falling back to one-shot.")
            self._solution.aot_failed_and_fell_back = True
            fallback_answer, fallback_stats = self._run_direct_oneshot(problem_description, is_fallback=True)
            self._solution.final_answer = fallback_answer
            self._solution.fallback_call_stats = fallback_stats
            self._solution.reasoning_trace = aot_result_data.reasoning_trace

        if self._solution.total_wall_clock_time_seconds is None:
            self._solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time
        self._process_summary = self._generate_process_summary(self._solution)

    def get_result(self) -> Tuple[Optional[Solution], Optional[str]]: # Matches ReasoningProcess Any, but more specific
        return self._solution, self._process_summary

    def _generate_process_summary(self, solution: Solution) -> str:
        output_buffer = io.StringIO()
        output_buffer.write("\n" + "="*20 + " AoTProcess Execution Summary " + "="*20 + "\n")
        if solution.aot_result:
            output_buffer.write(f"AoT Process Attempted: Yes\n")
            if solution.aot_summary_output:
                 output_buffer.write(f"--- AoT Processor Internal Summary ---\n{solution.aot_summary_output}\n----------------------------------\n")

            if solution.aot_result.succeeded:
                 output_buffer.write(f"AoT Succeeded (Reported by AoT Processor): Yes\n")
            elif solution.aot_failed_and_fell_back:
                output_buffer.write(f"AoT FAILED and Fell Back to One-Shot: Yes (AoT Failure Reason: {solution.aot_result.final_answer})\n")
                if solution.fallback_call_stats:
                    sfb = solution.fallback_call_stats
                    output_buffer.write(f"  Fallback One-Shot Call ({sfb.model_name}): C={sfb.completion_tokens}, P={sfb.prompt_tokens}, Time={sfb.call_duration_seconds:.2f}s\n")
        else:
            output_buffer.write(f"AoT Process Was Not Fully Attempted (e.g., AoTProcessor initialization error).\n")
            if solution.final_answer:
                 output_buffer.write(f"Status/Error: {solution.final_answer}\n")

        output_buffer.write(f"Total Completion Tokens (AoTProcess): {solution.total_completion_tokens}\n")
        output_buffer.write(f"Total Prompt Tokens (AoTProcess): {solution.total_prompt_tokens}\n")
        output_buffer.write(f"Grand Total Tokens (AoTProcess): {solution.grand_total_tokens}\n")
        output_buffer.write(f"Total LLM Interaction Time (AoTProcess): {solution.total_llm_interaction_time_seconds:.2f}s\n")
        output_buffer.write(f"Total Wall-Clock Time (AoTProcess Execution): {solution.total_wall_clock_time_seconds:.2f}s\n")

        if solution.reasoning_trace:
            output_buffer.write("\n--- Reasoning Trace (from AoTProcess) ---\n")
            for step_line in solution.reasoning_trace: output_buffer.write(step_line + "\n")

        if solution.final_answer and ( (solution.aot_result and solution.aot_result.succeeded) or solution.aot_failed_and_fell_back ):
            output_buffer.write(f"\nFinal Answer (from AoTProcess):\n{solution.final_answer}\n")
        elif not solution.final_answer:
            output_buffer.write("\nFinal answer not successfully extracted by AoTProcess.\n")
        output_buffer.write("="*60 + "\n")
        return output_buffer.getvalue()
# End of AoTProcess class definition

class InteractiveAoTOrchestrator: # Renamed from AoTOrchestrator for clarity
    def __init__(self,
                 llm_client: LLMClient, # Passed in
                 trigger_mode: AotTriggerMode,
                 aot_config: AoTRunnerConfig,
                 direct_oneshot_llm_config: LLMConfig, # Passed LLMConfig
                 assessment_llm_config: LLMConfig, # Passed LLMConfig
                 aot_main_llm_config: LLMConfig, # Passed LLMConfig for AoTProcessor
                 direct_oneshot_model_names: List[str],
                 assessment_model_names: List[str],
                 use_heuristic_shortcut: bool = True,
                 heuristic_detector: Optional[HeuristicDetector] = None,
                 enable_rate_limiting: bool = True, # Kept for consistency, but not used for LLMClient init here
                 enable_audit_logging: bool = True): # Kept for consistency, but not used for LLMClient init here

        self.llm_client = llm_client # Use passed-in client
        self.trigger_mode = trigger_mode
        self.use_heuristic_shortcut = use_heuristic_shortcut
        self.direct_oneshot_llm_config = direct_oneshot_llm_config
        self.direct_oneshot_model_names = direct_oneshot_model_names
        self.aot_main_llm_config = aot_main_llm_config # Store for AoTProcess

        self.heuristic_detector = heuristic_detector

        self.complexity_assessor = None
        if self.trigger_mode == AotTriggerMode.ASSESS_FIRST:
            self.complexity_assessor = ComplexityAssessor(
                llm_client=self.llm_client,
                small_model_names=assessment_model_names,
                llm_config=assessment_llm_config, # Pass LLMConfig
                use_heuristic_shortcut=self.use_heuristic_shortcut,
                heuristic_detector=self.heuristic_detector
            )

        self.aot_process_instance: Optional[AoTProcess] = None
        if self.trigger_mode == AotTriggerMode.ALWAYS_AOT or self.trigger_mode == AotTriggerMode.ASSESS_FIRST:
            self.aot_process_instance = AoTProcess(
                llm_client=self.llm_client, # Pass shared client
                aot_config=aot_config,
                aot_main_llm_config=self.aot_main_llm_config, # Pass LLMConfig
                direct_oneshot_llm_config=self.direct_oneshot_llm_config # Pass LLMConfig
            )

    def _run_direct_oneshot(self, problem_text: str, is_fallback:bool = False) -> Tuple[str, LLMCallStats]:
        mode = "FALLBACK ONESHOT (Orchestrator)" if is_fallback else "ONESHOT (Orchestrator)"
        logging.info(f"--- Proceeding with {mode} Answer ---")
        logging.info(f"Using models: {', '.join(self.direct_oneshot_model_names)}, LLMConfig: {self.direct_oneshot_llm_config}")

        response_content, stats = self.llm_client.call(
            prompt=problem_text, models=self.direct_oneshot_model_names, config=self.direct_oneshot_llm_config
        )
        logging.debug(f"Direct {mode} response from {stats.model_name}:\n{response_content}")
        logging.info(f"LLM call ({stats.model_name}) for {mode}: Duration: {stats.call_duration_seconds:.2f}s, Tokens (C:{stats.completion_tokens}, P:{stats.prompt_tokens})")
        return response_content, stats

    def solve(self, problem_text: str, model_name_for_aot: str = "default_aot_model") -> Tuple[Solution, str]:
        overall_start_time = time.monotonic()
        orchestrator_solution = Solution()
        aot_process_execution_summary: Optional[str] = None

        if self.trigger_mode == AotTriggerMode.NEVER_AOT:
            logging.info("Trigger mode: NEVER_AOT. Orchestrator performing direct one-shot call.")
            final_answer, oneshot_stats = self._run_direct_oneshot(problem_text)
            orchestrator_solution.final_answer = final_answer
            orchestrator_solution.main_call_stats = oneshot_stats

        elif self.trigger_mode == AotTriggerMode.ALWAYS_AOT:
            logging.info("Trigger mode: ALWAYS_AOT. Orchestrator delegating to AoTProcess.")
            if not self.aot_process_instance:
                logging.critical("AoTProcess not initialized for ALWAYS_AOT mode.")
                orchestrator_solution.final_answer = "Error: AoTProcess not initialized for ALWAYS_AOT mode."
            else:
                self.aot_process_instance.execute(problem_description=problem_text, model_name=model_name_for_aot)
                aot_solution_obj, aot_process_execution_summary = self.aot_process_instance.get_result()

                if aot_solution_obj:
                    orchestrator_solution.final_answer = aot_solution_obj.final_answer
                    orchestrator_solution.reasoning_trace = aot_solution_obj.reasoning_trace
                    orchestrator_solution.aot_result = aot_solution_obj.aot_result
                    orchestrator_solution.aot_summary_output = aot_solution_obj.aot_summary_output
                    orchestrator_solution.aot_failed_and_fell_back = aot_solution_obj.aot_failed_and_fell_back
                    if aot_solution_obj.fallback_call_stats:
                        orchestrator_solution.fallback_call_stats = aot_solution_obj.fallback_call_stats
                else:
                    orchestrator_solution.final_answer = "Error: AoTProcess executed but returned no solution object."
                    logging.error("AoTProcess returned None for solution object in ALWAYS_AOT mode.")


        elif self.trigger_mode == AotTriggerMode.ASSESS_FIRST:
            logging.info("Trigger mode: ASSESS_FIRST. Orchestrator performing complexity assessment.")
            if not self.complexity_assessor:
                logging.critical("ComplexityAssessor not initialized for ASSESS_FIRST mode.")
                orchestrator_solution.final_answer = "Error: ComplexityAssessor not initialized."
            else:
                assessment_decision, assessment_stats = self.complexity_assessor.assess(problem_text)
                orchestrator_solution.assessment_stats = assessment_stats
                orchestrator_solution.assessment_decision = assessment_decision

                if assessment_decision == AssessmentDecision.ONE_SHOT:
                    logging.info("Assessment: ONE_SHOT. Orchestrator performing direct one-shot call.")
                    final_answer, oneshot_stats = self._run_direct_oneshot(problem_text)
                    orchestrator_solution.final_answer = final_answer
                    orchestrator_solution.main_call_stats = oneshot_stats
                elif assessment_decision == AssessmentDecision.ADVANCED_REASONING:
                    logging.info("Assessment: ADVANCED_REASONING. Orchestrator delegating to AoTProcess.")
                    if not self.aot_process_instance:
                        logging.critical("AoTProcess not initialized for ASSESS_FIRST mode (ADVANCED_REASONING path).")
                        orchestrator_solution.final_answer = "Error: AoTProcess not initialized for ADVANCED_REASONING path."
                    else:
                        self.aot_process_instance.execute(problem_description=problem_text, model_name=model_name_for_aot)
                        aot_solution_obj, aot_process_execution_summary = self.aot_process_instance.get_result()

                        if aot_solution_obj:
                            orchestrator_solution.final_answer = aot_solution_obj.final_answer
                            orchestrator_solution.reasoning_trace = aot_solution_obj.reasoning_trace
                            orchestrator_solution.aot_result = aot_solution_obj.aot_result
                            orchestrator_solution.aot_summary_output = aot_solution_obj.aot_summary_output
                            orchestrator_solution.aot_failed_and_fell_back = aot_solution_obj.aot_failed_and_fell_back
                            if aot_solution_obj.fallback_call_stats:
                                orchestrator_solution.fallback_call_stats = aot_solution_obj.fallback_call_stats
                        else:
                             orchestrator_solution.final_answer = "Error: AoTProcess (post-assessment) returned no solution object."
                             logging.error("AoTProcess returned None for solution object in ASSESS_FIRST (ADVANCED_REASONING path) mode.")
                else:
                    logging.error("Complexity assessment failed. Orchestrator attempting one-shot call as a last resort.")
                    
                    fallback_answer, fallback_stats = self._run_direct_oneshot(problem_text, is_fallback=True)
                    orchestrator_solution.final_answer = fallback_answer
                    orchestrator_solution.fallback_call_stats = fallback_stats 

        if orchestrator_solution.total_wall_clock_time_seconds is None:
             orchestrator_solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time
        final_summary_output = self._generate_overall_summary(orchestrator_solution, 
                                                              aot_process_specific_summary=aot_process_execution_summary)
        return orchestrator_solution, final_summary_output

    def _generate_overall_summary(self, solution: Solution, aot_process_specific_summary: Optional[str] = None) -> str:
        output_buffer = io.StringIO()
        output_buffer.write("\n" + "="*20 + " ORCHESTRATOR OVERALL SUMMARY " + "="*20 + "\n")
        output_buffer.write(f"Orchestrator Trigger Mode: {self.trigger_mode.value}\n")
        output_buffer.write(f"Heuristic Shortcut Option for Assessor: {self.use_heuristic_shortcut}\n")
        
        if solution.assessment_stats:
            s = solution.assessment_stats
            decision_for_summary = solution.assessment_decision.value if solution.assessment_decision else 'N/A'
            time_str = f"{s.call_duration_seconds:.2f}s" if s.call_duration_seconds is not None else "N/A"
            output_buffer.write(f"Assessment Phase ({s.model_name if s else 'N/A'}): Decision='{decision_for_summary}', C={s.completion_tokens if s else 'N/A'}, P={s.prompt_tokens if s else 'N/A'}, Time={time_str}\n")
        
        if solution.main_call_stats: 
            s = solution.main_call_stats
            output_buffer.write(f"Orchestrator Main Model Call (Direct ONESHOT path) ({s.model_name}): C={s.completion_tokens}, P={s.prompt_tokens}, Time={s.call_duration_seconds:.2f}s\n")
        
        if aot_process_specific_summary: 
            output_buffer.write("--- Delegated to AoTProcess ---\n")
            output_buffer.write(aot_process_specific_summary) 
            output_buffer.write("-------------------------------\n")
            if solution.aot_result and solution.aot_result.succeeded:
                 output_buffer.write(f"AoTProcess Reported Success: Yes\n")
            elif solution.aot_failed_and_fell_back:
                 output_buffer.write(f"AoTProcess Reported Failure and Fallback: Yes (Reason: {solution.aot_result.final_answer if solution.aot_result else 'N/A'})\n")
        
        elif solution.fallback_call_stats and not aot_process_specific_summary:
            sfb = solution.fallback_call_stats
            output_buffer.write(f"Orchestrator Fallback One-Shot Call (e.g. due to Assessment Error) ({sfb.model_name}): C={sfb.completion_tokens}, P={sfb.prompt_tokens}, Time={sfb.call_duration_seconds:.2f}s\n")

        output_buffer.write(f"\n--- Orchestrator Perspective Totals ---\n")
        output_buffer.write(f"Total Completion Tokens (Orchestrator): {solution.total_completion_tokens}\n")
        output_buffer.write(f"Total Prompt Tokens (Orchestrator): {solution.total_prompt_tokens}\n")
        output_buffer.write(f"Grand Total Tokens (Orchestrator): {solution.grand_total_tokens}\n")
        output_buffer.write(f"Total LLM Interaction Time (Orchestrator, sum of calls it's aware of): {solution.total_llm_interaction_time_seconds:.2f}s\n")
        output_buffer.write(f"Total Orchestrator Wall-Clock Time: {solution.total_wall_clock_time_seconds:.2f}s\n")

        if solution.reasoning_trace: 
            output_buffer.write("\n--- Reasoning Trace (from AoTProcess, if run) ---\n")
            for step_line in solution.reasoning_trace: output_buffer.write(step_line + "\n")
        
        if solution.final_answer: 
            output_buffer.write(f"\nFinal Answer (Returned by Orchestrator):\n{solution.final_answer}\n")
        else: 
            output_buffer.write("\nFinal answer was not successfully extracted or an error occurred (Orchestrator).\n")
        output_buffer.write("="*67 + "\n")
        return output_buffer.getvalue()
