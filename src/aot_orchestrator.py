import time
import logging
import io
from typing import List, Tuple, Optional # Import Optional

from src.aot_enums import AotTriggerMode, AssessmentDecision
from src.aot_dataclasses import LLMCallStats, AoTRunnerConfig, Solution
from src.llm_client import LLMClient
from src.complexity_assessor import ComplexityAssessor
from src.aot_processor import AoTProcessor
from src.heuristic_detector import HeuristicDetector # Import HeuristicDetector

class InteractiveAoTOrchestrator:
    def __init__(self,
                 trigger_mode: AotTriggerMode,
                 aot_config: AoTRunnerConfig,
                 direct_oneshot_model_names: List[str],
                 direct_oneshot_temperature: float,
                 assessment_model_names: List[str],
                 assessment_temperature: float,
                 api_key: str,
                 use_heuristic_shortcut: bool = True, # New parameter
                 heuristic_detector: Optional[HeuristicDetector] = None): # New parameter

        self.trigger_mode = trigger_mode
        self.use_heuristic_shortcut = use_heuristic_shortcut # Store it
        self.aot_config = aot_config
        self.direct_oneshot_model_names = direct_oneshot_model_names
        self.direct_oneshot_temperature = direct_oneshot_temperature
        self.llm_client = LLMClient(api_key=api_key)
        self.heuristic_detector = heuristic_detector # Store the passed detector

        self.complexity_assessor = None
        if self.trigger_mode == AotTriggerMode.ASSESS_FIRST:
            self.complexity_assessor = ComplexityAssessor(
                llm_client=self.llm_client,
                small_model_names=assessment_model_names,
                temperature=assessment_temperature,
                use_heuristic_shortcut=self.use_heuristic_shortcut, # Pass the new parameter
                heuristic_detector=self.heuristic_detector # Pass the detector
            )
        self.aot_processor = None
        if self.trigger_mode != AotTriggerMode.NEVER_AOT:
             self.aot_processor = AoTProcessor(llm_client=self.llm_client, config=self.aot_config)

    def _run_direct_oneshot(self, problem_text: str, is_fallback:bool = False) -> Tuple[str, LLMCallStats]:
        mode = "FALLBACK ONESHOT" if is_fallback else "ONESHOT"
        logging.info(f"--- Proceeding with {mode} Answer ---")
        logging.info(f"Using models: {', '.join(self.direct_oneshot_model_names)}, Temperature: {self.direct_oneshot_temperature}")
        
        response_content, stats = self.llm_client.call(
            prompt=problem_text, models=self.direct_oneshot_model_names, temperature=self.direct_oneshot_temperature
        )
        logging.debug(f"Direct {mode} response from {stats.model_name}:\n{response_content}")
        logging.info(f"LLM call ({stats.model_name}) for {mode}: Duration: {stats.call_duration_seconds:.2f}s, Tokens (C:{stats.completion_tokens}, P:{stats.prompt_tokens})")
        return response_content, stats

    def solve(self, problem_text: str) -> Tuple[Solution, str]: # Modified return type
        overall_start_time = time.monotonic()
        solution = Solution()

        if self.trigger_mode == AotTriggerMode.NEVER_AOT:
            logging.info("Trigger mode: NEVER_AOT. Direct one-shot call.")
            final_answer, oneshot_stats = self._run_direct_oneshot(problem_text)
            solution.final_answer = final_answer
            solution.main_call_stats = oneshot_stats

        elif self.trigger_mode == AotTriggerMode.ALWAYS_AOT:
            logging.info("Trigger mode: ALWAYS_AOT. Direct AoT process.")
            if not self.aot_processor: 
                logging.critical("AoTProcessor not initialized for ALWAYS_AOT mode.")
                raise Exception("AoTProcessor not initialized for ALWAYS_AOT mode.") # Should not happen if constructor is correct
            
            aot_result_data, aot_summary_str = self.aot_processor.run(problem_text) # Unpack the tuple
            solution.aot_result = aot_result_data
            solution.aot_summary_output = aot_summary_str # Store the summary
            if aot_result_data.succeeded:
                solution.final_answer = aot_result_data.final_answer
                solution.reasoning_trace = aot_result_data.reasoning_trace
            else:
                logging.warning(f"AoT process (ALWAYS_AOT mode) failed (Reason: {aot_result_data.final_answer}). Falling back to one-shot.")
                solution.aot_failed_and_fell_back = True
                fallback_answer, fallback_stats = self._run_direct_oneshot(problem_text, is_fallback=True)
                solution.final_answer = fallback_answer
                solution.fallback_call_stats = fallback_stats
                solution.reasoning_trace = aot_result_data.reasoning_trace 

        elif self.trigger_mode == AotTriggerMode.ASSESS_FIRST:
            if not self.complexity_assessor:
                logging.critical("ComplexityAssessor not initialized for ASSESS_FIRST mode.")
                raise Exception("ComplexityAssessor not initialized for ASSESS_FIRST mode.")
            
            assessment_decision, assessment_stats = self.complexity_assessor.assess(problem_text)
            solution.assessment_stats = assessment_stats

            if assessment_decision == AssessmentDecision.ONESHOT:
                logging.info("Assessment: ONESHOT. Direct one-shot call.")
                final_answer, oneshot_stats = self._run_direct_oneshot(problem_text)
                solution.final_answer = final_answer
                solution.main_call_stats = oneshot_stats
            elif assessment_decision == AssessmentDecision.AOT:
                logging.info("Assessment: AOT. Proceeding with AoT process.")
                if not self.aot_processor:
                    logging.critical("AoTProcessor not initialized for ASSESS_FIRST mode (AOT path).")
                    raise Exception("AoTProcessor not initialized for ASSESS_FIRST mode (AOT path).")
                
                aot_result_data, aot_summary_str = self.aot_processor.run(problem_text) # Unpack the tuple
                solution.aot_result = aot_result_data
                solution.aot_summary_output = aot_summary_str # Store the summary
                if aot_result_data.succeeded:
                    solution.final_answer = aot_result_data.final_answer
                    solution.reasoning_trace = aot_result_data.reasoning_trace
                else:
                    logging.warning(f"AoT process (after ASSESS_FIRST) failed (Reason: {aot_result_data.final_answer}). Falling back to one-shot.")
                    solution.aot_failed_and_fell_back = True
                    fallback_answer, fallback_stats = self._run_direct_oneshot(problem_text, is_fallback=True)
                    solution.final_answer = fallback_answer
                    solution.fallback_call_stats = fallback_stats
                    solution.reasoning_trace = aot_result_data.reasoning_trace
            else: # AssessmentDecision.ERROR
                logging.error("Complexity assessment failed. Attempting one-shot call as a last resort.")
                solution.aot_failed_and_fell_back = True # Mark as a form of fallback due to assessment error
                fallback_answer, fallback_stats = self._run_direct_oneshot(problem_text, is_fallback=True)
                solution.final_answer = fallback_answer
                solution.fallback_call_stats = fallback_stats

        solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time
        summary_output = self._generate_overall_summary(solution) # Call the new method
        return solution, summary_output # Return both

    def _generate_overall_summary(self, solution: Solution) -> str: # New method name and return type
        output_buffer = io.StringIO()
        output_buffer.write("\n" + "="*20 + " OVERALL SUMMARY " + "="*20 + "\n")
        output_buffer.write(f"Heuristic Shortcut Enabled: {self.use_heuristic_shortcut}\n") # Add this line
        if solution.assessment_stats:
            s = solution.assessment_stats
            output_buffer.write(f"Assessment ({s.model_name}): C={s.completion_tokens}, P={s.prompt_tokens}, Time={s.call_duration_seconds:.2f}s\n")
        if solution.main_call_stats: # Direct ONESHOT or ONESHOT after assessment
            s = solution.main_call_stats
            output_buffer.write(f"Main Model Call (Direct ONESHOT path) ({s.model_name}): C={s.completion_tokens}, P={s.prompt_tokens}, Time={s.call_duration_seconds:.2f}s\n")
        
        if solution.aot_result: # If AoT was attempted
            output_buffer.write(f"AoT Process Attempted: Yes\n")
            if solution.aot_result.succeeded:
                 output_buffer.write(f"AoT Succeeded (as per AoT summary): Yes\n")
            elif solution.aot_failed_and_fell_back:
                output_buffer.write(f"AoT FAILED and Fell Back to One-Shot: Yes (AoT Failure Reason: {solution.aot_result.final_answer})\n")
                if solution.fallback_call_stats:
                    sfb = solution.fallback_call_stats
                    output_buffer.write(f"Fallback One-Shot Call ({sfb.model_name}): C={sfb.completion_tokens}, P={sfb.prompt_tokens}, Time={sfb.call_duration_seconds:.2f}s\n")
            elif not solution.aot_result.succeeded and solution.fallback_call_stats and not solution.aot_failed_and_fell_back:
                 output_buffer.write(f"Process led to Fallback One-Shot (e.g. due to Assessment Error): Yes\n")
                 sfb = solution.fallback_call_stats
                 output_buffer.write(f"Fallback One-Shot Call ({sfb.model_name}): C={sfb.completion_tokens}, P={sfb.prompt_tokens}, Time={sfb.call_duration_seconds:.2f}s\n")

        output_buffer.write(f"Total Completion Tokens (All Calls): {solution.total_completion_tokens}\n")
        output_buffer.write(f"Total Prompt Tokens (All Calls): {solution.total_prompt_tokens}\n")
        output_buffer.write(f"Grand Total Tokens (All Calls): {solution.grand_total_tokens}\n")
        output_buffer.write(f"Total LLM Interaction Time (All Calls): {solution.total_llm_interaction_time_seconds:.2f}s\n")
        output_buffer.write(f"Total Process Wall-Clock Time: {solution.total_wall_clock_time_seconds:.2f}s\n")

        if solution.reasoning_trace:
            output_buffer.write("\n--- AoT Reasoning Trace (Filtered) ---\n")
            for step_line in solution.reasoning_trace: output_buffer.write(step_line + "\n")
        
        if solution.final_answer: 
            output_buffer.write(f"\nFinal Answer:\n{solution.final_answer}\n")
        else: 
            output_buffer.write("\nFinal answer was not successfully extracted or an error occurred.\n")
        output_buffer.write("="*57 + "\n")
        return output_buffer.getvalue()
