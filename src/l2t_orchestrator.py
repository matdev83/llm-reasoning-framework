import time
import logging
import io
from typing import List, Tuple

from src.aot_dataclasses import LLMCallStats
from src.aot_enums import AssessmentDecision # Import AssessmentDecision
from src.complexity_assessor import ComplexityAssessor # Import ComplexityAssessor
from src.llm_client import LLMClient
from src.l2t_dataclasses import L2TConfig, L2TResult, L2TSolution # Import L2TSolution
from src.l2t_enums import L2TTriggerMode # Import L2TTriggerMode
from src.l2t_processor import L2TProcessor

# Configure basic logging if no handlers are present
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class L2TOrchestrator:
    def __init__(self,
                 trigger_mode: L2TTriggerMode, # New parameter
                 l2t_config: L2TConfig,
                 direct_oneshot_model_names: List[str], # New parameter
                 direct_oneshot_temperature: float, # New parameter
                 assessment_model_names: List[str], # New parameter
                 assessment_temperature: float, # New parameter
                 api_key: str,
                 use_heuristic_shortcut: bool = True): # New parameter

        self.trigger_mode = trigger_mode
        self.use_heuristic_shortcut = use_heuristic_shortcut
        self.l2t_config = l2t_config
        self.direct_oneshot_model_names = direct_oneshot_model_names
        self.direct_oneshot_temperature = direct_oneshot_temperature
        self.llm_client = LLMClient(api_key=api_key)
        
        self.complexity_assessor = None
        if self.trigger_mode == L2TTriggerMode.ASSESS_FIRST:
            self.complexity_assessor = ComplexityAssessor(
                llm_client=self.llm_client,
                small_model_names=assessment_model_names,
                temperature=assessment_temperature,
                use_heuristic_shortcut=self.use_heuristic_shortcut
            )
        
        self.l2t_processor = None
        if self.trigger_mode != L2TTriggerMode.NEVER_L2T:
             self.l2t_processor = L2TProcessor(llm_client=self.llm_client, config=self.l2t_config)

    def _generate_l2t_summary_from_result(self, result: L2TResult) -> str:
        """Generates a summary string for a single L2TResult object."""
        output_buffer = io.StringIO()
        output_buffer.write("\n" + "="*20 + " L2T PROCESS SUMMARY " + "="*20 + "\n")
        output_buffer.write(f"L2T Succeeded: {result.succeeded}\n")
        if result.error_message:
            output_buffer.write(f"Error Message: {result.error_message}\n")

        output_buffer.write(f"Total LLM Calls: {result.total_llm_calls}\n")
        output_buffer.write(f"Total Completion Tokens: {result.total_completion_tokens}\n")
        output_buffer.write(f"Total Prompt Tokens: {result.total_prompt_tokens}\n")
        output_buffer.write(f"Grand Total Tokens (All L2T Calls): {result.total_completion_tokens + result.total_prompt_tokens}\n")
        output_buffer.write(f"Total L2T LLM Interaction Time: {result.total_llm_interaction_time_seconds:.2f}s\n")
        output_buffer.write(f"Total L2T Process Wall-Clock Time: {result.total_process_wall_clock_time_seconds:.2f}s\n")

        if result.reasoning_graph and result.reasoning_graph.nodes:
            output_buffer.write(f"Number of nodes in graph: {len(result.reasoning_graph.nodes)}\n")
            if result.reasoning_graph.root_node_id:
                 output_buffer.write(f"Root node ID: {result.reasoning_graph.root_node_id[:8]}...\n")

        if result.final_answer:
            output_buffer.write(f"\nFinal Answer:\n{result.final_answer}\n")
        elif not result.succeeded:
            output_buffer.write("\nFinal answer was not successfully obtained.\n")

        output_buffer.write("="*59 + "\n")
        return output_buffer.getvalue()

    def _run_direct_oneshot(self, problem_text: str, is_fallback:bool = False) -> Tuple[str, LLMCallStats]:
        mode = "FALLBACK ONESHOT" if is_fallback else "ONESHOT"
        logger.info(f"--- Proceeding with {mode} Answer ---")
        logger.info(f"Using models: {', '.join(self.direct_oneshot_model_names)}, Temperature: {self.direct_oneshot_temperature}")
        
        response_content, stats = self.llm_client.call(
            prompt=problem_text, models=self.direct_oneshot_model_names, temperature=self.direct_oneshot_temperature
        )
        logger.debug(f"Direct {mode} response from {stats.model_name}:\n{response_content}")
        logger.info(f"LLM call ({stats.model_name}) for {mode}: Duration: {stats.call_duration_seconds:.2f}s, Tokens (C:{stats.completion_tokens}, P:{stats.prompt_tokens})")
        return response_content, stats

    def solve(self, problem_text: str) -> Tuple[L2TSolution, str]: # Modified return type
        overall_start_time = time.monotonic()
        solution = L2TSolution() # Use L2TSolution

        if self.trigger_mode == L2TTriggerMode.NEVER_L2T:
            logger.info("Trigger mode: NEVER_L2T. Direct one-shot call.")
            final_answer, oneshot_stats = self._run_direct_oneshot(problem_text)
            solution.final_answer = final_answer
            solution.main_call_stats = oneshot_stats

        elif self.trigger_mode == L2TTriggerMode.ALWAYS_L2T:
            logger.info("Trigger mode: ALWAYS_L2T. Direct L2T process.")
            if not self.l2t_processor: 
                logger.critical("L2TProcessor not initialized for ALWAYS_L2T mode.")
                raise Exception("L2TProcessor not initialized for ALWAYS_L2T mode.")
            l2t_result_data = self.l2t_processor.run(problem_text)
            solution.l2t_result = l2t_result_data
            solution.l2t_summary_output = self._generate_l2t_summary_from_result(l2t_result_data) # Generate summary from result
            if l2t_result_data.succeeded:
                solution.final_answer = l2t_result_data.final_answer
            else:
                logger.warning(f"L2T process (ALWAYS_L2T mode) failed (Reason: {l2t_result_data.error_message}). Falling back to one-shot.")
                solution.l2t_failed_and_fell_back = True
                fallback_answer, fallback_stats = self._run_direct_oneshot(problem_text, is_fallback=True)
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
                final_answer, oneshot_stats = self._run_direct_oneshot(problem_text)
                solution.final_answer = final_answer
                solution.main_call_stats = oneshot_stats
            elif assessment_decision == AssessmentDecision.AOT: # AOT here means L2T for this orchestrator
                logger.info("Assessment: L2T. Proceeding with L2T process.")
                if not self.l2t_processor:
                    logger.critical("L2TProcessor not initialized for ASSESS_FIRST mode (L2T path).")
                    raise Exception("L2TProcessor not initialized for ASSESS_FIRST mode (L2T path).")
                
                l2t_result_data = self.l2t_processor.run(problem_text)
                solution.l2t_result = l2t_result_data
                solution.l2t_summary_output = self._generate_l2t_summary_from_result(l2t_result_data) # Generate summary from result
                if l2t_result_data.succeeded:
                    solution.final_answer = l2t_result_data.final_answer
                else:
                    logger.warning(f"L2T process (after ASSESS_FIRST) failed (Reason: {l2t_result_data.error_message}). Falling back to one-shot.")
                    solution.l2t_failed_and_fell_back = True
                    fallback_answer, fallback_stats = self._run_direct_oneshot(problem_text, is_fallback=True)
                    solution.final_answer = fallback_answer
                    solution.fallback_call_stats = fallback_stats
            else: # AssessmentDecision.ERROR
                logger.error("Complexity assessment failed. Attempting one-shot call as a last resort.")
                solution.l2t_failed_and_fell_back = True # Mark as a form of fallback due to assessment error
                fallback_answer, fallback_stats = self._run_direct_oneshot(problem_text, is_fallback=True)
                solution.final_answer = fallback_answer
                solution.fallback_call_stats = fallback_stats

        solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time
        summary_output = self._generate_overall_summary(solution) # Call the new method
        return solution, summary_output # Return both

    def _generate_overall_summary(self, solution: L2TSolution) -> str: # New method name and return type
        output_buffer = io.StringIO()
        output_buffer.write("\n" + "="*20 + " OVERALL L2T ORCHESTRATOR SUMMARY " + "="*20 + "\n")
        output_buffer.write(f"Trigger Mode: {self.trigger_mode.value.upper()}\n")
        output_buffer.write(f"Heuristic Shortcut Enabled: {self.use_heuristic_shortcut}\n")

        if solution.assessment_stats:
            s = solution.assessment_stats
            output_buffer.write(f"Assessment ({s.model_name}): Decision={solution.assessment_decision.value if solution.assessment_decision else 'N/A'}, C={s.completion_tokens}, P={s.prompt_tokens}, Time={s.call_duration_seconds:.2f}s\n")
        
        if solution.main_call_stats: # Direct ONESHOT or ONESHOT after assessment
            s = solution.main_call_stats
            output_buffer.write(f"Main Model Call (Direct ONESHOT path) ({s.model_name}): C={s.completion_tokens}, P={s.prompt_tokens}, Time={s.call_duration_seconds:.2f}s\n")
        
        if solution.l2t_result: # If L2T was attempted
            output_buffer.write(f"L2T Process Attempted: Yes\n")
            if solution.l2t_result.succeeded:
                 output_buffer.write(f"L2T Succeeded (as per L2T summary): Yes\n")
            elif solution.l2t_failed_and_fell_back:
                output_buffer.write(f"L2T FAILED and Fell Back to One-Shot: Yes (L2T Failure Reason: {solution.l2t_result.error_message})\n")
                if solution.fallback_call_stats:
                    sfb = solution.fallback_call_stats
                    output_buffer.write(f"Fallback One-Shot Call ({sfb.model_name}): C={sfb.completion_tokens}, P={sfb.prompt_tokens}, Time={sfb.call_duration_seconds:.2f}s\n")
            elif not solution.l2t_result.succeeded and solution.fallback_call_stats and solution.assessment_decision == AssessmentDecision.ERROR:
                 output_buffer.write(f"Process led to Fallback One-Shot (e.g. due to Assessment Error): Yes\n")
                 sfb = solution.fallback_call_stats
                 output_buffer.write(f"Fallback One-Shot Call ({sfb.model_name}): C={sfb.completion_tokens}, P={sfb.prompt_tokens}, Time={sfb.call_duration_seconds:.2f}s\n")
        # Handle case where L2T was never attempted but fallback occurred due to assessment error
        elif solution.assessment_decision == AssessmentDecision.ERROR and solution.fallback_call_stats and not solution.l2t_result:
            output_buffer.write(f"Process led to Fallback One-Shot (due to Assessment Error, L2T not attempted): Yes\n")
            sfb = solution.fallback_call_stats
            output_buffer.write(f"Fallback One-Shot Call ({sfb.model_name}): C={sfb.completion_tokens}, P={sfb.prompt_tokens}, Time={sfb.call_duration_seconds:.2f}s\n")


        output_buffer.write(f"Total Completion Tokens (All Calls): {solution.total_completion_tokens}\n")
        output_buffer.write(f"Total Prompt Tokens (All Calls): {solution.total_prompt_tokens}\n")
        output_buffer.write(f"Grand Total Tokens (All Calls): {solution.grand_total_tokens}\n")
        output_buffer.write(f"Total LLM Interaction Time (All Calls): {solution.total_llm_interaction_time_seconds:.2f}s\n")
        output_buffer.write(f"Total Process Wall-Clock Time: {solution.total_wall_clock_time_seconds:.2f}s\n")

        if solution.final_answer: 
            output_buffer.write(f"\nFinal Answer:\n{solution.final_answer}\n")
        else: 
            output_buffer.write("\nFinal answer was not successfully extracted or an error occurred.\n")
        output_buffer.write("="*60 + "\n")
        return output_buffer.getvalue()
