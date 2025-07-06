import time
import logging
import io
from typing import List, Tuple, Optional, Any

from src.reasoning_process import ReasoningProcess
from src.llm_client import LLMClient
from src.llm_config import LLMConfig
from src.hybrid.dataclasses import HybridConfig, HybridSolution, HybridResult, LLMCallStats
from src.hybrid.processor import HybridProcessor

logger = logging.getLogger(__name__)

class HybridProcess(ReasoningProcess):
    def __init__(self,
                 hybrid_config: HybridConfig,
                 direct_oneshot_model_names: List[str], # For fallback
                 direct_oneshot_temperature: float,    # For fallback
                 api_key: str,
                 enable_rate_limiting: bool = True,
                 enable_audit_logging: bool = True):

        self.hybrid_config = hybrid_config
        self.direct_oneshot_model_names = direct_oneshot_model_names
        self.direct_oneshot_temperature = direct_oneshot_temperature

        self.llm_client = LLMClient( # HybridProcess has its own LLMClient for fallbacks etc.
            api_key=api_key,
            enable_rate_limiting=enable_rate_limiting,
            enable_audit_logging=enable_audit_logging
        )
        self.hybrid_processor = HybridProcessor(llm_client=self.llm_client, config=self.hybrid_config) # Processor uses the same client

        self._solution: Optional[HybridSolution] = None
        self._process_summary: Optional[str] = None

    def _run_direct_oneshot(self, problem_text: str, is_fallback:bool = False) -> Tuple[str, LLMCallStats]:
        mode = "FALLBACK ONESHOT (HybridProcess)" if is_fallback else "ONESHOT (HybridProcess)"
        logger.info(f"--- Proceeding with {mode} Answer ---")
        logger.info(f"Using models: {', '.join(self.direct_oneshot_model_names)}, Temperature: {self.direct_oneshot_temperature}")

        # Create LLMConfig for oneshot call
        oneshot_config = LLMConfig(
            temperature=self.direct_oneshot_temperature,
            max_tokens=2048  # Default max tokens for oneshot
        )
        
        response_content, stats = self.llm_client.call(
            prompt=problem_text,
            models=self.direct_oneshot_model_names,
            config=oneshot_config
        )
        logger.debug(f"Direct {mode} response from {stats.model_name}:\n{response_content}")
        logger.info(f"LLM call ({stats.model_name}) for {mode}: Duration: {stats.call_duration_seconds:.2f}s, Tokens (C:{stats.completion_tokens}, P:{stats.prompt_tokens})")
        return response_content, stats

    def execute(self, problem_description: str, model_name: str, *args, **kwargs) -> None:
        # model_name param is for compatibility with ReasoningProcess, actual models are in hybrid_config
        overall_start_time = time.monotonic()
        self._solution = HybridSolution()
        logger.info(f"HybridProcess executing for problem: {problem_description[:100]}...")

        if not self.hybrid_processor:
            logger.critical("HybridProcessor not initialized within HybridProcess.")
            self._solution.final_answer = "Error: HybridProcessor not initialized."
            # Ensure solution object tracks stats even for this failure path
            if self._solution.total_wall_clock_time_seconds is None:
                 self._solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time
            self._process_summary = self._generate_process_summary(self._solution)
            return

        hybrid_result_data = self.hybrid_processor.run(problem_description)
        self._solution.hybrid_result = hybrid_result_data

        # Generate a processor summary (can be simple for now)
        processor_summary_buffer = io.StringIO()
        processor_summary_buffer.write(f"Hybrid Processor Attempted: Yes\n")
        if hybrid_result_data.succeeded:
            processor_summary_buffer.write(f"Hybrid Processor Succeeded: Yes\n")
            if hybrid_result_data.reasoning_call_stats:
                rs = hybrid_result_data.reasoning_call_stats
                processor_summary_buffer.write(f"  Reasoning Call ({rs.model_name}): C={rs.completion_tokens}, P={rs.prompt_tokens}, Time={rs.call_duration_seconds:.2f}s\n")
            if hybrid_result_data.response_call_stats:
                resp_s = hybrid_result_data.response_call_stats
                processor_summary_buffer.write(f"  Response Call ({resp_s.model_name}): C={resp_s.completion_tokens}, P={resp_s.prompt_tokens}, Time={resp_s.call_duration_seconds:.2f}s\n")
            if hybrid_result_data.extracted_reasoning:
                 processor_summary_buffer.write(f"  Extracted Reasoning Length: {len(hybrid_result_data.extracted_reasoning)} chars\n")
        else:
            processor_summary_buffer.write(f"Hybrid Processor Failed: Yes (Reason: {hybrid_result_data.error_message})\n")
        self._solution.hybrid_summary_output = processor_summary_buffer.getvalue()


        if hybrid_result_data.succeeded:
            self._solution.final_answer = hybrid_result_data.final_answer
            # self._solution.reasoning_trace could potentially store the extracted_reasoning
            if hybrid_result_data.extracted_reasoning:
                self._solution.reasoning_trace = [f"Extracted Reasoning:\n{hybrid_result_data.extracted_reasoning}"]
        else:
            logger.warning(f"Hybrid process failed (Reason: {hybrid_result_data.error_message}). Falling back to one-shot.")
            self._solution.hybrid_failed_and_fell_back = True
            fallback_answer, fallback_stats = self._run_direct_oneshot(problem_description, is_fallback=True)
            self._solution.final_answer = fallback_answer
            self._solution.fallback_call_stats = fallback_stats
            # Preserve any partial reasoning if available from the failed hybrid attempt
            if hybrid_result_data.extracted_reasoning:
                 self._solution.reasoning_trace = [f"Extracted Reasoning (from failed Hybrid attempt):\n{hybrid_result_data.extracted_reasoning}"]


        if self._solution.total_wall_clock_time_seconds is None:
            self._solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time
        self._process_summary = self._generate_process_summary(self._solution)

    def get_result(self) -> Tuple[Optional[HybridSolution], Optional[str]]:
        return self._solution, self._process_summary

    def _generate_process_summary(self, solution: HybridSolution) -> str:
        output_buffer = io.StringIO()
        output_buffer.write("\n" + "="*20 + " HybridProcess Execution Summary " + "="*20 + "\n")

        if solution.hybrid_result:
            output_buffer.write(f"Hybrid Process Attempted: Yes\n")
            if solution.hybrid_summary_output:
                 output_buffer.write(f"--- Hybrid Processor Internal Summary ---\n{solution.hybrid_summary_output}\n----------------------------------\n")

            if solution.hybrid_result.succeeded:
                 output_buffer.write(f"Hybrid Succeeded (Reported by Hybrid Processor): Yes\n")
            elif solution.hybrid_failed_and_fell_back:
                output_buffer.write(f"Hybrid FAILED and Fell Back to One-Shot: Yes (Hybrid Failure Reason: {solution.hybrid_result.error_message})\n")
                if solution.fallback_call_stats:
                    sfb = solution.fallback_call_stats
                    output_buffer.write(f"  Fallback One-Shot Call ({sfb.model_name}): C={sfb.completion_tokens}, P={sfb.prompt_tokens}, Time={sfb.call_duration_seconds:.2f}s\n")
        else:
            output_buffer.write(f"Hybrid Process Was Not Fully Attempted (e.g., HybridProcessor initialization error).\n")
            if solution.final_answer:
                 output_buffer.write(f"Status/Error: {solution.final_answer}\n")

        output_buffer.write(f"Total Completion Tokens (HybridProcess): {solution.total_completion_tokens}\n")
        output_buffer.write(f"Total Prompt Tokens (HybridProcess): {solution.total_prompt_tokens}\n")
        output_buffer.write(f"Grand Total Tokens (HybridProcess): {solution.grand_total_tokens}\n")
        output_buffer.write(f"Total LLM Interaction Time (HybridProcess): {solution.total_llm_interaction_time_seconds:.2f}s\n")
        output_buffer.write(f"Total Wall-Clock Time (HybridProcess Execution): {solution.total_wall_clock_time_seconds:.2f}s\n")

        if solution.reasoning_trace: # This now contains extracted reasoning
            output_buffer.write("\n--- Reasoning Trace (from HybridProcess) ---\n")
            for step_line in solution.reasoning_trace: output_buffer.write(step_line + "\n")

        if solution.final_answer and ( (solution.hybrid_result and solution.hybrid_result.succeeded) or solution.hybrid_failed_and_fell_back ):
            output_buffer.write(f"\nFinal Answer (from HybridProcess):\n{solution.final_answer}\n")
        elif not solution.final_answer:
            output_buffer.write("\nFinal answer not successfully extracted by HybridProcess.\n")
        output_buffer.write("="*60 + "\n")
        return output_buffer.getvalue()


from src.aot.enums import AssessmentDecision # Reusing for now, might need a HybridAssessmentDecision
from src.complexity_assessor import ComplexityAssessor
from src.heuristic_detector import HeuristicDetector
from src.hybrid.enums import HybridTriggerMode # This enum will be created in the next step

# Assume HybridProcess, HybridConfig, HybridSolution, LLMCallStats are already imported in this file.
# Assume LLMClient is also imported.

class HybridOrchestrator:
    def __init__(self,
                 trigger_mode: HybridTriggerMode,
                 hybrid_config: HybridConfig,
                 direct_oneshot_model_names: List[str],
                 direct_oneshot_temperature: float,
                 api_key: str,
                 # Parameters for complexity assessment (optional)
                 assessment_model_names: Optional[List[str]] = None,
                 assessment_temperature: Optional[float] = None,
                 use_heuristic_shortcut: bool = True, # For assessor
                 heuristic_detector: Optional[HeuristicDetector] = None, # For assessor
                 enable_rate_limiting: bool = True,
                 enable_audit_logging: bool = True):

        self.trigger_mode = trigger_mode
        self.use_heuristic_shortcut = use_heuristic_shortcut
        self.direct_oneshot_model_names = direct_oneshot_model_names
        self.direct_oneshot_temperature = direct_oneshot_temperature

        self.llm_client = LLMClient( # For orchestrator's direct calls & assessor
            api_key=api_key,
            enable_rate_limiting=enable_rate_limiting,
            enable_audit_logging=enable_audit_logging
        )
        self.heuristic_detector = heuristic_detector

        self.complexity_assessor = None
        if self.trigger_mode == HybridTriggerMode.ASSESS_FIRST_HYBRID:
            if not assessment_model_names or assessment_temperature is None:
                logger.warning("ASSESS_FIRST trigger mode for HybridOrchestrator requires assessment_model_names and assessment_temperature.")
                # Fallback to ALWAYS_HYBRID or handle error, for now, it will fail if assessor is used without params
            else:
                assessment_config = LLMConfig(temperature=assessment_temperature)
                self.complexity_assessor = ComplexityAssessor(
                    llm_client=self.llm_client,
                    small_model_names=assessment_model_names,
                    llm_config=assessment_config,
                    use_heuristic_shortcut=self.use_heuristic_shortcut,
                    heuristic_detector=self.heuristic_detector
                )

        self.hybrid_process_instance: Optional[HybridProcess] = None
        # Instantiate HybridProcess if Hybrid is a possibility based on trigger_mode
        if self.trigger_mode == HybridTriggerMode.ALWAYS_HYBRID or self.trigger_mode == HybridTriggerMode.ASSESS_FIRST_HYBRID:
            self.hybrid_process_instance = HybridProcess(
                hybrid_config=hybrid_config,
                direct_oneshot_model_names=direct_oneshot_model_names,
                direct_oneshot_temperature=direct_oneshot_temperature,
                api_key=api_key,
                enable_rate_limiting=enable_rate_limiting,
                enable_audit_logging=enable_audit_logging
            )

    def _run_direct_oneshot(self, problem_text: str, is_fallback:bool = False) -> Tuple[str, LLMCallStats]:
        mode = "FALLBACK ONESHOT (HybridOrchestrator)" if is_fallback else "ONESHOT (HybridOrchestrator)"
        logger.info(f"--- Proceeding with {mode} Answer ---")
        logger.info(f"Using models: {', '.join(self.direct_oneshot_model_names)}, Temperature: {self.direct_oneshot_temperature}")

        # Create LLMConfig for oneshot call
        oneshot_config = LLMConfig(
            temperature=self.direct_oneshot_temperature,
            max_tokens=2048  # Default max tokens for oneshot
        )
        
        response_content, stats = self.llm_client.call(
            prompt=problem_text, 
            models=self.direct_oneshot_model_names, 
            config=oneshot_config
        )
        logger.debug(f"Direct {mode} response from {stats.model_name}:\n{response_content}")
        logger.info(f"LLM call ({stats.model_name}) for {mode}: Duration: {stats.call_duration_seconds:.2f}s, Tokens (C:{stats.completion_tokens}, P:{stats.prompt_tokens})")
        return response_content, stats

    def solve(self, problem_text: str, model_name_for_hybrid: str = "default_hybrid_model") -> Tuple[HybridSolution, str]:
        # model_name_for_hybrid is for consistency, actual models are in hybrid_config
        overall_start_time = time.monotonic()
        orchestrator_solution = HybridSolution()
        hybrid_process_execution_summary: Optional[str] = None

        if self.trigger_mode == HybridTriggerMode.NEVER_HYBRID:
            logger.info("Trigger mode: NEVER_HYBRID. Orchestrator performing direct one-shot call.")
            final_answer, oneshot_stats = self._run_direct_oneshot(problem_text)
            orchestrator_solution.final_answer = final_answer
            orchestrator_solution.main_call_stats = oneshot_stats

        elif self.trigger_mode == HybridTriggerMode.ALWAYS_HYBRID:
            logger.info("Trigger mode: ALWAYS_HYBRID. Orchestrator delegating to HybridProcess.")
            if not self.hybrid_process_instance:
                logger.critical("HybridProcess not initialized for ALWAYS_HYBRID mode.")
                orchestrator_solution.final_answer = "Error: HybridProcess not initialized for ALWAYS_HYBRID mode."
            else:
                self.hybrid_process_instance.execute(problem_description=problem_text, model_name=model_name_for_hybrid)
                hybrid_solution_obj, hybrid_process_execution_summary = self.hybrid_process_instance.get_result()

                if hybrid_solution_obj:
                    # Copy fields instead of replacing the orchestrator_solution object
                    orchestrator_solution.hybrid_result = hybrid_solution_obj.hybrid_result
                    orchestrator_solution.final_answer = hybrid_solution_obj.final_answer
                    orchestrator_solution.reasoning_trace = hybrid_solution_obj.reasoning_trace
                    # main_call_stats is for orchestrator's direct one-shot, not applicable here
                    orchestrator_solution.fallback_call_stats = hybrid_solution_obj.fallback_call_stats # if HybridProcess fell back
                    orchestrator_solution.hybrid_failed_and_fell_back = hybrid_solution_obj.hybrid_failed_and_fell_back
                    orchestrator_solution.hybrid_summary_output = hybrid_solution_obj.hybrid_summary_output
                    # assessment_stats and assessment_decision on orchestrator_solution are preserved
                else:
                    orchestrator_solution.final_answer = "Error: HybridProcess executed but returned no solution object."
                    logger.error("HybridProcess returned None for solution object in ALWAYS_HYBRID mode.")


        elif self.trigger_mode == HybridTriggerMode.ASSESS_FIRST_HYBRID:
            logger.info("Trigger mode: ASSESS_FIRST. Orchestrator performing complexity assessment.")
            if not self.complexity_assessor:
                logger.critical("ComplexityAssessor not initialized for ASSESS_FIRST mode (Hybrid).")
                orchestrator_solution.final_answer = "Error: ComplexityAssessor not initialized for Hybrid."
                 # Attempt direct one-shot as a final fallback if assessor is missing
                logger.warning("Falling back to direct one-shot due to missing ComplexityAssessor in ASSESS_FIRST mode.")
                final_answer, oneshot_stats = self._run_direct_oneshot(problem_text, is_fallback=True)
                orchestrator_solution.final_answer = final_answer
                orchestrator_solution.fallback_call_stats = oneshot_stats

            else:
                assessment_decision, assessment_stats = self.complexity_assessor.assess(problem_text)
                orchestrator_solution.assessment_stats = assessment_stats
                orchestrator_solution.assessment_decision = assessment_decision

                if assessment_decision == AssessmentDecision.ONE_SHOT:
                    logger.info("Assessment: ONE_SHOT. Orchestrator performing direct one-shot call for Hybrid path.")
                    final_answer, oneshot_stats = self._run_direct_oneshot(problem_text)
                    orchestrator_solution.final_answer = final_answer
                    orchestrator_solution.main_call_stats = oneshot_stats
                elif assessment_decision == AssessmentDecision.ADVANCED_REASONING: # This means use Hybrid
                    logger.info("Assessment: ADVANCED_REASONING. Orchestrator delegating to HybridProcess.")
                    if not self.hybrid_process_instance:
                        logger.critical("HybridProcess not initialized for ASSESS_FIRST mode (ADVANCED_REASONING path).")
                        orchestrator_solution.final_answer = "Error: HybridProcess not initialized for ADVANCED_REASONING path."
                    else:
                        self.hybrid_process_instance.execute(problem_description=problem_text, model_name=model_name_for_hybrid)
                        hybrid_solution_obj, hybrid_process_execution_summary = self.hybrid_process_instance.get_result()

                        if hybrid_solution_obj:
                            # Copy fields, preserving assessment_stats and assessment_decision
                            orchestrator_solution.hybrid_result = hybrid_solution_obj.hybrid_result
                            orchestrator_solution.final_answer = hybrid_solution_obj.final_answer
                            orchestrator_solution.reasoning_trace = hybrid_solution_obj.reasoning_trace
                            orchestrator_solution.fallback_call_stats = hybrid_solution_obj.fallback_call_stats
                            orchestrator_solution.hybrid_failed_and_fell_back = hybrid_solution_obj.hybrid_failed_and_fell_back
                            orchestrator_solution.hybrid_summary_output = hybrid_solution_obj.hybrid_summary_output
                            # assessment_stats and assessment_decision on orchestrator_solution are preserved
                        else:
                             orchestrator_solution.final_answer = "Error: HybridProcess (post-assessment) returned no solution object."
                             logger.error("HybridProcess returned None for solution object in ASSESS_FIRST (ADVANCED_REASONING path) mode.")
                else:  # AssessmentDecision.ERROR
                    logger.error("Complexity assessment failed for Hybrid. Orchestrator attempting one-shot call as a last resort.")
                    fallback_answer, fallback_stats = self._run_direct_oneshot(problem_text, is_fallback=True)
                    orchestrator_solution.final_answer = fallback_answer
                    orchestrator_solution.fallback_call_stats = fallback_stats

        if orchestrator_solution.total_wall_clock_time_seconds is None:
             orchestrator_solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time

        # Ensure the summary from HybridProcess is passed to the orchestrator's summary generation
        final_summary_output = self._generate_overall_summary(orchestrator_solution,
                                                              hybrid_process_specific_summary=hybrid_process_execution_summary)
        return orchestrator_solution, final_summary_output

    def _generate_overall_summary(self, solution: HybridSolution, hybrid_process_specific_summary: Optional[str] = None) -> str:
        output_buffer = io.StringIO()
        output_buffer.write("\n" + "="*20 + " HYBRID ORCHESTRATOR OVERALL SUMMARY " + "="*20 + "\n")
        output_buffer.write(f"Orchestrator Trigger Mode: {self.trigger_mode.value}\n") # Assumes HybridTriggerMode has .value

        if self.trigger_mode == HybridTriggerMode.ASSESS_FIRST_HYBRID:
            output_buffer.write(f"Heuristic Shortcut Option for Assessor: {self.use_heuristic_shortcut}\n")
            if solution.assessment_stats:
                s = solution.assessment_stats
                decision_for_summary = solution.assessment_decision.value if solution.assessment_decision else 'N/A'
                time_str = f"{s.call_duration_seconds:.2f}s" if s.call_duration_seconds is not None else "N/A"
                output_buffer.write(f"Assessment Phase ({s.model_name if s else 'N/A'}): Decision='{decision_for_summary}', C={s.completion_tokens if s else 'N/A'}, P={s.prompt_tokens if s else 'N/A'}, Time={time_str}\n")

        if solution.main_call_stats: # Direct one-shot by orchestrator
            s = solution.main_call_stats
            output_buffer.write(f"Orchestrator Main Model Call (Direct ONESHOT path) ({s.model_name}): C={s.completion_tokens}, P={s.prompt_tokens}, Time={s.call_duration_seconds:.2f}s\n")

        if hybrid_process_specific_summary:
            # This is the summary from HybridProcess.execute()
            output_buffer.write("--- Delegated to HybridProcess ---\n")
            output_buffer.write(hybrid_process_specific_summary)
            output_buffer.write("-------------------------------\n")
            # Redundant info if already in hybrid_process_specific_summary, but kept for consistency with AoT/L2T
            if solution.hybrid_result and solution.hybrid_result.succeeded:
                 output_buffer.write(f"HybridProcess Reported Success: Yes\n")
            elif solution.hybrid_failed_and_fell_back:
                 output_buffer.write(f"HybridProcess Reported Failure and Fallback: Yes (Reason: {solution.hybrid_result.error_message if solution.hybrid_result else 'N/A'})\n")

        elif solution.fallback_call_stats and not hybrid_process_specific_summary: # Fallback by orchestrator itself (e.g. assessment error)
            sfb = solution.fallback_call_stats
            output_buffer.write(f"Orchestrator Fallback One-Shot Call ({sfb.model_name}): C={sfb.completion_tokens}, P={sfb.prompt_tokens}, Time={sfb.call_duration_seconds:.2f}s\n")

        output_buffer.write(f"\n--- Orchestrator Perspective Totals ---\n")
        output_buffer.write(f"Total Completion Tokens (Orchestrator): {solution.total_completion_tokens}\n")
        output_buffer.write(f"Total Prompt Tokens (Orchestrator): {solution.total_prompt_tokens}\n")
        output_buffer.write(f"Grand Total Tokens (Orchestrator): {solution.grand_total_tokens}\n")
        output_buffer.write(f"Total LLM Interaction Time (Orchestrator, sum of calls it's aware of): {solution.total_llm_interaction_time_seconds:.2f}s\n")
        output_buffer.write(f"Total Orchestrator Wall-Clock Time: {solution.total_wall_clock_time_seconds:.2f}s\n")

        if solution.reasoning_trace: # Contains extracted reasoning if HybridProcess ran
            output_buffer.write("\n--- Reasoning Trace (from HybridProcess, if run) ---\n")
            for step_line in solution.reasoning_trace: output_buffer.write(step_line + "\n")

        if solution.final_answer:
            output_buffer.write(f"\nFinal Answer (Returned by Orchestrator):\n{solution.final_answer}\n")
        else:
            output_buffer.write("\nFinal answer was not successfully extracted or an error occurred (Orchestrator).\n")
        output_buffer.write("="*67 + "\n")
        return output_buffer.getvalue()
