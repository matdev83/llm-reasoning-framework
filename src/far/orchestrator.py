import time
import logging
import io
from typing import List, Tuple, Optional, Any

from src.reasoning_process import ReasoningProcess
from src.llm_client import LLMClient
from src.llm_config import LLMConfig
from src.far.dataclasses import FaRConfig, FaRSolution, FaRResult, LLMCallStats
from src.far.processor import FaRProcessor
from src.far.enums import FaRTriggerMode

# For complexity assessment
from src.complexity_assessor import ComplexityAssessor
from src.aot.enums import AssessmentDecision # Reusing existing enum for assessment outcome
from src.heuristic_detector import HeuristicDetector # For heuristic shortcut in assessor

# For communication logging
from src.communication_logger import log_llm_request, log_llm_response, log_stage, ModelRole

logger = logging.getLogger(__name__)

class FaRProcess(ReasoningProcess):
    """
    Implements the Fact-and-Reflection (FaR) reasoning process.
    This class is responsible for executing the FaR chain using FaRProcessor
    and handling fallbacks to a direct one-shot call if FaR fails.
    """
    def __init__(self,
                 llm_client: LLMClient,
                 far_config: FaRConfig,
                 # Configs for the direct one-shot call used as a fallback by this process
                 direct_oneshot_llm_config: LLMConfig,
                 direct_oneshot_model_names: List[str]):

        self.llm_client = llm_client
        self.far_config = far_config
        self.direct_oneshot_llm_config = direct_oneshot_llm_config
        self.direct_oneshot_model_names = direct_oneshot_model_names

        self.far_processor = FaRProcessor(llm_client=self.llm_client, config=self.far_config)

        self._solution: Optional[FaRSolution] = None
        self._process_summary: Optional[str] = None

    def _run_direct_oneshot_fallback(self, problem_text: str) -> Tuple[str, LLMCallStats]:
        stage_name = "Fallback One-Shot (FaRProcess)"
        log_stage("FaR", stage_name)
        logger.info(f"--- {stage_name}: FaR process failed, running direct one-shot. ---")
        logger.info(f"Using models: {', '.join(self.direct_oneshot_model_names)}, LLMConfig: {self.direct_oneshot_llm_config}")

        # Log the outgoing request for fallback one-shot
        config_info = {"temperature": self.direct_oneshot_llm_config.temperature, "max_tokens": self.direct_oneshot_llm_config.max_tokens}
        comm_id = log_llm_request("FaR", ModelRole.FAR_ONESHOT_FALLBACK, self.direct_oneshot_model_names,
                                  problem_text, stage_name, config_info)

        response_content, stats = self.llm_client.call(
            prompt=problem_text,
            models=self.direct_oneshot_model_names,
            config=self.direct_oneshot_llm_config
        )

        # Log the incoming response for fallback one-shot
        log_llm_response(comm_id, "FaR", ModelRole.FAR_ONESHOT_FALLBACK, stats.model_name,
                         response_content, stage_name, stats)

        logger.info(f"Fallback one-shot call ({stats.model_name}) completed. Duration: {stats.call_duration_seconds:.2f}s")
        return response_content, stats

    def execute(self, problem_description: str, model_name: str, *args, **kwargs) -> None:
        # model_name param is for compatibility, actual models are in far_config
        overall_start_time = time.monotonic()
        self._solution = FaRSolution()
        logger.info(f"FaRProcess executing for problem: {problem_description[:100]}...")

        if not self.far_processor: # Should not happen if constructor ran
            logger.critical("FaRProcessor not initialized within FaRProcess.")
            self._solution.final_answer = "Error: FaRProcessor not initialized in FaRProcess."
            if self._solution.total_wall_clock_time_seconds is None: # Ensure time is set
                 self._solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time
            self._process_summary = self._generate_process_summary(self._solution)
            return

        far_result_data = self.far_processor.run(problem_description)
        self._solution.far_result = far_result_data

        # Generate a simple summary from the processor's perspective
        processor_summary_buffer = io.StringIO()
        processor_summary_buffer.write(f"FaR Processor Attempted: Yes\n")
        if far_result_data.succeeded:
            processor_summary_buffer.write(f"  FaR Processor Succeeded: Yes\n")
            if far_result_data.fact_call_stats:
                fs = far_result_data.fact_call_stats
                processor_summary_buffer.write(f"    Fact Call ({fs.model_name}): C={fs.completion_tokens}, P={fs.prompt_tokens}, Time={fs.call_duration_seconds:.2f}s\n")
            if far_result_data.main_call_stats:
                ms = far_result_data.main_call_stats
                processor_summary_buffer.write(f"    Main Call ({ms.model_name}): C={ms.completion_tokens}, P={ms.prompt_tokens}, Time={ms.call_duration_seconds:.2f}s\n")
            if far_result_data.elicited_facts:
                 processor_summary_buffer.write(f"    Elicited Facts Length: {len(far_result_data.elicited_facts)} chars\n")
        else:
            processor_summary_buffer.write(f"  FaR Processor Failed: Yes (Reason: {far_result_data.error_message})\n")
        self._solution.far_summary_output = processor_summary_buffer.getvalue()


        if far_result_data.succeeded and far_result_data.final_answer is not None:
            self._solution.final_answer = far_result_data.final_answer
            if far_result_data.elicited_facts:
                self._solution.reasoning_trace.append(f"Elicited Facts:\n{far_result_data.elicited_facts}")
            self._solution.reasoning_trace.append(f"Final Answer Generation Process:\n(Reflected on facts to produce the answer)")

        else:
            logger.warning(f"FaR process failed (Reason: {far_result_data.error_message}). Falling back to one-shot.")
            self._solution.far_failed_and_fell_back = True
            fallback_answer, fallback_stats = self._run_direct_oneshot_fallback(problem_description)
            self._solution.final_answer = fallback_answer
            self._solution.fallback_call_stats = fallback_stats
            # Preserve any partial reasoning (facts) if available
            if far_result_data.elicited_facts:
                self._solution.reasoning_trace.append(f"Elicited Facts (from failed FaR attempt):\n{far_result_data.elicited_facts}")

        if self._solution.total_wall_clock_time_seconds is None:
            self._solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time
        self._process_summary = self._generate_process_summary(self._solution)

    def get_result(self) -> Tuple[Optional[FaRSolution], Optional[str]]:
        return self._solution, self._process_summary

    def _generate_process_summary(self, solution: FaRSolution) -> str:
        output_buffer = io.StringIO()
        output_buffer.write("\n" + "="*20 + " FaRProcess Execution Summary " + "="*20 + "\n")
        if solution.far_result:
            output_buffer.write(f"FaR Process Attempted: Yes\n")
            if solution.far_summary_output: # This is the processor's internal summary
                 output_buffer.write(f"--- FaR Processor Internal Summary ---\n{solution.far_summary_output}\n----------------------------------\n")

            if solution.far_result.succeeded:
                 output_buffer.write(f"FaR Succeeded (Reported by FaR Processor): Yes\n")
            elif solution.far_failed_and_fell_back:
                output_buffer.write(f"FaR FAILED and Fell Back to One-Shot: Yes (FaR Failure Reason: {solution.far_result.error_message})\n")
                if solution.fallback_call_stats:
                    sfb = solution.fallback_call_stats
                    output_buffer.write(f"  Fallback One-Shot Call ({sfb.model_name}): C={sfb.completion_tokens}, P={sfb.prompt_tokens}, Time={sfb.call_duration_seconds:.2f}s\n")
        else:
            output_buffer.write(f"FaR Process Was Not Fully Attempted (e.g., FaRProcessor initialization error).\n")
            if solution.final_answer: # This might be an error message if init failed
                 output_buffer.write(f"Status/Error: {solution.final_answer}\n")

        output_buffer.write(f"Total Completion Tokens (FaRProcess scope): {solution.total_completion_tokens}\n")
        output_buffer.write(f"Total Prompt Tokens (FaRProcess scope): {solution.total_prompt_tokens}\n")
        output_buffer.write(f"Grand Total Tokens (FaRProcess scope): {solution.grand_total_tokens}\n")
        output_buffer.write(f"Total LLM Interaction Time (FaRProcess scope): {solution.total_llm_interaction_time_seconds:.2f}s\n")
        output_buffer.write(f"Total Wall-Clock Time (FaRProcess Execution): {solution.total_wall_clock_time_seconds:.2f}s\n")

        if solution.reasoning_trace:
            output_buffer.write("\n--- Reasoning Trace (from FaRProcess) ---\n")
            for step_line in solution.reasoning_trace: output_buffer.write(step_line + "\n")

        if solution.final_answer and ( (solution.far_result and solution.far_result.succeeded) or solution.far_failed_and_fell_back ):
            output_buffer.write(f"\nFinal Answer (from FaRProcess):\n{solution.final_answer}\n")
        elif not solution.final_answer:
            output_buffer.write("\nFinal answer not successfully extracted by FaRProcess.\n")
        output_buffer.write("="*60 + "\n")
        return output_buffer.getvalue()


class FaROrchestrator:
    def __init__(self,
                 llm_client: LLMClient, # Shared client
                 trigger_mode: FaRTriggerMode,
                 far_config: FaRConfig, # Specific to FaR processor
                 # For direct one-shot by orchestrator (NEVER_FAR or fallback from assessment)
                 direct_oneshot_llm_config: LLMConfig,
                 direct_oneshot_model_names: List[str],
                 # For assessment phase (optional, only if ASSESS_FIRST is used)
                 assessment_llm_config: Optional[LLMConfig] = None,
                 assessment_model_names: Optional[List[str]] = None,
                 use_heuristic_shortcut: bool = True,
                 heuristic_detector: Optional[HeuristicDetector] = None):

        self.llm_client = llm_client
        self.trigger_mode = trigger_mode
        self.far_config = far_config # Stored for FaRProcess instantiation

        self.direct_oneshot_llm_config = direct_oneshot_llm_config
        self.direct_oneshot_model_names = direct_oneshot_model_names

        self.use_heuristic_shortcut = use_heuristic_shortcut
        self.heuristic_detector = heuristic_detector

        self.complexity_assessor: Optional[ComplexityAssessor] = None
        if self.trigger_mode == FaRTriggerMode.ASSESS_FIRST_FAR:
            if not assessment_llm_config or not assessment_model_names:
                logger.warning("ASSESS_FIRST_FAR mode selected but assessment_llm_config or assessment_model_names not provided. Assessment will be skipped.")
            else:
                self.complexity_assessor = ComplexityAssessor(
                    llm_client=self.llm_client,
                    small_model_names=assessment_model_names,
                    llm_config=assessment_llm_config,
                    use_heuristic_shortcut=self.use_heuristic_shortcut,
                    heuristic_detector=self.heuristic_detector
                )

        self.far_process_instance: Optional[FaRProcess] = None
        # Instance will be created on demand in solve() if FaR is chosen.
        # This avoids creating it if trigger_mode is NEVER_FAR or assessment chooses one-shot.

    def _get_or_create_far_process_instance(self) -> FaRProcess:
        if not self.far_process_instance:
            logger.info("Creating FaRProcess instance on demand.")
            self.far_process_instance = FaRProcess(
                llm_client=self.llm_client, # Pass shared client
                far_config=self.far_config,
                # Pass orchestrator's direct one-shot config for FaRProcess's internal fallback
                direct_oneshot_llm_config=self.direct_oneshot_llm_config,
                direct_oneshot_model_names=self.direct_oneshot_model_names
            )
        return self.far_process_instance

    def _run_orchestrator_direct_oneshot(self, problem_text: str, is_fallback_path:bool = False) -> Tuple[str, LLMCallStats]:
        mode_desc = "FALLBACK ONESHOT (FaROrchestrator)" if is_fallback_path else "DIRECT ONESHOT (FaROrchestrator)"
        stage_name = f"Orchestrator {'Fallback ' if is_fallback_path else ''}One-Shot"
        log_stage("FaR", stage_name)

        logger.info(f"--- Proceeding with {mode_desc} ---")
        logger.info(f"Using models: {', '.join(self.direct_oneshot_model_names)}, LLMConfig: {self.direct_oneshot_llm_config}")

        # Log outgoing request
        config_info = {"temperature": self.direct_oneshot_llm_config.temperature, "max_tokens": self.direct_oneshot_llm_config.max_tokens}
        # Use appropriate FaR roles for oneshot calls
        model_role = ModelRole.FAR_ONESHOT_FALLBACK if is_fallback_path else ModelRole.FAR_ONESHOT
        comm_id = log_llm_request("FaR", model_role, self.direct_oneshot_model_names,
                                  problem_text, stage_name, config_info)

        response_content, stats = self.llm_client.call(
            prompt=problem_text,
            models=self.direct_oneshot_model_names,
            config=self.direct_oneshot_llm_config
        )

        # Log incoming response
        log_llm_response(comm_id, "FaR", model_role, stats.model_name,
                         response_content, stage_name, stats)

        logger.info(f"LLM call ({stats.model_name}) for {mode_desc}: Duration: {stats.call_duration_seconds:.2f}s")
        return response_content, stats

    def solve(self, problem_text: str, model_name_for_far: str = "default_far_model") -> Tuple[FaRSolution, str]:
        # model_name_for_far is for compatibility, actual models from FaRConfig
        print(f"🔍 ORCHESTRATOR SOLVE CALLED with trigger_mode: {self.trigger_mode}")
        overall_start_time = time.monotonic()
        orchestrator_solution = FaRSolution() # This will accumulate all results
        far_process_execution_summary: Optional[str] = None # Summary from FaRProcess if run


        if self.trigger_mode == FaRTriggerMode.NEVER_FAR:
            logger.info("Trigger mode: NEVER_FAR. Orchestrator performing direct one-shot call.")
            final_answer, oneshot_stats = self._run_orchestrator_direct_oneshot(problem_text)
            orchestrator_solution.final_answer = final_answer
            orchestrator_solution.main_call_stats = oneshot_stats # Record as main_call_stats

        elif self.trigger_mode == FaRTriggerMode.ALWAYS_FAR:
            logger.info("Trigger mode: ALWAYS_FAR. Orchestrator delegating to FaRProcess.")
            logger.debug(f"About to get or create FaRProcess instance")
            current_far_process = self._get_or_create_far_process_instance()
            logger.debug(f"Got FaRProcess instance: {current_far_process}")
            logger.debug(f"About to execute FaRProcess with problem: {problem_text[:50]}...")
            current_far_process.execute(problem_description=problem_text, model_name=model_name_for_far) # model_name is for API consistency
            logger.debug(f"FaRProcess execute completed")

            # Get the solution object and summary string from FaRProcess
            far_solution_obj_from_process, far_process_execution_summary = current_far_process.get_result()

            # Debug logging
            logger.debug(f"FaRProcess get_result() returned: {far_solution_obj_from_process}")
            logger.debug(f"FaRProcess summary: {far_process_execution_summary}")

            if far_solution_obj_from_process:
                # Transfer results from FaRProcess's solution to the orchestrator's solution
                orchestrator_solution.far_result = far_solution_obj_from_process.far_result
                orchestrator_solution.final_answer = far_solution_obj_from_process.final_answer
                orchestrator_solution.reasoning_trace = far_solution_obj_from_process.reasoning_trace
                orchestrator_solution.far_failed_and_fell_back = far_solution_obj_from_process.far_failed_and_fell_back
                orchestrator_solution.fallback_call_stats = far_solution_obj_from_process.fallback_call_stats # if FaRProcess fell back
                orchestrator_solution.far_summary_output = far_solution_obj_from_process.far_summary_output
            else:
                err_msg = "Error: FaRProcess executed but returned no solution object."
                logger.error(err_msg)
                orchestrator_solution.final_answer = err_msg
                # Ensure far_result is at least an empty object indicating failure
                orchestrator_solution.far_result = FaRResult(succeeded=False, error_message=err_msg)


        elif self.trigger_mode == FaRTriggerMode.ASSESS_FIRST_FAR:
            print(f"🔍 Entering ASSESS_FIRST_FAR branch")
            logger.info("Trigger mode: ASSESS_FIRST_FAR. Orchestrator performing complexity assessment.")
            if not self.complexity_assessor:
                logger.warning("ComplexityAssessor not available for ASSESS_FIRST_FAR mode. Defaulting to ALWAYS_FAR behavior.")
                # Fallback to ALWAYS_FAR logic
                current_far_process = self._get_or_create_far_process_instance()
                current_far_process.execute(problem_description=problem_text, model_name=model_name_for_far)
                far_solution_obj_from_process, far_process_execution_summary = current_far_process.get_result()
                if far_solution_obj_from_process:
                    orchestrator_solution.far_result = far_solution_obj_from_process.far_result
                    orchestrator_solution.final_answer = far_solution_obj_from_process.final_answer
                    # ... copy other relevant fields as in ALWAYS_FAR ...
                    orchestrator_solution.reasoning_trace = far_solution_obj_from_process.reasoning_trace
                    orchestrator_solution.far_failed_and_fell_back = far_solution_obj_from_process.far_failed_and_fell_back
                    orchestrator_solution.fallback_call_stats = far_solution_obj_from_process.fallback_call_stats
                    orchestrator_solution.far_summary_output = far_solution_obj_from_process.far_summary_output

            else: # Assessor is available
                assessment_decision, assessment_stats = self.complexity_assessor.assess(problem_text)
                orchestrator_solution.assessment_stats = assessment_stats
                # orchestrator_solution.assessment_decision = assessment_decision # Store this if FaRSolution has this field

                if assessment_decision == AssessmentDecision.ONE_SHOT:
                    logger.info("Assessment: ONE_SHOT. Orchestrator performing direct one-shot call.")
                    final_answer, oneshot_stats = self._run_orchestrator_direct_oneshot(problem_text)
                    orchestrator_solution.final_answer = final_answer
                    orchestrator_solution.main_call_stats = oneshot_stats # Record as main_call_stats

                elif assessment_decision == AssessmentDecision.ADVANCED_REASONING: # This means use FaR
                    logger.info("Assessment: ADVANCED_REASONING. Orchestrator delegating to FaRProcess.")
                    current_far_process = self._get_or_create_far_process_instance()
                    current_far_process.execute(problem_description=problem_text, model_name=model_name_for_far)
                    far_solution_obj_from_process, far_process_execution_summary = current_far_process.get_result()
                    if far_solution_obj_from_process:
                        orchestrator_solution.far_result = far_solution_obj_from_process.far_result
                        orchestrator_solution.final_answer = far_solution_obj_from_process.final_answer
                        # ... copy other relevant fields ...
                        orchestrator_solution.reasoning_trace = far_solution_obj_from_process.reasoning_trace
                        orchestrator_solution.far_failed_and_fell_back = far_solution_obj_from_process.far_failed_and_fell_back
                        orchestrator_solution.fallback_call_stats = far_solution_obj_from_process.fallback_call_stats
                        orchestrator_solution.far_summary_output = far_solution_obj_from_process.far_summary_output
                    else: # Should not happen if execute always returns a solution object
                        err_msg = "Error: FaRProcess (post-assessment) returned no solution object."
                        logger.error(err_msg)
                        orchestrator_solution.final_answer = err_msg
                        orchestrator_solution.far_result = FaRResult(succeeded=False, error_message=err_msg)
                else: # AssessmentDecision.ERROR or other unexpected
                    logger.error(f"Complexity assessment resulted in unexpected decision: {assessment_decision}. Orchestrator attempting one-shot call as a last resort.")
                    final_answer, oneshot_stats = self._run_orchestrator_direct_oneshot(problem_text, is_fallback_path=True)
                    orchestrator_solution.final_answer = final_answer
                    # Store as fallback_call_stats because this is orchestrator's own fallback due to assessment error
                    orchestrator_solution.fallback_call_stats = oneshot_stats
                    orchestrator_solution.far_failed_and_fell_back = True # Indicate that the main FaR path was bypassed due to error

        if orchestrator_solution.total_wall_clock_time_seconds is None:
            orchestrator_solution.total_wall_clock_time_seconds = time.monotonic() - overall_start_time

        # Generate the final overall summary string
        final_summary_output = self._generate_overall_summary(
            orchestrator_solution,
            far_process_specific_summary=far_process_execution_summary # Pass summary from FaRProcess
        )
        return orchestrator_solution, final_summary_output

    def _generate_overall_summary(self, solution: FaRSolution, far_process_specific_summary: Optional[str] = None) -> str:
        output_buffer = io.StringIO()
        output_buffer.write("\n" + "="*20 + " FaR ORCHESTRATOR OVERALL SUMMARY " + "="*20 + "\n")
        output_buffer.write(f"Orchestrator Trigger Mode: {self.trigger_mode.value}\n")

        if self.trigger_mode == FaRTriggerMode.ASSESS_FIRST_FAR and self.complexity_assessor:
            output_buffer.write(f"Heuristic Shortcut Option for Assessor: {self.use_heuristic_shortcut}\n")
            if solution.assessment_stats:
                s = solution.assessment_stats
                # decision_for_summary = solution.assessment_decision.value if solution.assessment_decision else 'N/A' # If assessment_decision is stored
                # For now, we don't explicitly store assessment_decision on FaRSolution, but ComplexityAssessor returns it.
                # We can infer the decision from which path was taken.
                # This part of summary might need refinement based on how AssessmentDecision is tracked.
                time_str = f"{s.call_duration_seconds:.2f}s" if s.call_duration_seconds is not None else "N/A"
                # output_buffer.write(f"Assessment Phase ({s.model_name if s else 'N/A'}): Decision='{decision_for_summary}', C={s.completion_tokens if s else 'N/A'}, P={s.prompt_tokens if s else 'N/A'}, Time={time_str}\n")
                output_buffer.write(f"Assessment Phase ({s.model_name if s else 'N/A'}): C={s.completion_tokens if s else 'N/A'}, P={s.prompt_tokens if s else 'N/A'}, Time={time_str}\n")


        if solution.main_call_stats: # Direct one-shot by orchestrator (NEVER_FAR or assessment -> ONE_SHOT)
            s = solution.main_call_stats
            output_buffer.write(f"Orchestrator Main Model Call (Direct ONESHOT path) ({s.model_name}): C={s.completion_tokens}, P={s.prompt_tokens}, Time={s.call_duration_seconds:.2f}s\n")

        if far_process_specific_summary: # Summary from FaRProcess.execute()
            output_buffer.write("--- Delegated to FaRProcess ---\n")
            output_buffer.write(far_process_specific_summary)
            output_buffer.write("-------------------------------\n")
            # This info is likely already in far_process_specific_summary, but for consistency:
            if solution.far_result and solution.far_result.succeeded:
                 output_buffer.write(f"FaRProcess Reported Success: Yes\n")
            elif solution.far_failed_and_fell_back: # This means FaRProcess ran, failed, and FaRProcess itself fell back
                 output_buffer.write(f"FaRProcess Reported Failure and Internal Fallback: Yes (Reason: {solution.far_result.error_message if solution.far_result else 'N/A'})\n")

        # This is for orchestrator's own fallback (e.g., assessment error, not FaRProcess internal fallback)
        elif solution.fallback_call_stats and not far_process_specific_summary :
            sfb = solution.fallback_call_stats
            output_buffer.write(f"Orchestrator Fallback One-Shot Call (e.g., due to Assessment Error) ({sfb.model_name}): C={sfb.completion_tokens}, P={sfb.prompt_tokens}, Time={sfb.call_duration_seconds:.2f}s\n")

        output_buffer.write(f"\n--- Orchestrator Perspective Totals ---\n")
        output_buffer.write(f"Total Completion Tokens (Orchestrator): {solution.total_completion_tokens}\n")
        output_buffer.write(f"Total Prompt Tokens (Orchestrator): {solution.total_prompt_tokens}\n")
        output_buffer.write(f"Grand Total Tokens (Orchestrator): {solution.grand_total_tokens}\n")
        output_buffer.write(f"Total LLM Interaction Time (Orchestrator, sum of calls it's aware of): {solution.total_llm_interaction_time_seconds:.2f}s\n")
        output_buffer.write(f"Total Orchestrator Wall-Clock Time: {solution.total_wall_clock_time_seconds:.2f}s\n")

        if solution.reasoning_trace:
            output_buffer.write("\n--- Reasoning Trace (from FaRProcess, if run) ---\n")
            for step_line in solution.reasoning_trace: output_buffer.write(step_line + "\n")

        if solution.final_answer:
            output_buffer.write(f"\nFinal Answer (Returned by Orchestrator):\n{solution.final_answer}\n")
        else:
            output_buffer.write("\nFinal answer was not successfully extracted or an error occurred (Orchestrator).\n")
        output_buffer.write("="*67 + "\n")
        return output_buffer.getvalue()
