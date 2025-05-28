import io
from typing import Optional # Added Optional
from src.l2t_dataclasses import L2TResult, L2TSolution
from src.l2t_enums import L2TTriggerMode
from src.aot_enums import AssessmentDecision
from src.aot_dataclasses import LLMCallStats 

class L2TSummaryGenerator:
    def __init__(self, trigger_mode: L2TTriggerMode, use_heuristic_shortcut: bool):
        self.trigger_mode = trigger_mode
        self.use_heuristic_shortcut = use_heuristic_shortcut

    def generate_l2t_summary_from_result(self, result: L2TResult) -> str:
        """Generates a summary string for a single L2TResult object."""
        output_buffer = io.StringIO()
        output_buffer.write("\n" + "="*20 + " L2T PROCESSOR EXECUTION SUMMARY " + "="*20 + "\n") # Clarified title
        output_buffer.write(f"L2T Processor Succeeded: {result.succeeded}\n")
        if result.error_message:
            output_buffer.write(f"Error Message: {result.error_message}\n")

        output_buffer.write(f"Total LLM Calls (L2T Processor): {result.total_llm_calls}\n")
        output_buffer.write(f"Total Completion Tokens (L2T Processor): {result.total_completion_tokens}\n")
        output_buffer.write(f"Total Prompt Tokens (L2T Processor): {result.total_prompt_tokens}\n")
        output_buffer.write(f"Grand Total Tokens (L2T Processor): {result.total_completion_tokens + result.total_prompt_tokens}\n")
        output_buffer.write(f"Total LLM Interaction Time (L2T Processor): {result.total_llm_interaction_time_seconds:.2f}s\n")
        output_buffer.write(f"Total Wall-Clock Time (L2T Processor): {result.total_process_wall_clock_time_seconds:.2f}s\n")

        if result.reasoning_graph and result.reasoning_graph.nodes:
            output_buffer.write(f"Number of nodes in graph (L2T Processor): {len(result.reasoning_graph.nodes)}\n")
            if result.reasoning_graph.root_node_id:
                 output_buffer.write(f"Root node ID (L2T Processor): {result.reasoning_graph.root_node_id[:8]}...\n")

        if result.final_answer:
            output_buffer.write(f"\nFinal Answer (from L2T Processor):\n{result.final_answer}\n")
        elif not result.succeeded:
            output_buffer.write("\nFinal answer was not successfully obtained by L2T Processor.\n")

        output_buffer.write("="*60 + "\n") # Adjusted length
        return output_buffer.getvalue()

    def generate_l2t_process_summary(self, solution: L2TSolution) -> str:
        """Generates a summary string for an L2TProcess execution (which uses L2TProcessor)."""
        # This method is specifically for L2TProcess class if it needs its own summary string
        # before it's passed to the orchestrator.
        output_buffer = io.StringIO()
        output_buffer.write("\n" + "="*20 + " L2TProcess Execution Summary " + "="*20 + "\n")
        
        if solution.l2t_result: # L2TResult from L2TProcessor
            # Include the detailed summary from L2TProcessor
            output_buffer.write(self.generate_l2t_summary_from_result(solution.l2t_result))
            
            if solution.l2t_failed_and_fell_back and solution.fallback_call_stats:
                sfb = solution.fallback_call_stats
                output_buffer.write(f"L2TProcess: L2T Processor FAILED. Fell Back to One-Shot.\n")
                output_buffer.write(f"  L2TProcess Fallback One-Shot Call ({sfb.model_name}): C={sfb.completion_tokens}, P={sfb.prompt_tokens}, Time={sfb.call_duration_seconds:.2f}s\n")
        elif solution.final_answer: # Maybe an error message if l2t_result is None
            output_buffer.write(f"L2TProcess Status: Final answer present but no L2TResult. Answer: {solution.final_answer}\n")
        else:
            output_buffer.write("L2TProcess: No L2T result and no final answer.\n")

        # Add overall L2TProcess specific totals if they differ from L2TProcessor totals
        # (e.g. if L2TProcess itself adds more calls, though current design doesn't show that)
        output_buffer.write(f"L2TProcess Total Wall-Clock Time: {solution.total_wall_clock_time_seconds:.2f}s\n")
        if solution.final_answer:
            output_buffer.write(f"\nFinal Answer (from L2TProcess):\n{solution.final_answer}\n")
        output_buffer.write("="*60 + "\n")
        return output_buffer.getvalue()


    def generate_overall_summary(self, solution: L2TSolution, l2t_process_execution_summary: Optional[str] = None) -> str:
        output_buffer = io.StringIO()
        output_buffer.write("\n" + "="*20 + " OVERALL L2T ORCHESTRATOR SUMMARY " + "="*20 + "\n")
        output_buffer.write(f"Orchestrator Trigger Mode: {self.trigger_mode.value.upper()}\n")
        output_buffer.write(f"Orchestrator Heuristic Shortcut Option for Assessor: {self.use_heuristic_shortcut}\n")

        if solution.assessment_stats:
            s = solution.assessment_stats
            decision_val = getattr(s, 'assessment_decision', solution.assessment_decision) # Fallback to solution's decision
            decision_for_summary = getattr(s, 'assessment_decision_for_summary', decision_val.value if decision_val else 'N/A')
            time_str = f"{s.call_duration_seconds:.2f}s" if s.call_duration_seconds is not None else "N/A"
            output_buffer.write(f"Assessment Phase ({s.model_name if s else 'N/A'}): Decision='{decision_for_summary}', C={s.completion_tokens if s else 'N/A'}, P={s.prompt_tokens if s else 'N/A'}, Time={time_str}\n")
        
        if solution.main_call_stats: # Direct ONESHOT by orchestrator
            s = solution.main_call_stats
            output_buffer.write(f"Orchestrator Main Model Call (Direct ONESHOT path) ({s.model_name}): C={s.completion_tokens}, P={s.prompt_tokens}, Time={s.call_duration_seconds:.2f}s\n")
        
        if l2t_process_execution_summary: # If L2TProcess was invoked and returned its own summary
            output_buffer.write("--- Delegated to L2TProcess ---\n")
            output_buffer.write(l2t_process_execution_summary) 
            output_buffer.write("-------------------------------\n")
            # The L2TProcess summary should ideally contain details about L2T success/failure/fallback.
            # We can add a concluding line from orchestrator's perspective if needed.
            if solution.l2t_result and solution.l2t_result.succeeded:
                 output_buffer.write(f"L2T Orchestrator: L2TProcess reported success.\n")
            elif solution.l2t_failed_and_fell_back:
                 output_buffer.write(f"L2T Orchestrator: L2TProcess reported failure and fallback.\n")
        
        # This handles fallback by the orchestrator itself (e.g. assessment error, L2TProcess not called or failed early)
        elif solution.fallback_call_stats and not l2t_process_execution_summary : 
            sfb = solution.fallback_call_stats
            output_buffer.write(f"Orchestrator Fallback One-Shot Call (e.g. due to Assessment Error, L2TProcess not run or its summary not provided) ({sfb.model_name}): C={sfb.completion_tokens}, P={sfb.prompt_tokens}, Time={sfb.call_duration_seconds:.2f}s\n")


        output_buffer.write(f"\n--- Orchestrator Perspective Totals ---\n")
        output_buffer.write(f"Total Completion Tokens (Orchestrator): {solution.total_completion_tokens}\n")
        output_buffer.write(f"Total Prompt Tokens (Orchestrator): {solution.total_prompt_tokens}\n")
        output_buffer.write(f"Grand Total Tokens (Orchestrator): {solution.grand_total_tokens}\n")
        output_buffer.write(f"Total LLM Interaction Time (Orchestrator, sum of calls it's aware of): {solution.total_llm_interaction_time_seconds:.2f}s\n")
        output_buffer.write(f"Total Orchestrator Wall-Clock Time: {solution.total_wall_clock_time_seconds:.2f}s\n")

        # Final answer from the orchestrator's perspective
        if solution.final_answer: 
            output_buffer.write(f"\nFinal Answer (Returned by Orchestrator):\n{solution.final_answer}\n")
        else: 
            output_buffer.write("\nFinal answer was not successfully extracted or an error occurred (Orchestrator).\n")
        output_buffer.write("="*67 + "\n") # Adjusted length
        return output_buffer.getvalue()
