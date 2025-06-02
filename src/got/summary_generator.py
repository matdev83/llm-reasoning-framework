import io
from typing import Optional
from .dataclasses import GoTSolution, GoTResult, GoTThought
from .enums import GoTTriggerMode

class GoTSummaryGenerator:
    def __init__(self, trigger_mode: GoTTriggerMode, use_heuristic_shortcut: Optional[bool] = None):
        self.trigger_mode = trigger_mode
        self.use_heuristic_shortcut = use_heuristic_shortcut

    def _format_llm_stats(self, stats_name: str, stats: Optional[object]) -> str: # Actual type is LLMCallStats
        if not stats:
            return f"{stats_name}: Not available/not run.\n"
        # Assuming stats object has model_name, completion_tokens, prompt_tokens, call_duration_seconds
        return (f"{stats_name} ({getattr(stats, 'model_name', 'N/A')}): "
                f"C={getattr(stats, 'completion_tokens', 'N/A')}, P={getattr(stats, 'prompt_tokens', 'N/A')}, "
                f"Time={getattr(stats, 'call_duration_seconds', 0.0):.2f}s\n")

    def _format_got_result_summary(self, got_result: Optional[GoTResult]) -> str:
        if not got_result:
            return "GoT Processor Result: Not available (likely not run or failed early).\n"

        buffer = io.StringIO()
        buffer.write(f"GoT Processor Succeeded: {got_result.succeeded}\n")
        if got_result.error_message:
            buffer.write(f"GoT Processor Error: {got_result.error_message}\n")

        buffer.write(f"  LLM Calls: {got_result.total_llm_calls}\n")
        buffer.write(f"  Total Completion Tokens: {got_result.total_completion_tokens}\n")
        buffer.write(f"  Total Prompt Tokens: {got_result.total_prompt_tokens}\n")
        buffer.write(f"  Total LLM Interaction Time: {got_result.total_llm_interaction_time_seconds:.2f}s\n")
        buffer.write(f"  Total GoT Process Wall-Clock Time: {got_result.total_process_wall_clock_time_seconds:.2f}s\n")

        if got_result.final_graph and got_result.final_graph.thoughts:
            buffer.write(f"  Final Graph: {len(got_result.final_graph.thoughts)} thoughts\n")
            if got_result.solution_candidates:
                buffer.write(f"  Solution Candidates ({len(got_result.solution_candidates)}):\n")
                for cand in got_result.solution_candidates[:3]: # Show top 3
                    buffer.write(f"    - ID: {cand.id}, Score: {cand.score:.2f}, Content: {cand.content[:80]}...\n")
            elif got_result.final_answer:
                 buffer.write(f"  Final Answer derived from graph (no specific candidates above threshold).\n")

        return buffer.getvalue()

    def generate_got_process_summary(self, solution: GoTSolution) -> str:
        # Summary from the perspective of GoTProcess (when it's run directly or by orchestrator)
        output_buffer = io.StringIO()
        output_buffer.write("\n" + "="*20 + " GoTProcess Execution Summary " + "="*20 + "\n")

        if solution.got_result:
            output_buffer.write("GoT Process Attempted: Yes\n")
            output_buffer.write(self._format_got_result_summary(solution.got_result))
            if solution.got_failed_and_fell_back:
                output_buffer.write("GoT FAILED and Fell Back to One-Shot: Yes\n")
                if solution.fallback_call_stats:
                    output_buffer.write(self._format_llm_stats("  Fallback One-Shot Call", solution.fallback_call_stats))
        else:
            output_buffer.write("GoT Process Was Not Fully Attempted or Failed Very Early.\n")
            if solution.final_answer and not solution.got_failed_and_fell_back : # Likely error before got_result populated
                 output_buffer.write(f"Status/Error: {solution.final_answer}\n")


        output_buffer.write(f"Total Completion Tokens (GoTProcess scope): {solution.total_completion_tokens}\n")
        output_buffer.write(f"Total Prompt Tokens (GoTProcess scope): {solution.total_prompt_tokens}\n")
        output_buffer.write(f"Grand Total Tokens (GoTProcess scope): {solution.grand_total_tokens}\n")
        output_buffer.write(f"Total LLM Interaction Time (GoTProcess scope): {solution.total_llm_interaction_time_seconds:.2f}s\n")
        output_buffer.write(f"Total Wall-Clock Time (GoTProcess Execution): {solution.total_wall_clock_time_seconds:.2f}s\n")

        if solution.final_answer:
            output_buffer.write(f"\nFinal Answer (from GoTProcess perspective):\n{solution.final_answer}\n")
        else:
            output_buffer.write("\nFinal answer not successfully extracted by GoTProcess.\n")
        output_buffer.write("="*67 + "\n")
        return output_buffer.getvalue()

    def generate_overall_summary(self, solution: GoTSolution, got_process_specific_summary: Optional[str] = None) -> str:
        # Summary from the perspective of the GoTOrchestrator
        output_buffer = io.StringIO()
        output_buffer.write("\n" + "="*20 + " GoT ORCHESTRATOR OVERALL SUMMARY " + "="*20 + "\n")
        output_buffer.write(f"Orchestrator Trigger Mode: {self.trigger_mode.value}\n")
        if self.trigger_mode == GoTTriggerMode.ASSESS_FIRST_GOT:
            output_buffer.write(f"Heuristic Shortcut Option for Assessor: {self.use_heuristic_shortcut}\n")

        if solution.assessment_stats:
            decision_val = solution.assessment_decision.value if solution.assessment_decision else 'N/A'
            output_buffer.write(self._format_llm_stats(f"Assessment Phase (Decision='{decision_val}')", solution.assessment_stats))

        # If GoTProcess was run (either ALWAYS_GOT or after ASSESS_FIRST_GOT -> ADVANCED)
        if got_process_specific_summary:
            output_buffer.write("--- Delegated to GoTProcess ---\n")
            output_buffer.write(got_process_specific_summary)
            output_buffer.write("-------------------------------\n")
            if solution.got_result and solution.got_result.succeeded:
                 output_buffer.write("GoTProcess Reported Success: Yes\n")
            elif solution.got_failed_and_fell_back:
                 output_buffer.write(f"GoTProcess Reported Failure and Fallback: Yes (Reason: {solution.got_result.error_message if solution.got_result else 'N/A'})\n")

        # If orchestrator ran a one-shot directly (NEVER_GOT or ASSESS_FIRST_GOT -> ONE_SHOT)
        # This stat is not explicitly on GoTSolution yet, assuming it would be stored in fallback_call_stats or a new field
        # For now, let's assume if GoT didn't run, and there's a final answer, it came from orchestrator's direct one-shot.
        # This part needs alignment with how Orchestrator stores stats for its direct one-shot.
        # Let's assume for now it's captured in fallback_call_stats if it's the *only* main call.
        elif solution.fallback_call_stats and not got_process_specific_summary and self.trigger_mode != GoTTriggerMode.ALWAYS_GOT:
             # This implies it was a direct one-shot by orchestrator (NEVER_GOT or ASSESS_FIRST resulted in ONE_SHOT)
             output_buffer.write(self._format_llm_stats("Orchestrator Direct One-Shot Call", solution.fallback_call_stats))
        elif solution.final_answer and not got_process_specific_summary and not solution.got_failed_and_fell_back:
            # This case implies NEVER_GOT and the result should be from a direct call, stats might be missing in current GoTSolution
            output_buffer.write("Orchestrator executed a direct one-shot call (details might be in final answer section if stats not separated).\n")


        output_buffer.write(f"\n--- Orchestrator Perspective Totals ---\n")
        output_buffer.write(f"Total Completion Tokens (Orchestrator): {solution.total_completion_tokens}\n")
        output_buffer.write(f"Total Prompt Tokens (Orchestrator): {solution.total_prompt_tokens}\n")
        output_buffer.write(f"Grand Total Tokens (Orchestrator): {solution.grand_total_tokens}\n")
        output_buffer.write(f"Total LLM Interaction Time (Orchestrator): {solution.total_llm_interaction_time_seconds:.2f}s\n")
        output_buffer.write(f"Total Orchestrator Wall-Clock Time: {solution.total_wall_clock_time_seconds:.2f}s\n")

        if solution.final_answer:
            output_buffer.write(f"\nFinal Answer (Returned by Orchestrator):\n{solution.final_answer}\n")
        else:
            output_buffer.write("\nFinal answer was not successfully extracted or an error occurred (Orchestrator).\n")
        output_buffer.write("="*76 + "\n")
        return output_buffer.getvalue()
