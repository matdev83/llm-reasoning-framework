import io
from src.l2t_dataclasses import L2TResult, L2TSolution
from src.l2t_enums import L2TTriggerMode
from src.aot_enums import AssessmentDecision
from src.aot_dataclasses import LLMCallStats # Import LLMCallStats

class L2TSummaryGenerator:
    def __init__(self, trigger_mode: L2TTriggerMode, use_heuristic_shortcut: bool):
        self.trigger_mode = trigger_mode
        self.use_heuristic_shortcut = use_heuristic_shortcut

    def generate_l2t_summary_from_result(self, result: L2TResult) -> str:
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

    def generate_overall_summary(self, solution: L2TSolution) -> str:
        output_buffer = io.StringIO()
        output_buffer.write("\n" + "="*20 + " OVERALL L2T ORCHESTRATOR SUMMARY " + "="*20 + "\n")
        output_buffer.write(f"Trigger Mode: {self.trigger_mode.value.upper()}\n")
        output_buffer.write(f"Heuristic Shortcut Enabled: {self.use_heuristic_shortcut}\n")

        if solution.assessment_stats:
            s = solution.assessment_stats
            output_buffer.write(f"Assessment ({s.model_name}): Decision={solution.assessment_decision.value if solution.assessment_decision else 'N/A'}, C={s.completion_tokens}, P={s.prompt_tokens}, Time={s.call_duration_seconds:.2f}s\n")
        
        if solution.main_call_stats:
            s = solution.main_call_stats
            output_buffer.write(f"Main Model Call (Direct ONESHOT path) ({s.model_name}): C={s.completion_tokens}, P={s.prompt_tokens}, Time={s.call_duration_seconds:.2f}s\n")
        
        if solution.l2t_result:
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
