import time
import logging
import io # For string buffer if generating a summary string

from src.l2t_dataclasses import L2TConfig, L2TResult # LLMCallStats is aggregated in L2TResult
# Assuming llm_client will be available in the environment where this is run.
# For local testing, you might need to adjust this import or provide a mock.
# from src.llm_client import LLMClient
# Placeholder for LLMClient to allow linting and type checking
class LLMClient:
    def __init__(self, api_key: str): # type: ignore
        # This is a mock implementation. Replace with actual LLM client.
        self.api_key = api_key
        if not api_key:
             raise ValueError("API key is required for LLMClient")
        print(f"Mock LLMClient initialized with api_key: {'*' * len(api_key) if api_key else 'None'}")

    def call(self, prompt: str, models: list[str], temperature: float): # type: ignore
        # This is a mock implementation. Replace with actual LLM client.
        # This mock is simpler than the L2TProcessor's one as the orchestrator doesn't directly call it.
        # The L2TProcessor will use its own more detailed mock or a real client.
        print(f"Orchestrator's LLMClient.call (should not be directly called by orchestrator logic normally) with prompt: {prompt[:100]}")
        # This call method is primarily for the L2TProcessor's instantiation.
        # The L2TProcessor itself has a more detailed mock for its internal calls.
        return "Mock LLM Response from Orchestrator's client instance", None


from src.l2t_processor import L2TProcessor

# Configure basic logging if no handlers are present
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class L2TOrchestrator:
    def __init__(self,
                 l2t_config: L2TConfig,
                 api_key: str): # Or however LLMClient is initialized

        self.l2t_config = l2t_config
        # In a real scenario, LLMClient would be properly imported and initialized.
        # For now, using the placeholder defined above.
        self.llm_client = LLMClient(api_key=api_key) 
        self.l2t_processor = L2TProcessor(llm_client=self.llm_client, config=self.l2t_config)

    def _generate_l2t_summary(self, result: L2TResult) -> str:
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
            # Could add more graph details here if desired, e.g., number of terminal nodes, etc.
            if result.reasoning_graph.root_node_id:
                 output_buffer.write(f"Root node ID: {result.reasoning_graph.root_node_id[:8]}...\n")

        if result.final_answer:
            output_buffer.write(f"\nFinal Answer:\n{result.final_answer}\n")
        elif not result.succeeded:
            output_buffer.write("\nFinal answer was not successfully obtained.\n")

        output_buffer.write("="*59 + "\n") # Adjusted length
        return output_buffer.getvalue()

    def solve(self, problem_text: str) -> tuple[L2TResult, str]: # Returns result object and summary string
        logger.info("--- Starting L2T Orchestration ---")
        logger.info(f"Problem: {problem_text[:100].strip()}...")

        # Record start time for the solve method itself, if needed for overall orchestration timing
        # although L2TResult.total_process_wall_clock_time_seconds should cover the processor's work.
        # orchestration_start_time = time.monotonic()

        l2t_result = self.l2t_processor.run(problem_text)

        summary_output = self._generate_l2t_summary(l2t_result)
        # Log to the orchestrator's logger
        for line in summary_output.strip().split('\n'):
            logger.info(line)


        if l2t_result.succeeded:
            logger.info("L2T process completed successfully.")
        else:
            logger.warning(f"L2T process failed or did not find a final answer. Error: {l2t_result.error_message}")
        
        # orchestration_total_time = time.monotonic() - orchestration_start_time
        # logger.info(f"L2T Orchestration total wall-clock time: {orchestration_total_time:.2f}s (includes summary generation)")

        return l2t_result, summary_output

if __name__ == '__main__':
    # This basic test assumes L2TProcessor's mock LLM client is sufficient for its internal operations.
    # The LLMClient passed here is for the L2TProcessor's __init__ method.
    
    logger.info("--- Starting L2TOrchestrator basic test ---")

    # 1. Setup Config
    test_config = L2TConfig(
        max_steps=4, # Keep low for testing
        max_total_nodes=8,
        max_time_seconds=60,
        classification_model_names=["test-classify-model"],
        thought_generation_model_names=["test-thought-gen-model"],
        initial_prompt_model_names=["test-initial-model"]
    )

    # 2. Instantiate Orchestrator
    # Ensure you have a dummy API key or adjust LLMClient mock if needed
    try:
        orchestrator = L2TOrchestrator(l2t_config=test_config, api_key="dummy_api_key_for_l2t_orchestrator")
    except Exception as e:
        logger.error(f"Failed to initialize L2TOrchestrator: {e}")
        exit(1)

    # 3. Define a problem
    problem1 = "What is the capital of France, and what are two interesting facts about it?"
    logger.info(f"\n--- Test Case 1: Problem: '{problem1}' ---")

    # 4. Call solve
    l2t_result_obj, summary_str = orchestrator.solve(problem_text=problem1)

    # 5. Print parts of the result and summary (summary is already logged)
    print("\n--- Orchestrator Test Case 1 Results ---")
    print(f"Result Object - Succeeded: {l2t_result_obj.succeeded}")
    print(f"Result Object - Final Answer: {l2t_result_obj.final_answer[:200] if l2t_result_obj.final_answer else 'N/A'}...")
    print(f"Result Object - Error: {l2t_result_obj.error_message}")
    print(f"Summary String was logged above.")


    # Example: Problem that might lead to early termination if mock LLM is configured for it
    # For L2TProcessor's current mock, it will likely run until max_steps or max_nodes
    # unless specific keywords trigger FINAL_ANSWER or TERMINATE_BRANCH in its mock LLM.
    problem2 = "This is a problem designed to find a final answer quickly." # Mock needs to support this
    # To test this, the L2TProcessor's mock LLMClient would need to be adjusted to produce FINAL_ANSWER
    # for a thought containing "final answer" for example.
    # The current L2TProcessor mock LLMClient has:
    # if "final answer" in prompt.lower(): # test condition for final answer
    #    return "Your classification: FINAL_ANSWER", LLMCallStats(...)
    # So if a thought contains "final answer", it should be classified as such.
    # Let's assume the initial thought or a subsequent thought might contain this.

    logger.info(f"\n--- Test Case 2: Problem: '{problem2}' (testing for potential early FINAL_ANSWER) ---")
    # Re-using the same orchestrator instance
    l2t_result_obj_2, summary_str_2 = orchestrator.solve(problem_text=problem2)

    print("\n--- Orchestrator Test Case 2 Results ---")
    print(f"Result Object - Succeeded: {l2t_result_obj_2.succeeded}")
    print(f"Result Object - Final Answer: {l2t_result_obj_2.final_answer[:200] if l2t_result_obj_2.final_answer else 'N/A'}...")
    print(f"Result Object - Error: {l2t_result_obj_2.error_message}")

    logger.info("--- L2TOrchestrator basic test completed ---")
