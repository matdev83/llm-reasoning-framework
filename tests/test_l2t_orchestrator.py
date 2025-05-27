import unittest
from unittest.mock import patch, MagicMock

from src.l2t_orchestrator import L2TOrchestrator
from src.l2t_dataclasses import L2TConfig, L2TResult, L2TGraph
from src.aot_dataclasses import LLMCallStats # Import LLMCallStats from aot_dataclasses

# Suppress logging during tests
import logging
logging.disable(logging.CRITICAL)


# Mock for L2TProcessor to be used by the orchestrator tests
class MockL2TProcessor:
    def __init__(self, llm_client, config):
        self.llm_client = llm_client
        self.config = config
        # We'll mock the 'run' method using unittest.mock.patch on the class instance
        self.run = MagicMock()


class TestL2TOrchestrator(unittest.TestCase):
    def setUp(self):
        self.config = L2TConfig(
            max_steps=3, # Using some non-default values for testing
            classification_model_names=["test-classifier"],
            thought_generation_model_names=["test-generator"],
        )
        self.api_key = "test_api_key_for_orchestrator"

    # Patch 'src.l2t_orchestrator.L2TProcessor' so that when L2TOrchestrator
    # tries to instantiate it, it gets our MockL2TProcessor instead.
    @patch("src.l2t_orchestrator.L2TProcessor") # Patch the class itself
    def test_solve_successful_run(self, MockL2TProcessorClass):
        problem_text = "Test problem: orchestrator success."

        # Create a mock L2TResult for a successful run
        mock_successful_result = L2TResult(
            final_answer="This is the final answer from L2T.",
            reasoning_graph=L2TGraph(), # Minimal graph
            total_llm_calls=5,
            total_completion_tokens=100,
            total_prompt_tokens=200,
            total_llm_interaction_time_seconds=2.5,
            total_process_wall_clock_time_seconds=3.0,
            succeeded=True,
            error_message=None,
        )
        
        # Configure the mock L2TProcessor class to return a MockL2TProcessor instance
        # when it's instantiated by the orchestrator.
        mock_processor_instance = MockL2TProcessor(llm_client=MagicMock(), config=self.config)
        MockL2TProcessorClass.return_value = mock_processor_instance
        
        # Configure the 'run' method of the L2TProcessor instance *inside* the orchestrator
        mock_processor_instance.run.return_value = mock_successful_result

        # Instantiate the orchestrator. This will use the mocked L2TProcessor class,
        # which in turn returns our configured mock_processor_instance.
        orchestrator = L2TOrchestrator(l2t_config=self.config, api_key=self.api_key)
        
        # Call solve
        result_obj, summary_str = orchestrator.solve(problem_text)

        # Assert that L2TProcessor was instantiated correctly
        MockL2TProcessorClass.assert_called_once_with(llm_client=orchestrator.llm_client, config=self.config)
        
        # Assert that the run method on the *instance* was called correctly
        mock_processor_instance.run.assert_called_once_with(problem_text)

        # Assert results
        self.assertEqual(result_obj, mock_successful_result)
        self.assertTrue(result_obj.succeeded)
        self.assertIn("L2T PROCESS SUMMARY", summary_str)
        self.assertIn("L2T Succeeded: True", summary_str)
        self.assertIn(f"Final Answer:\n{mock_successful_result.final_answer}", summary_str)
        self.assertIn(f"Total LLM Calls: {mock_successful_result.total_llm_calls}", summary_str)
        self.assertNotIn("Error Message:", summary_str)


    @patch("src.l2t_orchestrator.L2TProcessor") # Patch the class itself
    def test_solve_failed_run(self, MockL2TProcessorClass):
        problem_text = "Test problem: orchestrator failure."

        # Create a mock L2TResult for a failed run
        mock_failed_result = L2TResult(
            final_answer=None,
            reasoning_graph=L2TGraph(),
            total_llm_calls=2,
            total_completion_tokens=50,
            total_prompt_tokens=70,
            total_llm_interaction_time_seconds=1.0,
            total_process_wall_clock_time_seconds=1.2,
            succeeded=False,
            error_message="Max steps reached during L2T processing.",
        )

        mock_processor_instance = MockL2TProcessor(llm_client=MagicMock(), config=self.config)
        MockL2TProcessorClass.return_value = mock_processor_instance
        mock_processor_instance.run.return_value = mock_failed_result

        orchestrator = L2TOrchestrator(l2t_config=self.config, api_key=self.api_key)
        result_obj, summary_str = orchestrator.solve(problem_text)

        MockL2TProcessorClass.assert_called_once_with(llm_client=orchestrator.llm_client, config=self.config)
        mock_processor_instance.run.assert_called_once_with(problem_text)

        self.assertEqual(result_obj, mock_failed_result)
        self.assertFalse(result_obj.succeeded)
        self.assertIsNone(result_obj.final_answer)
        self.assertIn("L2T PROCESS SUMMARY", summary_str)
        self.assertIn("L2T Succeeded: False", summary_str)
        self.assertIn(f"Error Message: {mock_failed_result.error_message}", summary_str)
        self.assertIn("Final answer was not successfully obtained.", summary_str)
        self.assertNotIn("Final Answer:\n", summary_str) # Check that final answer section is not there if no answer

    @patch("src.l2t_orchestrator.L2TProcessor") # Patch the class itself
    def test_summary_generation_content(self, MockL2TProcessorClass):
        problem_text = "Test problem for summary details."
        
        reasoning_graph_mock = L2TGraph()
        mock_detailed_result = L2TResult(
            final_answer="Detailed answer.",
            reasoning_graph=reasoning_graph_mock, # Minimal graph for now
            total_llm_calls=3,
            total_completion_tokens=123,
            total_prompt_tokens=456,
            total_llm_interaction_time_seconds=1.23,
            total_process_wall_clock_time_seconds=2.34,
            succeeded=True
        )
        # Add a root node to test graph summary details
        reasoning_graph_mock.add_node(MagicMock(id="root12345"), is_root=True)

        mock_processor_instance = MockL2TProcessor(llm_client=MagicMock(), config=self.config)
        MockL2TProcessorClass.return_value = mock_processor_instance
        mock_processor_instance.run.return_value = mock_detailed_result
        
        orchestrator = L2TOrchestrator(l2t_config=self.config, api_key=self.api_key)
        _, summary_str = orchestrator.solve(problem_text)

        self.assertIn(f"Total LLM Calls: {mock_detailed_result.total_llm_calls}", summary_str)
        self.assertIn(f"Total Completion Tokens: {mock_detailed_result.total_completion_tokens}", summary_str)
        self.assertIn(f"Total Prompt Tokens: {mock_detailed_result.total_prompt_tokens}", summary_str)
        grand_total = mock_detailed_result.total_completion_tokens + mock_detailed_result.total_prompt_tokens
        self.assertIn(f"Grand Total Tokens (All L2T Calls): {grand_total}", summary_str)
        self.assertIn(f"Total L2T LLM Interaction Time: {mock_detailed_result.total_llm_interaction_time_seconds:.2f}s", summary_str)
        self.assertIn(f"Total L2T Process Wall-Clock Time: {mock_detailed_result.total_process_wall_clock_time_seconds:.2f}s", summary_str)
        self.assertIn(f"Number of nodes in graph: {len(mock_detailed_result.reasoning_graph.nodes)}", summary_str)
        self.assertIn(f"Root node ID: {mock_detailed_result.reasoning_graph.root_node_id[:8]}", summary_str)
        self.assertIn(f"Final Answer:\n{mock_detailed_result.final_answer}", summary_str)
