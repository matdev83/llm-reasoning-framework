import unittest
from unittest.mock import patch, MagicMock, call
import uuid

from src.l2t_processor import L2TProcessor
from src.l2t_dataclasses import (
    L2TConfig,
    L2TResult,
    L2TGraph,
    L2TNodeCategory,
    L2TNode,
)
from src.aot_dataclasses import LLMCallStats # Import LLMCallStats from aot_dataclasses
# Use the mock LLMClient from l2t_processor for testing if it's suitable,
# or define a more controlled one here.
# from src.l2t_processor import LLMClient # Mock LLMClient
# For more control, we'll use MagicMock for LLMClient
# from src.llm_client import LLMClient # Assuming this is the actual client path

from src.llm_client import LLMClient # Import the actual LLMClient

# Suppress logging during tests
import logging
logging.disable(logging.CRITICAL)


class TestL2TProcessor(unittest.TestCase):
    def setUp(self):
        self.config = L2TConfig(
            max_steps=5,
            max_total_nodes=10,
            max_time_seconds=60,
            classification_model_names=["mock-classifier"],
            thought_generation_model_names=["mock-generator"],
            initial_prompt_model_names=["mock-initial"],
        )
        # self.mock_llm_client = MagicMock(spec=LLMClient)
        # The L2TProcessor instantiates its own PromptGenerator and Parser.
        # So, we don't mock those directly unless we are testing those units.
        # For testing L2TProcessor, we mock the LLMClient's `call` method,
        # and the Parser static methods.

    @patch("src.l2t_response_parser.L2TResponseParser.parse_l2t_thought_generation_response")
    @patch("src.l2t_response_parser.L2TResponseParser.parse_l2t_node_classification_response")
    @patch("src.l2t_response_parser.L2TResponseParser.parse_l2t_initial_response")
    @patch("src.l2t_processor.LLMClient") # Patch the class itself
    def test_run_successful_path_with_final_answer(
        self,
        MockLLMClient, # This is the mock class
        mock_parse_initial,
        mock_parse_classify,
        mock_parse_thought_gen,
    ):
        problem_text = "Test problem: Find the final answer."

        # --- Mock LLM Responses and Stats ---
        initial_thought_content = "This is the initial thought for the problem."
        generated_thought_content = "This is the generated thought, which is the final answer."

        stats_initial = LLMCallStats(completion_tokens=10, prompt_tokens=5, call_duration_seconds=0.1, model_name="initial_model")
        stats_classify1 = LLMCallStats(completion_tokens=5, prompt_tokens=5, call_duration_seconds=0.05, model_name="classify_model")
        stats_thought_gen = LLMCallStats(completion_tokens=15, prompt_tokens=10, call_duration_seconds=0.1, model_name="gen_model")
        stats_classify2 = LLMCallStats(completion_tokens=5, prompt_tokens=5, call_duration_seconds=0.05, model_name="classify_model")

        # Configure mock LLMClient.call responses
        MockLLMClient.return_value.call.side_effect = [
            ("Your thought: " + initial_thought_content, stats_initial),  # Initial thought
            ("Your classification: CONTINUE", stats_classify1),            # Classification of initial thought
            ("Your new thought: " + generated_thought_content, stats_thought_gen), # Generated thought
            ("Your classification: FINAL_ANSWER", stats_classify2),       # Classification of generated thought
        ]

        # Configure mock L2TResponseParser responses
        mock_parse_initial.return_value = initial_thought_content
        mock_parse_classify.side_effect = [
            L2TNodeCategory.CONTINUE,
            L2TNodeCategory.FINAL_ANSWER,
        ]
        mock_parse_thought_gen.return_value = generated_thought_content
        
        # --- Instantiate and Run ---
        # The @patch above targets 'src.l2t_processor.LLMClient',
        # so MockLLMClient is the mock class. We pass its instance to L2TProcessor.
        processor = L2TProcessor(llm_client=MockLLMClient.return_value, config=self.config)
        result = processor.run(problem_text)

        # --- Assertions ---
        self.assertTrue(result.succeeded)
        self.assertEqual(result.final_answer, generated_thought_content)
        self.assertIsNone(result.error_message)

        self.assertEqual(result.total_llm_calls, 4)
        expected_completion_tokens = stats_initial.completion_tokens + stats_classify1.completion_tokens + stats_thought_gen.completion_tokens + stats_classify2.completion_tokens
        expected_prompt_tokens = stats_initial.prompt_tokens + stats_classify1.prompt_tokens + stats_thought_gen.prompt_tokens + stats_classify2.prompt_tokens
        self.assertEqual(result.total_completion_tokens, expected_completion_tokens)
        self.assertEqual(result.total_prompt_tokens, expected_prompt_tokens)

        self.assertIsNotNone(result.reasoning_graph)
        graph = result.reasoning_graph
        self.assertEqual(len(graph.nodes), 2) # Root node + 1 generated node

        root_node_id = graph.root_node_id
        self.assertIsNotNone(root_node_id)
        root_node = graph.get_node(root_node_id)
        self.assertIsNotNone(root_node) # Add check for None
        self.assertEqual(root_node.content, initial_thought_content)
        self.assertEqual(root_node.category, L2TNodeCategory.CONTINUE)
        self.assertEqual(len(root_node.children_ids), 1)

        child_node_id = root_node.children_ids[0]
        child_node = graph.get_node(child_node_id)
        self.assertIsNotNone(child_node) # Add check for None
        self.assertEqual(child_node.content, generated_thought_content)
        self.assertEqual(child_node.category, L2TNodeCategory.FINAL_ANSWER)
        self.assertEqual(child_node.parent_id, root_node_id)

        # Check PromptGenerator calls (optional, but good for completeness)
        # This requires mocking the PromptGenerator methods if we want to assert their calls.
        # For now, focusing on LLM calls and Parser mocks.

    @patch("src.l2t_response_parser.L2TResponseParser.parse_l2t_thought_generation_response")
    @patch("src.l2t_response_parser.L2TResponseParser.parse_l2t_node_classification_response")
    @patch("src.l2t_response_parser.L2TResponseParser.parse_l2t_initial_response")
    @patch("src.l2t_processor.LLMClient") # Patch the class itself
    def test_run_max_steps_reached(
        self,
        MockLLMClient, # This is the mock class
        mock_parse_initial,
        mock_parse_classify,
        mock_parse_thought_gen,
    ):
        problem_text = "Test problem: Max steps."
        self.config.max_steps = 2 # Set small max_steps

        initial_thought_content = "Initial thought."
        thought_step1_content = "Generated thought step 1."
        thought_step2_content = "Generated thought step 2 (should not be fully processed if max_steps=2)."


        stats = LLMCallStats(completion_tokens=1, prompt_tokens=1, call_duration_seconds=0.01)

        # Mock LLM calls to always continue generating
        MockLLMClient.return_value.call.side_effect = [
            ("Mocked LLM Response", stats), # Initial thought
            ("Mocked LLM Response", stats), # Classification of initial thought
            ("Mocked LLM Response", stats), # Generated thought 1
            ("Mocked LLM Response", stats), # Classification of thought 1
            ("Mocked LLM Response", stats), # Generated thought 2
        ]

        # Mock Parser responses
        mock_parse_initial.return_value = initial_thought_content
        mock_parse_classify.return_value = L2TNodeCategory.CONTINUE # Always continue
        mock_parse_thought_gen.side_effect = [thought_step1_content, thought_step2_content, "Should not reach here"]


        processor = L2TProcessor(llm_client=MockLLMClient.return_value, config=self.config)
        result = processor.run(problem_text)

        self.assertFalse(result.succeeded)
        self.assertIsNone(result.final_answer)
        self.assertIn("Max steps reached", result.error_message)
        
        # Initial + Classify_Initial + Gen_Thought_1 + Classify_Thought_1 + Gen_Thought_2 (processor step for T1 is 1, for T2 is 2)
        # Step 0: Initial thought (1 LLM call)
        # Step 1: Classify initial (CONTINUE), Gen thought 1 (2 LLM calls for this step)
        # Step 2: Classify thought 1 (CONTINUE), Gen thought 2 (2 LLM calls for this step)
        # Loop terminates because current_process_step (which is 2) >= self.config.max_steps (which is 2)
        # So, thought 2 is generated, but not classified.
        # Total calls: 1 (initial) + 1 (classify initial) + 1 (gen T1) + 1 (classify T1) + 1 (gen T2) = 5 calls
        self.assertEqual(MockLLMClient.return_value.call.call_count, 5) 
        self.assertEqual(result.total_llm_calls, 5)

        graph = result.reasoning_graph
        self.assertIsNotNone(graph)
        # Root, Thought1, Thought2 = 3 nodes
        # Root node (initial_thought_content) -> classified CONTINUE
        # Child 1 (thought_step1_content) from Root -> classified CONTINUE
        # Child 2 (thought_step2_content) from Child 1 -> category is None (not classified)
        self.assertEqual(len(graph.nodes), 3)
        root_node = graph.get_node(graph.root_node_id)
        self.assertIsNotNone(root_node) # Add check for None
        self.assertEqual(root_node.category, L2TNodeCategory.CONTINUE)
        self.assertEqual(len(root_node.children_ids), 1)
        node1 = graph.get_node(root_node.children_ids[0])
        self.assertIsNotNone(node1) # Add check for None
        self.assertEqual(node1.content, thought_step1_content)
        self.assertEqual(node1.category, L2TNodeCategory.CONTINUE)
        self.assertEqual(len(node1.children_ids), 1)
        node2 = graph.get_node(node1.children_ids[0])
        self.assertIsNotNone(node2) # Add check for None
        self.assertEqual(node2.content, thought_step2_content)
        self.assertIsNone(node2.category) # Not classified due to max_steps


    @patch("src.l2t_response_parser.L2TResponseParser.parse_l2t_initial_response")
    @patch("src.l2t_processor.LLMClient") # Patch the class itself
    def test_run_initial_thought_generation_failure_parsing(
        self, MockLLMClient, mock_parse_initial
    ):
        problem_text = "Test problem: Initial failure."
        stats_initial = LLMCallStats(completion_tokens=1, prompt_tokens=1, call_duration_seconds=0.01)

        # Mock LLM to return something unparseable by the initial parser
        MockLLMClient.return_value.call.return_value = ("This is not a valid initial thought format.", stats_initial)
        mock_parse_initial.return_value = None # Simulate parser failure

        processor = L2TProcessor(llm_client=MockLLMClient.return_value, config=self.config)
        result = processor.run(problem_text)

        self.assertFalse(result.succeeded)
        self.assertIsNone(result.final_answer)
        self.assertIn("Failed during initial thought generation", result.error_message)
        self.assertEqual(result.total_llm_calls, 1)
        self.assertEqual(result.total_completion_tokens, stats_initial.completion_tokens)
        self.assertIsNotNone(result.reasoning_graph) # Graph should exist but be empty/minimal
        self.assertEqual(len(result.reasoning_graph.nodes), 0) # No root node added


    @patch("src.l2t_processor.LLMClient") # Patch the class itself
    def test_run_initial_thought_generation_failure_llm_error(
        self, MockLLMClient
    ):
        problem_text = "Test problem: Initial LLM error."

        # Mock LLM to return an error string
        MockLLMClient.return_value.call.return_value = ("Error: LLM unavailable", None) # No stats if LLM call fails

        processor = L2TProcessor(llm_client=MockLLMClient.return_value, config=self.config)
        result = processor.run(problem_text)

        self.assertFalse(result.succeeded)
        self.assertIsNone(result.final_answer)
        self.assertIn("Failed during initial thought generation", result.error_message)
        # The L2TProcessor's _update_result_stats won't be called if stats is None
        self.assertEqual(result.total_llm_calls, 0) 
        self.assertEqual(result.total_completion_tokens, 0)
        self.assertIsNotNone(result.reasoning_graph)
        self.assertEqual(len(result.reasoning_graph.nodes), 0)
