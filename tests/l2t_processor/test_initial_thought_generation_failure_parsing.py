import unittest
from unittest.mock import patch, MagicMock, call, ANY

from src.l2t.processor import L2TProcessor
from src.l2t.dataclasses import (
    L2TConfig,
    L2TResult,
    L2TGraph,
    L2TNodeCategory,
    L2TNode,
)
from src.aot.dataclasses import LLMCallStats
from src.llm_config import LLMConfig # Added LLMConfig

from src.llm_client import LLMClient
from src.l2t_processor_utils.node_processor import NodeProcessor

import logging
# logging.disable(logging.CRITICAL) # Re-enable logging disable for cleaner output

class TestL2TProcessor_InitialThoughtFailureParsing(unittest.TestCase):
    def setUp(self):
        self.l2t_config = L2TConfig( # Renamed from self.config
            max_steps=5,
            max_total_nodes=10,
            max_time_seconds=60,
            classification_model_names=["mock-classifier"],
            thought_generation_model_names=["mock-generator"],
            initial_prompt_model_names=["mock-initial"],
        )
        # Define LLMConfig objects for L2TProcessor
        self.initial_thought_llm_config = LLMConfig(temperature=0.7)
        self.node_processor_llm_config = LLMConfig(temperature=0.1)

    @patch("src.l2t.response_parser.L2TResponseParser.parse_l2t_initial_response")
    @patch("src.l2t.processor.LLMClient")
    @patch("src.l2t.processor.NodeProcessor") # Patch the NodeProcessor as it's imported in processor.py
    def test_run_initial_thought_generation_failure_parsing(
        self, MockNodeProcessor, MockL2TProcessorLLMClient, mock_parse_initial
    ):
        problem_text = "Test problem: Initial failure."
        # Create a MagicMock that behaves like LLMCallStats
        mock_stats_initial = MagicMock(spec=LLMCallStats)
        mock_stats_initial.completion_tokens = 1
        mock_stats_initial.prompt_tokens = 1
        mock_stats_initial.call_duration_seconds = 0.01
        mock_stats_initial.model_name = None # Or a specific mock model name if needed

        # Set mock return values before processor instantiation
        def llm_client_call_side_effect(*args, **kwargs):
            # print(f"DEBUG: In llm_client_call_side_effect, mock_stats_initial.completion_tokens: {mock_stats_initial.completion_tokens}")
            return "This is not a valid initial thought format.", mock_stats_initial
        MockL2TProcessorLLMClient.return_value.call.side_effect = llm_client_call_side_effect
        mock_parse_initial.return_value = None

        processor = L2TProcessor(
            api_key="mock_api_key",
            l2t_config=self.l2t_config, # Use l2t_config
            initial_thought_llm_config=self.initial_thought_llm_config,
            node_processor_llm_config=self.node_processor_llm_config,
        )
        # Get the actual node_processor instance from the instantiated processor
        mock_node_processor_instance = processor.node_processor
        # Explicitly mock the _update_result_stats method on the instance
        mock_node_processor_instance._update_result_stats = MagicMock()
        
        # Setup the _update_result_stats side effect on the actual mock instance
        # Define mock_update_stats_effect inside the test method to avoid scope issues
        def mock_update_stats_effect_inner(result_obj, stats):
            # print(f"DEBUG: Inside mock_update_stats_effect, id(result_obj): {id(result_obj)}")
            # print(f"DEBUG: Inside mock_update_stats_effect, stats: {stats!r}, type: {type(stats)}, stats is mock_stats_initial: {stats is mock_stats_initial}")
            # print(f"DEBUG: Inside mock_update_stats_effect, stats.completion_tokens: {getattr(stats, 'completion_tokens', 'NO ATTRIBUTE')}")
            if stats:
                result_obj.total_llm_calls += 1
                result_obj.total_completion_tokens += stats.completion_tokens
                result_obj.total_prompt_tokens += stats.prompt_tokens
                result_obj.total_llm_interaction_time_seconds += stats.call_duration_seconds
                # print(f"DEBUG: Inside mock_update_stats_effect, result_obj.total_completion_tokens after update: {result_obj.total_completion_tokens}")
        mock_node_processor_instance._update_result_stats.side_effect = mock_update_stats_effect_inner


        result = processor.run(problem_text)
        # Remove debug prints
        # print(f"DEBUG: In test method, id(result) after processor.run(): {id(result)}")
        # print(f"DEBUG: In test method, result.total_completion_tokens before assertion: {result.total_completion_tokens}")

        self.assertFalse(result.succeeded)
        self.assertIsNone(result.final_answer)
        self.assertIsNotNone(result.error_message)
        if result.error_message: # Ensure it's not None before checking content
            self.assertIn("Failed during initial thought generation", result.error_message)
        self.assertEqual(result.total_llm_calls, 1) # One call made for initial thought
        self.assertEqual(result.total_completion_tokens, mock_stats_initial.completion_tokens)
        self.assertIsNotNone(result.reasoning_graph)
        if result.reasoning_graph: # Ensure it's not None before accessing nodes
            self.assertEqual(len(result.reasoning_graph.nodes), 0) # No nodes should be added if initial parsing fails
        mock_node_processor_instance._update_result_stats.assert_called_once_with(result, mock_stats_initial)
