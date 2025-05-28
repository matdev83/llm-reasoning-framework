import unittest
from unittest.mock import patch, MagicMock, call

from src.l2t_processor import L2TProcessor
from src.l2t_dataclasses import (
    L2TConfig,
    L2TResult,
    L2TGraph,
    L2TNodeCategory,
    L2TNode,
)
from src.aot_dataclasses import LLMCallStats

from src.llm_client import LLMClient
from src.l2t_processor_utils.node_processor import NodeProcessor

import logging
logging.disable(logging.CRITICAL)

class TestL2TProcessor_InitialThoughtFailureParsing(unittest.TestCase):
    def setUp(self):
        self.config = L2TConfig(
            max_steps=5,
            max_total_nodes=10,
            max_time_seconds=60,
            classification_model_names=["mock-classifier"],
            thought_generation_model_names=["mock-generator"],
            initial_prompt_model_names=["mock-initial"],
        )

    @patch("src.l2t_processor.NodeProcessor")
    @patch("src.l2t_response_parser.L2TResponseParser.parse_l2t_initial_response")
    @patch("src.l2t_processor.LLMClient")
    def test_run_initial_thought_generation_failure_parsing(
        self, MockL2TProcessorLLMClient, mock_parse_initial, MockNodeProcessor
    ):
        problem_text = "Test problem: Initial failure."
        stats_initial = LLMCallStats(completion_tokens=1, prompt_tokens=1, call_duration_seconds=0.01)

        MockL2TProcessorLLMClient.return_value.call.return_value = ("This is not a valid initial thought format.", stats_initial)
        mock_parse_initial.return_value = None

        processor = L2TProcessor(api_key="mock_api_key", config=self.config)
        mock_node_processor_instance = processor.node_processor

        def mock_update_stats_effect(result_obj, stats):
            if stats:
                result_obj.total_llm_calls += 1
                result_obj.total_completion_tokens += stats.completion_tokens
                result_obj.total_prompt_tokens += stats.prompt_tokens
                result_obj.total_llm_interaction_time_seconds += stats.call_duration_seconds
        mock_node_processor_instance._update_result_stats.side_effect = mock_update_stats_effect

        result = processor.run(problem_text)

        self.assertFalse(result.succeeded)
        self.assertIsNone(result.final_answer)
        self.assertIsNotNone(result.error_message)
        self.assertIn("Failed during initial thought generation", result.error_message)
        self.assertEqual(result.total_llm_calls, 1)
        self.assertEqual(result.total_completion_tokens, stats_initial.completion_tokens)
        self.assertIsNotNone(result.reasoning_graph)
        self.assertEqual(len(result.reasoning_graph.nodes), 0)
        mock_node_processor_instance._update_result_stats.assert_any_call(result, stats_initial)
