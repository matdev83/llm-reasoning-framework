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

class TestL2TProcessor_MaxSteps(unittest.TestCase):
    def setUp(self):
        self.config = L2TConfig(
            max_steps=2,
            max_total_nodes=10,
            max_time_seconds=60,
            classification_model_names=["mock-classifier"],
            thought_generation_model_names=["mock-generator"],
            initial_prompt_model_names=["mock-initial"],
        )

    @patch("src.l2t_processor.NodeProcessor")
    @patch("src.l2t_response_parser.L2TResponseParser.parse_l2t_initial_response")
    @patch("src.l2t_processor.LLMClient")
    def test_run_max_steps_reached(
        self,
        MockL2TProcessorLLMClient,
        mock_parse_initial,
        MockNodeProcessor
    ):
        problem_text = "Test problem: Max steps."
        initial_thought_content = "Initial thought."
        thought_step1_content = "Generated thought step 1."
        thought_step2_content = "Generated thought step 2 (should not be fully processed if max_steps=2)."

        stats = LLMCallStats(completion_tokens=1, prompt_tokens=1, call_duration_seconds=0.01)

        MockL2TProcessorLLMClient.return_value.call.return_value = ("Mocked LLM Response", stats)
        mock_parse_initial.return_value = initial_thought_content

        processor = L2TProcessor(api_key="mock_api_key", config=self.config)
        mock_node_processor_instance = processor.node_processor

        def process_node_side_effect_max_steps(*args, **kwargs):
            call_count = getattr(process_node_side_effect_max_steps, "call_count", 0)
            node_id, graph, result_obj, step = args
            if call_count == 0:
                process_node_side_effect_max_steps.call_count = 1
                mock_node_processor_instance._update_result_stats(result_obj, stats)  # classify initial
                graph.classify_node(node_id, L2TNodeCategory.CONTINUE)
                mock_node_processor_instance._update_result_stats(result_obj, stats)  # gen thought 1
                graph.add_node(L2TNode(id="child1", content=thought_step1_content, parent_id=node_id, generation_step=1))
                graph.move_to_hist(node_id)
            elif call_count == 1:
                process_node_side_effect_max_steps.call_count = 2
                mock_node_processor_instance._update_result_stats(result_obj, stats)  # classify thought 1
                graph.classify_node(node_id, L2TNodeCategory.CONTINUE)
                mock_node_processor_instance._update_result_stats(result_obj, stats)  # gen thought 2
                graph.add_node(L2TNode(id="child2", content=thought_step2_content, parent_id=node_id, generation_step=2))
                graph.move_to_hist(node_id)
            else:
                pass
        process_node_side_effect_max_steps.call_count = 0
        mock_node_processor_instance.process_node.side_effect = process_node_side_effect_max_steps

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
        self.assertIn("Max steps reached", result.error_message)
        self.assertEqual(result.total_llm_calls, 1 + 4) # Initial + 4 calls from NodeProcessor

        graph = result.reasoning_graph
        self.assertIsNotNone(graph)
        self.assertIsNotNone(graph.nodes)
        self.assertEqual(len(graph.nodes), 3)
        root_node_id = graph.root_node_id
        self.assertIsNotNone(root_node_id)
        root_node = graph.get_node(root_node_id)
        self.assertIsNotNone(root_node)
        self.assertEqual(root_node.category, L2TNodeCategory.CONTINUE)
        self.assertIsNotNone(root_node.children_ids)
        self.assertEqual(len(root_node.children_ids), 1)
        node1_id = root_node.children_ids[0]
        self.assertIsNotNone(node1_id)
        node1 = graph.get_node(node1_id)
        self.assertIsNotNone(node1)
        self.assertEqual(node1.content, thought_step1_content)
        self.assertEqual(node1.category, L2TNodeCategory.CONTINUE)
        self.assertIsNotNone(node1.children_ids)
        self.assertEqual(len(node1.children_ids), 1)
        node2_id = node1.children_ids[0]
        self.assertIsNotNone(node2_id)
        node2 = graph.get_node(node2_id)
        self.assertIsNotNone(node2)
        self.assertEqual(node2.content, thought_step2_content)
        self.assertIsNone(node2.category)
        mock_node_processor_instance._update_result_stats.assert_any_call(result, stats)
