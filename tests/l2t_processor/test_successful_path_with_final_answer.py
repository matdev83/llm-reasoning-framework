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

class TestL2TProcessor_SuccessfulPath(unittest.TestCase):
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
    def test_run_successful_path_with_final_answer(
        self,
        MockL2TProcessorLLMClient,
        mock_parse_initial,
        MockNodeProcessor
    ):
        problem_text = "Test problem: Find the final answer."

        initial_thought_content = "This is the initial thought for the problem."
        generated_thought_content = "This is the generated thought, which is the final answer."

        stats_initial = LLMCallStats(completion_tokens=10, prompt_tokens=5, call_duration_seconds=0.1, model_name="initial_model")
        stats_classify1 = LLMCallStats(completion_tokens=5, prompt_tokens=5, call_duration_seconds=0.05, model_name="classify_model")
        stats_thought_gen = LLMCallStats(completion_tokens=15, prompt_tokens=10, call_duration_seconds=0.1, model_name="gen_model")
        stats_classify2 = LLMCallStats(completion_tokens=5, prompt_tokens=5, call_duration_seconds=0.05, model_name="classify_model")

        MockL2TProcessorLLMClient.return_value.call.return_value = ("Your thought: " + initial_thought_content, stats_initial)
        mock_parse_initial.return_value = initial_thought_content

        processor = L2TProcessor(api_key="mock_api_key", config=self.config)
        mock_node_processor_instance = processor.node_processor

        def process_node_side_effect(*args, **kwargs):
            call_count = getattr(process_node_side_effect, "call_count", 0)
            node_id, graph, result_obj, step = args
            if call_count == 0:
                process_node_side_effect.call_count = 1
                mock_node_processor_instance._update_result_stats(result_obj, stats_classify1)
                graph.classify_node(node_id, L2TNodeCategory.CONTINUE)
                mock_node_processor_instance._update_result_stats(result_obj, stats_thought_gen)
                graph.add_node(L2TNode(id="child1", content=generated_thought_content, parent_id=node_id, generation_step=1))
                graph.move_to_hist(node_id)
            elif call_count == 1:
                process_node_side_effect.call_count = 2
                mock_node_processor_instance._update_result_stats(result_obj, stats_classify2)
                graph.classify_node(node_id, L2TNodeCategory.FINAL_ANSWER)
                setattr(result_obj, 'final_answer', generated_thought_content)
                setattr(result_obj, 'succeeded', True)
                graph.move_to_hist(node_id)
            else:
                pass
        process_node_side_effect.call_count = 0
        mock_node_processor_instance.process_node.side_effect = process_node_side_effect

        def mock_update_stats_effect(result_obj, stats):
            if stats:
                result_obj.total_llm_calls += 1
                result_obj.total_completion_tokens += stats.completion_tokens
                result_obj.total_prompt_tokens += stats.prompt_tokens
                result_obj.total_llm_interaction_time_seconds += stats.call_duration_seconds
        mock_node_processor_instance._update_result_stats.side_effect = mock_update_stats_effect

        result = processor.run(problem_text)

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
        self.assertIsNotNone(graph.nodes)
        self.assertEqual(len(graph.nodes), 2)
        root_node_id = graph.root_node_id
        self.assertIsNotNone(root_node_id)
        root_node = graph.get_node(root_node_id)
        self.assertIsNotNone(root_node)
        self.assertEqual(root_node.content, initial_thought_content)
        self.assertEqual(root_node.category, L2TNodeCategory.CONTINUE)
        self.assertIsNotNone(root_node.children_ids)
        self.assertEqual(len(root_node.children_ids), 1)
        child_node_id = root_node.children_ids[0]
        self.assertIsNotNone(child_node_id)
        child_node = graph.get_node(child_node_id)
        self.assertIsNotNone(child_node)
        self.assertEqual(child_node.content, generated_thought_content)
        self.assertEqual(child_node.category, L2TNodeCategory.FINAL_ANSWER)
        self.assertEqual(child_node.parent_id, root_node_id)
        mock_node_processor_instance._update_result_stats.assert_any_call(result, stats_initial)
