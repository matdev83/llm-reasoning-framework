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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestL2TProcessor_MaxSteps(unittest.TestCase):
    def setUp(self):
        self.l2t_config = L2TConfig( # Renamed from self.config
            max_steps=2,
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
    @patch("src.l2t.processor.LLMClient") # Patch LLMClient where it's used in L2TProcessor
    @patch("src.l2t.processor.NodeProcessor") # Patch NodeProcessor where it's used in L2TProcessor
    def test_run_max_steps_reached(
        self,
        MockNodeProcessor,
        MockL2TProcessorLLMClient,
        mock_parse_initial
    ):
        problem_text = "Test problem: Max steps."
        initial_thought_content = "Initial thought."
        thought_step1_content = "Generated thought step 1."
        thought_step2_content = "Generated thought step 2 (should not be fully processed if max_steps=2)."

        stats = LLMCallStats(completion_tokens=1, prompt_tokens=1, call_duration_seconds=0.01)

        # Set mock return values before processor instantiation
        MockL2TProcessorLLMClient.return_value.call.return_value = ("Mocked LLM Response", stats)
        mock_parse_initial.return_value = initial_thought_content

        processor = L2TProcessor(
            api_key="mock_api_key",
            l2t_config=self.l2t_config, # Use l2t_config
            initial_thought_llm_config=self.initial_thought_llm_config,
            node_processor_llm_config=self.node_processor_llm_config,
        )
        # Get the actual node_processor instance from the instantiated processor
        mock_node_processor_instance = processor.node_processor
        # Explicitly mock the methods on the instance
        mock_node_processor_instance._update_result_stats = MagicMock()
        mock_node_processor_instance.process_node = MagicMock()

        # Use a simple counter for the side effect logic
        process_node_call_counter = 0

        def process_node_side_effect_max_steps(*args, **kwargs):
            nonlocal process_node_call_counter # Declare as nonlocal to modify the outer variable
            node_id, graph, result_obj, step = args
            if process_node_call_counter == 0:
                process_node_call_counter = 1
                mock_node_processor_instance._update_result_stats(result_obj, stats)  # classify initial
                graph.classify_node(node_id, L2TNodeCategory.CONTINUE)
                mock_node_processor_instance._update_result_stats(result_obj, stats)  # gen thought 1
                graph.add_node(L2TNode(id="child1", content=thought_step1_content, parent_id=node_id, generation_step=1))
                graph.move_to_hist(node_id)
            elif process_node_call_counter == 1:
                process_node_call_counter = 2
                mock_node_processor_instance._update_result_stats(result_obj, stats)  # classify thought 1
                graph.classify_node(node_id, L2TNodeCategory.CONTINUE)
                mock_node_processor_instance._update_result_stats(result_obj, stats)  # gen thought 2
                graph.add_node(L2TNode(id="child2", content=thought_step2_content, parent_id=node_id, generation_step=2))
                graph.move_to_hist(node_id)
            else:
                pass
        mock_node_processor_instance.process_node.side_effect = process_node_side_effect_max_steps

        # Re-introduce the side effect for _update_result_stats
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
        if result.error_message: # Add check for None
            self.assertIsInstance(result.error_message, str)
            self.assertTrue("Max steps reached" in result.error_message)
        self.assertEqual(result.total_llm_calls, 1 + 4) # Initial + 4 calls from NodeProcessor

        graph = result.reasoning_graph
        self.assertIsNotNone(graph)
        if graph: # Add check for None
            self.assertEqual(len(graph.nodes), 3)
            root_node_id = graph.root_node_id
            self.assertIsNotNone(root_node_id)
            root_node = graph.get_node(root_node_id) if root_node_id else None
            self.assertIsNotNone(root_node)
            if root_node: # Add check for None
                self.assertEqual(root_node.category, L2TNodeCategory.CONTINUE)
                self.assertIsNotNone(root_node.children_ids)
                self.assertEqual(len(root_node.children_ids), 1)
                node1_id = root_node.children_ids[0] if root_node.children_ids else None
                self.assertIsNotNone(node1_id)
                node1 = graph.get_node(node1_id) if node1_id else None
                self.assertIsNotNone(node1)
                if node1: # Add check for None
                    self.assertEqual(node1.content, thought_step1_content)
                    self.assertEqual(node1.category, L2TNodeCategory.CONTINUE)
                    self.assertIsNotNone(node1.children_ids)
                    self.assertEqual(len(node1.children_ids), 1)
                    node2_id = node1.children_ids[0] if node1.children_ids else None
                    self.assertIsNotNone(node2_id)
                    node2 = graph.get_node(node2_id) if node2_id else None
                    self.assertIsNotNone(node2)
                    if node2: # Add check for None
                        self.assertEqual(node2.content, thought_step2_content)
                        self.assertIsNone(node2.category)
        mock_node_processor_instance._update_result_stats.assert_any_call(result, stats)
