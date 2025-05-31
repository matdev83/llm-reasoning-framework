import unittest
from unittest.mock import patch, MagicMock, call
from typing import cast

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
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.disable(logging.CRITICAL) # Re-disable logging for cleaner test output

class TestL2TProcessor_BacktrackLogic(unittest.TestCase):
    def setUp(self):
        self.l2t_config = L2TConfig( # Renamed from self.config
            max_steps=5,  # Allow enough steps for backtracking and re-exploration
            max_total_nodes=10,
            max_time_seconds=60,
            classification_model_names=["mock-classifier"],
            thought_generation_model_names=["mock-generator"],
            initial_prompt_model_names=["mock-initial"],
        )
        # Define LLMConfig objects for L2TProcessor
        self.initial_thought_llm_config = LLMConfig(temperature=0.7)
        self.node_processor_llm_config = LLMConfig(temperature=0.1)


    @patch("src.l2t.processor.L2TResponseParser.parse_l2t_initial_response")
    @patch("src.l2t.processor.LLMClient")
    def test_backtrack_re_adds_parent_to_v_pres(self, MockL2TProcessorLLMClient, mock_parse_initial):
        problem_text = "Test problem: Demonstrate backtracking."

        initial_thought_content = "This is the initial thought."
        unfruitful_thought_content = "This thought leads to a dead end and should be backtracked from."
        re_explored_thought_content = "This is a new thought generated after backtracking from the unfruitful path."
        final_answer_content = "This is the final answer after successful re-exploration."

        # Mock initial LLM call
        MockL2TProcessorLLMClient.return_value.call.return_value = ("Your thought: " + initial_thought_content, MagicMock(spec=LLMCallStats))
        mock_parse_initial.return_value = initial_thought_content

        # Mock the LLMClient for the L2TProcessor itself
        mock_llm_client_instance = MockL2TProcessorLLMClient.return_value
        mock_llm_client_instance.call.return_value = ("Your thought: " + initial_thought_content, MagicMock(spec=LLMCallStats))

        processor = L2TProcessor(
            api_key="mock_api_key",
            l2t_config=self.l2t_config, # Use l2t_config
            initial_thought_llm_config=self.initial_thought_llm_config,
            node_processor_llm_config=self.node_processor_llm_config,
        )

        # Attributes to keep track of node IDs across mock calls
        self.root_node_id = None
        self.unfruitful_node_id = None
        self.call_count = 0  # To track calls to the mocked process_node

        from src.l2t_processor_utils.node_processor import NodeProcessor
        node_processor_mock = MagicMock(spec=NodeProcessor)

        def side_effect_process_node(node_id_to_classify: str, graph: L2TGraph, result: L2TResult, current_process_step: int) -> None:
            logging.debug(f"SIDE_EFFECT: Call count: {self.call_count}, Node: {node_id_to_classify}, Step: {current_process_step}, result.final_answer (start): {result.final_answer}, result ID: {id(result)}")
            node_processor_mock._update_result_stats(result, MagicMock(spec=LLMCallStats, completion_tokens=1, prompt_tokens=1, call_duration_seconds=0.1))
    
            if self.call_count == 0:  # Processing root node
                self.root_node_id = node_id_to_classify
                graph.classify_node(node_id_to_classify, L2TNodeCategory.CONTINUE)
                new_node = L2TNode(id="unfruitful_child", content=unfruitful_thought_content, parent_id=node_id_to_classify, generation_step=1)
                graph.add_node(new_node)
                self.unfruitful_node_id = new_node.id
                graph.move_to_hist(node_id_to_classify)
    
            elif node_id_to_classify == self.unfruitful_node_id:
                graph.classify_node(node_id_to_classify, L2TNodeCategory.BACKTRACK)
                # Simulate moving the backtracked node to history
                graph.move_to_hist(node_id_to_classify)
                # Simulate moving its parent back to v_pres for re-exploration
                if self.root_node_id: # Ensure root_node_id is set
                    graph.re_add_to_v_pres(self.root_node_id)
    
            elif node_id_to_classify == self.root_node_id and self.call_count > 1: # This condition will now be met after backtracking
                graph.classify_node(node_id_to_classify, L2TNodeCategory.CONTINUE)
                new_node = L2TNode(id="re_explored_child", content=re_explored_thought_content, parent_id=node_id_to_classify, generation_step=2)
                graph.add_node(new_node)
                graph.move_to_hist(node_id_to_classify)
    
            elif node_id_to_classify == "re_explored_child":
                graph.classify_node(node_id_to_classify, L2TNodeCategory.FINAL_ANSWER)
                logging.debug(f"SIDE_EFFECT: Setting final_answer for {node_id_to_classify}, result ID: {id(result)}")
                result.final_answer = final_answer_content
                result.succeeded = True
                logging.debug(f"SIDE_EFFECT: result.final_answer after setting: {result.final_answer}, result.succeeded: {result.succeeded}, result ID: {id(result)}")
                graph.move_to_hist(node_id_to_classify)
    
            else:
                graph.classify_node(node_id_to_classify, L2TNodeCategory.TERMINATE_BRANCH)
                graph.move_to_hist(node_id_to_classify)
    
            self.call_count += 1
            logging.debug(f"SIDE_EFFECT: End of process_node. result.final_answer: {result.final_answer}, result.succeeded: {result.succeeded}, result ID: {id(result)}")
    
        node_processor_mock.process_node.side_effect = side_effect_process_node
        processor.node_processor = node_processor_mock
        result = processor.run(problem_text)
        logging.debug(f"TEST: After processor.run(). result.final_answer: {result.final_answer}, result.succeeded: {result.succeeded}, result ID: {id(result)}")

        # Assertions
        self.assertTrue(result.succeeded)
        self.assertEqual(result.final_answer, final_answer_content)
        self.assertIsNone(result.error_message)
        
        graph = cast(L2TGraph, result.reasoning_graph)
        
        # Verify nodes exist and cast them
        root_node = cast(L2TNode, graph.get_node(cast(str, self.root_node_id)))
        unfruitful_node = cast(L2TNode, graph.get_node(cast(str, self.unfruitful_node_id)))
        re_explored_node = cast(L2TNode, graph.get_node("re_explored_child"))

        self.assertIsNotNone(root_node)
        self.assertIsNotNone(unfruitful_node)
        self.assertIsNotNone(re_explored_node)

        # Verify categories
        self.assertEqual(root_node.category, L2TNodeCategory.CONTINUE)  # Classified twice, last one sticks
        self.assertEqual(unfruitful_node.category, L2TNodeCategory.BACKTRACK)
        self.assertEqual(re_explored_node.category, L2TNodeCategory.FINAL_ANSWER)

        # Verify graph state after backtracking
        # Unfruitful node should be in v_hist
        self.assertIn(unfruitful_node.id, graph.v_hist)
        self.assertNotIn(unfruitful_node.id, graph.v_pres)

        # Root node should have been re-added to v_pres and then processed again
        # It should now be in v_hist after its second processing
        self.assertIn(root_node.id, graph.v_hist)
        self.assertNotIn(root_node.id, graph.v_pres)

        # Re-explored node should be in v_hist (as it led to final answer)
        self.assertIn(re_explored_node.id, graph.v_hist)
        self.assertNotIn(re_explored_node.id, graph.v_pres)

        # Verify parent-child relationships
        self.assertIn(unfruitful_node.id, root_node.children_ids)
        self.assertIn(re_explored_node.id, root_node.children_ids)  # Root node should have two children now
        self.assertEqual(unfruitful_node.parent_id, root_node.id)
        self.assertEqual(re_explored_node.parent_id, root_node.id)

        # Verify generation steps
        self.assertEqual(root_node.generation_step, 0)
        self.assertEqual(unfruitful_node.generation_step, 1)
        self.assertEqual(re_explored_node.generation_step, 2)  # This is important for correct pathing

if __name__ == '__main__':
    unittest.main()
