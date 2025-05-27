import time
import logging
import uuid  # For generating unique node IDs
from typing import Optional, Tuple

from src.llm_client import LLMClient # Use the actual LLMClient
from src.aot_dataclasses import LLMCallStats # Import LLMCallStats from aot_dataclasses

from src.l2t_dataclasses import (
    L2TConfig,
    L2TGraph,
    L2TNode,
    L2TNodeCategory,
    L2TResult,
)
from src.l2t_prompt_generator import L2TPromptGenerator
from src.l2t_response_parser import L2TResponseParser

logger = logging.getLogger(__name__)


class L2TProcessor:
    def __init__(self, llm_client: LLMClient, config: Optional[L2TConfig] = None):
        self.llm_client = llm_client
        self.config = config if config else L2TConfig()
        self.prompt_generator = L2TPromptGenerator(self.config)
        # Ensure logger is configured to show messages
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    def _update_result_stats(self, result: L2TResult, stats: LLMCallStats): # stats is no longer Optional here
        if stats: # This check is technically redundant if stats is not Optional, but harmless
            result.total_llm_calls += 1
            result.total_completion_tokens += stats.completion_tokens
            result.total_prompt_tokens += stats.prompt_tokens
            result.total_llm_interaction_time_seconds += stats.call_duration_seconds

    def run(self, problem_text: str) -> L2TResult:
        result = L2TResult()
        graph = L2TGraph()
        process_start_time = time.monotonic()
        current_process_step = 0  # Overall process step

        # First Step
        initial_prompt = self.prompt_generator.construct_l2t_initial_prompt(
            problem_text, self.config.x_fmt_default, self.config.x_eva_default
        )
        initial_response_content, initial_stats = self.llm_client.call(
            initial_prompt,
            models=self.config.initial_prompt_model_names,
            temperature=self.config.initial_prompt_temperature,
        )
        self._update_result_stats(result, initial_stats)

        parsed_initial_thought = (
            L2TResponseParser.parse_l2t_initial_response(initial_response_content)
        )

        if initial_response_content.startswith("Error:") or parsed_initial_thought is None:
            error_msg = f"Failed during initial thought generation. LLM Response: {initial_response_content}"
            logger.error(error_msg)
            result.error_message = error_msg
            result.succeeded = False
            result.total_process_wall_clock_time_seconds = (
                time.monotonic() - process_start_time
            )
            result.reasoning_graph = graph # Store graph even on early failure
            return result

        root_node_id = str(uuid.uuid4())
        root_node = L2TNode(
            id=root_node_id,
            content=parsed_initial_thought,
            parent_id=None,
            generation_step=0, # Initial node is at generation_step 0
        )
        graph.add_node(root_node, is_root=True)
        logger.info(f"Initial thought node {root_node_id} created: '{parsed_initial_thought[:100]}...'")


        # Reasoning Loop
        while (
            current_process_step < self.config.max_steps
            and len(graph.v_pres) > 0
            and (time.monotonic() - process_start_time) < self.config.max_time_seconds
            and len(graph.nodes) < self.config.max_total_nodes
            and result.final_answer is None
        ):
            current_process_step += 1
            logger.info(
                f"--- L2T Process Step {current_process_step}/{self.config.max_steps} --- "
                f"V_pres size: {len(graph.v_pres)}, Total nodes: {len(graph.nodes)} ---"
            )

            nodes_to_process_this_round = list(graph.v_pres) # Iterate over a copy
            # graph.v_pres.clear() # Clear v_pres; nodes will be added back if they lead to new thoughts or explicitly re-added

            for node_id_to_classify in nodes_to_process_this_round:
                if result.final_answer is not None:
                    break # Exit inner loop if final answer found

                node_to_classify = graph.get_node(node_id_to_classify)
                if not node_to_classify:
                    logger.error(f"Node {node_id_to_classify} not found in graph during processing round.")
                    if node_id_to_classify in graph.v_pres: # defensive removal
                        graph.v_pres.remove(node_id_to_classify)
                    graph.move_to_hist(node_id_to_classify) # Ensure it's moved to hist if somehow still there
                    continue

                # Ensure node hasn't been processed already in this step via another path (e.g. complex backtrack later)
                if node_to_classify.category is not None:
                    logger.debug(f"Node {node_id_to_classify} already classified as {node_to_classify.category}. Moving to hist.")
                    graph.move_to_hist(node_id_to_classify) # Ensure it's removed from v_pres
                    continue


                # Node Classification
                parent_node = graph.get_parent(node_to_classify.id)
                parent_content = parent_node.content if parent_node else "This is the initial thought."
                # More detailed context for classification
                ancestor_path_str = ""
                current_ancestor = parent_node
                path_list = [f"Current thought: '{node_to_classify.content}'"]
                if parent_node:
                    path_list.append(f"Direct parent thought: '{parent_node.content}'")
                temp_count = 0
                while current_ancestor and temp_count < 3: # Limit context length
                    grandparent = graph.get_parent(current_ancestor.id)
                    if grandparent:
                        path_list.append(f"Grandparent thought: '{grandparent.content}'")
                    current_ancestor = grandparent
                    temp_count +=1
                ancestor_path_str = "\n".join(reversed(path_list))

                graph_context_for_classification = (
                    f"Reasoning path leading to current thought:\n{ancestor_path_str}\n\n"
                    f"You are classifying the 'Current thought'."
                )

                classification_prompt = self.prompt_generator.construct_l2t_node_classification_prompt(
                    graph_context_for_classification,
                    node_to_classify.content, # This is already included in graph_context
                    self.config.x_eva_default,
                )
                (
                    classification_response_content,
                    classification_stats,
                ) = self.llm_client.call(
                    classification_prompt,
                    models=self.config.classification_model_names,
                    temperature=self.config.classification_temperature,
                )
                self._update_result_stats(result, classification_stats)
                node_category = L2TResponseParser.parse_l2t_node_classification_response(
                    classification_response_content
                )

                if classification_response_content.startswith("Error:") or node_category is None:
                    logger.warning(
                        f"Node classification failed for node {node_to_classify.id}. "
                        f"Defaulting to TERMINATE_BRANCH. Response: {classification_response_content}"
                    )
                    node_category = L2TNodeCategory.TERMINATE_BRANCH
                
                graph.classify_node(node_to_classify.id, node_category)
                logger.info(f"Node {node_to_classify.id} classified as {node_category.name}")

                # Thought Generation / Graph Update
                if node_category == L2TNodeCategory.CONTINUE:
                    # Context for generation: just the parent is usually enough as per many ToT papers
                    thought_gen_context = (
                        f"The parent thought, which you should build upon, is: '{node_to_classify.content}'. "
                        "Generate the next single thought in the reasoning chain."
                    )
                    thought_gen_prompt = self.prompt_generator.construct_l2t_thought_generation_prompt(
                        thought_gen_context, # graph_context
                        node_to_classify.content, # parent_node_content (current node becomes parent for next)
                        self.config.x_fmt_default,
                        self.config.x_eva_default,
                    )
                    (
                        new_thought_response_content,
                        new_thought_stats,
                    ) = self.llm_client.call(
                        thought_gen_prompt,
                        models=self.config.thought_generation_model_names,
                        temperature=self.config.thought_generation_temperature,
                    )
                    self._update_result_stats(result, new_thought_stats)
                    new_thought_content = L2TResponseParser.parse_l2t_thought_generation_response(
                        new_thought_response_content
                    )

                    if new_thought_response_content.startswith("Error:") or new_thought_content is None:
                        logger.warning(
                            f"Thought generation failed for parent node {node_to_classify.id}. "
                            f"Response: {new_thought_response_content}"
                        )
                        # No new node, current node to_classify will be moved to hist.
                    else:
                        new_node_id = str(uuid.uuid4())
                        # New node's generation_step is the current_process_step
                        new_node = L2TNode(
                            id=new_node_id,
                            content=new_thought_content,
                            parent_id=node_to_classify.id,
                            generation_step=node_to_classify.generation_step + 1, # generation_step of the node
                        )
                        graph.add_node(new_node) # Adds to v_pres for the *next* round
                        # node_to_classify.children_ids.append(new_node_id) # L2TGraph.add_node handles this
                        logger.info(
                            f"Generated new thought node {new_node_id} (gen_step {new_node.generation_step}) from parent {node_to_classify.id}"
                        )
                
                elif node_category == L2TNodeCategory.FINAL_ANSWER:
                    result.final_answer = node_to_classify.content
                    result.succeeded = True
                    logger.info(
                        f"Final answer found at node {node_to_classify.id}: {result.final_answer[:100]}..."
                    )
                    # Outer loop condition `result.final_answer is None` will handle termination.
                
                elif node_category == L2TNodeCategory.TERMINATE_BRANCH:
                    logger.info(
                        f"Terminating branch at node {node_to_classify.id}: '{node_to_classify.content[:100]}...'"
                    )
                
                elif node_category == L2TNodeCategory.BACKTRACK:
                    logger.info(
                        f"Backtrack requested at node {node_to_classify.id}. Basic: Treating as TERMINATE_BRANCH for now."
                    )
                    # Basic implementation: treat as terminate for now.
                    # Future: could re-add parent to v_pres if not too deep or not processed recently.
                    # For example:
                    # parent_of_backtrack_node = graph.get_parent(node_to_classify.id)
                    # if parent_of_backtrack_node and parent_of_backtrack_node.id not in graph.v_pres and parent_of_backtrack_node.id in graph.v_hist:
                    #    logger.info(f"Attempting to reactivate parent {parent_of_backtrack_node.id} due to backtrack.")
                    #    graph.v_hist.remove(parent_of_backtrack_node.id)
                    #    graph.v_pres.append(parent_of_backtrack_node.id)
                    #    # Reset category of parent? Or allow re-classification?
                    #    parent_of_backtrack_node.category = None # Allow re-evaluation if needed

                graph.move_to_hist(node_id_to_classify) # Move processed node from v_pres to v_hist

            # Check if any new nodes were added to v_pres in this round.
            # If v_pres is empty after processing all nodes in nodes_to_process_this_round,
            # it means no CONTINUE nodes generated valid new thoughts.
            if not graph.v_pres and result.final_answer is None:
                logger.info("No new thoughts generated in this step and no final answer. Terminating early.")
                break


        # Finalization
        result.reasoning_graph = graph
        if result.final_answer is None:
            result.succeeded = False
            if len(graph.nodes) >= self.config.max_total_nodes:
                result.error_message = "L2T process completed: Max total nodes reached."
            elif current_process_step >= self.config.max_steps:
                result.error_message = "L2T process completed: Max steps reached."
            elif (time.monotonic() - process_start_time) >= self.config.max_time_seconds:
                result.error_message = "L2T process completed: Max time reached."
            elif not graph.v_pres:
                result.error_message = "L2T process completed: No more thoughts to process and no final answer."
            else: # Should not happen if loop conditions are correct
                result.error_message = "L2T process completed without a final answer for an unknown reason."
            logger.info(f"L2T process finished without a final answer. Reason: {result.error_message}")
        
        result.total_process_wall_clock_time_seconds = (
            time.monotonic() - process_start_time
        )
        logger.info(
            f"L2T run finished. Success: {result.succeeded}. LLM Calls: {result.total_llm_calls}. "
            f"Total time: {result.total_process_wall_clock_time_seconds:.2f}s. "
            f"Final Answer: {result.final_answer[:200] if result.final_answer else 'None'}"
        )
        return result
