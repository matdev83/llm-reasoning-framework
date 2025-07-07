import logging
import uuid
from typing import Optional, Tuple

from src.llm_client import LLMClient
from src.llm_config import LLMConfig # Added
from src.aot.dataclasses import LLMCallStats
from src.l2t.dataclasses import L2TConfig, L2TGraph, L2TNode, L2TNodeCategory, L2TResult
from src.l2t.prompt_generator import L2TPromptGenerator
from src.l2t.response_parser import L2TResponseParser
from src.communication_logger import log_llm_request, log_llm_response, ModelRole

logger = logging.getLogger(__name__)

class NodeProcessor:
    def __init__(self, llm_client: LLMClient, l2t_config: L2TConfig, prompt_generator: L2TPromptGenerator, llm_config: LLMConfig): # Modified
        self.llm_client = llm_client
        self.l2t_config = l2t_config # Modified
        self.prompt_generator = prompt_generator
        self.llm_config = llm_config # Added

    def process_node(self, node_id_to_classify: str, graph: L2TGraph, result: L2TResult, current_process_step: int) -> None:
        node_to_classify = graph.get_node(node_id_to_classify)
        if not node_to_classify:
            logger.error(f"Node {node_id_to_classify} not found in graph during processing round.")
            if node_id_to_classify in graph.v_pres:
                graph.v_pres.remove(node_id_to_classify)
            graph.move_to_hist(node_id_to_classify)
            return

        if node_to_classify.category is not None:
            logger.debug(f"Node {node_to_classify.id} already classified as {node_to_classify.category}. Moving to hist.")
            graph.move_to_hist(node_id_to_classify)
            return

        # Calculate remaining steps and check for budget hint
        remaining_steps_hint: Optional[int] = None
        if self.l2t_config.pass_remaining_steps_pct is not None and self.l2t_config.max_steps > 0: # Modified
            remaining_steps = self.l2t_config.max_steps - current_process_step # Modified
            if remaining_steps <= (self.l2t_config.max_steps * self.l2t_config.pass_remaining_steps_pct): # Modified
                remaining_steps_hint = remaining_steps
                logger.info(f"Budget hint: {remaining_steps} steps remaining (threshold {self.l2t_config.pass_remaining_steps_pct*100}%).") # Modified

        # Node Classification
        parent_node = graph.get_parent(node_to_classify.id)
        parent_content = parent_node.content if parent_node else "This is the initial thought."
        ancestor_path_str = ""
        current_ancestor = parent_node
        path_list = [f"Current thought: '{node_to_classify.content}'"]
        if parent_node:
            path_list.append(f"Direct parent thought: '{parent_node.content}'")
        temp_count = 0
        while current_ancestor and temp_count < 3:
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
            node_to_classify.content,
            self.l2t_config.x_eva_default, # Modified
            remaining_steps_hint
        )
        
        # Log the outgoing classification request
        config_info = {"temperature": self.llm_config.temperature, "max_tokens": self.llm_config.max_tokens}
        step_info = f"Step {current_process_step} - Classify Node {node_to_classify.id[:8]}"
        comm_id = log_llm_request("L2T", ModelRole.L2T_CLASSIFICATION, 
                                 self.l2t_config.classification_model_names, 
                                 classification_prompt, step_info, config_info)
        
        (
            classification_response_content,
            classification_stats,
        ) = self.llm_client.call(
            classification_prompt,
            models=self.l2t_config.classification_model_names, # Modified
            config=self.llm_config, # Modified
        )
        
        # Log the incoming classification response
        log_llm_response(comm_id, "L2T", ModelRole.L2T_CLASSIFICATION, 
                        classification_stats.model_name, classification_response_content, 
                        step_info, classification_stats)
        
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
            thought_gen_context = (
                f"The parent thought, which you should build upon, is: '{node_to_classify.content}'. "
                "Generate the next single thought in the reasoning chain."
            )
            thought_gen_prompt = self.prompt_generator.construct_l2t_thought_generation_prompt(
                thought_gen_context,
                node_to_classify.content,
                self.l2t_config.x_fmt_default, # Modified
                self.l2t_config.x_eva_default, # Modified
                remaining_steps_hint
            )
            
            # Log the outgoing thought generation request
            step_info = f"Step {current_process_step} - Generate from Node {node_to_classify.id[:8]}"
            comm_id = log_llm_request("L2T", ModelRole.L2T_THOUGHT_GENERATION, 
                                     self.l2t_config.thought_generation_model_names, 
                                     thought_gen_prompt, step_info, config_info)
            
            (
                new_thought_response_content,
                new_thought_stats,
            ) = self.llm_client.call(
                thought_gen_prompt,
                models=self.l2t_config.thought_generation_model_names, # Modified
                config=self.llm_config, # Modified
            )
            
            # Log the incoming thought generation response
            log_llm_response(comm_id, "L2T", ModelRole.L2T_THOUGHT_GENERATION, 
                            new_thought_stats.model_name, new_thought_response_content, 
                            step_info, new_thought_stats)
            
            self._update_result_stats(result, new_thought_stats)
            new_thought_content = L2TResponseParser.parse_l2t_thought_generation_response(
                new_thought_response_content
            )

            if new_thought_response_content.startswith("Error:") or new_thought_content is None:
                logger.warning(
                    f"Thought generation failed for parent node {node_to_classify.id}. "
                    f"Response: {new_thought_response_content}"
                )
            else:
                new_node_id = str(uuid.uuid4())
                new_node = L2TNode(
                    id=new_node_id,
                    content=new_thought_content,
                    parent_id=node_to_classify.id,
                    generation_step=node_to_classify.generation_step + 1,
                )
                graph.add_node(new_node)
                logger.info(
                    f"Generated new thought node {new_node_id} (gen_step {new_node.generation_step}) from parent {node_to_classify.id}"
                )

        elif node_category == L2TNodeCategory.FINAL_ANSWER:
            result.final_answer = node_to_classify.content
            result.succeeded = True
            logger.info(
                f"Final answer found at node {node_to_classify.id}: {result.final_answer[:100]}..."
            )

        elif node_category == L2TNodeCategory.TERMINATE_BRANCH:
            logger.info(
                f"Terminating branch at node {node_to_classify.id}: '{node_to_classify.content[:100]}...'"
            )

        elif node_category == L2TNodeCategory.BACKTRACK:
            logger.info(f"Backtrack requested at node {node_to_classify.id}.")
            parent_node = graph.get_parent(node_to_classify.id)
            if parent_node:
                # Re-add parent to v_pres to allow for new thought generation from that point
                graph.re_add_to_v_pres(parent_node.id)
                logger.info(f"Parent node {parent_node.id} re-added to v_pres for re-exploration.")
            else:
                logger.info(f"Node {node_to_classify.id} is a root node or has no parent. Cannot backtrack further.")

            # The current node that led to BACKTRACK is considered processed (unfruitful path)
            graph.move_to_hist(node_id_to_classify)

    def _update_result_stats(self, result: L2TResult, stats: LLMCallStats):
        if stats:
            result.total_llm_calls += 1
            result.total_completion_tokens += stats.completion_tokens
            result.total_prompt_tokens += stats.prompt_tokens
            result.total_llm_interaction_time_seconds += stats.call_duration_seconds
