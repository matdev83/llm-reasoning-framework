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

        # Improved error handling - don't immediately default to TERMINATE_BRANCH
        if classification_response_content.startswith("Error:"):
            logger.warning(
                f"Node classification API error for node {node_to_classify.id}. "
                f"Response: {classification_response_content}"
            )
            # For API errors, try to continue reasoning rather than terminating
            # This allows the process to continue even with transient API issues
            if "429" in classification_response_content or "Too Many Requests" in classification_response_content:
                logger.info(f"Rate limit detected, classifying node {node_to_classify.id} as CONTINUE to allow progression")
                node_category = L2TNodeCategory.CONTINUE
            else:
                logger.info(f"API error detected, classifying node {node_to_classify.id} as TERMINATE_BRANCH")
                node_category = L2TNodeCategory.TERMINATE_BRANCH
        elif node_category is None:
            logger.warning(
                f"Node classification parsing failed for node {node_to_classify.id}. "
                f"Response: {classification_response_content}"
            )
            # If parsing failed but we have a response, try to infer from content
            response_lower = classification_response_content.lower()
            if any(word in response_lower for word in ["final", "answer", "solution", "conclude", "decision"]):
                logger.info(f"Inferring FINAL_ANSWER from response content for node {node_to_classify.id}")
                node_category = L2TNodeCategory.FINAL_ANSWER
            elif any(word in response_lower for word in ["continue", "proceed", "next", "further"]):
                logger.info(f"Inferring CONTINUE from response content for node {node_to_classify.id}")
                node_category = L2TNodeCategory.CONTINUE
            elif any(word in response_lower for word in ["terminate", "stop", "end", "dead"]):
                logger.info(f"Inferring TERMINATE_BRANCH from response content for node {node_to_classify.id}")
                node_category = L2TNodeCategory.TERMINATE_BRANCH
            else:
                logger.info(f"Cannot infer classification, defaulting to CONTINUE for node {node_to_classify.id}")
                node_category = L2TNodeCategory.CONTINUE

        graph.classify_node(node_to_classify.id, node_category)
        logger.info(f"Node {node_to_classify.id} classified as {node_category.name}")

        # Add convergence detection - override classification if content suggests final answer
        if node_category == L2TNodeCategory.CONTINUE:
            if self._is_likely_final_answer(node_to_classify.content):
                logger.info(f"Convergence detected: Node {node_to_classify.id} content suggests final answer, overriding classification")
                node_category = L2TNodeCategory.FINAL_ANSWER
                graph.classify_node(node_to_classify.id, node_category)

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

    def _is_likely_final_answer(self, content: str) -> bool:
        """
        Analyze thought content to determine if it likely represents a final answer.
        Returns True if the content suggests this is a conclusive answer.
        """
        content_lower = content.lower()
        
        # Check for definitive conclusion patterns
        conclusion_patterns = [
            "should choose",
            "the answer is",
            "the solution is",
            "therefore",
            "in conclusion",
            "the decision should be",
            "the result is",
            "the correct",
            "the best",
            "the optimal",
            "should be selected",
            "is the right",
            "must be",
            "final decision",
            "recommended",
            "conclusion:",
            "my recommendation",
            "i recommend",
            "the company should",
            "should go with",
            "should pick",
            "is better",
            "is superior",
            "is preferable",
            "would be wise",
            "makes sense",
            "clear choice",
            "obvious choice"
        ]
        
        # Check for decision-making patterns (especially for ethical dilemmas)
        decision_patterns = [
            "maneuver",
            "option",
            "choice",
            "alternative",
            "select",
            "pick",
            "decide",
            "software",
            "solution",
            "approach",
            "method",
            "strategy"
        ]
        
        # Check for final answer indicators
        final_indicators = [
            "final",
            "ultimate",
            "definitive",
            "conclusive",
            "end result",
            "bottom line",
            "summary",
            "overall"
        ]
        
        # Check for comparative language that suggests a final choice
        comparative_patterns = [
            "better than",
            "worse than",
            "more than",
            "less than",
            "higher than",
            "lower than",
            "superior to",
            "inferior to",
            "outperforms",
            "exceeds"
        ]
        
        # Count matches
        conclusion_matches = sum(1 for pattern in conclusion_patterns if pattern in content_lower)
        decision_matches = sum(1 for pattern in decision_patterns if pattern in content_lower)
        final_matches = sum(1 for pattern in final_indicators if pattern in content_lower)
        comparative_matches = sum(1 for pattern in comparative_patterns if pattern in content_lower)
        
        # Check for specific answer format (e.g., "The AV should choose Maneuver B")
        has_specific_answer = any(pattern in content_lower for pattern in ["should choose", "choose", "select", "recommend"]) and \
                             any(pattern in content_lower for pattern in ["maneuver", "option", "alternative", "solution", "software"])
        
        # Check for value/cost analysis conclusions
        has_value_conclusion = any(pattern in content_lower for pattern in ["best value", "most cost-effective", "cheapest", "most expensive", "worth it"]) and \
                              any(pattern in content_lower for pattern in ["therefore", "so", "thus", "hence"])
        
        # Check content length - very short content is less likely to be final
        is_substantial = len(content.strip()) > 30  # Lowered threshold
        
        # Check for definitive statements
        has_definitive_statement = any(pattern in content_lower for pattern in ["clearly", "obviously", "definitely", "certainly", "undoubtedly"])
        
        # Check for any form of recommendation or conclusion
        has_any_conclusion = any(pattern in content_lower for pattern in ["should", "recommend", "suggest", "advise", "propose", "conclude"])
        
        # Check for direct answers to questions
        has_direct_answer = any(pattern in content_lower for pattern in ["yes", "no", "true", "false", "correct", "incorrect"])
        
        # Decision logic - made more sensitive
        if has_specific_answer and is_substantial:
            return True
        
        if has_value_conclusion and is_substantial:
            return True
            
        if conclusion_matches >= 2 and is_substantial:
            return True
            
        if (conclusion_matches >= 1 and decision_matches >= 1 and is_substantial):
            return True
            
        if final_matches >= 1 and conclusion_matches >= 1 and is_substantial:
            return True
            
        if comparative_matches >= 1 and conclusion_matches >= 1 and is_substantial:
            return True
            
        if has_definitive_statement and conclusion_matches >= 1 and is_substantial:
            return True
            
        # More aggressive: any conclusion-like language
        if has_any_conclusion and is_substantial and len(content.strip()) > 100:
            return True
            
        if has_direct_answer and conclusion_matches >= 1 and is_substantial:
            return True
            
        return False
