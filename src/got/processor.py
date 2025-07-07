import time
import logging
import uuid # For generating unique thought IDs
from typing import List, Optional, Tuple, Set

from src.reasoning_process import ReasoningProcess
from src.llm_client import LLMClient
from src.aot.dataclasses import LLMCallStats # Ensure this is imported
from .dataclasses import (
    GoTConfig,
    GoTModelConfigs,
    GoTGraph,
    GoTThought,
    GoTThoughtStatus,
    GoTResult,
)
from .prompt_generator import GoTPromptGenerator
from .response_parser import GoTResponseParser
from src.communication_logger import log_llm_request, log_llm_response, log_stage, ModelRole

# Configure logging if not already configured
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

class GoTProcessor: # Removed ReasoningProcess from inheritance
    def __init__(self,
                 llm_client: LLMClient,
                 config: GoTConfig,
                 model_configs: GoTModelConfigs):
        self.llm_client = llm_client
        self.config = config
        self.model_configs = model_configs
        self.prompt_generator = GoTPromptGenerator()
        self.response_parser = GoTResponseParser()

    def _generate_thought_id(self) -> str:
        return str(uuid.uuid4())

    def _update_result_stats(self, result: GoTResult, stats: Optional[LLMCallStats]):
        if stats:
            result.total_llm_calls += 1
            result.total_completion_tokens += stats.completion_tokens
            result.reasoning_completion_tokens += stats.completion_tokens  # All GoT operations are reasoning
            result.total_prompt_tokens += stats.prompt_tokens
            result.total_llm_interaction_time_seconds += stats.call_duration_seconds
        # If stats is None, it means an error likely occurred before or during the call,
        # or the LLMClient didn't return stats. This is logged in _call_llm.

    def _call_llm(self, prompt: str, models: List[str], llm_config_type: str, step_info: Optional[str] = None) -> Tuple[str, Optional[LLMCallStats]]:
        selected_llm_config = getattr(self.model_configs, f"{llm_config_type}_config", None)
        if not selected_llm_config:
            err_msg = f"Error: Invalid LLM configuration type '{llm_config_type}'"
            logger.error(err_msg)
            return err_msg, None

        # Map config types to model roles
        role_mapping = {
            "thought_generation": ModelRole.GOT_THOUGHT_GENERATION,
            "scoring": ModelRole.GOT_SCORING,
            "aggregation": ModelRole.GOT_AGGREGATION,
            "refinement": ModelRole.GOT_REFINEMENT
        }
        model_role = role_mapping.get(llm_config_type, ModelRole.GOT_THOUGHT_GENERATION)

        # Log the outgoing request
        config_info = {"temperature": selected_llm_config.temperature, "max_tokens": selected_llm_config.max_tokens}
        comm_id = log_llm_request("GoT", model_role, models, prompt, step_info, config_info)

        response_content, stats = self.llm_client.call(
            prompt=prompt,
            models=models,
            config=selected_llm_config
        )

        # Log the incoming response
        if stats:
            log_llm_response(comm_id, "GoT", model_role, stats.model_name, 
                            response_content, step_info, stats)
        else:
            # This case implies an issue with the LLMClient call itself if stats are missing without an error in response_content
            log_llm_response(comm_id, "GoT", model_role, "unknown", 
                            response_content, step_info, None, error="No stats returned from LLM call")

        return response_content, stats

    def _generate_initial_thoughts(self, problem_description: str, graph: GoTGraph, result: GoTResult) -> None:
        prompt = self.prompt_generator.construct_initial_thought_prompt(problem_description)

        response_content, stats = self._call_llm(
            prompt, self.config.thought_generation_model_names, "thought_generation", "Initial Generation"
        )
        self._update_result_stats(result, stats)

        if response_content.startswith("Error:"):
            logger.error(f"Failed to generate initial thoughts: {response_content}")
            result.error_message = f"Initial thought generation failed: {response_content}"
            return

        parsed_thoughts = self.response_parser.parse_initial_thoughts(response_content)
        if not parsed_thoughts:
            logger.warning("No initial thoughts parsed from LLM response.")
            result.error_message = "No initial thoughts parsed from LLM response."
            return

        for i, thought_content in enumerate(parsed_thoughts):
            if len(graph.thoughts) >= self.config.max_thoughts:
                logger.warning("Max thoughts limit reached during initial thought generation.")
                break
            thought_id = self._generate_thought_id()
            thought = GoTThought(
                id=thought_id,
                content=thought_content,
                generation_step=0, # Iteration 0 for initial thoughts
                status=GoTThoughtStatus.ACTIVE
            )
            graph.add_thought(thought)
            logger.info(f"Added initial thought {thought_id}: {thought_content[:100]}...")
            self._score_thought(problem_description, thought, graph, result)


    def _expand_thought(self, problem_description: str, parent_thought: GoTThought, graph: GoTGraph, result: GoTResult, current_step: int) -> None:
        # Calculate how many more children can be generated for this parent
        allowed_new_children = self.config.max_children_per_thought - len(parent_thought.children_ids)
        if allowed_new_children <= 0:
            logger.debug(f"Thought {parent_thought.id} already has max children ({len(parent_thought.children_ids)}/{self.config.max_children_per_thought}). Skipping expansion.")
            return

        prompt = self.prompt_generator.construct_expand_thought_prompt(
            problem_description, parent_thought, allowed_new_children
        )
        step_info = f"Step {current_step} - Expand {parent_thought.id[:8]}"
        response_content, stats = self._call_llm(
            prompt, self.config.thought_generation_model_names, "thought_generation", step_info
        )
        self._update_result_stats(result, stats)

        if response_content.startswith("Error:"):
            logger.error(f"Failed to expand thought {parent_thought.id}: {response_content}")
            return

        new_thought_contents = self.response_parser.parse_expanded_thoughts(response_content)
        for content in new_thought_contents:
            if len(graph.thoughts) >= self.config.max_thoughts:
                logger.warning("Max thoughts limit reached. Cannot add new expanded thought.")
                break
            if len(parent_thought.children_ids) >= self.config.max_children_per_thought:
                logger.debug(f"Thought {parent_thought.id} reached max children ({len(parent_thought.children_ids)}/{self.config.max_children_per_thought}) during expansion loop.")
                break

            child_id = self._generate_thought_id()
            child_thought = GoTThought(
                id=child_id,
                content=content,
                parent_ids={parent_thought.id},
                generation_step=current_step,
                status=GoTThoughtStatus.ACTIVE
            )
            graph.add_thought(child_thought)
            # GoTGraph.add_thought handles linking parent to child and child to parent

            logger.info(f"Added expanded thought {child_id} from parent {parent_thought.id}: {content[:100]}...")
            self._score_thought(problem_description, child_thought, graph, result)

    def _aggregate_thoughts(self, problem_description: str, thoughts_to_aggregate: List[GoTThought], graph: GoTGraph, result: GoTResult, current_step: int) -> Optional[GoTThought]:
        if not self.config.enable_aggregation or not thoughts_to_aggregate or len(thoughts_to_aggregate) < 2:
            return None

        # Filter out thoughts not in graph just in case list is stale, though unlikely with current flow
        valid_thoughts_to_aggregate = [t for t in thoughts_to_aggregate if t.id in graph.thoughts]
        if len(valid_thoughts_to_aggregate) < 2 : return None

        valid_thoughts_to_aggregate.sort(key=lambda t: t.score, reverse=True)

        prompt = self.prompt_generator.construct_aggregate_thoughts_prompt(
            problem_description, valid_thoughts_to_aggregate[:self.config.max_parents_for_aggregation]
        )
        step_info = f"Step {current_step} - Aggregate {len(valid_thoughts_to_aggregate)} thoughts"
        response_content, stats = self._call_llm(
            prompt, self.config.aggregation_model_names, "aggregation", step_info
        )
        self._update_result_stats(result, stats)

        if response_content.startswith("Error:"):
            logger.error(f"Failed to aggregate thoughts: {response_content}")
            return None

        aggregated_content = self.response_parser.parse_aggregated_thought(response_content)
        if aggregated_content:
            if len(graph.thoughts) >= self.config.max_thoughts:
                logger.warning("Max thoughts limit reached. Cannot add new aggregated thought.")
                return None

            aggregated_thought_id = self._generate_thought_id()
            parent_ids = {t.id for t in valid_thoughts_to_aggregate}

            new_thought = GoTThought(
                id=aggregated_thought_id,
                content=aggregated_content,
                parent_ids=parent_ids,
                generation_step=current_step,
                status=GoTThoughtStatus.ACTIVE
            )
            graph.add_thought(new_thought)
            # GoTGraph.add_thought handles linking

            logger.info(f"Added aggregated thought {aggregated_thought_id}: {aggregated_content[:100]}...")
            self._score_thought(problem_description, new_thought, graph, result)

            for t in valid_thoughts_to_aggregate: # Mark original thoughts that were aggregated
                if t.id in graph.thoughts: # Ensure they still exist
                     graph.update_thought_status(t.id, GoTThoughtStatus.AGGREGATED)
            return new_thought
        return None

    def _refine_thought(self, problem_description: str, thought_to_refine: GoTThought, graph: GoTGraph, result: GoTResult, current_step: int) -> Optional[GoTThought]:
        if not self.config.enable_refinement:
            return None

        prompt = self.prompt_generator.construct_refine_thought_prompt(problem_description, thought_to_refine)

        step_info = f"Step {current_step} - Refine {thought_to_refine.id[:8]}"
        response_content, stats = self._call_llm(
            prompt, self.config.refinement_model_names, "refinement", step_info
        )
        self._update_result_stats(result, stats)

        if response_content.startswith("Error:"):
            logger.error(f"Failed to refine thought {thought_to_refine.id}: {response_content}")
            return None

        refined_content = self.response_parser.parse_refined_thought(response_content)
        # Check if content actually changed, case-insensitively
        if refined_content and refined_content.strip().lower() != thought_to_refine.content.strip().lower():
            if len(graph.thoughts) >= self.config.max_thoughts:
                logger.warning("Max thoughts limit reached. Cannot add new refined thought.")
                return None

            refined_thought_id = self._generate_thought_id()
            # Refined thought becomes a child of the original thought
            new_parent_ids = {thought_to_refine.id}

            new_thought = GoTThought(
                id=refined_thought_id,
                content=refined_content,
                parent_ids=new_parent_ids,
                generation_step=current_step,
                status=GoTThoughtStatus.ACTIVE
            )
            new_thought.history.append(f"Refined from (ID {thought_to_refine.id}): {thought_to_refine.content}")

            graph.add_thought(new_thought)
            # GoTGraph.add_thought handles linking

            logger.info(f"Added refined thought {refined_thought_id} from {thought_to_refine.id}: {refined_content[:100]}...")
            self._score_thought(problem_description, new_thought, graph, result)

            graph.update_thought_status(thought_to_refine.id, GoTThoughtStatus.REFINED)
            return new_thought
        elif refined_content and refined_content.strip().lower() == thought_to_refine.content.strip().lower():
            logger.info(f"Refinement of thought {thought_to_refine.id} did not produce a significant change.")
        else:
            logger.info(f"No refined content parsed for thought {thought_to_refine.id}.")
        return None

    def _score_thought(self, problem_description: str, thought: GoTThought, graph: GoTGraph, result: GoTResult) -> None:
        logger.debug(f"Scoring thought {thought.id}...")
        prompt = self.prompt_generator.construct_score_thought_prompt(problem_description, thought)

        response_content, stats = self._call_llm(
            prompt, self.config.scoring_model_names, "scoring"
        )
        self._update_result_stats(result, stats)

        if response_content.startswith("Error:"):
            logger.error(f"Failed to score thought {thought.id}: {response_content}")
            graph.update_thought_score(thought.id, 0.0) # Assign default low score
            return

        score, justification = self.response_parser.parse_scored_thought(response_content)
        if score is not None:
            graph.update_thought_score(thought.id, score)
            logger.info(f"Scored thought {thought.id}: {score:.2f} (Justification: {justification[:50] if justification else 'N/A'}...)")
            if score >= self.config.solution_found_score_threshold:
                # Avoid duplicate candidates if re-scored
                if not any(c.id == thought.id for c in result.solution_candidates):
                    graph.update_thought_status(thought.id, GoTThoughtStatus.SOLUTION_CANDIDATE)
                    result.solution_candidates.append(thought) # Add to list of candidates
                    logger.info(f"Thought {thought.id} marked as solution candidate with score {score:.2f}.")
                # else: thought might already be a candidate, score updated.
        else:
            logger.warning(f"Could not parse score for thought {thought.id}. Assigning default score 0.0.")
            graph.update_thought_score(thought.id, 0.0)

    def _prune_thoughts(self, graph: GoTGraph, result: GoTResult) -> None:
        if not self.config.enable_pruning or self.config.pruning_threshold_score is None:
            return

        logger.info("Pruning thoughts...")
        pruned_count = 0
        # Iterate over a copy of thought IDs if modifying graph.thoughts directly,
        # but here we only update status.
        for thought_id in list(graph.thoughts.keys()):
            thought = graph.get_thought(thought_id)
            if not thought: continue

            # Do not prune if it's already marked as a solution candidate, refined, or aggregated.
            # Or if it's already pruned.
            if thought.status != GoTThoughtStatus.ACTIVE:
                continue

            if thought.score < self.config.pruning_threshold_score:
                graph.update_thought_status(thought.id, GoTThoughtStatus.PRUNED)
                pruned_count +=1
                logger.info(f"Pruned thought {thought.id} (score: {thought.score:.2f}).")

        if pruned_count > 0:
            logger.info(f"Pruned {pruned_count} thoughts.")


    def run(self, problem_description: str) -> GoTResult:
        result = GoTResult()
        graph = GoTGraph()
        result.final_graph = graph
        process_start_time = time.monotonic()

        self._generate_initial_thoughts(problem_description, graph, result)

        if not graph.thoughts:
            logger.error("GoT process failed: No initial thoughts were generated or survived initial processing.")
            result.succeeded = False
            if not result.error_message:
                 result.error_message = "Failed to generate any initial thoughts that passed scoring."
            result.total_process_wall_clock_time_seconds = time.monotonic() - process_start_time
            return result

        for current_iter in range(self.config.max_iterations):
            iteration_number = current_iter + 1
            logger.info(f"--- GoT Iteration {iteration_number}/{self.config.max_iterations} ---")

            # Termination checks
            if (time.monotonic() - process_start_time) >= self.config.max_time_seconds:
                logger.info("Max process time limit reached. Stopping GoT iterations.")
                break
            # Check token budget limit
            if (self.config.max_reasoning_tokens and 
                result.reasoning_completion_tokens >= self.config.max_reasoning_tokens):
                logger.info(f"Token limit ({self.config.max_reasoning_tokens}) reached. Stopping GoT iterations.")
                break
            # Check overall thought limit before starting operations that add thoughts
            if len(graph.thoughts) >= self.config.max_thoughts:
                logger.info("Max thoughts limit (overall) reached. Stopping GoT iterations.")
                break

            # Check for high-scoring solution candidate
            if result.solution_candidates:
                result.solution_candidates.sort(key=lambda t: t.score, reverse=True)
                if result.solution_candidates[0].score >= self.config.solution_found_score_threshold:
                    logger.info(f"High-scoring solution candidate ({result.solution_candidates[0].id}, score: {result.solution_candidates[0].score:.2f}) found. Stopping GoT iterations early.")
                    break

            active_thoughts_for_step = [t for t in graph.thoughts.values() if t.status == GoTThoughtStatus.ACTIVE]
            if not active_thoughts_for_step:
                logger.info("No active thoughts available for current iteration. Stopping.")
                break
            active_thoughts_for_step.sort(key=lambda t: t.score, reverse=True)

            # --- Operations for the current iteration ---
            expanded_this_iteration = 0
            # 1. Expansion
            for thought_to_expand in active_thoughts_for_step:
                if len(graph.thoughts) >= self.config.max_thoughts:
                    logger.warning("Max thoughts limit reached during expansion phase of iteration.")
                    break
                if thought_to_expand.score < self.config.min_score_for_expansion:
                    continue # Skip low-scoring thoughts for expansion

                self._expand_thought(problem_description, thought_to_expand, graph, result, iteration_number)
                expanded_this_iteration += 1
                # Optional: Limit number of expansions per iteration step if desired
                # if expanded_this_iteration >= SOME_ITERATION_EXPANSION_LIMIT: break

            # 2. Aggregation (conditionally)
            if self.config.enable_aggregation and iteration_number % 2 == 0 :
                if len(graph.thoughts) < self.config.max_thoughts:
                    agg_candidates = [t for t in graph.thoughts.values() if t.status == GoTThoughtStatus.ACTIVE and t.score > 0.6] # Example threshold
                    agg_candidates.sort(key=lambda t: t.score, reverse=True)
                    if len(agg_candidates) >= 2:
                        self._aggregate_thoughts(problem_description, agg_candidates[:self.config.max_parents_for_aggregation], graph, result, iteration_number)

            # 3. Refinement (conditionally)
            if self.config.enable_refinement:
                 if len(graph.thoughts) < self.config.max_thoughts:
                    ref_candidates = [t for t in graph.thoughts.values() if t.status == GoTThoughtStatus.ACTIVE and t.score > 0.7] # Example threshold
                    ref_candidates.sort(key=lambda t: t.score, reverse=True)
                    for thought_to_refine in ref_candidates[:2]: # Refine top N
                        if len(graph.thoughts) >= self.config.max_thoughts: break
                        self._refine_thought(problem_description, thought_to_refine, graph, result, iteration_number)

            # 4. Pruning
            if self.config.enable_pruning:
                self._prune_thoughts(graph, result)

            # Post-transformation check for solution candidates
            if result.solution_candidates:
                result.solution_candidates.sort(key=lambda t: t.score, reverse=True)
                if result.solution_candidates[0].score >= self.config.solution_found_score_threshold:
                    logger.info(f"High-scoring solution candidate ({result.solution_candidates[0].id}) identified post-transformation. Stopping.")
                    break

            # Check for stable state (no expansions and no new active thoughts from agg/refine)
            current_active_thoughts_after_ops = [t for t in graph.thoughts.values() if t.status == GoTThoughtStatus.ACTIVE]
            if expanded_this_iteration == 0 and not current_active_thoughts_after_ops and not result.solution_candidates:
                logger.info("No expansions this iteration and no active thoughts remaining. Stopping.")
                break
        # --- End of Iteration Loop ---

        # Determine final answer
        if result.solution_candidates: # Already sorted by score desc
            best_candidate = result.solution_candidates[0]
            if best_candidate.score >= self.config.min_score_for_expansion:
                result.final_answer = best_candidate.content
                result.succeeded = True
                logger.info(f"GoT process finished. Best solution candidate: {best_candidate.id} (Score: {best_candidate.score:.2f})")
            else:
                result.succeeded = False
                result.final_answer = None
                result.error_message = f"GoT: Best candidate score ({best_candidate.score:.2f}) was below quality threshold ({self.config.min_score_for_expansion})."
                logger.warning(result.error_message)
        else:
            all_thoughts_sorted = sorted([t for t in graph.thoughts.values() if t.status != GoTThoughtStatus.PRUNED], key=lambda t: t.score, reverse=True)
            if all_thoughts_sorted and all_thoughts_sorted[0].score >= self.config.min_score_for_expansion * 0.7:
                fallback_thought = all_thoughts_sorted[0]
                result.final_answer = fallback_thought.content
                result.succeeded = True
                logger.info(f"GoT process finished. No strong solution candidate. Using highest-scored non-pruned thought as fallback: {fallback_thought.id} (Score: {fallback_thought.score:.2f})")
            else:
                result.succeeded = False
                result.error_message = "GoT process completed without a viable final answer."
                logger.warning(result.error_message)

        result.total_process_wall_clock_time_seconds = time.monotonic() - process_start_time
        return result

    # Removed get_result as it was tied to ReasoningProcess interface and not suitable here.
    # The run() method returns the GoTResult directly.
