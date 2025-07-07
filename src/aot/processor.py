import time
import logging
import io
from typing import List, Optional, Tuple, TYPE_CHECKING

from .dataclasses import LLMCallStats, ParsedLLMOutput, AoTRunnerConfig, AoTResult
from src.prompt_generator import PromptGenerator
from src.response_parser import ResponseParser
from .constants import MIN_PREDICTED_STEP_TOKENS_FALLBACK, MIN_PREDICTED_STEP_DURATION_FALLBACK

if TYPE_CHECKING:
    from src.llm_client import LLMClient # For type hinting only

from src.llm_config import LLMConfig # Added

class AoTProcessor:
    def __init__(self, llm_client: 'LLMClient', runner_config: AoTRunnerConfig, llm_config: LLMConfig): # Modified
        self.llm_client = llm_client
        self.runner_config = runner_config # Modified
        self.llm_config = llm_config # Added

    def run(self, problem_text: str) -> Tuple[AoTResult, str]:
        """
        Implements the proper Answer On Thought (AoT) process:
        1. Generate initial answer
        2. Reflect on the answer
        3. Refine the answer based on reflection
        4. Repeat reflection-refinement cycle until satisfied or limits reached
        """
        result = AoTResult()
        process_start_time = time.monotonic()
        
        logging.info(f"Starting proper AoT process for problem: '{problem_text[:100].strip()}...'")
        logging.info(f"Main Models: {', '.join(self.runner_config.main_model_names)}, LLMConfig: {self.llm_config}")
        if self.runner_config.max_reasoning_tokens:
            logging.info(f"Reasoning Token Limit: {self.runner_config.max_reasoning_tokens}")
        logging.info(f"Max Time Limit: {self.runner_config.max_time_seconds}s")
        logging.info(f"Max Iterations: {self.runner_config.max_steps}")

        # Phase 1: Generate Initial Answer
        logging.info("--- Phase 1: Generating Initial Answer ---")
        initial_prompt = PromptGenerator.construct_aot_initial_prompt(problem_text)
        initial_reply, initial_stats = self.llm_client.call(
            initial_prompt, models=self.runner_config.main_model_names, config=self.llm_config
        )
        
        # Update statistics
        result.total_llm_interaction_time_seconds += initial_stats.call_duration_seconds
        result.total_completion_tokens += initial_stats.completion_tokens
        result.reasoning_completion_tokens += initial_stats.completion_tokens
        result.total_prompt_tokens += initial_stats.prompt_tokens
        
        logging.debug(f"Initial Response from {initial_stats.model_name}:\n{initial_reply}")
        logging.info(f"LLM call ({initial_stats.model_name}): Duration: {initial_stats.call_duration_seconds:.2f}s, Tokens (C:{initial_stats.completion_tokens}, P:{initial_stats.prompt_tokens})")
        
        if initial_reply.startswith("Error:"):
            logging.critical(f"LLM call failed for initial answer. Error: {initial_reply}")
            result.succeeded = False
            result.final_answer = f"Error: Initial answer generation failed - {initial_reply}"
            result.total_process_wall_clock_time_seconds = time.monotonic() - process_start_time
            return result, self._generate_aot_summary(result)
        
        # Parse initial answer
        parsed_initial = ResponseParser.parse_llm_output(initial_reply)
        if parsed_initial.initial_answer:
            result.initial_answer = parsed_initial.initial_answer
            result.full_history_for_context.append(f"Initial Answer: {parsed_initial.initial_answer}")
            result.reasoning_trace.append(f"Initial Answer: {parsed_initial.initial_answer}")
            logging.info(f"Initial answer extracted: {parsed_initial.initial_answer[:100]}...")
        else:
            # Fallback: use the entire response as initial answer
            result.initial_answer = initial_reply
            result.full_history_for_context.append(f"Initial Answer: {initial_reply}")
            result.reasoning_trace.append(f"Initial Answer: {initial_reply}")
            logging.warning("No structured initial answer found, using full response")
        
        # Phase 2: Iterative Reflection and Refinement
        current_answer = result.initial_answer
        iteration = 1
        max_iterations = self.runner_config.max_steps
        
        while iteration <= max_iterations:
            elapsed_time = time.monotonic() - process_start_time
            
            # Check time limit
            if self.runner_config.max_time_seconds > 0 and elapsed_time >= self.runner_config.max_time_seconds:
                logging.info(f"Time limit ({self.runner_config.max_time_seconds}s) reached. Stopping iterations.")
                break
            
            # Check token limit
            if (self.runner_config.max_reasoning_tokens and 
                result.reasoning_completion_tokens >= self.runner_config.max_reasoning_tokens):
                logging.info(f"Token limit ({self.runner_config.max_reasoning_tokens}) reached. Stopping iterations.")
                break
            
            logging.info(f"--- Iteration {iteration}/{max_iterations} ---")
            
            # Sub-phase 2a: Reflection
            logging.info(f"--- Reflection Phase {iteration} ---")
            reflection_prompt = PromptGenerator.construct_aot_reflection_prompt(problem_text, result.full_history_for_context)
            reflection_reply, reflection_stats = self.llm_client.call(
                reflection_prompt, models=self.runner_config.main_model_names, config=self.llm_config
            )
            
            # Update statistics
            result.total_llm_interaction_time_seconds += reflection_stats.call_duration_seconds
            result.total_completion_tokens += reflection_stats.completion_tokens
            result.reasoning_completion_tokens += reflection_stats.completion_tokens
            result.total_prompt_tokens += reflection_stats.prompt_tokens
            
            logging.debug(f"Reflection Response from {reflection_stats.model_name}:\n{reflection_reply}")
            logging.info(f"LLM call ({reflection_stats.model_name}): Duration: {reflection_stats.call_duration_seconds:.2f}s, Tokens (C:{reflection_stats.completion_tokens}, P:{reflection_stats.prompt_tokens})")
            
            if reflection_reply.startswith("Error:"):
                logging.warning(f"Reflection failed for iteration {iteration}: {reflection_reply}")
                break
            
            # Parse reflection
            parsed_reflection = ResponseParser.parse_llm_output(reflection_reply)
            if parsed_reflection.reflection_text:
                result.reflections.append(parsed_reflection.reflection_text)
                result.full_history_for_context.append(f"Reflection {iteration}: {parsed_reflection.reflection_text}")
                result.reasoning_trace.append(f"Reflection {iteration}: {parsed_reflection.reflection_text}")
                logging.info(f"Reflection {iteration} captured: {parsed_reflection.reflection_text[:100]}...")
            else:
                # Fallback: use entire response
                result.reflections.append(reflection_reply)
                result.full_history_for_context.append(f"Reflection {iteration}: {reflection_reply}")
                result.reasoning_trace.append(f"Reflection {iteration}: {reflection_reply}")
                logging.warning(f"No structured reflection found for iteration {iteration}, using full response")
            
            # Sub-phase 2b: Refinement
            logging.info(f"--- Refinement Phase {iteration} ---")
            refinement_prompt = PromptGenerator.construct_aot_refinement_prompt(problem_text, result.full_history_for_context)
            refinement_reply, refinement_stats = self.llm_client.call(
                refinement_prompt, models=self.runner_config.main_model_names, config=self.llm_config
            )
            
            # Update statistics
            result.total_llm_interaction_time_seconds += refinement_stats.call_duration_seconds
            result.total_completion_tokens += refinement_stats.completion_tokens
            result.reasoning_completion_tokens += refinement_stats.completion_tokens
            result.total_prompt_tokens += refinement_stats.prompt_tokens
            
            logging.debug(f"Refinement Response from {refinement_stats.model_name}:\n{refinement_reply}")
            logging.info(f"LLM call ({refinement_stats.model_name}): Duration: {refinement_stats.call_duration_seconds:.2f}s, Tokens (C:{refinement_stats.completion_tokens}, P:{refinement_stats.prompt_tokens})")
            
            if refinement_reply.startswith("Error:"):
                logging.warning(f"Refinement failed for iteration {iteration}: {refinement_reply}")
                break
            
            # Parse refinement
            parsed_refinement = ResponseParser.parse_llm_output(refinement_reply)
            if parsed_refinement.refined_answer:
                result.refined_answers.append(parsed_refinement.refined_answer)
                result.full_history_for_context.append(f"Refined Answer {iteration}: {parsed_refinement.refined_answer}")
                result.reasoning_trace.append(f"Refined Answer {iteration}: {parsed_refinement.refined_answer}")
                current_answer = parsed_refinement.refined_answer
                logging.info(f"Refined answer {iteration} captured: {parsed_refinement.refined_answer[:100]}...")
            else:
                # Fallback: use entire response
                result.refined_answers.append(refinement_reply)
                result.full_history_for_context.append(f"Refined Answer {iteration}: {refinement_reply}")
                result.reasoning_trace.append(f"Refined Answer {iteration}: {refinement_reply}")
                current_answer = refinement_reply
                logging.warning(f"No structured refined answer found for iteration {iteration}, using full response")
            
            result.iterations_completed = iteration
            
            # Check for early termination based on no progress
            if iteration > 1 and len(result.refined_answers) >= 2:
                if result.refined_answers[-1] == result.refined_answers[-2]:
                    logging.info(f"No change in refined answer after iteration {iteration}. Stopping early.")
                    break
            
            iteration += 1
            time.sleep(0.2)  # Brief pause between iterations
        
        # Phase 3: Generate Final Answer
        if not result.final_answer:
            logging.info("--- Phase 3: Generating Final Answer ---")
            elapsed_time = time.monotonic() - process_start_time
            can_make_final_call = True
            
            if (self.runner_config.max_time_seconds > 0 and 
                (self.runner_config.max_time_seconds - elapsed_time) < MIN_PREDICTED_STEP_DURATION_FALLBACK * 1.5):
                logging.warning("Very low time budget remaining for final AoT call. Skipping.")
                can_make_final_call = False
            
            if (self.runner_config.max_reasoning_tokens and 
                self.runner_config.max_reasoning_tokens > 0 and
                (self.runner_config.max_reasoning_tokens - result.reasoning_completion_tokens) < MIN_PREDICTED_STEP_TOKENS_FALLBACK * 1.5):
                logging.warning("Very low token budget remaining for final AoT call. Skipping.")
                can_make_final_call = False
            
            if can_make_final_call:
                final_prompt = PromptGenerator.construct_aot_final_prompt(problem_text, result.full_history_for_context)
                final_reply, final_stats = self.llm_client.call(
                    final_prompt, models=self.runner_config.main_model_names, config=self.llm_config
                )
                
                result.total_llm_interaction_time_seconds += final_stats.call_duration_seconds
                result.total_completion_tokens += final_stats.completion_tokens
                result.total_prompt_tokens += final_stats.prompt_tokens
                
                logging.debug(f"Final Response from {final_stats.model_name}:\n{final_reply}")
                logging.info(f"LLM call ({final_stats.model_name}): Duration: {final_stats.call_duration_seconds:.2f}s, Tokens (C:{final_stats.completion_tokens}, P:{final_stats.prompt_tokens})")
                
                if not final_reply.startswith("Error:"):
                    parsed_final = ResponseParser.parse_llm_output(final_reply)
                    if parsed_final.final_answer_text and parsed_final.is_final_answer_marked_done:
                        result.final_answer = parsed_final.final_answer_text
                    else:
                        logging.warning("Could not parse final answer from explicit call. Using full response as fallback.")
                        result.final_answer = final_reply
                else:
                    logging.warning(f"LLM call for final answer failed: {final_reply}")
                    result.final_answer = current_answer  # Use last refined answer
            else:
                logging.info("Skipped explicit final answer generation due to budget constraints.")
                result.final_answer = current_answer  # Use last refined answer
        
        # If we still don't have a final answer, use the most recent refined answer
        if not result.final_answer:
            if result.refined_answers:
                result.final_answer = result.refined_answers[-1]
                logging.info("Using last refined answer as final answer")
            else:
                result.final_answer = result.initial_answer
                logging.info("Using initial answer as final answer")
        
        # Determine success
        if result.final_answer and not result.final_answer.startswith("Error:"):
            result.succeeded = True
        else:
            result.succeeded = False
            if not result.final_answer:
                result.final_answer = "Error: AoT process did not yield a final answer due to limits or parsing issues."
                logging.warning(result.final_answer)
        
        result.total_process_wall_clock_time_seconds = time.monotonic() - process_start_time
        aot_summary_output = self._generate_aot_summary(result)
        return result, aot_summary_output

    def _generate_aot_summary(self, result: AoTResult) -> str:
        output_buffer = io.StringIO()
        output_buffer.write("\n--- AoT Process Summary ---\n")
        output_buffer.write(f"AoT Succeeded: {result.succeeded}\n")
        output_buffer.write(f"Iterations Completed: {result.iterations_completed}\n")
        output_buffer.write(f"Initial Answer: {result.initial_answer[:100] if result.initial_answer else 'None'}...\n")
        output_buffer.write(f"Reflections Generated: {len(result.reflections)}\n")
        output_buffer.write(f"Refined Answers Generated: {len(result.refined_answers)}\n")
        output_buffer.write(f"Total reasoning completion tokens: {result.reasoning_completion_tokens}\n")
        output_buffer.write(f"Total completion tokens (AoT phase: reasoning + final AoT call): {result.total_completion_tokens}\n")
        output_buffer.write(f"Total prompt tokens (all AoT calls): {result.total_prompt_tokens}\n")
        output_buffer.write(f"Grand total AoT tokens: {result.total_completion_tokens + result.total_prompt_tokens}\n")
        output_buffer.write(f"Total AoT LLM interaction time: {result.total_llm_interaction_time_seconds:.2f}s\n")
        output_buffer.write(f"Total AoT process wall-clock time: {result.total_process_wall_clock_time_seconds:.2f}s\n")
        return output_buffer.getvalue()
