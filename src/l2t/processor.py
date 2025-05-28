import time
import logging
import uuid  # For generating unique node IDs
from typing import Optional, Tuple

from src.llm_client import LLMClient # Use the actual LLMClient
from src.aot.dataclasses import LLMCallStats # Import LLMCallStats from aot_dataclasses

from .dataclasses import (
    L2TConfig,
    L2TGraph,
    L2TNode,
    L2TNodeCategory,
    L2TResult,
)
from .prompt_generator import L2TPromptGenerator
from .response_parser import L2TResponseParser
from src.l2t_processor_utils.node_processor import NodeProcessor # Import NodeProcessor

logger = logging.getLogger(__name__)


from src.aot.constants import OPENROUTER_API_URL, HTTP_REFERER, APP_TITLE # Import these constants

class L2TProcessor:
    def __init__(self, api_key: str, config: Optional[L2TConfig] = None,
                 enable_rate_limiting: bool = True, enable_audit_logging: bool = True):
        self.llm_client = LLMClient(
            api_key=api_key,
            api_url=OPENROUTER_API_URL,
            http_referer=HTTP_REFERER,
            app_title=APP_TITLE,
            enable_rate_limiting=enable_rate_limiting,
            enable_audit_logging=enable_audit_logging
        )
        self.config = config if config else L2TConfig()
        self.prompt_generator = L2TPromptGenerator(self.config)
        self.node_processor = NodeProcessor(self.llm_client, self.config, self.prompt_generator)
        # Ensure logger is configured to show messages
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def run(self, problem_text: str) -> L2TResult:
        result = L2TResult()
        graph = L2TGraph()
        process_start_time = time.monotonic()
        current_process_step = 0  # Overall process step

        # First Step
        # No budget hints for the initial prompt
        initial_prompt = self.prompt_generator.construct_l2t_initial_prompt(
            problem_text, self.config.x_fmt_default, self.config.x_eva_default
        )
        initial_response_content, initial_stats = self.llm_client.call(
            initial_prompt,
            models=self.config.initial_prompt_model_names,
            temperature=self.config.initial_prompt_temperature,
        )
        self.node_processor._update_result_stats(result, initial_stats) # Use node_processor's method

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
        # logger.info(f"DEBUG: After adding root node, graph.v_pres = {list(graph.v_pres)}")
        # logger.info(f"DEBUG: About to enter while loop with result.final_answer = {result.final_answer}")
        # logger.warning(f"INIT: root_node_id={root_node_id}, graph.v_pres={list(graph.v_pres)}")


        # Reasoning Loop
        termination_reason = None
        while (
            current_process_step < self.config.max_steps
            and len(graph.v_pres) > 0
            and (time.monotonic() - process_start_time) < self.config.max_time_seconds
            and len(graph.nodes) < self.config.max_total_nodes
            and result.final_answer is None
        ):
            current_process_step += 1
            logger.warning(
                f"--- L2T Process Step {current_process_step}/{self.config.max_steps} --- "
                f"V_pres size: {len(graph.v_pres)}, Total nodes: {len(graph.nodes)} ---"
            )

            nodes_to_process_this_round = list(graph.v_pres) # Iterate over a copy

            for node_id_to_classify in nodes_to_process_this_round:
                if result.final_answer is not None:
                    break # Exit inner loop if final answer found

                self.node_processor.process_node(node_id_to_classify, graph, result, current_process_step)

            # Check if any new nodes were added to v_pres in this round.
            if not graph.v_pres and result.final_answer is None:
                logger.info("No new thoughts generated in this step and no final answer. Terminating early.")
                break # Keep termination_reason as None for now, let the final block determine it
        
        # After loop: determine termination reason
        # Prioritize max_steps if it was reached
        if current_process_step >= self.config.max_steps and result.final_answer is None:
            termination_reason = "max_steps"
        elif len(graph.nodes) >= self.config.max_total_nodes and result.final_answer is None:
            termination_reason = "max_total_nodes"
        elif (time.monotonic() - process_start_time) >= self.config.max_time_seconds and result.final_answer is None:
            termination_reason = "max_time"
        elif result.final_answer is None: # If no other reason was met and no final answer
            termination_reason = "no_more_thoughts"

        # Finalization
        result.reasoning_graph = graph
        # logger.debug(f"Finalization - current_process_step: {current_process_step}, max_steps: {self.config.max_steps}, termination_reason: {termination_reason}, final_answer: {result.final_answer}")
        if result.final_answer is None:
            result.succeeded = False
            # logger.debug(f"Setting error message. Current termination_reason: {termination_reason}")
            # logger.debug(f"About to assign error message. termination_reason is: {termination_reason}") # New debug print
            if termination_reason == "max_steps":
                result.error_message = "L2T process completed: Max steps reached."
            elif termination_reason == "max_total_nodes":
                result.error_message = "L2T process completed: Max total nodes reached."
            elif termination_reason == "max_time":
                result.error_message = "L2T process completed: Max time reached."
            elif termination_reason == "no_more_thoughts":
                result.error_message = "L2T process completed: No more thoughts to process and no final answer."
            else: # Should not happen if loop conditions are correct
                result.error_message = "L2T process completed without a final answer for an unknown reason."
            logger.info(f"L2T process finished without a final answer. Reason: {result.error_message}")
        else:
            result.succeeded = True
        
        result.total_process_wall_clock_time_seconds = (
            time.monotonic() - process_start_time
        )
        logger.info(
            f"L2T run finished. Success: {result.succeeded}. LLM Calls: {result.total_llm_calls}. "
            f"Total time: {result.total_process_wall_clock_time_seconds:.2f}s. "
            f"Final Answer: {result.final_answer[:200] if result.final_answer else 'None'}"
        )
        return result
