import time
import logging
import uuid  # For generating unique node IDs
from typing import Optional, Tuple

from src.llm_client import LLMClient
from src.llm_config import LLMConfig
from src.aot.dataclasses import LLMCallStats

from .dataclasses import (
    L2TConfig,
    L2TGraph,
    L2TNode,
    L2TNodeCategory,
    L2TResult,
)
from .prompt_generator import L2TPromptGenerator
from .response_parser import L2TResponseParser
from src.l2t_processor_utils.node_processor import NodeProcessor # Moved to __init__
from src.communication_logger import log_llm_request, log_llm_response, log_stage, ModelRole

logger = logging.getLogger(__name__)

from src.aot.constants import OPENROUTER_API_URL, HTTP_REFERER, APP_TITLE

class L2TProcessor:
    def __init__(self,
                 api_key: str,
                 l2t_config: Optional[L2TConfig] = None,
                 initial_thought_llm_config: Optional[LLMConfig] = None, # Changed
                 node_processor_llm_config: Optional[LLMConfig] = None,  # Added
                 enable_rate_limiting: bool = True,
                 enable_audit_logging: bool = True):
        self.llm_client = LLMClient(
            api_key=api_key,
            api_url=OPENROUTER_API_URL,
            http_referer=HTTP_REFERER,
            app_title=APP_TITLE,
            enable_rate_limiting=enable_rate_limiting,
            enable_audit_logging=enable_audit_logging
        )
        self.l2t_config = l2t_config if l2t_config else L2TConfig()
        # Store specific LLMConfigs, providing defaults if None
        self.initial_thought_llm_config = initial_thought_llm_config if initial_thought_llm_config else LLMConfig()
        self.node_processor_llm_config = node_processor_llm_config if node_processor_llm_config else LLMConfig()

        self.prompt_generator = L2TPromptGenerator(self.l2t_config)
        self.node_processor = NodeProcessor(
            llm_client=self.llm_client,
            l2t_config=self.l2t_config,
            prompt_generator=self.prompt_generator,
            llm_config=self.node_processor_llm_config # Pass the dedicated config for NodeProcessor
        )
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def run(self, problem_text: str) -> L2TResult:
        result = L2TResult()
        graph = L2TGraph()
        process_start_time = time.monotonic()
        current_process_step = 0

        log_stage("L2T", "Initial Thought Generation")
        
        initial_prompt = self.prompt_generator.construct_l2t_initial_prompt(
            problem_text, self.l2t_config.x_fmt_default, self.l2t_config.x_eva_default
        )
        
        # Log the outgoing initial request
        config_info = {"temperature": self.initial_thought_llm_config.temperature, "max_tokens": self.initial_thought_llm_config.max_tokens}
        comm_id = log_llm_request("L2T", ModelRole.L2T_INITIAL_THOUGHT, 
                                 self.l2t_config.initial_prompt_model_names, 
                                 initial_prompt, "Initial Thought", config_info)
        
        initial_response_content, initial_stats = self.llm_client.call(
            initial_prompt,
            models=self.l2t_config.initial_prompt_model_names,
            config=self.initial_thought_llm_config, # Use specific config for initial thought
        )
        
        # Log the incoming initial response
        log_llm_response(comm_id, "L2T", ModelRole.L2T_INITIAL_THOUGHT, 
                        initial_stats.model_name, initial_response_content, 
                        "Initial Thought", initial_stats)
        
        self.node_processor._update_result_stats(result, initial_stats)

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
            result.reasoning_graph = graph
            return result

        root_node_id = str(uuid.uuid4())
        root_node = L2TNode(
            id=root_node_id,
            content=parsed_initial_thought,
            parent_id=None,
            generation_step=0,
        )
        graph.add_node(root_node, is_root=True)
        logger.info(f"Initial thought node {root_node_id} created: '{parsed_initial_thought[:100]}...'")

        log_stage("L2T", "Iterative Processing")
        
        termination_reason = None
        while (
            current_process_step < self.l2t_config.max_steps
            and len(graph.v_pres) > 0
            and (time.monotonic() - process_start_time) < self.l2t_config.max_time_seconds
            and len(graph.nodes) < self.l2t_config.max_total_nodes
            and result.final_answer is None
        ):
            current_process_step += 1
            logger.info(
                f"L2T Process Step {current_process_step}/{self.l2t_config.max_steps} - "
                f"V_pres size: {len(graph.v_pres)}, Total nodes: {len(graph.nodes)}"
            )
            logger.debug(f"PROCESSOR: Start of step {current_process_step}. graph.v_pres: {list(graph.v_pres)}")

            nodes_to_process_this_round = list(graph.v_pres)

            for node_id_to_classify in nodes_to_process_this_round:
                if result.final_answer is not None:
                    break
                self.node_processor.process_node(node_id_to_classify, graph, result, current_process_step)
                logger.debug(f"DEBUG: After processing {node_id_to_classify}. graph.v_pres: {list(graph.v_pres)}")

            if result.final_answer is not None:
                break
            if not graph.v_pres:
                logger.info("No new thoughts generated in this step and no final answer. Terminating early.")
                break
        
        if current_process_step >= self.l2t_config.max_steps and result.final_answer is None:
            termination_reason = "max_steps"
        elif len(graph.nodes) >= self.l2t_config.max_total_nodes and result.final_answer is None:
            termination_reason = "max_total_nodes"
        elif (time.monotonic() - process_start_time) >= self.l2t_config.max_time_seconds and result.final_answer is None:
            termination_reason = "max_time"
        elif result.final_answer is None:
            termination_reason = "no_more_thoughts"

        result.reasoning_graph = graph
        if result.final_answer is None:
            result.succeeded = False
            if termination_reason == "max_steps":
                result.error_message = "L2T process completed: Max steps reached."
            elif termination_reason == "max_total_nodes":
                result.error_message = "L2T process completed: Max total nodes reached."
            elif termination_reason == "max_time":
                result.error_message = "L2T process completed: Max time reached."
            elif termination_reason == "no_more_thoughts":
                result.error_message = "L2T process completed: No more thoughts to process and no final answer."
            else:
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
