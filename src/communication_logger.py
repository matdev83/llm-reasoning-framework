"""
Unified Communication Logger for LLM Reasoning Framework

This module provides a consistent logging interface for all LLM communications
across different reasoning processes (AoT, L2T, GoT, Hybrid). It ensures
transparency by clearly marking the process name, model role, and actual model
being used for each communication.
"""

import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

# Import the existing LLMCallStats for consistency
from src.aot.dataclasses import LLMCallStats


class ModelRole(Enum):
    """Enumeration of different model roles in the reasoning processes"""
    # AoT roles
    AOT_MAIN = "aot_main"
    AOT_ASSESSMENT = "aot_assessment"
    AOT_ONESHOT = "aot_oneshot"
    
    # L2T roles
    L2T_INITIAL_THOUGHT = "l2t_initial_thought"
    L2T_CLASSIFICATION = "l2t_classification"
    L2T_THOUGHT_GENERATION = "l2t_thought_generation"
    L2T_ONESHOT = "l2t_oneshot"
    
    # Hybrid roles
    HYBRID_REASONING = "hybrid_reasoning"
    HYBRID_RESPONSE = "hybrid_response"
    HYBRID_ASSESSMENT = "hybrid_assessment"
    HYBRID_ONESHOT = "hybrid_oneshot"
    
    # GoT roles
    GOT_THOUGHT_GENERATION = "got_thought_generation"
    GOT_SCORING = "got_scoring"
    GOT_AGGREGATION = "got_aggregation"
    GOT_REFINEMENT = "got_refinement"
    GOT_ASSESSMENT = "got_assessment"
    GOT_ONESHOT = "got_oneshot"
    
    # FaR roles
    FAR_FACT_EXTRACTION = "far_fact_extraction"
    FAR_REFLECTION_ANSWER = "far_reflection_answer"
    FAR_ASSESSMENT = "far_assessment"
    FAR_ONESHOT = "far_oneshot"
    FAR_ONESHOT_FALLBACK = "far_oneshot_fallback"
    
    # General roles
    COMPLEXITY_ASSESSMENT = "complexity_assessment"
    ONESHOT_FALLBACK = "oneshot_fallback"


@dataclass
class CommunicationContext:
    """Context information for a communication event"""
    process_name: str
    model_role: ModelRole
    step_info: Optional[str] = None  # e.g., "iteration 3/10", "phase 2"
    additional_info: Optional[Dict[str, Any]] = None


class CommunicationLogger:
    """
    Unified logger for all LLM communications in the reasoning framework.
    
    This class provides consistent logging of:
    - Outgoing prompts/requests to LLMs
    - Incoming responses from LLMs
    - Model identification and role clarification
    - Process stage information
    """
    
    def __init__(self, logger_name: str = "communication"):
        self.logger = logging.getLogger(logger_name)
        self._communication_counter = 0
    
    def log_outgoing_request(self, 
                           context: CommunicationContext,
                           actual_model_names: List[str],
                           prompt: str,
                           config_info: Optional[Dict[str, Any]] = None) -> int:
        """
        Log an outgoing request to an LLM.
        
        Args:
            context: Communication context with process name and model role
            actual_model_names: List of actual model names being called
            prompt: The prompt being sent
            config_info: Optional configuration information (temperature, max_tokens, etc.)
            
        Returns:
            Communication ID for tracking this request/response pair
        """
        self._communication_counter += 1
        comm_id = self._communication_counter
        
        # Format model names for display
        models_str = ", ".join(actual_model_names) if len(actual_model_names) <= 3 else f"{', '.join(actual_model_names[:3])}... (+{len(actual_model_names)-3} more)"
        
        # Build step info string
        step_info = f" [{context.step_info}]" if context.step_info else ""
        
        # Build config info string
        config_str = ""
        if config_info:
            config_parts = []
            if "temperature" in config_info:
                config_parts.append(f"temp={config_info['temperature']}")
            if "max_tokens" in config_info:
                config_parts.append(f"max_tokens={config_info['max_tokens']}")
            if config_parts:
                config_str = f" (config: {', '.join(config_parts)})"
        
        # Main log message
        self.logger.info(f"[{comm_id:03d}] ({context.process_name}){step_info} Send to model with role: {context.model_role.value}, actual model: {models_str}{config_str}")
        
        # Log prompt details at debug level
        prompt_preview = prompt[:200].replace('\n', ' ') if len(prompt) > 200 else prompt.replace('\n', ' ')
        self.logger.debug(f"[{comm_id:03d}] Prompt preview: {prompt_preview}...")
        
        # Log full prompt at trace level if available
        if self.logger.isEnabledFor(5):  # TRACE level (custom)
            self.logger.log(5, f"[{comm_id:03d}] Full prompt:\n{prompt}")
        
        return comm_id
    
    def log_incoming_response(self,
                            comm_id: int,
                            context: CommunicationContext,
                            actual_model_name: str,
                            response: str,
                            stats: Optional[LLMCallStats] = None,
                            error: Optional[str] = None) -> None:
        """
        Log an incoming response from an LLM.
        
        Args:
            comm_id: Communication ID from the corresponding outgoing request
            context: Communication context with process name and model role
            actual_model_name: The actual model that generated the response
            response: The response content
            stats: Optional LLM call statistics
            error: Optional error message if the call failed
        """
        # Build step info string
        step_info = f" [{context.step_info}]" if context.step_info else ""
        
        if error:
            self.logger.warning(f"[{comm_id:03d}] ({context.process_name}){step_info} Error from model {actual_model_name} (role: {context.model_role.value}): {error}")
            return
        
        # Build stats string
        stats_str = ""
        if stats:
            try:
                # Handle potential mock objects or missing attributes gracefully
                duration = getattr(stats, 'call_duration_seconds', 0.0)
                prompt_tokens = getattr(stats, 'prompt_tokens', 0)
                completion_tokens = getattr(stats, 'completion_tokens', 0)
                
                # Format with safe conversion to float/int
                duration_val = float(duration) if duration is not None else 0.0
                prompt_val = int(prompt_tokens) if prompt_tokens is not None else 0
                completion_val = int(completion_tokens) if completion_tokens is not None else 0
                
                stats_str = f" (duration: {duration_val:.2f}s, tokens: P:{prompt_val}, C:{completion_val})"
            except (TypeError, ValueError, AttributeError):
                # Fallback for mock objects or invalid stats
                stats_str = " (stats: unavailable)"
        
        # Main log message
        self.logger.info(f"[{comm_id:03d}] ({context.process_name}){step_info} Received from model {actual_model_name} (role: {context.model_role.value}){stats_str}")
        
        # Log response details at debug level
        response_preview = response[:200].replace('\n', ' ') if len(response) > 200 else response.replace('\n', ' ')
        self.logger.debug(f"[{comm_id:03d}] Response preview: {response_preview}...")
        
        # Log full response at trace level if available
        if self.logger.isEnabledFor(5):  # TRACE level (custom)
            self.logger.log(5, f"[{comm_id:03d}] Full response:\n{response}")
    
    def log_process_stage(self, 
                         process_name: str, 
                         stage_name: str, 
                         stage_info: Optional[str] = None) -> None:
        """
        Log a process stage change.
        
        Args:
            process_name: Name of the reasoning process
            stage_name: Name of the current stage
            stage_info: Optional additional information about the stage
        """
        info_str = f" - {stage_info}" if stage_info else ""
        self.logger.info(f"({process_name}) === {stage_name}{info_str} ===")
    
    def log_process_summary(self,
                          process_name: str,
                          summary_info: Dict[str, Any]) -> None:
        """
        Log a process summary.
        
        Args:
            process_name: Name of the reasoning process
            summary_info: Dictionary containing summary information
        """
        self.logger.info(f"({process_name}) Process Summary:")
        for key, value in summary_info.items():
            self.logger.info(f"({process_name})   {key}: {value}")


# Global communication logger instance
_global_comm_logger: Optional[CommunicationLogger] = None


def get_communication_logger() -> CommunicationLogger:
    """Get the global communication logger instance."""
    global _global_comm_logger
    if _global_comm_logger is None:
        _global_comm_logger = CommunicationLogger()
    return _global_comm_logger


def set_communication_logger(logger: CommunicationLogger) -> None:
    """Set a custom communication logger instance."""
    global _global_comm_logger
    _global_comm_logger = logger


# Convenience functions for common operations
def log_llm_request(process_name: str, 
                   model_role: ModelRole, 
                   actual_model_names: List[str],
                   prompt: str,
                   step_info: Optional[str] = None,
                   config_info: Optional[Dict[str, Any]] = None) -> int:
    """Convenience function to log an LLM request."""
    context = CommunicationContext(
        process_name=process_name,
        model_role=model_role,
        step_info=step_info
    )
    return get_communication_logger().log_outgoing_request(context, actual_model_names, prompt, config_info)


def log_llm_response(comm_id: int,
                    process_name: str,
                    model_role: ModelRole,
                    actual_model_name: str,
                    response: str,
                    step_info: Optional[str] = None,
                    stats: Optional[LLMCallStats] = None,
                    error: Optional[str] = None) -> None:
    """Convenience function to log an LLM response."""
    context = CommunicationContext(
        process_name=process_name,
        model_role=model_role,
        step_info=step_info
    )
    get_communication_logger().log_incoming_response(comm_id, context, actual_model_name, response, stats, error)


def log_stage(process_name: str, stage_name: str, stage_info: Optional[str] = None) -> None:
    """Convenience function to log a process stage."""
    get_communication_logger().log_process_stage(process_name, stage_name, stage_info)


def log_summary(process_name: str, summary_info: Dict[str, Any]) -> None:
    """Convenience function to log a process summary."""
    get_communication_logger().log_process_summary(process_name, summary_info) 