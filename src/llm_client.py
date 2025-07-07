import time
import requests
import logging
import json
from typing import List, Optional, Tuple, cast, Dict, Any, Iterator

# Assume llm_accounting is installed and available
from llm_accounting import LLMAccounting
from llm_accounting.audit_log import AuditLogger
from llm_accounting.models.limits import LimitScope, LimitType, TimeInterval
from llm_accounting.backends.sqlite import SQLiteBackend
from llm_accounting.backends.base import BaseBackend

LLM_ACCOUNTING_AVAILABLE = True # Always true as per new instructions

from src.aot.dataclasses import LLMCallStats
from src.aot.constants import OPENROUTER_API_URL, HTTP_REFERER, APP_TITLE

from src.llm_config import LLMConfig # Added import

class StreamingResponse:
    """Container for streaming response data"""
    def __init__(self):
        self.content = ""
        self.reasoning = ""
        self.usage = {}
        self.model_name = ""
        self.finish_reason = None
        
    def add_content(self, text: str):
        """Add content chunk"""
        self.content += text
        
    def add_reasoning(self, text: str):
        """Add reasoning chunk"""
        self.reasoning += text

class LLMClient:
    def __init__(self, api_key: str, api_url: str = OPENROUTER_API_URL, http_referer: str = HTTP_REFERER, app_title: str = APP_TITLE,
                 enable_rate_limiting: bool = True, enable_audit_logging: bool = True, db_path: str = "data/audit_log.sqlite"):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY must be provided.")
        self.api_key = api_key
        self.api_url = api_url
        self.http_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": http_referer,
            "X-Title": app_title,
            "Content-Type": "application/json"
        }

        # Always use llm_accounting components
        sqlite_backend_instance = SQLiteBackend(db_path=db_path)
        self.accounting = LLMAccounting(backend=sqlite_backend_instance)
        
        # enable_rate_limiting and enable_audit_logging can still be controlled by parameters
        self.enable_rate_limiting = enable_rate_limiting
        
        if enable_audit_logging:
            self.audit_logger = AuditLogger(backend=sqlite_backend_instance)
        else:
            self.audit_logger = None
        self.enable_audit_logging = enable_audit_logging # This should be set based on the parameter

    def call_with_reasoning(self, prompt: str, models: List[str], config: LLMConfig, 
                           reasoning_config: Optional[Dict[str, Any]] = None,
                           use_streaming: bool = False,
                           model_headers: Optional[Dict[str, str]] = None) -> Tuple[str, str, LLMCallStats]:
        """
        Enhanced call method with reasoning token support.
        
        Returns:
            Tuple of (content, reasoning, stats)
        """
        if use_streaming:
            return self._call_streaming_with_reasoning(prompt, models, config, reasoning_config, model_headers)
        else:
            return self._call_non_streaming_with_reasoning(prompt, models, config, reasoning_config, model_headers)
    
    def _call_streaming_with_reasoning(self, prompt: str, models: List[str], config: LLMConfig, 
                                     reasoning_config: Optional[Dict[str, Any]] = None,
                                     model_headers: Optional[Dict[str, str]] = None) -> Tuple[str, str, LLMCallStats]:
        """Handle streaming calls with reasoning token support"""
        if not models:
            logging.error("No models provided for LLM call.")
            return "Error: No models configured for call.", "", LLMCallStats(model_name="N/A")
        
        for model_name in models:
            # Prepare headers
            headers = self.http_headers.copy()
            if model_headers:
                headers.update(model_headers)
            
            # Construct payload
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                **config.to_payload_dict()
            }
            
            # Add reasoning configuration if provided
            if reasoning_config:
                payload["reasoning"] = reasoning_config
            
            # ------------------------------------------------------------------
            # PRE–REQUEST USAGE LOGGING (approximate prompt token count)
            # ------------------------------------------------------------------
            try:
                self.accounting.track_usage(
                    model=model_name,
                    prompt_tokens=len(prompt.split()),  # Rough estimate – sufficient for tests
                    caller_name="LLMClient",
                    username="api_user"
                )
            except Exception as log_exc:
                # Usage tracking should never break the main flow
                logging.debug(f"Pre-request usage logging failed: {log_exc}")
            
            current_call_stats = LLMCallStats(model_name=model_name)
            call_start_time = time.monotonic()
            
            try:
                logging.info(f"Attempting streaming LLM call with model: {model_name}")
                
                response = requests.post(self.api_url, headers=headers, json=payload, stream=True, timeout=90)
                response.raise_for_status()
                
                # Process streaming response
                streaming_response = StreamingResponse()
                streaming_response.model_name = model_name
                
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            data_text = line_text[6:]  # Remove 'data: ' prefix
                            
                            if data_text.strip() == '[DONE]':
                                break
                                
                            try:
                                chunk_data = json.loads(data_text)
                                self._process_streaming_chunk(chunk_data, streaming_response)
                            except json.JSONDecodeError:
                                continue
                
                current_call_stats.call_duration_seconds = time.monotonic() - call_start_time
                
                # Extract final usage if available
                if streaming_response.usage:
                    current_call_stats.completion_tokens = streaming_response.usage.get("completion_tokens", 0)
                    current_call_stats.prompt_tokens = streaming_response.usage.get("prompt_tokens", 0)
                
                # Track usage
                self.accounting.track_usage(
                    model=model_name,
                    prompt_tokens=current_call_stats.prompt_tokens,
                    completion_tokens=current_call_stats.completion_tokens,
                    execution_time=current_call_stats.call_duration_seconds,
                    caller_name="LLMClient",
                    username="api_user"
                )
                
                return streaming_response.content, streaming_response.reasoning, current_call_stats
                
            except Exception as e:
                current_call_stats.call_duration_seconds = time.monotonic() - call_start_time
                logging.warning(f"Streaming call to {model_name} failed: {e}. Trying next model if available.")
                
                # ------------------------------------------------------------------
                # POST-REQUEST USAGE LOGGING FOR FAILURE (no token counts expected)
                # ------------------------------------------------------------------
                try:
                    self.accounting.track_usage(
                        model=model_name,
                        execution_time=current_call_stats.call_duration_seconds,
                        caller_name="LLMClient",
                        username="api_user"
                    )
                except Exception as log_exc:
                    logging.debug(f"Post-request usage logging failed: {log_exc}")
                
                # If last model, return error tuple
                if model_name == models[-1]:
                    # Create an informative error message similar to non-streaming branch
                    if isinstance(e, requests.exceptions.Timeout):
                        err_msg = f"Error: API call to {model_name} timed out"
                    else:
                        err_msg = f"Error: Streaming call to {model_name} failed: {e}"
                    return err_msg, "", current_call_stats
                continue
        
        return "Error: All models failed.", "", LLMCallStats(model_name=models[0] if models else "N/A")
    
    def _call_non_streaming_with_reasoning(self, prompt: str, models: List[str], config: LLMConfig, 
                                         reasoning_config: Optional[Dict[str, Any]] = None,
                                         model_headers: Optional[Dict[str, str]] = None) -> Tuple[str, str, LLMCallStats]:
        """Handle non-streaming calls with reasoning token support"""
        if not models:
            logging.error("No models provided for LLM call.")
            return "Error: No models configured for call.", "", LLMCallStats(model_name="N/A")
        
        for model_name in models:
            # Prepare headers
            headers = self.http_headers.copy()
            if model_headers:
                headers.update(model_headers)
            
            # Construct payload
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                **config.to_payload_dict()
            }
            
            # Add reasoning configuration if provided
            if reasoning_config:
                payload["reasoning"] = reasoning_config
            
            # ------------------------------------------------------------------
            # PRE–REQUEST USAGE LOGGING (approximate prompt token count)
            # ------------------------------------------------------------------
            try:
                self.accounting.track_usage(
                    model=model_name,
                    prompt_tokens=len(prompt.split()),  # Rough estimate as per test expectations
                    caller_name="LLMClient",
                    username="api_user"
                )
            except Exception as log_exc:
                logging.debug(f"Pre-request usage logging failed: {log_exc}")
            
            current_call_stats = LLMCallStats(model_name=model_name)
            call_start_time = time.monotonic()
            
            try:
                logging.info(f"Attempting non-streaming LLM call with model: {model_name}")
                
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=90)
                current_call_stats.call_duration_seconds = time.monotonic() - call_start_time
                
                response.raise_for_status()
                response_data = response.json()
                
                # Extract content and reasoning
                content = ""
                reasoning = ""
                
                if "choices" in response_data and response_data["choices"]:
                    choice = response_data["choices"][0]
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    reasoning = message.get("reasoning", "")
                
                # Extract usage
                usage = response_data.get("usage", {})
                current_call_stats.completion_tokens = usage.get("completion_tokens", 0)
                current_call_stats.prompt_tokens = usage.get("prompt_tokens", 0)
                
                # Track usage AFTER successful response
                try:
                    self.accounting.track_usage(
                        model=model_name,
                        prompt_tokens=current_call_stats.prompt_tokens,
                        completion_tokens=current_call_stats.completion_tokens,
                        execution_time=current_call_stats.call_duration_seconds,
                        caller_name="LLMClient",
                        username="api_user"
                    )
                except Exception as log_exc:
                    logging.debug(f"Post-request usage logging failed: {log_exc}")
                
                return content, reasoning, current_call_stats
                
            except Exception as e:
                current_call_stats.call_duration_seconds = time.monotonic() - call_start_time
                logging.warning(f"Non-streaming call to {model_name} failed: {e}. Trying next model if available.")
                
                # Attempt to extract usage even in error responses
                usage_prompt_tokens = None
                usage_completion_tokens = None
                http_status_code: Optional[int] = None
                if isinstance(e, requests.exceptions.HTTPError) and getattr(e, 'response', None):
                    http_status_code = e.response.status_code
                    try:
                        error_data = e.response.json()
                        usage = error_data.get("usage", {})
                        usage_prompt_tokens = usage.get("prompt_tokens")
                        usage_completion_tokens = usage.get("completion_tokens")
                        current_call_stats.prompt_tokens = usage_prompt_tokens or 0
                        current_call_stats.completion_tokens = usage_completion_tokens or 0
                    except Exception:
                        pass
                
                # ------------------------------------------------------------------
                # POST-REQUEST USAGE LOGGING FOR FAILURE (tokens only if available)
                # ------------------------------------------------------------------
                try:
                    failure_usage_kwargs = {
                        "model": model_name,
                        "execution_time": current_call_stats.call_duration_seconds,
                        "caller_name": "LLMClient",
                        "username": "api_user"
                    }
                    if usage_prompt_tokens is not None:
                        failure_usage_kwargs["prompt_tokens"] = usage_prompt_tokens
                    if usage_completion_tokens is not None:
                        failure_usage_kwargs["completion_tokens"] = usage_completion_tokens
                    self.accounting.track_usage(**failure_usage_kwargs)
                except Exception as log_exc:
                    logging.debug(f"Post-error usage logging failed: {log_exc}")
                
                # Build error message if this was the last model attempt
                if model_name == models[-1]:
                    if isinstance(e, requests.exceptions.Timeout):
                        err_msg = f"Error: API call to {model_name} timed out"
                    elif isinstance(e, requests.exceptions.HTTPError):
                        status = http_status_code or "unknown"
                        err_msg = f"Error: API call to {model_name} (HTTP {status}) {str(e)}"
                    else:
                        err_msg = f"Error: API call to {model_name} failed: {e}"
                    return err_msg, "", current_call_stats
                continue
        
        # Should not reach here, but fallback
        return "Error: All models failed.", "", LLMCallStats(model_name=models[0] if models else "N/A")
    
    def _process_streaming_chunk(self, chunk_data: Dict[str, Any], streaming_response: StreamingResponse):
        """Process individual streaming chunks"""
        if "choices" not in chunk_data or not chunk_data["choices"]:
            return
        
        choice = chunk_data["choices"][0]
        delta = choice.get("delta", {})
        
        # Handle content
        if "content" in delta and delta["content"]:
            streaming_response.add_content(delta["content"])
        
        # Handle reasoning
        if "reasoning" in delta and delta["reasoning"]:
            streaming_response.add_reasoning(delta["reasoning"])
        
        # Handle finish reason
        if "finish_reason" in choice and choice["finish_reason"]:
            streaming_response.finish_reason = choice["finish_reason"]
        
        # Handle usage (usually comes at the end)
        if "usage" in chunk_data:
            streaming_response.usage = chunk_data["usage"]

    def call(self, prompt: str, models: List[str], config: LLMConfig) -> Tuple[str, LLMCallStats]:
        """Legacy call method for backward compatibility"""
        content, reasoning, stats = self.call_with_reasoning(prompt, models, config)
        
        # If reasoning was extracted, combine it with content for backward compatibility
        if reasoning:
            # For backward compatibility, combine reasoning and content
            combined_content = f"{reasoning}\n\n{content}" if content else reasoning
            return combined_content, stats
        
        return content, stats
