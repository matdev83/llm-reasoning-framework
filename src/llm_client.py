import time
import requests
import logging
from typing import List, Optional, Tuple, cast

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

    def call(self, prompt: str, models: List[str], config: LLMConfig) -> Tuple[str, LLMCallStats]:
        if not models:
            logging.error("No models provided for LLM call.")
            return "Error: No models configured for call.", LLMCallStats(model_name="N/A")
        
        last_error_content_for_failover = f"Error: All models ({', '.join(models)}) in the list failed."
        last_error_stats_for_failover = LLMCallStats(model_name=models[0] if models else "N/A")
        request_id = None # Initialize request_id

        for model_name in models:
            # Construct payload using LLMConfig
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                **config.to_payload_dict() # Unpack parameters from config
            }
            current_call_stats = LLMCallStats(model_name=model_name)
            call_start_time = time.monotonic()
            
            try:
                if self.enable_rate_limiting: # Now always uses real accounting
                    allowed, reason = self.accounting.check_quota(
                        model=model_name,
                        username="api_user",
                        caller_name="LLMClient",
                        input_tokens=len(prompt.split()) # Approximate token count for quota check
                    )
                    if not allowed:
                        logging.warning(f"Rate limit exceeded for model {model_name}: {reason}. Trying next model if available.")
                        last_error_content_for_failover = f"Error: Rate limit exceeded - {reason}"
                        last_error_stats_for_failover = current_call_stats
                        continue

                # Log the request before making the API call
                self.accounting.track_usage( # Now always uses real accounting
                    model=model_name,
                    prompt_tokens=len(prompt.split()),
                    caller_name="LLMClient",
                    username="api_user"
                )
                logging.info(f"Attempting LLM call with model: {model_name}")

                response = requests.post(self.api_url, headers=self.http_headers, json=payload, timeout=90)
                current_call_stats.call_duration_seconds = time.monotonic() - call_start_time
                
                response_payload = None
                content = None # Initialize content here

                try:
                    response_payload = response.json()
                    # If JSON parsing succeeds, extract content as before
                    if "error" in response_payload:
                        error_msg = response_payload['error'].get('message', str(response_payload['error']))
                        logging.error(f"LLM API Error ({model_name}) in 2xx response: {error_msg}")
                        content = f"Error: API returned an error - {error_msg}"
                    elif not response_payload.get("choices") or not response_payload["choices"][0].get("message"):
                        logging.error(f"Unexpected API response structure from {model_name}: {response_payload}")
                        content = "Error: Unexpected API response structure"
                    else:
                        content = response_payload["choices"][0]["message"]["content"]
                    
                    usage = response_payload.get("usage", {})
                except requests.exceptions.JSONDecodeError:
                    # If JSON parsing fails, but status is OK, use raw text
                    if response.status_code == 200:
                        logging.warning(f"Could not parse JSON response from {model_name}. Status: {response.status_code}, Body: {response.text}. Returning raw text.")
                        content = response.text # <--- CHANGE HERE: Use raw text
                        usage = {} # No usage info if JSON parsing failed
                    else:
                        logging.warning(f"Could not parse JSON response from {model_name}. Status: {response.status_code}, Body: {response.text}. Treating as error.")
                        content = f"Error: Could not parse response from {model_name} (Status: {response.status_code})"
                        usage = {}

                if 400 <= response.status_code < 600:
                    error_body_text = response.text
                    if response_payload is not None:
                        response_dict = response_payload
                        if 'error' in response_dict:
                            error_body_text = str(response_dict.get('error', {}).get('message', error_body_text))
                    
                    logging.warning(f"API call to {model_name} failed with HTTP status {response.status_code}: {error_body_text}. Trying next model if available.")
                    last_error_content_for_failover = f"Error: API call to {model_name} (HTTP {response.status_code}) - {error_body_text}"
                    
                    if response_payload is not None: # Use response_payload if available for usage
                        usage = response_payload.get("usage", {})
                        current_call_stats.completion_tokens = usage.get("completion_tokens", 0)
                        current_call_stats.prompt_tokens = usage.get("prompt_tokens", 0)
                    else: # If no JSON payload, usage is unknown
                        current_call_stats.completion_tokens = 0
                        current_call_stats.prompt_tokens = 0
                    
                    self.accounting.track_usage( # Now always uses real accounting
                        model=model_name,
                        prompt_tokens=current_call_stats.prompt_tokens,
                        completion_tokens=current_call_stats.completion_tokens,
                        execution_time=current_call_stats.call_duration_seconds,
                        caller_name="LLMClient",
                        username="api_user"
                    )
                    last_error_stats_for_failover = current_call_stats
                    continue 

                response.raise_for_status()
                
                # After content is determined, proceed with usage tracking and audit logging
                current_call_stats.completion_tokens = usage.get("completion_tokens", 0)
                current_call_stats.prompt_tokens = usage.get("prompt_tokens", 0)

                self.accounting.track_usage( # Now always uses real accounting
                    model=model_name,
                    prompt_tokens=current_call_stats.prompt_tokens,
                    completion_tokens=current_call_stats.completion_tokens,
                    execution_time=current_call_stats.call_duration_seconds,
                    caller_name="LLMClient",
                    username="api_user"
                )

                if self.enable_audit_logging and self.audit_logger: # audit_logger will be None if not available
                    self.audit_logger.log_prompt(
                        app_name="LLMClient",
                        user_name="api_user",
                        model=model_name,
                        prompt_text=prompt
                    )
                    # Only log response if it was a valid JSON payload, otherwise it's raw text
                    if response_payload is not None:
                        response_data = cast(dict, response_payload) 
                        self.audit_logger.log_response(
                            app_name="LLMClient",
                            user_name="api_user",
                            model=model_name,
                            response_text=content,
                            remote_completion_id=response_data.get("id")
                        )
                    else:
                        logging.debug(f"Skipping audit log for raw text response from {model_name}.")

                if content.startswith("Error:"): 
                    if model_name != models[-1]: 
                         logging.warning(f"LLM call to {model_name} resulted in error: {content}. Trying next model.")
                         last_error_content_for_failover = content
                         last_error_stats_for_failover = current_call_stats
                         continue
                else: 
                    if current_call_stats.completion_tokens == 0 and content:
                         logging.warning(f"completion_tokens reported as 0 from {model_name}, but content was received.")
                
                return content, current_call_stats

            except requests.exceptions.HTTPError as e:
                current_call_stats.call_duration_seconds = time.monotonic() - call_start_time
                response_data_for_log = None
                if e.response is not None:
                    try:
                        response_data_for_log = e.response.json()
                        usage = response_data_for_log.get("usage", {})
                        current_call_stats.completion_tokens = usage.get("completion_tokens", 0)
                        current_call_stats.prompt_tokens = usage.get("prompt_tokens", 0)
                    except: 
                        response_data_for_log = {"error": str(e), "status_code": e.response.status_code, "text": e.response.text}
                else:
                    response_data_for_log = {"error": str(e)}

                self.accounting.track_usage( # Now always uses real accounting
                    model=model_name,
                    prompt_tokens=current_call_stats.prompt_tokens,
                    completion_tokens=current_call_stats.completion_tokens,
                    execution_time=current_call_stats.call_duration_seconds,
                    caller_name="LLMClient",
                    username="api_user"
                )
                
                if e.response is not None and (400 <= e.response.status_code < 500 and e.response.status_code not in [401, 403, 429]):
                    logging.warning(f"API request to {model_name} failed with HTTPError (status {e.response.status_code}): {e}. Trying next model.")
                    last_error_content_for_failover = f"Error: API call to {model_name} (HTTP {e.response.status_code}) - {e}"
                    last_error_stats_for_failover = current_call_stats
                    continue
                else: 
                    logging.error(f"API request to {model_name} failed with non-failover HTTPError: {e}")
                    return f"Error: API call failed - {e}", current_call_stats

            except requests.exceptions.Timeout as e:
                current_call_stats.call_duration_seconds = time.monotonic() - call_start_time
                self.accounting.track_usage( # Now always uses real accounting
                    model=model_name,
                    execution_time=current_call_stats.call_duration_seconds,
                    caller_name="LLMClient",
                    username="api_user"
                )
                logging.warning(f"API request to {model_name} timed out after {current_call_stats.call_duration_seconds:.2f}s. Trying next model if available.")
                last_error_content_for_failover = f"Error: API call to {model_name} timed out"
                last_error_stats_for_failover = current_call_stats
                continue

            except requests.exceptions.RequestException as e: 
                current_call_stats.call_duration_seconds = time.monotonic() - call_start_time
                self.accounting.track_usage( # Now always uses real accounting
                    model=model_name,
                    execution_time=current_call_stats.call_duration_seconds,
                    caller_name="LLMClient",
                    username="api_user"
                )
                logging.warning(f"API request to {model_name} failed with RequestException: {e}. Trying next model if available.")
                last_error_content_for_failover = f"Error: API call to {model_name} failed - {e}"
                last_error_stats_for_failover = current_call_stats
                continue
            
            finally:
                if model_name != models[-1]:
                    request_id = None
        
        logging.error(f"All models ({', '.join(models)}) failed. Last error from model '{last_error_stats_for_failover.model_name}': {last_error_content_for_failover}")
        return last_error_content_for_failover, last_error_stats_for_failover
