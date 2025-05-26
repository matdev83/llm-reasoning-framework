import time
import requests
import logging
from typing import List, Optional, Tuple

from src.aot_dataclasses import LLMCallStats
from src.aot_constants import OPENROUTER_API_URL, HTTP_REFERER, APP_TITLE

class LLMClient:
    def __init__(self, api_key: str, api_url: str = OPENROUTER_API_URL, http_referer: str = HTTP_REFERER, app_title: str = APP_TITLE):
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

    def call(self, prompt: str, models: List[str], temperature: float) -> Tuple[str, LLMCallStats]:
        if not models:
            logging.error("No models provided for LLM call.")
            return "Error: No models configured for call.", LLMCallStats(model_name="N/A")
        
        last_error_content_for_failover = f"Error: All models ({', '.join(models)}) in the list failed."
        last_error_stats_for_failover = LLMCallStats(model_name=models[0] if models else "N/A")

        for model_name in models:
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "reasoning": {"effort": "high"} # Assuming this is a custom param for OpenRouter
            }
            current_call_stats = LLMCallStats(model_name=model_name)
            call_start_time = time.monotonic()
            logging.info(f"Attempting LLM call with model: {model_name}")
            try:
                response = requests.post(self.api_url, headers=self.http_headers, json=payload, timeout=90)
                current_call_stats.call_duration_seconds = time.monotonic() - call_start_time

                if 400 <= response.status_code < 600: # Client or Server error that OpenRouter might return with usage
                    error_body_text = response.text
                    try:
                        error_json = response.json()
                        error_body_text = error_json.get('error', {}).get('message', str(error_json.get('error', error_body_text)))
                    except requests.exceptions.JSONDecodeError: 
                        pass # Keep original error_body_text
                    logging.warning(f"API call to {model_name} failed with HTTP status {response.status_code}: {error_body_text}. Trying next model if available.")
                    last_error_content_for_failover = f"Error: API call to {model_name} (HTTP {response.status_code}) - {error_body_text}"
                    try: # Try to get usage even from error responses
                        data = response.json()
                        usage = data.get("usage", {})
                        current_call_stats.completion_tokens = usage.get("completion_tokens", 0)
                        current_call_stats.prompt_tokens = usage.get("prompt_tokens", 0)
                    except: # Ignore if cannot parse usage from error
                        pass
                    last_error_stats_for_failover = current_call_stats
                    continue 

                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx) if not handled above
                
                data = response.json()

                if "error" in data: # Error in a 2xx response payload
                    error_msg = data['error'].get('message', str(data['error']))
                    logging.error(f"LLM API Error ({model_name}) in 2xx response: {error_msg}")
                    content = f"Error: API returned an error - {error_msg}"
                elif not data.get("choices") or not data["choices"][0].get("message"):
                    logging.error(f"Unexpected API response structure from {model_name}: {data}")
                    content = "Error: Unexpected API response structure"
                else:
                    content = data["choices"][0]["message"]["content"]
                
                # Common logic for usage stats if successful or error in 2xx
                usage = data.get("usage", {})
                current_call_stats.completion_tokens = usage.get("completion_tokens", 0)
                current_call_stats.prompt_tokens = usage.get("prompt_tokens", 0)

                if content.startswith("Error:"): # If content is an error string we constructed
                    # If it's an error we are returning, and it was a failover candidate, log and continue
                    if model_name != models[-1]: # If not the last model
                         logging.warning(f"LLM call to {model_name} resulted in error: {content}. Trying next model.")
                         last_error_content_for_failover = content
                         last_error_stats_for_failover = current_call_stats
                         continue
                    # If it IS the last model, this error will be returned
                else: # Successful content
                    if current_call_stats.completion_tokens == 0 and content:
                         logging.warning(f"completion_tokens reported as 0 from {model_name}, but content was received.")
                
                return content, current_call_stats

            except requests.exceptions.HTTPError as e:
                current_call_stats.call_duration_seconds = time.monotonic() - call_start_time
                # Try to get usage from error response if possible
                if e.response is not None:
                    try:
                        data = e.response.json()
                        usage = data.get("usage", {})
                        current_call_stats.completion_tokens = usage.get("completion_tokens", 0)
                        current_call_stats.prompt_tokens = usage.get("prompt_tokens", 0)
                    except: pass
                
                # Only treat specific client errors (like 429, 401 if not already caught, etc.) as non-failover potentially.
                # For now, most HTTP errors might be per-model issues.
                if e.response is not None and (400 <= e.response.status_code < 500 and e.response.status_code not in [401, 403, 429]): # e.g. 400 Bad Request due to model specific issues
                    logging.warning(f"API request to {model_name} failed with HTTPError (status {e.response.status_code}): {e}. Trying next model.")
                    last_error_content_for_failover = f"Error: API call to {model_name} (HTTP {e.response.status_code}) - {e}"
                    last_error_stats_for_failover = current_call_stats
                    continue
                else: # Server errors, auth, rate limits, or unexpected HTTP errors
                    logging.error(f"API request to {model_name} failed with non-failover HTTPError: {e}")
                    return f"Error: API call failed - {e}", current_call_stats # Return error, stop trying models

            except requests.exceptions.Timeout as e:
                current_call_stats.call_duration_seconds = time.monotonic() - call_start_time
                logging.warning(f"API request to {model_name} timed out after {current_call_stats.call_duration_seconds:.2f}s. Trying next model if available.")
                last_error_content_for_failover = f"Error: API call to {model_name} timed out"
                last_error_stats_for_failover = current_call_stats
                continue
            except requests.exceptions.RequestException as e: # Other network issues
                current_call_stats.call_duration_seconds = time.monotonic() - call_start_time
                logging.warning(f"API request to {model_name} failed with RequestException: {e}. Trying next model if available.")
                last_error_content_for_failover = f"Error: API call to {model_name} failed - {e}"
                last_error_stats_for_failover = current_call_stats
                continue
        
        logging.error(f"All models ({', '.join(models)}) failed. Last error from model '{last_error_stats_for_failover.model_name}': {last_error_content_for_failover}")
        return last_error_content_for_failover, last_error_stats_for_failover
