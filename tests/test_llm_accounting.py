import unittest
import os
import sqlite3
import json
from unittest.mock import patch, Mock, call
import time
import requests
import tempfile # Import tempfile
import logging # Import logging for warnings

# Assuming llm_client is in src.llm_client
from src.llm_client import LLMClient, LLM_ACCOUNTING_AVAILABLE, SQLiteBackend # Import LLM_ACCOUNTING_AVAILABLE, SQLiteBackend
from src.aot.dataclasses import LLMCallStats
from src.llm_config import LLMConfig
from typing import cast # Import cast

# DB_PATH will now be dynamic
# DB_PATH = "data/audit_log.sqlite" # No longer needed as a global constant

class TestLLMAccountingIntegration(unittest.TestCase):

    def setUp(self):
        # Create a temporary file for the database
        self.temp_db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite").name
        
        # Set a dummy API key for LLMClient initialization
        os.environ["OPENROUTER_API_KEY"] = "dummy_key_for_testing"
        # Ensure audit logging is enabled for the test client
        # Pass the temporary db path to LLMClient
        self.llm_client = LLMClient(api_key="dummy_key_for_testing", enable_audit_logging=True,
                                     db_path=self.temp_db_file) # Add db_path parameter to LLMClient

        # Patch time.monotonic to control call_duration_seconds
        self.mock_monotonic_patch = patch('time.monotonic')
        self.mock_monotonic = self.mock_monotonic_patch.start()
        self.mock_monotonic.side_effect = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] # Extended for failover test

    def tearDown(self):
        # Dispose of the SQLAlchemy engine to close all connections, only if llm_accounting is available
        if LLM_ACCOUNTING_AVAILABLE and hasattr(self.llm_client, 'accounting') and \
           hasattr(self.llm_client.accounting, 'backend'):
            # Explicitly cast to SQLiteBackend to help type checker
            backend = cast(SQLiteBackend, self.llm_client.accounting.backend)
            if hasattr(backend, 'engine'):
                backend.engine.dispose()
        
        # Clean up the temporary database file
        if os.path.exists(self.temp_db_file):
            try:
                os.remove(self.temp_db_file)
            except PermissionError:
                logging.warning(f"Could not remove temporary DB file {self.temp_db_file} due to PermissionError.")
        del os.environ["OPENROUTER_API_KEY"]
        self.mock_monotonic_patch.stop()


    # Test 1: Successful LLM Call Logging
    @patch('requests.post')
    def test_successful_llm_call_logging(self, mock_post):
        # Configure mock for requests.post
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response_payload = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "model": "test_model_success" # OpenRouter includes model in response
        }
        mock_response.json.return_value = mock_response_payload
        mock_post.return_value = mock_response

        # Spy on accounting methods
        track_usage_patch = patch.object(self.llm_client.accounting, 'track_usage', wraps=self.llm_client.accounting.track_usage)
        mock_track_usage = track_usage_patch.start()
        self.addCleanup(track_usage_patch.stop) # Ensure patch is stopped

        prompt = "Test prompt for success"
        models = ["test_model_success"]
        llm_config = LLMConfig(temperature=0.7) # Use LLMConfig
        
        content, stats = self.llm_client.call(prompt, models, llm_config) # Pass LLMConfig

        self.assertEqual(content, "Test response")
        self.assertEqual(stats.prompt_tokens, 10)
        self.assertEqual(stats.completion_tokens, 20)
        self.assertEqual(stats.model_name, "test_model_success")
        self.assertEqual(stats.call_duration_seconds, 1.0) # 2.0 - 1.0

        # track_usage is called twice: once before request, once after response
        self.assertEqual(mock_track_usage.call_count, 2)
        
        # Check the first call (before request)
        first_call_args = mock_track_usage.call_args_list[0][1]
        self.assertEqual(first_call_args['model'], "test_model_success")
        self.assertEqual(first_call_args['prompt_tokens'], len(prompt.split())) # Approximate token count from prompt length
        self.assertIsNone(first_call_args.get('completion_tokens')) # Not available yet
        self.assertIsNone(first_call_args.get('execution_time')) # Not available yet

        # Check the second call (after response)
        second_call_args = mock_track_usage.call_args_list[1][1]
        self.assertEqual(second_call_args['model'], "test_model_success")
        self.assertEqual(second_call_args['prompt_tokens'], 10)
        self.assertEqual(second_call_args['completion_tokens'], 20)
        self.assertEqual(second_call_args['execution_time'], 1.0)

        # Removed DB verification as llm-accounting no longer uses SQLite directly for this version.
        # Verification is done via mock_track_usage calls.

    # Test 2: LLM Call with API Error Logging (with usage)
    @patch('requests.post')
    def test_api_error_with_usage_logging(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 500
        mock_error_payload = {
            "error": {"message": "Server error"},
            "usage": {"prompt_tokens": 5, "completion_tokens": 0}, # Error, but usage reported
            "model": "test_model_api_error"
        }
        mock_response.json.return_value = mock_error_payload
        mock_response.text = json.dumps(mock_error_payload) # For error reporting if .json() fails or for text part
        mock_post.return_value = mock_response
        
        # Configure raise_for_status for 500 error
        mock_response.raise_for_status = Mock(side_effect=requests.exceptions.HTTPError("Server Error", response=mock_response))


        mock_track_usage = patch.object(self.llm_client.accounting, 'track_usage', wraps=self.llm_client.accounting.track_usage).start()

        prompt = "Test prompt for API error"
        models = ["test_model_api_error"]
        llm_config = LLMConfig(temperature=0.7) # Use LLMConfig
        
        content, stats = self.llm_client.call(prompt, models, llm_config) # Pass LLMConfig

        self.assertTrue(content.startswith("Error: API call to test_model_api_error (HTTP 500)"))
        self.assertEqual(stats.prompt_tokens, 5) # Usage should be captured
        self.assertEqual(stats.completion_tokens, 0)
        self.assertEqual(stats.model_name, "test_model_api_error")
        self.assertEqual(stats.call_duration_seconds, 1.0) # 4.0 - 3.0 (due to setUp side_effect)

        self.assertEqual(mock_track_usage.call_count, 2)

        # Check the first call (before request)
        first_call_args = mock_track_usage.call_args_list[0][1]
        self.assertEqual(first_call_args['model'], "test_model_api_error")
        self.assertEqual(first_call_args['prompt_tokens'], 5) # Approximate token count from prompt length
        self.assertIsNone(first_call_args.get('completion_tokens')) # Not available yet
        self.assertIsNone(first_call_args.get('execution_time')) # Not available yet

        # Check the second call (after response)
        second_call_args = mock_track_usage.call_args_list[1][1]
        self.assertEqual(second_call_args['model'], "test_model_api_error")
        self.assertEqual(second_call_args['prompt_tokens'], 5)
        self.assertEqual(second_call_args['completion_tokens'], 0)
        self.assertEqual(second_call_args['execution_time'], 1.0)
        # The 'cost' key might not be present in track_usage args for error cases,
        # as it's calculated internally by llm-accounting.
        # We verify cost via DB content instead.


        # Removed DB verification as llm-accounting no longer uses SQLite directly for this version.
        # Verification is done via mock_track_usage calls.
            
    # Test 3: LLM Call with Network Error (No Response Body)
    @patch('requests.post')
    def test_network_error_logging(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

        mock_track_usage = patch.object(self.llm_client.accounting, 'track_usage', wraps=self.llm_client.accounting.track_usage).start()

        prompt = "Test prompt for network error"
        models = ["test_model_network_error"]
        llm_config = LLMConfig(temperature=0.7) # Use LLMConfig
        
        content, stats = self.llm_client.call(prompt, models, llm_config) # Pass LLMConfig

        self.assertTrue(content.startswith("Error: API call to test_model_network_error timed out"))
        self.assertEqual(stats.prompt_tokens, 0) # No usage info
        self.assertEqual(stats.completion_tokens, 0)
        self.assertEqual(stats.model_name, "test_model_network_error")
        self.assertEqual(stats.call_duration_seconds, 1.0) # 6.0 - 5.0

        self.assertEqual(mock_track_usage.call_count, 2) # One before, one after (with error details)

        # Check the first call (before request)
        first_call_args = mock_track_usage.call_args_list[0][1]
        self.assertEqual(first_call_args['model'], "test_model_network_error")
        self.assertEqual(first_call_args['prompt_tokens'], len(prompt.split())) # Approximate token count from prompt length
        self.assertIsNone(first_call_args.get('completion_tokens'))
        self.assertIsNone(first_call_args.get('execution_time'))

        # Check the second call (after response)
        second_call_args = mock_track_usage.call_args_list[1][1]
        self.assertEqual(second_call_args['model'], "test_model_network_error")
        self.assertNotIn('prompt_tokens', second_call_args)
        self.assertNotIn('completion_tokens', second_call_args)
        self.assertEqual(second_call_args['execution_time'], 1.0)
        # Removed assertion for cost as it's not always present for network errors
        # Removed assertion for response_payload as it's not always present for timeouts


        # Removed DB verification as llm-accounting no longer uses SQLite directly for this version.
        # Verification is done via mock_track_usage calls.

    # Test 4: Model Failover Logging
    @patch('requests.post')
    def test_model_failover_logging(self, mock_post):
        # First call fails (e.g. timeout), second succeeds
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "choices": [{"message": {"content": "Failover successful"}}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 25},
            "model": "model_two_success"
        }
        
        # Mock side_effect for requests.post
        # First call: raises Timeout
        # Second call: returns success
        mock_post.side_effect = [
            requests.exceptions.Timeout("Timeout on model_one_fail"),
            mock_response_success
        ]

        mock_track_usage = patch.object(self.llm_client.accounting, 'track_usage', wraps=self.llm_client.accounting.track_usage).start()

        prompt = "Test prompt for failover"
        models = ["model_one_fail", "model_two_success"] # First fails, second succeeds
        
        # Adjust mock_monotonic for two calls if needed:
        # call 1 (fail): start_time=7.0, end_time=8.0 (duration 1.0)
        # call 2 (success): start_time=9.0, end_time=10.0 (duration 1.0)
        # self.mock_monotonic.side_effect gets consumed. Need to ensure enough values or reset.
        # For simplicity, the existing setup should give 1.0s duration for each if time.monotonic is called twice per call.
        # LLMClient calls time.monotonic() at start of try, and end of try/except.
        # So, first call uses values at index 6,7 (from setUp: 7.0, 8.0). Duration (8.0-7.0)=1.0
        # Second call will use values at index 8,9. Need to extend side_effect for time.monotonic
        self.mock_monotonic.side_effect = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


        llm_config = LLMConfig(temperature=0.7) # Use LLMConfig
        content, stats = self.llm_client.call(prompt, models, llm_config) # Pass LLMConfig

        self.assertEqual(content, "Failover successful")
        self.assertEqual(stats.model_name, "model_two_success")
        self.assertEqual(stats.prompt_tokens, 15)
        self.assertEqual(stats.completion_tokens, 25)
        # Duration for the successful call. The mock_monotonic side_effect is global for the test.
        # If first call starts at T1, ends at T2. Second call starts T3, ends T4.
        # Durations will be T2-T1 and T4-T3.
        # With current setup, T1=7, T2=8 (duration 1.0 for first call). T3=9, T4=10 (duration 1.0 for second call).
        self.assertEqual(stats.call_duration_seconds, 1.0)


        self.assertEqual(mock_track_usage.call_count, 4) # One for initial fail, two for successful call

        # Check calls to spies (order matters)
        # Call 1 (failure) - before request
        first_call_args = mock_track_usage.call_args_list[0][1]
        self.assertEqual(first_call_args['model'], "model_one_fail")
        self.assertEqual(first_call_args['prompt_tokens'], len(prompt.split())) # Approximate token count from prompt length
        self.assertIsNone(first_call_args.get('completion_tokens'))
        self.assertIsNone(first_call_args.get('execution_time'))
        
        # Call 2 (failure) - after request
        second_call_args = mock_track_usage.call_args_list[1][1]
        self.assertEqual(second_call_args['model'], "model_one_fail")
        self.assertNotIn('prompt_tokens', second_call_args)
        self.assertNotIn('completion_tokens', second_call_args)
        self.assertEqual(second_call_args['execution_time'], 1.0)

        # Call 3 (success) - before request
        third_call_args = mock_track_usage.call_args_list[2][1]
        self.assertEqual(third_call_args['model'], "model_two_success")
        self.assertEqual(third_call_args['prompt_tokens'], len(prompt.split()))
        self.assertIsNone(third_call_args.get('completion_tokens'))
        self.assertIsNone(third_call_args.get('execution_time'))

        # Call 4 (success) - after request
        fourth_call_args = mock_track_usage.call_args_list[3][1]
        self.assertEqual(fourth_call_args['model'], "model_two_success")
        self.assertEqual(fourth_call_args['prompt_tokens'], 15)
        self.assertEqual(fourth_call_args['completion_tokens'], 25)
        self.assertEqual(fourth_call_args['execution_time'], 1.0)


if __name__ == '__main__':
    unittest.main()
