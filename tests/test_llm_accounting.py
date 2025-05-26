import unittest
import os
import sqlite3
import json
from unittest.mock import patch, Mock, call # Added call here
import time # Will be needed for mocking time.monotonic

# Assuming llm_client is in src.llm_client
from src.llm_client import LLMClient
from src.aot_dataclasses import LLMCallStats # LLMClient returns this

# Default DB name used by llm-accounting
DB_NAME = "llm_accounting.db"

class TestLLMAccountingIntegration(unittest.TestCase):

    def setUp(self):
        # Set a dummy API key for LLMClient initialization
        os.environ["OPENROUTER_API_KEY"] = "dummy_key_for_testing"
        self.llm_client = LLMClient(api_key="dummy_key_for_testing")

        # Ensure a clean slate for the database for each test
        if os.path.exists(DB_NAME):
            os.remove(DB_NAME)
        
        # Patch time.monotonic to control call_duration_seconds
        self.mock_monotonic_patch = patch('time.monotonic')
        self.mock_monotonic = self.mock_monotonic_patch.start()
        self.mock_monotonic.side_effect = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] # Provide enough values

    def tearDown(self):
        # Clean up the database file after each test
        if os.path.exists(DB_NAME):
            os.remove(DB_NAME)
        del os.environ["OPENROUTER_API_KEY"]
        self.mock_monotonic_patch.stop()

    def _get_db_connection(self):
        return sqlite3.connect(DB_NAME)

    def _get_last_request_log(self, conn):
        cursor = conn.cursor()
        cursor.execute("SELECT id, model_name, prompt, request_payload, provider, call_type, timestamp FROM requests ORDER BY timestamp DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0], "model_name": row[1], "prompt": row[2], 
                "request_payload": json.loads(row[3]), "provider": row[4], 
                "call_type": row[5], "timestamp": row[6]
            }
        return None

    def _get_response_log_by_request_id(self, conn, request_id):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT request_id, model_name, response_payload, prompt_tokens, completion_tokens, cost, call_duration_seconds, timestamp 
            FROM responses WHERE request_id = ?
        """, (request_id,))
        row = cursor.fetchone()
        if row:
            return {
                "request_id": row[0], "model_name": row[1], "response_payload": json.loads(row[2]),
                "prompt_tokens": row[3], "completion_tokens": row[4], "cost": row[5],
                "call_duration_seconds": row[6], "timestamp": row[7]
            }
        return None

    def _get_all_request_logs(self, conn):
        cursor = conn.cursor()
        cursor.execute("SELECT id, model_name, prompt, request_payload, provider, call_type FROM requests ORDER BY timestamp ASC")
        return [{
            "id": r[0], "model_name": r[1], "prompt": r[2], 
            "request_payload": json.loads(r[3]), "provider": r[4], "call_type": r[5]
        } for r in cursor.fetchall()]

    def _get_all_response_logs(self, conn):
        cursor = conn.cursor()
        cursor.execute("SELECT request_id, model_name, response_payload, prompt_tokens, completion_tokens, cost, call_duration_seconds FROM responses ORDER BY timestamp ASC")
        return [{
            "request_id": r[0], "model_name": r[1], "response_payload": json.loads(r[2]),
            "prompt_tokens": r[3], "completion_tokens": r[4], "cost": r[5], "call_duration_seconds": r[6]
        } for r in cursor.fetchall()]

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
        log_request_spy = patch.object(self.llm_client.accounting, 'log_request', wraps=self.llm_client.accounting.log_request).start()
        log_response_spy = patch.object(self.llm_client.accounting, 'log_response', wraps=self.llm_client.accounting.log_response).start()

        prompt = "Test prompt for success"
        models = ["test_model_success"]
        temperature = 0.7
        
        content, stats = self.llm_client.call(prompt, models, temperature)

        self.assertEqual(content, "Test response")
        self.assertEqual(stats.prompt_tokens, 10)
        self.assertEqual(stats.completion_tokens, 20)
        self.assertEqual(stats.model_name, "test_model_success")
        self.assertEqual(stats.call_duration_seconds, 1.0) # 2.0 - 1.0

        log_request_spy.assert_called_once()
        log_response_spy.assert_called_once()
        
        request_args = log_request_spy.call_args[1]
        self.assertEqual(request_args['model_name'], "test_model_success")
        self.assertEqual(request_args['prompt'], prompt)
        self.assertEqual(request_args['provider'], "openrouter")

        response_args = log_response_spy.call_args[1]
        self.assertEqual(response_args['model_name'], "test_model_success")
        self.assertEqual(response_args['prompt_tokens'], 10)
        self.assertEqual(response_args['completion_tokens'], 20)
        self.assertGreater(response_args['cost'], 0) # llm-accounting should calculate cost
        self.assertEqual(response_args['call_duration_seconds'], 1.0)

        # Verify DB content
        conn = self._get_db_connection()
        try:
            db_request_log = self._get_last_request_log(conn)
            self.assertIsNotNone(db_request_log)
            self.assertEqual(db_request_log['model_name'], "test_model_success")
            self.assertEqual(db_request_log['prompt'], prompt)
            self.assertEqual(db_request_log['request_payload']['messages'][0]['content'], prompt)
            
            db_response_log = self._get_response_log_by_request_id(conn, db_request_log['id'])
            self.assertIsNotNone(db_response_log)
            self.assertEqual(db_response_log['model_name'], "test_model_success")
            self.assertEqual(db_response_log['response_payload']['choices'][0]['message']['content'], "Test response")
            self.assertEqual(db_response_log['prompt_tokens'], 10)
            self.assertEqual(db_response_log['completion_tokens'], 20)
            self.assertGreater(db_response_log['cost'], 0) 
            self.assertEqual(db_response_log['call_duration_seconds'], 1.0)
        finally:
            conn.close()
            log_request_spy.stop()
            log_response_spy.stop()

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


        log_request_spy = patch.object(self.llm_client.accounting, 'log_request', wraps=self.llm_client.accounting.log_request).start()
        log_response_spy = patch.object(self.llm_client.accounting, 'log_response', wraps=self.llm_client.accounting.log_response).start()

        prompt = "Test prompt for API error"
        models = ["test_model_api_error"]
        
        content, stats = self.llm_client.call(prompt, models, 0.7)

        self.assertTrue(content.startswith("Error: API call to test_model_api_error (HTTP 500)"))
        self.assertEqual(stats.prompt_tokens, 5) # Usage should be captured
        self.assertEqual(stats.completion_tokens, 0)
        self.assertEqual(stats.model_name, "test_model_api_error")
        self.assertEqual(stats.call_duration_seconds, 1.0) # 4.0 - 3.0 (due to setUp side_effect)

        log_request_spy.assert_called_once()
        log_response_spy.assert_called_once()

        response_args = log_response_spy.call_args[1]
        self.assertEqual(response_args['model_name'], "test_model_api_error")
        self.assertEqual(response_args['prompt_tokens'], 5)
        self.assertEqual(response_args['completion_tokens'], 0)
        # Cost might be 0 or calculated based on prompt tokens, depends on llm-accounting logic for errors
        # For now, let's check it's not None
        self.assertIsNotNone(response_args['cost']) 
        self.assertEqual(response_args['call_duration_seconds'], 1.0)

        conn = self._get_db_connection()
        try:
            db_request_log = self._get_last_request_log(conn)
            self.assertIsNotNone(db_request_log)
            self.assertEqual(db_request_log['model_name'], "test_model_api_error")
            
            db_response_log = self._get_response_log_by_request_id(conn, db_request_log['id'])
            self.assertIsNotNone(db_response_log)
            self.assertEqual(db_response_log['model_name'], "test_model_api_error")
            self.assertTrue("Server error" in db_response_log['response_payload']['error']['message'])
            self.assertEqual(db_response_log['prompt_tokens'], 5)
            self.assertEqual(db_response_log['completion_tokens'], 0)
        finally:
            conn.close()
            log_request_spy.stop()
            log_response_spy.stop()
            
    # Test 3: LLM Call with Network Error (No Response Body)
    @patch('requests.post')
    def test_network_error_logging(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

        log_request_spy = patch.object(self.llm_client.accounting, 'log_request', wraps=self.llm_client.accounting.log_request).start()
        log_response_spy = patch.object(self.llm_client.accounting, 'log_response', wraps=self.llm_client.accounting.log_response).start()

        prompt = "Test prompt for network error"
        models = ["test_model_network_error"]
        
        content, stats = self.llm_client.call(prompt, models, 0.7)

        self.assertTrue(content.startswith("Error: API call to test_model_network_error timed out"))
        self.assertEqual(stats.prompt_tokens, 0) # No usage info
        self.assertEqual(stats.completion_tokens, 0)
        self.assertEqual(stats.model_name, "test_model_network_error")
        self.assertEqual(stats.call_duration_seconds, 1.0) # 6.0 - 5.0

        log_request_spy.assert_called_once()
        log_response_spy.assert_called_once()
        
        response_args = log_response_spy.call_args[1]
        self.assertEqual(response_args['model_name'], "test_model_network_error")
        self.assertEqual(response_args['prompt_tokens'], 0)
        self.assertEqual(response_args['completion_tokens'], 0)
        self.assertEqual(response_args['cost'], 0) # No tokens, so cost should be 0
        self.assertEqual(response_args['call_duration_seconds'], 1.0)
        self.assertEqual(response_args['response_payload']['error'], "Timeout")


        conn = self._get_db_connection()
        try:
            db_request_log = self._get_last_request_log(conn)
            self.assertIsNotNone(db_request_log)
            self.assertEqual(db_request_log['model_name'], "test_model_network_error")
            
            db_response_log = self._get_response_log_by_request_id(conn, db_request_log['id'])
            self.assertIsNotNone(db_response_log)
            self.assertEqual(db_response_log['model_name'], "test_model_network_error")
            self.assertEqual(db_response_log['response_payload']['error'], "Timeout")
            self.assertTrue("Connection timed out" in db_response_log['response_payload']['details'])
            self.assertEqual(db_response_log['prompt_tokens'], 0)
            self.assertEqual(db_response_log['completion_tokens'], 0)
            self.assertEqual(db_response_log['cost'], 0)
        finally:
            conn.close()
            log_request_spy.stop()
            log_response_spy.stop()

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

        log_request_spy = patch.object(self.llm_client.accounting, 'log_request', wraps=self.llm_client.accounting.log_request).start()
        log_response_spy = patch.object(self.llm_client.accounting, 'log_response', wraps=self.llm_client.accounting.log_response).start()

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


        content, stats = self.llm_client.call(prompt, models, 0.7)

        self.assertEqual(content, "Failover successful")
        self.assertEqual(stats.model_name, "model_two_success")
        self.assertEqual(stats.prompt_tokens, 15)
        self.assertEqual(stats.completion_tokens, 25)
        # Duration for the successful call. The mock_monotonic side_effect is global for the test.
        # If first call starts at T1, ends at T2. Second call starts T3, ends T4.
        # Durations will be T2-T1 and T4-T3.
        # With current setup, T1=7, T2=8 (duration 1.0 for first call). T3=9, T4=10 (duration 1.0 for second call).
        self.assertEqual(stats.call_duration_seconds, 1.0)


        self.assertEqual(log_request_spy.call_count, 2)
        self.assertEqual(log_response_spy.call_count, 2)

        # Check calls to spies (order matters)
        # Call 1 (failure)
        request_args_fail = log_request_spy.call_args_list[0][1]
        self.assertEqual(request_args_fail['model_name'], "model_one_fail")
        
        response_args_fail = log_response_spy.call_args_list[0][1]
        self.assertEqual(response_args_fail['model_name'], "model_one_fail")
        self.assertEqual(response_args_fail['response_payload']['error'], "Timeout")
        self.assertEqual(response_args_fail['prompt_tokens'], 0)
        self.assertEqual(response_args_fail['call_duration_seconds'], 1.0) # 8.0 - 7.0

        # Call 2 (success)
        request_args_success = log_request_spy.call_args_list[1][1]
        self.assertEqual(request_args_success['model_name'], "model_two_success")

        response_args_success = log_response_spy.call_args_list[1][1]
        self.assertEqual(response_args_success['model_name'], "model_two_success")
        self.assertEqual(response_args_success['prompt_tokens'], 15)
        self.assertEqual(response_args_success['completion_tokens'], 25)
        self.assertGreater(response_args_success['cost'], 0)
        self.assertEqual(response_args_success['call_duration_seconds'], 1.0) # 10.0 - 9.0

        # Verify DB Content (order by timestamp to be sure)
        conn = self._get_db_connection()
        try:
            db_requests = self._get_all_request_logs(conn)
            db_responses = self._get_all_response_logs(conn)

            self.assertEqual(len(db_requests), 2)
            self.assertEqual(len(db_responses), 2)

            # First logged request/response (failure)
            self.assertEqual(db_requests[0]['model_name'], "model_one_fail")
            self.assertEqual(db_responses[0]['request_id'], db_requests[0]['id'])
            self.assertEqual(db_responses[0]['model_name'], "model_one_fail")
            self.assertEqual(db_responses[0]['response_payload']['error'], "Timeout")
            self.assertEqual(db_responses[0]['prompt_tokens'], 0)
            self.assertEqual(db_responses[0]['call_duration_seconds'], 1.0)


            # Second logged request/response (success)
            self.assertEqual(db_requests[1]['model_name'], "model_two_success")
            self.assertEqual(db_responses[1]['request_id'], db_requests[1]['id'])
            self.assertEqual(db_responses[1]['model_name'], "model_two_success")
            self.assertEqual(db_responses[1]['response_payload']['choices'][0]['message']['content'], "Failover successful")
            self.assertEqual(db_responses[1]['prompt_tokens'], 15)
            self.assertEqual(db_responses[1]['completion_tokens'], 25)
            self.assertGreater(db_responses[1]['cost'], 0)
            self.assertEqual(db_responses[1]['call_duration_seconds'], 1.0)

        finally:
            conn.close()
            log_request_spy.stop()
            log_response_spy.stop()

if __name__ == '__main__':
    unittest.main()
