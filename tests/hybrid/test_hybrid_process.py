import unittest
from unittest.mock import MagicMock, patch, ANY
import logging
import time # For checking timing if needed, though not explicitly tested here

# Ensure src path is available
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.hybrid.orchestrator import HybridProcess # HybridProcess is in orchestrator.py
from src.hybrid.processor import HybridProcessor
from src.hybrid.dataclasses import HybridConfig, HybridResult, HybridSolution, LLMCallStats
from src.llm_client import LLMClient # For mocking HybridProcess's own client

# Suppress most logging output during tests
logging.basicConfig(level=logging.CRITICAL)

class TestHybridProcess(unittest.TestCase):

    def setUp(self):
        self.api_key = "test_api_key"
        self.default_hybrid_config = HybridConfig(
            reasoning_model_name="reasoning/dummy",
            response_model_name="response/dummy"
            # Other HybridConfig defaults are used
        )
        self.direct_oneshot_models = ["oneshot/dummy"]
        self.direct_oneshot_temp = 0.5

        # This mock will be for the HybridProcessor instance *within* HybridProcess
        self.mock_hybrid_processor_instance = MagicMock(spec=HybridProcessor)

        # This mock is for the LLMClient instance created *by* HybridProcess for its own fallbacks
        self.mock_process_llm_client = MagicMock(spec=LLMClient)

    @patch('src.hybrid.orchestrator.HybridProcessor') # Patch where HybridProcessor is looked up for HybridProcess
    @patch('src.hybrid.orchestrator.LLMClient')      # Patch where LLMClient is looked up for HybridProcess
    def test_execute_successful_hybrid(self, MockLLMClient, MockHybridProcessor):
        # Configure mocks
        MockHybridProcessor.return_value = self.mock_hybrid_processor_instance
        MockLLMClient.return_value = self.mock_process_llm_client

        mock_reasoning_stats = LLMCallStats(model_name="reasoning/dummy", completion_tokens=10, prompt_tokens=5, call_duration_seconds=0.5)
        mock_response_stats = LLMCallStats(model_name="response/dummy", completion_tokens=8, prompt_tokens=12, call_duration_seconds=0.4)

        successful_hybrid_result = HybridResult(
            succeeded=True,
            final_answer="Hybrid success answer.",
            extracted_reasoning="Some extracted thoughts.",
            reasoning_call_stats=mock_reasoning_stats,
            response_call_stats=mock_response_stats
        )
        self.mock_hybrid_processor_instance.run.return_value = successful_hybrid_result

        process = HybridProcess(
            hybrid_config=self.default_hybrid_config,
            direct_oneshot_model_names=self.direct_oneshot_models,
            direct_oneshot_temperature=self.direct_oneshot_temp,
            api_key=self.api_key
        )

        problem_desc = "Test problem"
        process.execute(problem_desc, model_name="placeholder_model") # model_name is not used by HybridProcess directly

        solution, summary = process.get_result()

        self.assertIsNotNone(solution)
        self.assertTrue(solution.hybrid_result.succeeded)
        self.assertEqual(solution.final_answer, "Hybrid success answer.")
        self.assertIn("Extracted Reasoning:", solution.reasoning_trace[0])
        self.assertIn("Some extracted thoughts.", solution.reasoning_trace[0])
        self.assertFalse(solution.hybrid_failed_and_fell_back)
        self.assertIsNone(solution.fallback_call_stats)

        self.mock_hybrid_processor_instance.run.assert_called_once_with(problem_desc)
        self.mock_process_llm_client.call.assert_not_called() # Fallback client should not be called

        self.assertIn("HybridProcess Execution Summary", summary)
        self.assertIn("Hybrid Processor Succeeded: Yes", summary)


    @patch('src.hybrid.orchestrator.HybridProcessor')
    @patch('src.hybrid.orchestrator.LLMClient')
    def test_execute_hybrid_fails_fallback_succeeds(self, MockLLMClient, MockHybridProcessor):
        MockHybridProcessor.return_value = self.mock_hybrid_processor_instance
        MockLLMClient.return_value = self.mock_process_llm_client

        failed_hybrid_result = HybridResult(
            succeeded=False,
            error_message="Hybrid processor failed",
            extracted_reasoning="Partial thoughts before failure." # Still might have some reasoning
        )
        self.mock_hybrid_processor_instance.run.return_value = failed_hybrid_result

        fallback_answer = "Fallback successful answer."
        mock_fallback_stats = LLMCallStats(model_name="oneshot/dummy", completion_tokens=20, prompt_tokens=10, call_duration_seconds=0.8) # Removed extra arg
        self.mock_process_llm_client.call.return_value = (fallback_answer, mock_fallback_stats)

        process = HybridProcess(
            hybrid_config=self.default_hybrid_config,
            direct_oneshot_model_names=self.direct_oneshot_models,
            direct_oneshot_temperature=self.direct_oneshot_temp,
            api_key=self.api_key
        )

        problem_desc = "Test problem for fallback"
        process.execute(problem_desc, model_name="placeholder_model")

        solution, summary = process.get_result()

        self.assertIsNotNone(solution)
        self.assertFalse(solution.hybrid_result.succeeded)
        self.assertEqual(solution.final_answer, fallback_answer)
        self.assertTrue(solution.hybrid_failed_and_fell_back)
        self.assertEqual(solution.fallback_call_stats, mock_fallback_stats)
        self.assertIn("Partial thoughts before failure.", solution.reasoning_trace[0]) # Check reasoning trace preserved

        self.mock_hybrid_processor_instance.run.assert_called_once_with(problem_desc)
        self.mock_process_llm_client.call.assert_called_once_with(
            prompt=problem_desc,
            models=self.direct_oneshot_models,
            temperature=self.direct_oneshot_temp
        )
        self.assertIn("Hybrid FAILED and Fell Back to One-Shot: Yes", summary)
        self.assertIn(f"Fallback One-Shot Call ({mock_fallback_stats.model_name})", summary)


    @patch('src.hybrid.orchestrator.HybridProcessor')
    @patch('src.hybrid.orchestrator.LLMClient')
    def test_execute_hybrid_fails_fallback_also_fails(self, MockLLMClient, MockHybridProcessor):
        MockHybridProcessor.return_value = self.mock_hybrid_processor_instance
        MockLLMClient.return_value = self.mock_process_llm_client

        failed_hybrid_result = HybridResult(succeeded=False, error_message="Hybrid processor error")
        self.mock_hybrid_processor_instance.run.return_value = failed_hybrid_result

        # Configure mock for LLMClient.call to return an error string and stats
        fallback_error_answer = "Error during fallback call"
        mock_fallback_error_stats = LLMCallStats(model_name="oneshot/dummy", completion_tokens=0, prompt_tokens=10, call_duration_seconds=0.1)
        self.mock_process_llm_client.call.return_value = (fallback_error_answer, mock_fallback_error_stats)

        process = HybridProcess(
            hybrid_config=self.default_hybrid_config,
            direct_oneshot_model_names=self.direct_oneshot_models,
            direct_oneshot_temperature=self.direct_oneshot_temp,
            api_key=self.api_key
        )

        problem_desc = "Test problem for double failure"
        process.execute(problem_desc, model_name="placeholder_model")
        solution, summary = process.get_result()

        self.assertIsNotNone(solution)
        self.assertFalse(solution.hybrid_result.succeeded)
        self.assertTrue(solution.hybrid_failed_and_fell_back)
        self.assertEqual(solution.final_answer, fallback_error_answer)
        self.assertEqual(solution.fallback_call_stats, mock_fallback_error_stats)

        self.assertIn("Hybrid FAILED and Fell Back to One-Shot: Yes", summary)


    def test_summary_generation_content(self):
        MockHybridProcessorInstance = MagicMock(spec=HybridProcessor)
        MockProcessLLMClientInstance = MagicMock(spec=LLMClient)

        with patch('src.hybrid.orchestrator.HybridProcessor', return_value=MockHybridProcessorInstance):
            with patch('src.hybrid.orchestrator.LLMClient', return_value=MockProcessLLMClientInstance):
                mock_reasoning_stats = LLMCallStats(model_name="reasoning/dummy", completion_tokens=10, prompt_tokens=5, call_duration_seconds=0.5)
                mock_response_stats = LLMCallStats(model_name="response/dummy", completion_tokens=8, prompt_tokens=12, call_duration_seconds=0.4)
                successful_hybrid_result = HybridResult(
                    succeeded=True,
                    final_answer="Summary test answer.",
                    extracted_reasoning="Reasoning for summary.",
                    reasoning_call_stats=mock_reasoning_stats,
                    response_call_stats=mock_response_stats
                )
                MockHybridProcessorInstance.run.return_value = successful_hybrid_result

                process = HybridProcess(
                    hybrid_config=self.default_hybrid_config,
                    direct_oneshot_model_names=self.direct_oneshot_models,
                    direct_oneshot_temperature=self.direct_oneshot_temp,
                    api_key=self.api_key
                )
                process.execute("summary problem", "placeholder")
                solution, summary = process.get_result()

                self.assertIsNotNone(solution)
                self.assertIn("HybridProcess Execution Summary", summary)
                self.assertIn("Hybrid Processor Succeeded: Yes", summary)
                self.assertIn(f"Reasoning Call ({mock_reasoning_stats.model_name})", summary)
                self.assertIn(f"Response Call ({mock_response_stats.model_name})", summary)
                self.assertIn("Extracted Reasoning Length: 22 chars", summary)
                self.assertEqual(solution.total_completion_tokens, 18)
                self.assertEqual(solution.total_prompt_tokens, 17)
                self.assertEqual(solution.grand_total_tokens, 35)
                self.assertAlmostEqual(solution.total_llm_interaction_time_seconds, 0.90)
                self.assertIn("Reasoning Trace (from HybridProcess)", summary)
                self.assertIn("Reasoning for summary.", summary)
                self.assertIn("Final Answer (from HybridProcess):\nSummary test answer.", summary)


if __name__ == '__main__':
    unittest.main()
