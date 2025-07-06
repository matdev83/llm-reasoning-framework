import unittest
from unittest.mock import MagicMock, patch, call
import logging

# Ensure src path is available for imports, or adjust as per your test setup
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.llm_client import LLMClient
from src.llm_config import LLMConfig
from src.hybrid.processor import HybridProcessor
from src.hybrid.dataclasses import HybridConfig, HybridResult, LLMCallStats

# Suppress most logging output during tests
logging.basicConfig(level=logging.CRITICAL)

class TestHybridProcessor(unittest.TestCase):

    def setUp(self):
        self.mock_llm_client = MagicMock(spec=LLMClient)
        self.default_config = HybridConfig(
            reasoning_model_name="reasoning/dummy",
            response_model_name="response/dummy",
            reasoning_complete_token="<DONE>",
            reasoning_prompt_template="Reason: {problem_description} {reasoning_complete_token}",
            response_prompt_template="Problem: {problem_description} Reasoning: {extracted_reasoning} Answer:",
            max_reasoning_tokens=50,
            max_response_tokens=50
        )
        self.processor = HybridProcessor(llm_client=self.mock_llm_client, config=self.default_config)

    def test_run_successful_extraction_and_response(self):
        problem_desc = "What is 2+2?"
        reasoning_output_text = "Thinking... 2+2 is 4. <DONE> Some extra text ignored."
        final_answer_text = "The answer is 4."

        # Mock LLM calls
        mock_reasoning_stats = LLMCallStats(model_name="reasoning/dummy", completion_tokens=10, prompt_tokens=5, call_duration_seconds=0.5)
        mock_response_stats = LLMCallStats(model_name="response/dummy", completion_tokens=8, prompt_tokens=12, call_duration_seconds=0.4)

        self.mock_llm_client.call.side_effect = [
            (reasoning_output_text, mock_reasoning_stats),
            (final_answer_text, mock_response_stats)
        ]

        result = self.processor.run(problem_desc)

        self.assertTrue(result.succeeded)
        self.assertEqual(result.extracted_reasoning, "Thinking... 2+2 is 4.")
        self.assertEqual(result.final_answer, "The answer is 4.")
        self.assertEqual(result.reasoning_call_stats, mock_reasoning_stats)
        self.assertEqual(result.response_call_stats, mock_response_stats)
        self.assertIsNone(result.error_message)
        self.assertIsNotNone(result.detected_reasoning_format)  # Should have detected a format

        # Verify LLM client was called correctly
        expected_reasoning_prompt = f"Reason: {problem_desc} {self.default_config.reasoning_complete_token}"
        expected_response_prompt = f"Problem: {problem_desc} Reasoning: Thinking... 2+2 is 4. Answer:"

        calls = self.mock_llm_client.call.call_args_list
        self.assertEqual(len(calls), 2)

        # Reasoning call - check the new LLMConfig structure
        self.assertEqual(calls[0][1]['prompt'], expected_reasoning_prompt)
        self.assertEqual(calls[0][1]['models'], [self.default_config.reasoning_model_name])
        reasoning_config = calls[0][1]['config']
        self.assertIsInstance(reasoning_config, LLMConfig)
        self.assertEqual(reasoning_config.temperature, self.default_config.reasoning_model_temperature)
        self.assertEqual(reasoning_config.max_tokens, self.default_config.max_reasoning_tokens)
        self.assertEqual(reasoning_config.stop, [self.default_config.reasoning_complete_token])

        # Response call - check the new LLMConfig structure
        self.assertEqual(calls[1][1]['prompt'], expected_response_prompt)
        self.assertEqual(calls[1][1]['models'], [self.default_config.response_model_name])
        response_config = calls[1][1]['config']
        self.assertIsInstance(response_config, LLMConfig)
        self.assertEqual(response_config.temperature, self.default_config.response_model_temperature)
        self.assertEqual(response_config.max_tokens, self.default_config.max_response_tokens)


    def test_run_reasoning_call_fails(self):
        problem_desc = "A complex problem."
        self.mock_llm_client.call.side_effect = Exception("Reasoning LLM Error")

        result = self.processor.run(problem_desc)

        self.assertFalse(result.succeeded)
        self.assertIn("Reasoning model call failed: Reasoning LLM Error", result.error_message)
        self.assertIsNone(result.final_answer)
        self.assertIsNotNone(result.reasoning_call_stats) # Should have placeholder stats
        self.assertEqual(result.reasoning_call_stats.model_name, self.default_config.reasoning_model_name)
        self.mock_llm_client.call.assert_called_once() # Only reasoning call should be attempted

    def test_run_response_call_fails(self):
        problem_desc = "Another problem."
        reasoning_output_text = "Some reasoning. <DONE>"
        mock_reasoning_stats = LLMCallStats(model_name="reasoning/dummy", completion_tokens=5, prompt_tokens=5, call_duration_seconds=0.2)

        self.mock_llm_client.call.side_effect = [
            (reasoning_output_text, mock_reasoning_stats),
            Exception("Response LLM Error")
        ]

        result = self.processor.run(problem_desc)

        self.assertFalse(result.succeeded)
        self.assertIn("Response model call failed: Response LLM Error", result.error_message)
        self.assertIsNone(result.final_answer)
        self.assertEqual(result.extracted_reasoning, "Some reasoning.")
        self.assertEqual(result.reasoning_call_stats, mock_reasoning_stats)
        self.assertIsNotNone(result.response_call_stats) # Should have placeholder stats
        self.assertEqual(result.response_call_stats.model_name, self.default_config.response_model_name)
        self.assertEqual(self.mock_llm_client.call.call_count, 2)

    def test_reasoning_extraction_no_token(self):
        problem_desc = "Problem with no end token in reasoning."
        reasoning_output_text = "All reasoning, but no end token."
        # This should still extract everything if the token is not found, as per split behavior

        mock_reasoning_stats = LLMCallStats(model_name="reasoning/dummy", completion_tokens=10, prompt_tokens=5, call_duration_seconds=0.5)
        mock_response_stats = LLMCallStats(model_name="response/dummy", completion_tokens=8, prompt_tokens=12, call_duration_seconds=0.4)

        self.mock_llm_client.call.side_effect = [
            (reasoning_output_text, mock_reasoning_stats),
            ("Final answer based on incomplete reasoning.", mock_response_stats)
        ]

        result = self.processor.run(problem_desc)

        self.assertTrue(result.succeeded) # Processor currently continues
        self.assertEqual(result.extracted_reasoning, "All reasoning, but no end token.")
        # The response stage will use this potentially incomplete reasoning.

    def test_reasoning_extraction_empty_output(self):
        problem_desc = "Problem leading to empty reasoning."
        reasoning_output_text = "<DONE> Only token."
        mock_reasoning_stats = LLMCallStats(model_name="reasoning/dummy", completion_tokens=1, prompt_tokens=5, call_duration_seconds=0.1)
        mock_response_stats = LLMCallStats(model_name="response/dummy", completion_tokens=8, prompt_tokens=12, call_duration_seconds=0.4)

        self.mock_llm_client.call.side_effect = [
            (reasoning_output_text, mock_reasoning_stats),
            ("Final answer based on empty reasoning.", mock_response_stats)
        ]

        result = self.processor.run(problem_desc)

        self.assertTrue(result.succeeded) # Processor currently continues
        self.assertEqual(result.extracted_reasoning, "") # Empty string before token
        self.assertEqual(result.final_answer, "Final answer based on empty reasoning.")

    def test_stop_sequence_for_early_cancellation(self):
        """Test that stop sequences are properly configured for early cancellation to save tokens"""
        problem_desc = "Test problem for stop sequence"
        reasoning_output_text = "Some reasoning here"
        final_answer_text = "Final answer"

        mock_reasoning_stats = LLMCallStats(model_name="reasoning/dummy", completion_tokens=5, prompt_tokens=5, call_duration_seconds=0.2)
        mock_response_stats = LLMCallStats(model_name="response/dummy", completion_tokens=8, prompt_tokens=12, call_duration_seconds=0.4)

        self.mock_llm_client.call.side_effect = [
            (reasoning_output_text, mock_reasoning_stats),
            (final_answer_text, mock_response_stats)
        ]

        result = self.processor.run(problem_desc)

        # Verify that the reasoning call includes the stop sequence
        calls = self.mock_llm_client.call.call_args_list
        reasoning_config = calls[0][1]['config']
        self.assertIsInstance(reasoning_config, LLMConfig)
        self.assertIn(self.default_config.reasoning_complete_token, reasoning_config.stop)

    def test_deepseek_r1_thinking_tags_extraction(self):
        """Test extraction of DeepSeek-R1 style thinking tags"""
        problem_desc = "What is the square root of 16?"
        reasoning_output_text = """<THINKING>
Mathematical reasoning process here.
Step by step calculation approach.
Final verification of the result.
</THINKING>

The answer is 4."""
        
        mock_reasoning_stats = LLMCallStats(model_name="deepseek/r1", completion_tokens=20, prompt_tokens=5, call_duration_seconds=0.5)
        mock_response_stats = LLMCallStats(model_name="response/dummy", completion_tokens=8, prompt_tokens=12, call_duration_seconds=0.4)

        # Configure for DeepSeek-R1 model
        config = HybridConfig(
            reasoning_model_name="deepseek/r1",
            response_model_name="response/dummy",
            reasoning_complete_token="<DONE>",  # Won't be used for THINKING tags
        )
        processor = HybridProcessor(llm_client=self.mock_llm_client, config=config)

        self.mock_llm_client.call.side_effect = [
            (reasoning_output_text, mock_reasoning_stats),
            ("The final answer is 4.", mock_response_stats)
        ]

        result = processor.run(problem_desc)

        self.assertTrue(result.succeeded)
        # Test structure rather than exact content
        self.assertGreater(len(result.extracted_reasoning.strip()), 0)  # Should extract non-empty reasoning
        self.assertGreaterEqual(len(result.extracted_reasoning.split()), 3)  # Should have multiple words/tokens
        self.assertNotIn("<THINKING>", result.extracted_reasoning)  # Tags should be removed
        self.assertNotIn("</THINKING>", result.extracted_reasoning)  # Tags should be removed
        self.assertEqual(result.detected_reasoning_format, "deepseek_r1")
        self.assertEqual(result.final_answer, "The final answer is 4.")

    def test_format_hint_based_on_model_name(self):
        """Test that format hints are correctly determined from model names"""
        # Test DeepSeek-R1 model name detection
        config_deepseek = HybridConfig(
            reasoning_model_name="deepseek/deepseek-r1-distill-qwen-7b",
            response_model_name="response/dummy"
        )
        processor_deepseek = HybridProcessor(llm_client=self.mock_llm_client, config=config_deepseek)
        hint = processor_deepseek._get_format_hint()
        self.assertEqual(hint.value, "deepseek_r1")

        # Test OpenAI o1 model name detection
        config_openai = HybridConfig(
            reasoning_model_name="openai/o1-preview",
            response_model_name="response/dummy"
        )
        processor_openai = HybridProcessor(llm_client=self.mock_llm_client, config=config_openai)
        hint = processor_openai._get_format_hint()
        self.assertEqual(hint.value, "openai_o1")

        # Test Gemini model name detection
        config_gemini = HybridConfig(
            reasoning_model_name="google/gemini-2.0-flash-thinking",
            response_model_name="response/dummy"
        )
        processor_gemini = HybridProcessor(llm_client=self.mock_llm_client, config=config_gemini)
        hint = processor_gemini._get_format_hint()
        self.assertEqual(hint.value, "gemini_thinking")

if __name__ == '__main__':
    unittest.main()
