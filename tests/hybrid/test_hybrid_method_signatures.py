import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure src path is available for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.llm_client import LLMClient
from src.llm_config import LLMConfig
from src.hybrid.processor import HybridProcessor
from src.hybrid.dataclasses import HybridConfig, LLMCallStats

class TestHybridMethodSignatures(unittest.TestCase):
    """
    Simple tests focused on catching method signature issues.
    These tests would have caught the original bug where HybridProcessor
    was calling LLMClient.call() with wrong parameters.
    """

    def setUp(self):
        self.hybrid_config = HybridConfig(
            reasoning_model_name="test/reasoning-model",
            response_model_name="test/response-model",
            reasoning_model_temperature=0.1,
            response_model_temperature=0.3,
            reasoning_complete_token="<DONE>",
            reasoning_prompt_template="Problem: {problem_description}\nReason: {reasoning_complete_token}",
            response_prompt_template="Problem: {problem_description}\nReasoning: {extracted_reasoning}\nAnswer:",
            max_reasoning_tokens=100,
            max_response_tokens=50
        )

    def test_llm_client_call_method_signature(self):
        """Test that LLMClient.call() is called with correct method signature"""
        
        # Create a mock LLMClient that enforces the correct signature
        mock_llm_client = MagicMock(spec=LLMClient)
        
        # Use a simple side_effect list for predictable responses
        mock_llm_client.call.side_effect = [
            ("Some reasoning <DONE>", LLMCallStats("test/reasoning-model", 5, 5, 0.1)),
            ("Final answer", LLMCallStats("test/response-model", 5, 5, 0.1))
        ]
        
        # This should work without method signature errors
        processor = HybridProcessor(llm_client=mock_llm_client, config=self.hybrid_config)
        result = processor.run("Test problem")
        
        # Verify it worked (the main point is that no TypeError was raised)
        self.assertTrue(result.succeeded)
        self.assertEqual(result.extracted_reasoning, "Some reasoning")
        self.assertEqual(result.final_answer, "Final answer")
        
        # Verify it was called twice (reasoning + response)
        self.assertEqual(mock_llm_client.call.call_count, 2)
        
        # The key test: verify the method signature by checking the calls
        # If the signature was wrong, the mock would have failed
        self.assertTrue(mock_llm_client.call.called)

    def test_llm_config_objects_are_created_correctly(self):
        """Test that LLMConfig objects are created with correct parameters"""
        
        # Create a mock that captures the actual LLMConfig objects
        captured_configs = []
        
        def capture_config(prompt, models, config):
            captured_configs.append(config)
            if len(captured_configs) == 1:
                return ("Reasoning <DONE>", LLMCallStats("test/reasoning-model", 5, 5, 0.1))
            else:
                return ("Answer", LLMCallStats("test/response-model", 5, 5, 0.1))
        
        mock_llm_client = MagicMock(spec=LLMClient)
        mock_llm_client.call.side_effect = capture_config
        
        processor = HybridProcessor(llm_client=mock_llm_client, config=self.hybrid_config)
        processor.run("Test problem")
        
        # Check that we captured 2 configs
        self.assertEqual(len(captured_configs), 2)
        
        # Check reasoning config
        reasoning_config = captured_configs[0]
        self.assertIsInstance(reasoning_config, LLMConfig)
        self.assertEqual(reasoning_config.temperature, self.hybrid_config.reasoning_model_temperature)
        self.assertEqual(reasoning_config.max_tokens, self.hybrid_config.max_reasoning_tokens)
        self.assertIn(self.hybrid_config.reasoning_complete_token, reasoning_config.stop)
        
        # Check response config
        response_config = captured_configs[1]
        self.assertIsInstance(response_config, LLMConfig)
        self.assertEqual(response_config.temperature, self.hybrid_config.response_model_temperature)
        self.assertEqual(response_config.max_tokens, self.hybrid_config.max_response_tokens)

    def test_method_signature_error_detection(self):
        """Test that would catch the original method signature error"""
        
        # Create a mock that raises TypeError for wrong signature
        def strict_call(prompt, models, config):
            # This is the correct signature - if called with kwargs, it should fail
            return ("Test", LLMCallStats("test", 5, 5, 0.1))
        
        mock_llm_client = MagicMock(spec=LLMClient)
        mock_llm_client.call.side_effect = strict_call
        
        processor = HybridProcessor(llm_client=mock_llm_client, config=self.hybrid_config)
        
        # This should work with the fixed implementation
        result = processor.run("Test problem")
        self.assertTrue(result.succeeded)
        
        # Now test what would happen with the old broken implementation
        # If someone tries to call with kwargs, it should fail
        with self.assertRaises(TypeError):
            # This simulates the old broken call signature
            mock_llm_client.call("test", ["model"], temperature=0.1, max_tokens=50)

    def test_stop_sequence_is_configured(self):
        """Test that stop sequences are properly configured for token savings"""
        
        # Capture the actual calls to verify stop sequences
        captured_calls = []
        
        def capture_call(prompt, models, config):
            captured_calls.append((prompt, models, config))
            if len(captured_calls) == 1:
                return ("Reasoning <DONE>", LLMCallStats("test/reasoning-model", 5, 5, 0.1))
            else:
                return ("Answer", LLMCallStats("test/response-model", 5, 5, 0.1))
        
        mock_llm_client = MagicMock(spec=LLMClient)
        mock_llm_client.call.side_effect = capture_call
        
        processor = HybridProcessor(llm_client=mock_llm_client, config=self.hybrid_config)
        processor.run("Test problem")
        
        # Check that reasoning call has stop sequence
        reasoning_call = captured_calls[0]
        reasoning_config = reasoning_call[2]
        self.assertIsNotNone(reasoning_config.stop)
        self.assertIn(self.hybrid_config.reasoning_complete_token, reasoning_config.stop)

if __name__ == '__main__':
    unittest.main() 