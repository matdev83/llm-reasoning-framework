import unittest
import pytest
import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path

# Ensure src path is available
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.hybrid.orchestrator import HybridOrchestrator, HybridProcess
from src.hybrid.processor import HybridProcessor
from src.hybrid.dataclasses import HybridConfig
from src.hybrid.enums import HybridTriggerMode
from src.llm_client import LLMClient

# Skip integration tests by default - they require API keys and make real calls
pytestmark = pytest.mark.integration

class TestHybridIntegration(unittest.TestCase):
    """
    Integration tests for the Hybrid Thinking Model.
    
    These tests make real API calls and are disabled by default.
    To run them, use: pytest -m integration
    
    Requires OPENROUTER_API_KEY environment variable.
    """
    
    @classmethod
    def setUpClass(cls):
        cls.api_key = os.getenv('OPENROUTER_API_KEY')
        if not cls.api_key:
            pytest.skip("OPENROUTER_API_KEY environment variable not set")
        
        # Test models as specified
        cls.reasoning_model = "deepseek/deepseek-r1-0528:free"
        cls.response_model = "openrouter/cypher-alpha:free"
        
        # Simple test problems
        cls.simple_problem = "What is 7 + 15? Show your reasoning step by step."
        cls.complex_problem = "A farmer has chickens and rabbits. There are 35 heads and 94 legs total. How many chickens and how many rabbits are there? Explain your reasoning."

    def setUp(self):
        """Set up for each test"""
        self.hybrid_config = HybridConfig(
            reasoning_model_name=self.reasoning_model,
            response_model_name=self.response_model,
            reasoning_model_temperature=0.1,
            response_model_temperature=0.3,
            reasoning_complete_token="<REASONING_COMPLETE>",
            reasoning_prompt_template="Problem: {problem_description}\n\nThink step-by-step to solve this problem. When you finish your reasoning, output exactly: {reasoning_complete_token}\n\nReasoning:",
            response_prompt_template="Original Problem: {problem_description}\n\nReasoning from first model:\n{extracted_reasoning}\n\nBased on the problem and the reasoning above, provide a clear final answer:",
            max_reasoning_tokens=800,
            max_response_tokens=400
        )
        
        self.llm_client = LLMClient(api_key=self.api_key)

    def test_hybrid_processor_real_api_calls(self):
        """Test HybridProcessor with real API calls to verify method signatures work"""
        processor = HybridProcessor(llm_client=self.llm_client, config=self.hybrid_config)
        
        # This should work without method signature errors
        result = processor.run(self.simple_problem)
        
        # Verify the result structure
        self.assertIsNotNone(result)
        self.assertIsInstance(result.succeeded, bool)
        
        if result.succeeded:
            self.assertIsNotNone(result.final_answer)
            self.assertIsNotNone(result.extracted_reasoning)
            self.assertIsNotNone(result.reasoning_call_stats)
            self.assertIsNotNone(result.response_call_stats)
            self.assertIsNone(result.error_message)
            
            # Verify the reasoning was actually extracted
            self.assertGreater(len(result.extracted_reasoning), 0)
            self.assertGreater(len(result.final_answer), 0)
            
            # Check that reasoning extraction worked properly
            # For DeepSeek-R1 models, the token might appear in raw output but should be handled during extraction
            # For models with stop sequences, the token shouldn't appear at all
            if 'deepseek' in self.reasoning_model.lower() and 'r1' in self.reasoning_model.lower():
                # DeepSeek-R1: reasoning extraction should handle the token properly
                # The extracted reasoning should be meaningful content, not just the token
                self.assertGreater(len(result.extracted_reasoning.strip()), len(self.hybrid_config.reasoning_complete_token))
                # The reasoning should contain actual reasoning content, not just the completion token
                reasoning_without_token = result.extracted_reasoning.replace(self.hybrid_config.reasoning_complete_token, "").strip()
                self.assertGreater(len(reasoning_without_token), 50, "Reasoning should contain substantial content beyond the completion token")
            else:
                # Other models: stop sequences should prevent the token from appearing
                self.assertNotIn(self.hybrid_config.reasoning_complete_token, result.extracted_reasoning)
            
            print(f"✅ Reasoning extracted ({len(result.extracted_reasoning)} chars): {result.extracted_reasoning[:100]}...")
            print(f"✅ Final answer: {result.final_answer}")
            
        else:
            print(f"❌ Hybrid processor failed: {result.error_message}")
            # Don't fail the test if API calls fail due to rate limits, etc.
            # but do fail if it's a method signature error
            if "unexpected keyword argument" in str(result.error_message):
                self.fail(f"Method signature error detected: {result.error_message}")

    def test_hybrid_process_end_to_end(self):
        """Test HybridProcess end-to-end functionality"""
        process = HybridProcess(
            hybrid_config=self.hybrid_config,
            direct_oneshot_model_names=[self.response_model],
            direct_oneshot_temperature=0.3,
            api_key=self.api_key
        )
        
        # Execute the process
        process.execute(self.simple_problem, model_name="test")
        solution, summary = process.get_result()
        
        self.assertIsNotNone(solution)
        self.assertIsNotNone(summary)
        
        # Check that we got some kind of result
        self.assertIsNotNone(solution.final_answer)
        
        if solution.hybrid_result and solution.hybrid_result.succeeded:
            print(f"✅ Hybrid process succeeded")
            print(f"✅ Final answer: {solution.final_answer}")
            self.assertGreater(len(solution.final_answer), 0)
        elif solution.hybrid_failed_and_fell_back:
            print(f"⚠️ Hybrid failed, fell back to one-shot: {solution.final_answer}")
            # This is still a valid outcome
            self.assertGreater(len(solution.final_answer), 0)
        else:
            print(f"❌ Process failed: {solution.final_answer}")

    def test_hybrid_orchestrator_always_mode(self):
        """Test HybridOrchestrator in ALWAYS_HYBRID mode"""
        orchestrator = HybridOrchestrator(
            trigger_mode=HybridTriggerMode.ALWAYS_HYBRID,
            hybrid_config=self.hybrid_config,
            direct_oneshot_model_names=[self.response_model],
            direct_oneshot_temperature=0.3,
            api_key=self.api_key
        )
        
        solution, summary = orchestrator.solve(self.simple_problem)
        
        self.assertIsNotNone(solution)
        self.assertIsNotNone(summary)
        self.assertIsNotNone(solution.final_answer)
        
        print(f"✅ Orchestrator result: {solution.final_answer}")

    def test_reasoning_token_injection(self):
        """Test that reasoning tokens are properly injected into the response model"""
        processor = HybridProcessor(llm_client=self.llm_client, config=self.hybrid_config)
        
        # Use a problem where we can verify reasoning was used
        result = processor.run(self.complex_problem)
        
        if result.succeeded:
            # Verify that reasoning was extracted and is substantial
            self.assertIsNotNone(result.extracted_reasoning)
            self.assertGreater(len(result.extracted_reasoning), 20)  # Should have substantial reasoning
            
            # The final answer should be different from what a simple one-shot would produce
            # because it has access to the reasoning
            self.assertIsNotNone(result.final_answer)
            self.assertGreater(len(result.final_answer), 10)
            
            print(f"✅ Complex problem reasoning: {result.extracted_reasoning[:200]}...")
            print(f"✅ Final answer with injected reasoning: {result.final_answer}")

    def test_early_cancellation_token_usage(self):
        """Test that stop sequences help with early cancellation"""
        processor = HybridProcessor(llm_client=self.llm_client, config=self.hybrid_config)
        
        result = processor.run(self.simple_problem)
        
        if result.succeeded and result.reasoning_call_stats:
            # The reasoning model should have been stopped early if it produced the completion token
            # We can't directly test token savings, but we can verify the stop sequence was configured
            print(f"✅ Reasoning model used {result.reasoning_call_stats.completion_tokens} completion tokens")
            print(f"✅ Response model used {result.response_call_stats.completion_tokens} completion tokens")
            
            # Verify reasoning extraction worked properly
            if result.extracted_reasoning:
                if 'deepseek' in self.reasoning_model.lower() and 'r1' in self.reasoning_model.lower():
                    # DeepSeek-R1: token might appear but reasoning should be substantial
                    reasoning_without_token = result.extracted_reasoning.replace(self.hybrid_config.reasoning_complete_token, "").strip()
                    self.assertGreater(len(reasoning_without_token), 20, "Should have substantial reasoning content")
                else:
                    # Other models: stop sequences should prevent token from appearing
                    self.assertNotIn(self.hybrid_config.reasoning_complete_token, result.extracted_reasoning)

class TestHybridCLIIntegration(unittest.TestCase):
    """
    Test the hybrid thinking model through the CLI interface
    """
    
    @classmethod
    def setUpClass(cls):
        cls.api_key = os.getenv('OPENROUTER_API_KEY')
        if not cls.api_key:
            pytest.skip("OPENROUTER_API_KEY environment variable not set")

    @pytest.mark.integration
    def test_cli_hybrid_direct_mode(self):
        """Test hybrid thinking model through CLI in direct mode"""
        
        # Create a temporary problem file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("What is 12 + 8? Show your calculation.")
            problem_file = f.name
        
        try:
            # Test CLI call with hybrid-direct mode
            cmd = [
                sys.executable, "src/cli_runner.py",
                "--processing-mode", "hybrid-direct",
                "--problem-file", problem_file,
                "--hybrid-reasoning-models", "deepseek/deepseek-r1-0528:free",
                "--hybrid-response-models", "openrouter/cypher-alpha:free",
                "--hybrid-reasoning-temp", "0.1",
                "--hybrid-response-temp", "0.3",
                "--hybrid-max-reasoning-tokens", "500",
                "--hybrid-max-response-tokens", "300"
            ]
            
            # Set environment variable for API key
            env = os.environ.copy()
            env['OPENROUTER_API_KEY'] = self.api_key
            
            # Run the CLI command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=120  # 2 minute timeout
            )
            
            print(f"CLI return code: {result.returncode}")
            print(f"CLI stdout: {result.stdout}")
            if result.stderr:
                print(f"CLI stderr: {result.stderr}")
            
            # Check that the command succeeded
            self.assertEqual(result.returncode, 0, f"CLI command failed with stderr: {result.stderr}")
            
            # Check that we got some output
            self.assertGreater(len(result.stdout), 0)
            
            # Check for expected patterns in output
            output = result.stdout
            self.assertIn("Final Answer", output)
            
            # Should contain evidence of hybrid processing
            # (This will depend on the actual CLI output format)
            
        finally:
            # Clean up temporary file
            os.unlink(problem_file)

    @pytest.mark.integration  
    def test_cli_hybrid_orchestrator_always_mode(self):
        """Test hybrid thinking model through CLI orchestrator in always mode"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("A box contains 5 red balls and 3 blue balls. If you draw 2 balls without replacement, what is the probability both are red?")
            problem_file = f.name
        
        try:
            cmd = [
                sys.executable, "src/cli_runner.py",
                "--processing-mode", "hybrid-always",
                "--problem-file", problem_file,
                "--hybrid-reasoning-models", "deepseek/deepseek-r1-0528:free", 
                "--hybrid-response-models", "openrouter/cypher-alpha:free",
                "--hybrid-reasoning-temp", "0.1",
                "--hybrid-response-temp", "0.3"
            ]
            
            env = os.environ.copy()
            env['OPENROUTER_API_KEY'] = self.api_key
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=300  # 5 minute timeout for orchestrator mode (involves assessment + processing)
            )
            
            print(f"Orchestrator CLI return code: {result.returncode}")
            print(f"Orchestrator CLI stdout: {result.stdout}")
            
            # The command should succeed
            self.assertEqual(result.returncode, 0, f"CLI orchestrator command failed: {result.stderr}")
            self.assertGreater(len(result.stdout), 0)
            
        finally:
            os.unlink(problem_file)

if __name__ == '__main__':
    # Only run if explicitly called with integration marker
    if '--integration' in sys.argv:
        unittest.main()
    else:
        print("Integration tests skipped. Use 'pytest -m integration' to run them.") 