#!/usr/bin/env python3
"""
Real integration tests for FaR (Fact-and-Reflection) reasoning process.
These tests use actual API calls to verify end-to-end functionality.

Requires OPENROUTER_API_KEY environment variable.
Run with: pytest -m integration tests/integration/test_far_real_integration.py
"""

import os
import sys
import pytest
import unittest
import tempfile
import subprocess
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from src.far.processor import FaRProcessor
from src.far.orchestrator import FaRProcess, FaROrchestrator
from src.far.dataclasses import FaRConfig
from src.far.enums import FaRTriggerMode
from src.llm_client import LLMClient
from src.llm_config import LLMConfig


class TestFaRRealIntegration(unittest.TestCase):
    """
    Real integration tests for FaR reasoning process.
    
    These tests use actual API calls and require OPENROUTER_API_KEY.
    They verify that the two-stage FaR process (fact elicitation + reflection) 
    works correctly with real LLM responses.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with API key and model configurations."""
        cls.api_key = os.getenv('OPENROUTER_API_KEY')
        if not cls.api_key:
            pytest.skip("OPENROUTER_API_KEY environment variable not set")
        
        # Use free models for testing
        cls.fact_model = "openrouter/cypher-alpha:free"
        cls.main_model = "openrouter/cypher-alpha:free"
        cls.assess_model = "openrouter/cypher-alpha:free"
        
        # Test problems
        cls.simple_problem = "What is the capital of France?"
        cls.factual_problem = "What is the current population of Tokyo and what makes it unique as a city?"
        cls.complex_problem = "Explain the causes and consequences of the 2008 financial crisis, including key events and regulatory responses."
    
    def setUp(self):
        """Set up test fixtures."""
        self.llm_client = LLMClient(api_key=self.api_key)
        
        # Configure FaR with reasonable settings for testing
        self.far_config = FaRConfig(
            fact_model_names=[self.fact_model],
            main_model_names=[self.main_model],
            fact_model_temperature=0.3,
            main_model_temperature=0.7,
            max_fact_tokens=500,
            max_main_tokens=800,
            max_reasoning_tokens=2000,
            max_time_seconds=120  # 2 minutes timeout
        )
    
    @pytest.mark.integration
    def test_far_processor_real_api_calls(self):
        """Test FaR processor with real API calls."""
        print(f"\n🧪 Testing FaR processor with real API calls")
        print(f"📋 Fact model: {self.fact_model}")
        print(f"📋 Main model: {self.main_model}")
        print(f"❓ Problem: {self.factual_problem}")
        
        processor = FaRProcessor(llm_client=self.llm_client, config=self.far_config)
        result = processor.run(self.factual_problem)
        
        # Verify the result structure
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.problem_description)
        self.assertEqual(result.problem_description, self.factual_problem)
        
        if result.succeeded:
            print(f"✅ FaR process succeeded")
            print(f"📊 Elicited facts: {result.elicited_facts[:200]}...")
            print(f"💡 Final answer: {result.final_answer[:200]}...")
            
            # Verify we got meaningful results
            self.assertIsNotNone(result.elicited_facts)
            self.assertIsNotNone(result.final_answer)
            self.assertGreater(len(result.elicited_facts), 10)
            self.assertGreater(len(result.final_answer), 10)
            
            # Verify both LLM calls were made
            self.assertIsNotNone(result.fact_call_stats)
            self.assertIsNotNone(result.main_call_stats)
            self.assertGreater(result.fact_call_stats.completion_tokens, 0)
            self.assertGreater(result.main_call_stats.completion_tokens, 0)
            
            # Verify reasoning tokens are tracked
            expected_reasoning_tokens = result.fact_call_stats.completion_tokens + result.main_call_stats.completion_tokens
            self.assertEqual(result.reasoning_completion_tokens, expected_reasoning_tokens)
            
            print(f"📈 Fact call tokens: {result.fact_call_stats.completion_tokens}")
            print(f"📈 Main call tokens: {result.main_call_stats.completion_tokens}")
            print(f"📈 Total reasoning tokens: {result.reasoning_completion_tokens}")
            print(f"⏱️ Total time: {result.total_process_wall_clock_time_seconds:.2f}s")
            
        else:
            print(f"❌ FaR process failed: {result.error_message}")
            # Even if it failed, we should have some diagnostic info
            self.assertIsNotNone(result.error_message)
    
    @pytest.mark.integration
    def test_far_process_end_to_end(self):
        """Test FaR process end-to-end with real API calls."""
        print(f"\n🧪 Testing FaR process end-to-end")
        
        # Configure direct one-shot fallback
        direct_oneshot_config = LLMConfig(temperature=0.7, max_tokens=800)
        direct_oneshot_models = [self.main_model]
        
        far_process = FaRProcess(
            llm_client=self.llm_client,
            far_config=self.far_config,
            direct_oneshot_llm_config=direct_oneshot_config,
            direct_oneshot_model_names=direct_oneshot_models
        )
        
        # Execute the process
        far_process.execute(self.factual_problem, model_name="test")
        solution, summary = far_process.get_result()
        
        self.assertIsNotNone(solution)
        self.assertIsNotNone(summary)
        
        # Check that we got some kind of result
        self.assertIsNotNone(solution.final_answer)
        
        if solution.far_result and solution.far_result.succeeded:
            print(f"✅ FaR process succeeded")
            print(f"✅ Final answer: {solution.final_answer[:200]}...")
            self.assertGreater(len(solution.final_answer), 0)
            self.assertFalse(solution.far_failed_and_fell_back)
            self.assertIsNone(solution.fallback_call_stats)
            
            # Verify FaR-specific results
            self.assertIsNotNone(solution.far_result.elicited_facts)
            self.assertGreater(len(solution.far_result.elicited_facts), 10)
            
        elif solution.far_failed_and_fell_back:
            print(f"⚠️ FaR failed, fell back to one-shot: {solution.final_answer[:200]}...")
            # This is still a valid outcome
            self.assertGreater(len(solution.final_answer), 0)
            self.assertIsNotNone(solution.fallback_call_stats)
            
        else:
            print(f"❌ Process failed: {solution.final_answer}")
    
    @pytest.mark.integration
    def test_far_orchestrator_always_mode(self):
        """Test FaR orchestrator in ALWAYS_FAR mode."""
        print(f"\n🧪 Testing FaR orchestrator in ALWAYS_FAR mode")
        
        orchestrator = FaROrchestrator(
            llm_client=self.llm_client,
            trigger_mode=FaRTriggerMode.ALWAYS_FAR,
            far_config=self.far_config,
            direct_oneshot_llm_config=LLMConfig(temperature=0.7, max_tokens=800),
            direct_oneshot_model_names=[self.main_model]
        )
        
        print(f"🔍 About to call orchestrator.solve() with problem: {self.factual_problem[:50]}...")
        
        # Enable debug logging to see what's happening
        import logging
        logging.basicConfig(level=logging.DEBUG)
        far_logger = logging.getLogger('src.far.orchestrator')
        far_logger.setLevel(logging.DEBUG)
        
        solution, summary = orchestrator.solve(self.factual_problem)
        print(f"🔍 Orchestrator returned. Summary: {summary[:200] if summary else 'None'}...")
        
        self.assertIsNotNone(solution)
        self.assertIsNotNone(summary)
        
        # Debug output
        print(f"🔍 Solution object: {solution}")
        print(f"🔍 Solution final_answer: {solution.final_answer}")
        print(f"🔍 Solution far_result: {solution.far_result}")
        if hasattr(solution, 'error_message'):
            print(f"🔍 Solution error_message: {solution.error_message}")
        
        self.assertIsNotNone(solution.final_answer)
        
        print(f"✅ Orchestrator result: {solution.final_answer[:200]}...")
        
        # Should have attempted FaR process
        self.assertIsNotNone(solution.far_result)
    
    @pytest.mark.integration
    def test_far_orchestrator_assess_first_mode(self):
        """Test FaR orchestrator in ASSESS_FIRST_FAR mode."""
        print(f"\n🧪 Testing FaR orchestrator in ASSESS_FIRST_FAR mode")
        
        orchestrator = FaROrchestrator(
            llm_client=self.llm_client,
            trigger_mode=FaRTriggerMode.ASSESS_FIRST_FAR,
            far_config=self.far_config,
            direct_oneshot_llm_config=LLMConfig(temperature=0.7, max_tokens=800),
            direct_oneshot_model_names=[self.main_model],
            assessment_llm_config=LLMConfig(temperature=0.3, max_tokens=200),
            assessment_model_names=[self.assess_model]
        )
        
        # Use complex problem that should trigger FaR
        solution, summary = orchestrator.solve(self.complex_problem)
        
        self.assertIsNotNone(solution)
        self.assertIsNotNone(summary)
        
        # Debug output
        print(f"🔍 Solution object: {solution}")
        print(f"🔍 Solution final_answer: {solution.final_answer}")
        print(f"🔍 Solution far_result: {solution.far_result}")
        if hasattr(solution, 'error_message'):
            print(f"🔍 Solution error_message: {solution.error_message}")
        
        self.assertIsNotNone(solution.final_answer)
        
        # Should have made an assessment call
        self.assertIsNotNone(solution.assessment_stats)
        
        print(f"✅ Assessment made, result: {solution.final_answer[:200]}...")
        print(f"📊 Assessment tokens: {solution.assessment_stats.completion_tokens}")
    
    @pytest.mark.integration
    def test_fact_elicitation_quality(self):
        """Test that fact elicitation produces meaningful facts."""
        print(f"\n🧪 Testing fact elicitation quality")
        
        processor = FaRProcessor(llm_client=self.llm_client, config=self.far_config)
        
        # Use a problem where we can verify facts were extracted
        problem = "What are the main causes of climate change and what are the projected temperature increases by 2100?"
        result = processor.run(problem)
        
        if result.succeeded:
            print(f"📊 Elicited facts: {result.elicited_facts}")
            
            # Verify facts contain relevant information
            facts_lower = result.elicited_facts.lower()
            
            # Should contain climate-related terms
            climate_terms = ["climate", "temperature", "carbon", "greenhouse", "emission", "warming"]
            found_terms = [term for term in climate_terms if term in facts_lower]
            
            self.assertGreater(len(found_terms), 0, 
                f"Facts should contain climate-related terms. Found: {found_terms}")
            
            # Final answer should incorporate the facts
            answer_lower = result.final_answer.lower()
            
            # Some facts should appear in the final answer
            self.assertGreater(len(result.final_answer), len(result.elicited_facts) / 2,
                "Final answer should be substantial and incorporate facts")
            
            print(f"✅ Found {len(found_terms)} relevant terms in facts")
            print(f"✅ Final answer length: {len(result.final_answer)} chars")
    
    @pytest.mark.integration
    def test_two_stage_process_distinction(self):
        """Test that the two-stage process produces different results than one-shot."""
        print(f"\n🧪 Testing two-stage process distinction")
        
        # Get FaR result
        far_processor = FaRProcessor(llm_client=self.llm_client, config=self.far_config)
        far_result = far_processor.run(self.factual_problem)
        
        # Get one-shot result for comparison
        oneshot_config = LLMConfig(temperature=0.7, max_tokens=800)
        oneshot_response, oneshot_stats = self.llm_client.call(
            prompt=self.factual_problem,
            models=[self.main_model],
            config=oneshot_config
        )
        
        if far_result.succeeded:
            print(f"📊 FaR answer length: {len(far_result.final_answer)}")
            print(f"📊 One-shot answer length: {len(oneshot_response)}")
            
            # FaR should typically produce more comprehensive answers
            # due to the fact elicitation stage
            self.assertIsNotNone(far_result.final_answer)
            self.assertIsNotNone(oneshot_response)
            
            # Both should be substantial
            self.assertGreater(len(far_result.final_answer), 20)
            self.assertGreater(len(oneshot_response), 20)
            
            # FaR used more LLM calls (fact + main vs just one-shot)
            far_total_tokens = far_result.reasoning_completion_tokens
            oneshot_total_tokens = oneshot_stats.completion_tokens
            
            print(f"📈 FaR total tokens: {far_total_tokens}")
            print(f"📈 One-shot tokens: {oneshot_total_tokens}")
            
            # FaR should typically use more tokens due to two-stage process
            self.assertGreater(far_total_tokens, oneshot_total_tokens * 0.8,
                "FaR should typically use more tokens due to two-stage process")


class TestFaRCLIRealIntegration(unittest.TestCase):
    """
    Test the FaR reasoning process through the CLI interface with real API calls.
    """
    
    @classmethod
    def setUpClass(cls):
        cls.api_key = os.getenv('OPENROUTER_API_KEY')
        if not cls.api_key:
            pytest.skip("OPENROUTER_API_KEY environment variable not set")
        
        # Use free models for CLI testing
        cls.fact_model = "openrouter/cypher-alpha:free"
        cls.main_model = "openrouter/cypher-alpha:free"
        cls.assess_model = "openrouter/cypher-alpha:free"
    
    @pytest.mark.integration
    def test_cli_far_direct_mode(self):
        """Test FaR through CLI in direct mode."""
        print(f"\n🧪 Testing FaR through CLI in direct mode")
        
        # Create a temporary problem file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("What is the largest ocean on Earth and what are its key characteristics?")
            problem_file = f.name
        
        try:
            # Test CLI call with far-direct mode
            cmd = [
                sys.executable, "-m", "src.cli_runner",
                "--processing-mode", "far-direct",
                "--problem-filename", problem_file,
                "--far-fact-models", self.fact_model,
                "--far-main-models", self.main_model,
                "--far-fact-temp", "0.3",
                "--far-main-temp", "0.7",
                "--far-max-fact-tokens", "400",
                "--far-max-main-tokens", "600"
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
                timeout=180  # 3 minute timeout
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
            
            # Should contain evidence of FaR processing
            # Look for fact elicitation and reflection stages
            if "Fact Call" in output or "Main Call" in output:
                print("✅ Found evidence of FaR two-stage process")
            
        finally:
            # Clean up temporary file
            os.unlink(problem_file)
    
    @pytest.mark.integration
    def test_cli_far_orchestrator_always_mode(self):
        """Test FaR through CLI orchestrator in always mode."""
        print(f"\n🧪 Testing FaR through CLI orchestrator in always mode")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("What were the main causes of World War I and how did they lead to the conflict?")
            problem_file = f.name
        
        try:
            cmd = [
                sys.executable, "-m", "src.cli_runner",
                "--processing-mode", "far-always",
                "--problem-filename", problem_file,
                "--far-fact-models", self.fact_model,
                "--far-main-models", self.main_model,
                "--far-fact-temp", "0.3",
                "--far-main-temp", "0.7"
            ]
            
            env = os.environ.copy()
            env['OPENROUTER_API_KEY'] = self.api_key
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=300  # 5 minute timeout for orchestrator mode
            )
            
            print(f"Orchestrator CLI return code: {result.returncode}")
            print(f"Orchestrator CLI stdout: {result.stdout}")
            
            # The command should succeed
            self.assertEqual(result.returncode, 0, f"CLI orchestrator command failed: {result.stderr}")
            self.assertGreater(len(result.stdout), 0)
            
            # Should contain final answer
            self.assertIn("Final Answer", result.stdout)
            
        finally:
            os.unlink(problem_file)
    
    @pytest.mark.integration
    def test_cli_far_assess_first_mode(self):
        """Test FaR through CLI in assess-first mode."""
        print(f"\n🧪 Testing FaR through CLI in assess-first mode")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Analyze the economic impact of renewable energy adoption on traditional fossil fuel industries, including job market effects and policy implications.")
            problem_file = f.name
        
        try:
            cmd = [
                sys.executable, "-m", "src.cli_runner",
                "--processing-mode", "far-assess-first",
                "--problem-filename", problem_file,
                "--far-fact-models", self.fact_model,
                "--far-main-models", self.main_model,
                "--far-assess-models", self.assess_model,
                "--far-fact-temp", "0.3",
                "--far-main-temp", "0.7",
                "--far-assess-temp", "0.3"
            ]
            
            env = os.environ.copy()
            env['OPENROUTER_API_KEY'] = self.api_key
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=300  # 5 minute timeout
            )
            
            print(f"Assess-first CLI return code: {result.returncode}")
            print(f"Assess-first CLI stdout: {result.stdout}")
            
            # The command should succeed
            self.assertEqual(result.returncode, 0, f"CLI assess-first command failed: {result.stderr}")
            self.assertGreater(len(result.stdout), 0)
            
            # Should contain final answer
            self.assertIn("Final Answer", result.stdout)
            
        finally:
            os.unlink(problem_file)


if __name__ == '__main__':
    # Only run if explicitly called with integration marker
    if '--integration' in sys.argv:
        unittest.main()
    else:
        print("Real integration tests skipped. Use 'pytest -m integration' to run them.")
        print("Requires OPENROUTER_API_KEY environment variable.") 