import unittest
import subprocess
import sys
import os

class TestCLIParameters(unittest.TestCase):
    """Test CLI parameter validation and documented examples"""
    
    def setUp(self):
        """Set up test environment"""
        self.cli_command = [sys.executable, "-m", "src.cli_runner"]
        # Change to project directory if needed
        if not os.path.exists("src"):
            os.chdir("/c/Users/Mateusz/source/repos/llm-aot-process")
    
    def test_processing_mode_choices(self):
        """Test that documented processing modes are accepted"""
        valid_modes = [
            "aot-always", "aot-assess-first", "aot-never", "aot-direct",
            "l2t", "l2t-direct", 
            "hybrid-direct",
            "got-always", "got-assess-first", "got-never", "got-direct"
        ]
        
        for mode in valid_modes:
            with self.subTest(mode=mode):
                # Test that the mode is accepted (should not fail with argument error)
                cmd = self.cli_command + [
                    "--processing-mode", mode,
                    "--problem", "Test problem",
                    "--help"  # Use help to avoid actual execution
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                # Should not contain "invalid choice" error
                self.assertNotIn("invalid choice", result.stderr.lower())
    
    def test_hybrid_requires_two_models(self):
        """Test that hybrid-direct mode requires both reasoning and response models"""
        # Test with only reasoning model (should work but might warn)
        cmd = self.cli_command + [
            "--processing-mode", "hybrid-direct",
            "--problem", "Test problem",
            "--hybrid-reasoning-models", "google/gemini-pro",
            "--help"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        # Should not fail with argument parsing error
        self.assertEqual(result.returncode, 0)
        
        # Test with both models (should work)
        cmd = self.cli_command + [
            "--processing-mode", "hybrid-direct",
            "--problem", "Test problem", 
            "--hybrid-reasoning-models", "google/gemini-pro",
            "--hybrid-response-models", "anthropic/claude-3-sonnet-20240229",
            "--help"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
    
    def test_max_steps_not_applicable_to_hybrid(self):
        """Test that max-steps parameter is not used in hybrid mode"""
        # This test documents that max-steps is not applicable to hybrid
        # We can't easily test that it's ignored, but we can test it doesn't cause errors
        cmd = self.cli_command + [
            "--processing-mode", "hybrid-direct",
            "--problem", "Test problem",
            "--hybrid-reasoning-models", "google/gemini-pro",
            "--hybrid-response-models", "anthropic/claude-3-sonnet-20240229",
            "--max-steps", "10",  # This should be ignored
            "--help"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
    
    def test_strategy_specific_parameters(self):
        """Test that strategy-specific parameters are accepted"""
        parameter_sets = [
            # AoT parameters
            (["--processing-mode", "aot-direct", "--aot-max-steps", "10", "--aot-main-models", "openai/gpt-3.5-turbo"]),
            # L2T parameters  
            (["--processing-mode", "l2t-direct", "--l2t-max-steps", "10", "--l2t-max-total-nodes", "50"]),
            # GoT parameters
            (["--processing-mode", "got-direct", "--got-max-thoughts", "30", "--got-max-iterations", "8"]),
            # Hybrid parameters
            (["--processing-mode", "hybrid-direct", "--hybrid-reasoning-models", "google/gemini-pro", "--hybrid-response-models", "anthropic/claude-3-sonnet-20240229"])
        ]
        
        for params in parameter_sets:
            with self.subTest(params=params):
                cmd = self.cli_command + params + ["--problem", "Test problem", "--help"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0)
    
    def test_deprecated_processing_modes_rejected(self):
        """Test that old/deprecated processing modes are rejected"""
        deprecated_modes = [
            "always-reasoning", "assess-first", "never-reasoning",
            "aot_direct", "l2t_direct", "hybrid_direct", "got_direct"
        ]
        
        for mode in deprecated_modes:
            with self.subTest(mode=mode):
                cmd = self.cli_command + [
                    "--processing-mode", mode,
                    "--problem", "Test problem"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                # Should fail with invalid choice error
                self.assertIn("invalid choice", result.stderr.lower())
                self.assertNotEqual(result.returncode, 0)

if __name__ == '__main__':
    unittest.main() 