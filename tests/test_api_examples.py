import unittest
from unittest.mock import Mock, patch

class TestAPIExamples(unittest.TestCase):
    """Test that API examples from README work correctly"""
    
    def test_hybrid_imports(self):
        """Test that Hybrid API imports work"""
        try:
            from src.hybrid.orchestrator import HybridOrchestrator, HybridProcess
            from src.hybrid.dataclasses import HybridConfig
            from src.hybrid.enums import HybridTriggerMode
        except ImportError as e:
            self.fail(f"Failed to import Hybrid classes: {e}")
    
    def test_hybrid_config_creation(self):
        """Test that HybridConfig can be created with documented parameters"""
        from src.hybrid.dataclasses import HybridConfig
        
        try:
            config = HybridConfig(
                reasoning_model_name="google/gemini-pro",
                reasoning_model_temperature=0.1,
                response_model_name="anthropic/claude-3-sonnet-20240229",
                response_model_temperature=0.7,
                max_reasoning_tokens=2000,
                max_response_tokens=2000
            )
            self.assertIsNotNone(config)
            self.assertEqual(config.reasoning_model_name, "google/gemini-pro")
            self.assertEqual(config.response_model_name, "anthropic/claude-3-sonnet-20240229")
        except Exception as e:
            self.fail(f"Failed to create HybridConfig: {e}")
    
    def test_hybrid_process_creation(self):
        """Test that HybridProcess can be instantiated with documented parameters"""
        from src.hybrid.orchestrator import HybridProcess
        from src.hybrid.dataclasses import HybridConfig
        
        config = HybridConfig(
            reasoning_model_name="google/gemini-pro",
            reasoning_model_temperature=0.1,
            response_model_name="anthropic/claude-3-sonnet-20240229",
            response_model_temperature=0.7,
            max_reasoning_tokens=2000,
            max_response_tokens=2000
        )
        
        try:
            process = HybridProcess(
                hybrid_config=config,
                direct_oneshot_model_names=["anthropic/claude-3-sonnet-20240229"],
                direct_oneshot_temperature=0.7,
                api_key="test_key",
                enable_rate_limiting=True,
                enable_audit_logging=True
            )
            self.assertIsNotNone(process)
        except Exception as e:
            self.fail(f"Failed to create HybridProcess: {e}")
    
    def test_hybrid_orchestrator_creation(self):
        """Test that HybridOrchestrator can be instantiated with documented parameters"""
        from src.hybrid.orchestrator import HybridOrchestrator
        from src.hybrid.dataclasses import HybridConfig
        from src.hybrid.enums import HybridTriggerMode
        
        config = HybridConfig(
            reasoning_model_name="google/gemini-pro",
            reasoning_model_temperature=0.1,
            response_model_name="anthropic/claude-3-sonnet-20240229",
            response_model_temperature=0.7,
            max_reasoning_tokens=2000,
            max_response_tokens=2000
        )
        
        try:
            orchestrator = HybridOrchestrator(
                trigger_mode=HybridTriggerMode.ALWAYS_HYBRID,
                hybrid_config=config,
                direct_oneshot_model_names=["anthropic/claude-3-sonnet-20240229"],
                direct_oneshot_temperature=0.7,
                api_key="test_key"
            )
            self.assertIsNotNone(orchestrator)
        except Exception as e:
            self.fail(f"Failed to create HybridOrchestrator: {e}")
    
    def test_hybrid_process_interface(self):
        """Test that HybridProcess has the documented interface"""
        from src.hybrid.orchestrator import HybridProcess
        from src.hybrid.dataclasses import HybridConfig
        
        config = HybridConfig(
            reasoning_model_name="test/reasoning",
            response_model_name="test/response"
        )
        process = HybridProcess(
            hybrid_config=config,
            direct_oneshot_model_names=["test/model"],
            direct_oneshot_temperature=0.7,
            api_key="test_key"
        )
        
        # Check that the documented methods exist
        self.assertTrue(hasattr(process, 'execute'))
        self.assertTrue(hasattr(process, 'get_result'))
        
        # Check execute method signature
        import inspect
        sig = inspect.signature(process.execute)
        self.assertIn('problem_description', sig.parameters)
        self.assertIn('model_name', sig.parameters)
    
    def test_hybrid_orchestrator_interface(self):
        """Test that HybridOrchestrator has the documented interface"""
        from src.hybrid.orchestrator import HybridOrchestrator
        from src.hybrid.dataclasses import HybridConfig
        from src.hybrid.enums import HybridTriggerMode
        
        config = HybridConfig(
            reasoning_model_name="test/reasoning",
            response_model_name="test/response"
        )
        orchestrator = HybridOrchestrator(
            trigger_mode=HybridTriggerMode.ALWAYS_HYBRID,
            hybrid_config=config,
            direct_oneshot_model_names=["test/model"],
            direct_oneshot_temperature=0.7,
            api_key="test_key"
        )
        
        # Check that the documented methods exist
        self.assertTrue(hasattr(orchestrator, 'solve'))
        
        # Check solve method signature
        import inspect
        sig = inspect.signature(orchestrator.solve)
        self.assertIn('problem_text', sig.parameters)
    
    def test_aot_imports(self):
        """Test that AoT API imports work"""
        try:
            from src.aot.orchestrator import InteractiveAoTOrchestrator
            from src.aot.dataclasses import AoTRunnerConfig
            from src.aot.enums import AotTriggerMode
        except ImportError as e:
            self.fail(f"Failed to import AoT classes: {e}")
    
    def test_l2t_imports(self):
        """Test that L2T API imports work"""
        try:
            from src.l2t.orchestrator import L2TOrchestrator
            from src.l2t.dataclasses import L2TConfig
            from src.l2t.enums import L2TTriggerMode
        except ImportError as e:
            self.fail(f"Failed to import L2T classes: {e}")
    
    def test_got_imports(self):
        """Test that GoT API imports work"""
        try:
            from src.got.orchestrator import GoTOrchestrator
            from src.got.dataclasses import GoTConfig
            from src.got.enums import GoTTriggerMode
        except ImportError as e:
            self.fail(f"Failed to import GoT classes: {e}")

if __name__ == '__main__':
    unittest.main() 