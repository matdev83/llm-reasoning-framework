import unittest
from unittest.mock import patch, MagicMock

from src.aot_orchestrator import InteractiveAoTOrchestrator
from src.aot_dataclasses import AoTRunnerConfig, LLMCallStats, Solution
from src.aot_enums import AotTriggerMode, AssessmentDecision
from src.heuristic_detector import HeuristicDetector
from typing import Optional

# Suppress logging during tests
import logging
logging.disable(logging.CRITICAL)

# Mock for AoTProcessor
class MockAoTProcessor:
    def __init__(self, llm_client, config):
        self.llm_client = llm_client
        self.config = config
        self.run = MagicMock(return_value=(MagicMock(succeeded=True, final_answer="Mocked AoT Answer", reasoning_trace=[]), "Mocked AoT Summary"))

# Mock for ComplexityAssessor
class MockComplexityAssessor:
    def __init__(self, llm_client, small_model_names, temperature, use_heuristic_shortcut, heuristic_detector: Optional[HeuristicDetector] = None):
        self.llm_client = llm_client
        self.small_model_names = small_model_names
        self.temperature = temperature
        self.use_heuristic_shortcut = use_heuristic_shortcut
        self.heuristic_detector = heuristic_detector # Store the passed detector
        self.assess = MagicMock()

# Mock for HeuristicDetector
class MockHeuristicDetector(HeuristicDetector):
    def __init__(self):
        super().__init__()
        self.should_trigger_complex_process_heuristically = MagicMock(return_value=True)

class TestInteractiveAoTOrchestrator(unittest.TestCase):
    def setUp(self):
        self.aot_config = AoTRunnerConfig(
            main_model_names=["test-aot-model"],
            max_steps=3,
            max_time_seconds=60,
            max_reasoning_tokens=1000,
            no_progress_limit=2,
            temperature=0.7
        )
        self.api_key = "test_api_key_for_aot_orchestrator"
        self.direct_oneshot_model_names = ["test-oneshot-model"]
        self.direct_oneshot_temperature = 0.7
        self.assessment_model_names = ["test-assessment-model"]
        self.assessment_temperature = 0.0

    def _create_orchestrator(self, trigger_mode: AotTriggerMode, use_heuristic_shortcut: bool = True, heuristic_detector: Optional[HeuristicDetector] = None):
        return InteractiveAoTOrchestrator(
            trigger_mode=trigger_mode,
            aot_config=self.aot_config,
            direct_oneshot_model_names=self.direct_oneshot_model_names,
            direct_oneshot_temperature=self.direct_oneshot_temperature,
            assessment_model_names=self.assessment_model_names,
            assessment_temperature=self.assessment_temperature,
            api_key=self.api_key,
            use_heuristic_shortcut=use_heuristic_shortcut,
            heuristic_detector=heuristic_detector
        )

    @patch("src.aot_orchestrator.AoTProcessor")
    @patch("src.aot_orchestrator.ComplexityAssessor")
    @patch("src.aot_orchestrator.LLMClient")
    def test_solve_assess_first_with_custom_heuristic_detector(self, MockLLMClient, MockComplexityAssessorClass, MockAoTProcessorClass):
        problem_text = "Test problem with custom heuristic detector for AoT."
        
        custom_heuristic_detector = MockHeuristicDetector()
        custom_heuristic_detector.should_trigger_complex_process_heuristically.return_value = True

        mock_assessment_stats = LLMCallStats(model_name="assess-model", prompt_tokens=5, completion_tokens=1, call_duration_seconds=0.2)

        # Set the heuristic_detector attribute on the mocked ComplexityAssessor instance
        MockComplexityAssessorClass.return_value.heuristic_detector = custom_heuristic_detector

        # Define a side effect for the assess method of the mocked ComplexityAssessor instance
        def mock_assess_side_effect(problem_text: str):
            if MockComplexityAssessorClass.return_value.use_heuristic_shortcut:
                if MockComplexityAssessorClass.return_value.heuristic_detector.should_trigger_complex_process_heuristically(problem_text):
                    return AssessmentDecision.AOT, mock_assessment_stats
            return AssessmentDecision.ONESHOT, LLMCallStats(model_name="mock_llm_assess", prompt_tokens=1, completion_tokens=1, call_duration_seconds=0.1)

        MockComplexityAssessorClass.return_value.assess.side_effect = mock_assess_side_effect

        # Set the return value for the mocked AoTProcessor instance's run method
        MockAoTProcessorClass.return_value.run.return_value = (
            MagicMock(
                succeeded=True,
                final_answer="Mocked AoT Answer",
                reasoning_trace=[],
                total_llm_calls=1,
                total_completion_tokens=10,
                total_prompt_tokens=20,
                total_llm_interaction_time_seconds=0.1,
                total_process_wall_clock_time_seconds=0.2
            ),
            "Mocked AoT Summary"
        )

        # Create the orchestrator, passing the custom heuristic detector
        orchestrator = self._create_orchestrator(
            trigger_mode=AotTriggerMode.ASSESS_FIRST,
            use_heuristic_shortcut=True,
            heuristic_detector=custom_heuristic_detector
        )
        
        # Call solve
        solution, summary_str = orchestrator.solve(problem_text)

        # Assert that the custom heuristic detector's method was called
        custom_heuristic_detector.should_trigger_complex_process_heuristically.assert_called_once_with(problem_text)
        
        # Assert that ComplexityAssessor was initialized with the custom detector
        MockComplexityAssessorClass.assert_called_once_with(
            llm_client=orchestrator.llm_client,
            small_model_names=self.assessment_model_names,
            temperature=self.assessment_temperature,
            use_heuristic_shortcut=True,
            heuristic_detector=custom_heuristic_detector
        )

        # Further assertions to ensure the flow was as expected (e.g., AoT path taken)
        self.assertEqual(solution.aot_result.final_answer, "Mocked AoT Answer")
        self.assertIn("AoT Process Attempted: Yes", summary_str)
        self.assertIn("Heuristic Shortcut Enabled: True", summary_str)
