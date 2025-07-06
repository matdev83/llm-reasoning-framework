import unittest
from unittest.mock import MagicMock, patch, ANY
import logging

# Ensure src path is available
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.hybrid.orchestrator import HybridOrchestrator, HybridProcess
from src.hybrid.dataclasses import HybridConfig, HybridSolution, LLMCallStats, HybridResult
from src.hybrid.enums import HybridTriggerMode
from src.llm_client import LLMClient
from src.complexity_assessor import ComplexityAssessor # Corrected: AssessmentDecision is part of ComplexityAssessor module or imported there
from src.aot.enums import AssessmentDecision # Actual import for AssessmentDecision
from src.heuristic_detector import HeuristicDetector # Corrected: HeuristicDetector is imported from src.heuristic_detector
from src.llm_config import LLMConfig

# Suppress most logging output during tests
logging.basicConfig(level=logging.CRITICAL)

class TestHybridOrchestrator(unittest.TestCase):

    def setUp(self):
        self.api_key = "test_api_key"
        self.problem_text = "A test problem for the orchestrator."

        self.default_hybrid_config = HybridConfig(
            reasoning_model_name="reasoning/dummy",
            response_model_name="response/dummy"
        )
        self.direct_oneshot_models = ["direct/fallback"]
        self.direct_oneshot_temp = 0.6

        self.assessment_models = ["assess/dummy"]
        self.assessment_temp = 0.3

        # Mock for HybridProcess instance that would be created by the orchestrator
        self.mock_hybrid_process_instance = MagicMock(spec=HybridProcess)

        # Mock for LLMClient instance created by orchestrator (for its own one-shots or assessor)
        self.mock_orchestrator_llm_client = MagicMock(spec=LLMClient)

        # Mock for ComplexityAssessor instance
        self.mock_complexity_assessor_instance = MagicMock(spec=ComplexityAssessor)

        # Mock for HeuristicDetector
        self.mock_heuristic_detector_instance = MagicMock(spec=HeuristicDetector)


    @patch('src.hybrid.orchestrator.HybridProcess') # Patches HybridProcess where orchestrator instantiates it
    @patch('src.hybrid.orchestrator.LLMClient')     # Patches LLMClient where orchestrator instantiates its own
    @patch('src.hybrid.orchestrator.ComplexityAssessor') # Patches ComplexityAssessor
    @patch('src.hybrid.orchestrator.HeuristicDetector') # Patches HeuristicDetector
    def test_solve_always_hybrid_mode(self, MockHeuristicDetector, MockComplexityAssessor, MockLLMClient, MockHybridProcess):
        MockHybridProcess.return_value = self.mock_hybrid_process_instance
        MockLLMClient.return_value = self.mock_orchestrator_llm_client
        # HeuristicDetector and ComplexityAssessor are not used in ALWAYS_HYBRID

        mock_hybrid_solution = HybridSolution(
            final_answer="Always hybrid answer",
            hybrid_result=HybridResult(succeeded=True)
        )
        mock_summary = "Summary from HybridProcess"
        self.mock_hybrid_process_instance.get_result.return_value = (mock_hybrid_solution, mock_summary)

        orchestrator = HybridOrchestrator(
            trigger_mode=HybridTriggerMode.ALWAYS_HYBRID,
            hybrid_config=self.default_hybrid_config,
            direct_oneshot_model_names=self.direct_oneshot_models,
            direct_oneshot_temperature=self.direct_oneshot_temp,
            api_key=self.api_key
            # assessment params not needed for this mode
        )

        solution, summary_str = orchestrator.solve(self.problem_text)

        self.mock_hybrid_process_instance.execute.assert_called_once_with(
            problem_description=self.problem_text, model_name="default_hybrid_model"
        )
        self.assertEqual(solution.final_answer, "Always hybrid answer")
        self.assertIn(mock_summary, summary_str) # Orchestrator summary should include process summary
        MockLLMClient.assert_called_once()
        # Orchestrator creates one LLMClient for itself.
        # HybridProcess (if instantiated and not fully mocked out at class level) would create another.
        # Since MockHybridProcess is a mock of the class, its __init__ (which creates an LLMClient) isn't called.
        # The LLMClient mock here is for the orchestrator's own client.

    @patch('src.hybrid.orchestrator.HybridProcess')
    @patch('src.hybrid.orchestrator.LLMClient')
    @patch('src.hybrid.orchestrator.ComplexityAssessor')
    @patch('src.hybrid.orchestrator.HeuristicDetector')
    def test_solve_never_hybrid_mode(self, MockHeuristicDetector, MockComplexityAssessor, MockLLMClient, MockHybridProcess):
        MockLLMClient.return_value = self.mock_orchestrator_llm_client
        # HybridProcess should not be instantiated or used
        # HeuristicDetector and ComplexityAssessor are not used in NEVER_HYBRID

        oneshot_answer = "Never hybrid, direct oneshot answer."
        mock_oneshot_stats = LLMCallStats(model_name="direct/fallback", completion_tokens=15, prompt_tokens=7, call_duration_seconds=0.3)
        self.mock_orchestrator_llm_client.call.return_value = (oneshot_answer, mock_oneshot_stats)

        orchestrator = HybridOrchestrator(
            trigger_mode=HybridTriggerMode.NEVER_HYBRID,
            hybrid_config=self.default_hybrid_config,
            direct_oneshot_model_names=self.direct_oneshot_models,
            direct_oneshot_temperature=self.direct_oneshot_temp,
            api_key=self.api_key
        )

        solution, summary_str = orchestrator.solve(self.problem_text)

        MockHybridProcess.assert_not_called()
        self.mock_hybrid_process_instance.execute.assert_not_called()

        # Check the new LLMConfig structure
        calls = self.mock_orchestrator_llm_client.call.call_args_list
        self.assertEqual(len(calls), 1)
        call_args = calls[0]
        self.assertEqual(call_args[1]['prompt'], self.problem_text)
        self.assertEqual(call_args[1]['models'], self.direct_oneshot_models)
        config = call_args[1]['config']
        self.assertIsInstance(config, LLMConfig)
        self.assertEqual(config.temperature, self.direct_oneshot_temp)

        self.assertEqual(solution.final_answer, oneshot_answer)
        self.assertEqual(solution.main_call_stats, mock_oneshot_stats)
        self.assertIsNone(solution.hybrid_result)

    @patch('src.hybrid.orchestrator.HybridProcess')
    @patch('src.hybrid.orchestrator.LLMClient')
    @patch('src.hybrid.orchestrator.ComplexityAssessor')
    @patch('src.hybrid.orchestrator.HeuristicDetector')
    def test_solve_assess_first_chooses_hybrid(self, MockHeuristicDetector, MockComplexityAssessor, MockLLMClient, MockHybridProcess):
        MockHybridProcess.return_value = self.mock_hybrid_process_instance
        MockLLMClient.return_value = self.mock_orchestrator_llm_client
        MockComplexityAssessor.return_value = self.mock_complexity_assessor_instance
        MockHeuristicDetector.return_value = self.mock_heuristic_detector_instance

        mock_assessment_stats = LLMCallStats(model_name="assess/dummy", completion_tokens=3, prompt_tokens=2, call_duration_seconds=0.1)
        self.mock_complexity_assessor_instance.assess.return_value = (AssessmentDecision.ADVANCED_REASONING, mock_assessment_stats)

        mock_hybrid_solution = HybridSolution(final_answer="Assessed then Hybrid answer", hybrid_result=HybridResult(succeeded=True))
        mock_process_summary = "Hybrid process summary after assessment"
        self.mock_hybrid_process_instance.get_result.return_value = (mock_hybrid_solution, mock_process_summary)

        orchestrator = HybridOrchestrator(
            trigger_mode=HybridTriggerMode.ASSESS_FIRST_HYBRID,
            hybrid_config=self.default_hybrid_config,
            direct_oneshot_model_names=self.direct_oneshot_models,
            direct_oneshot_temperature=self.direct_oneshot_temp,
            assessment_model_names=self.assessment_models,
            assessment_temperature=self.assessment_temp,
            api_key=self.api_key,
            use_heuristic_shortcut=False
        )

        solution, summary_str = orchestrator.solve(self.problem_text)

        self.mock_complexity_assessor_instance.assess.assert_called_once_with(self.problem_text)
        self.mock_hybrid_process_instance.execute.assert_called_once_with(problem_description=self.problem_text, model_name="default_hybrid_model")
        self.mock_orchestrator_llm_client.call.assert_not_called()

        self.assertEqual(solution.final_answer, "Assessed then Hybrid answer")
        self.assertEqual(solution.assessment_stats, mock_assessment_stats)
        self.assertIn("Assessment Phase", summary_str)
        self.assertIn("Decision='ADVANCED_REASONING'", summary_str)
        self.assertIn(mock_process_summary, summary_str)


    @patch('src.hybrid.orchestrator.HybridProcess')
    @patch('src.hybrid.orchestrator.LLMClient')
    @patch('src.hybrid.orchestrator.ComplexityAssessor')
    @patch('src.hybrid.orchestrator.HeuristicDetector')
    def test_solve_assess_first_chooses_oneshot(self, MockHeuristicDetector, MockComplexityAssessor, MockLLMClient, MockHybridProcess):
        MockLLMClient.return_value = self.mock_orchestrator_llm_client
        MockComplexityAssessor.return_value = self.mock_complexity_assessor_instance
        MockHeuristicDetector.return_value = self.mock_heuristic_detector_instance
        MockHybridProcess.return_value = self.mock_hybrid_process_instance

        mock_assessment_stats = LLMCallStats(model_name="assess/dummy", completion_tokens=3, prompt_tokens=2, call_duration_seconds=0.1)
        self.mock_complexity_assessor_instance.assess.return_value = (AssessmentDecision.ONE_SHOT, mock_assessment_stats)

        oneshot_answer = "Assessed then OneShot answer."
        mock_oneshot_stats = LLMCallStats(model_name="direct/fallback", completion_tokens=10, prompt_tokens=8, call_duration_seconds=0.2)
        self.mock_orchestrator_llm_client.call.return_value = (oneshot_answer, mock_oneshot_stats)


        orchestrator = HybridOrchestrator(
            trigger_mode=HybridTriggerMode.ASSESS_FIRST_HYBRID,
            hybrid_config=self.default_hybrid_config,
            direct_oneshot_model_names=self.direct_oneshot_models,
            direct_oneshot_temperature=self.direct_oneshot_temp,
            assessment_model_names=self.assessment_models,
            assessment_temperature=self.assessment_temp,
            api_key=self.api_key,
            use_heuristic_shortcut=True
        )

        solution, summary_str = orchestrator.solve(self.problem_text)

        self.mock_complexity_assessor_instance.assess.assert_called_once_with(self.problem_text)
        self.mock_hybrid_process_instance.execute.assert_not_called()

        # Check the new LLMConfig structure
        calls = self.mock_orchestrator_llm_client.call.call_args_list
        self.assertEqual(len(calls), 1)
        call_args = calls[0]
        self.assertEqual(call_args[1]['prompt'], self.problem_text)
        self.assertEqual(call_args[1]['models'], self.direct_oneshot_models)
        config = call_args[1]['config']
        self.assertIsInstance(config, LLMConfig)
        self.assertEqual(config.temperature, self.direct_oneshot_temp)

        self.assertEqual(solution.final_answer, oneshot_answer)
        self.assertEqual(solution.main_call_stats, mock_oneshot_stats)
        self.assertEqual(solution.assessment_stats, mock_assessment_stats)
        self.assertEqual(solution.assessment_decision, AssessmentDecision.ONE_SHOT)
        self.assertIn("Assessment Phase", summary_str)
        self.assertIn("Decision='ONE_SHOT'", summary_str)
        self.assertIn("Orchestrator Main Model Call (Direct ONESHOT path)", summary_str)

    @patch('src.hybrid.orchestrator.HybridProcess')
    @patch('src.hybrid.orchestrator.LLMClient')
    @patch('src.hybrid.orchestrator.ComplexityAssessor')
    @patch('src.hybrid.orchestrator.HeuristicDetector')
    def test_solve_assess_first_assessment_fails(self, MockHeuristicDetector, MockComplexityAssessor, MockLLMClient, MockHybridProcess):
        MockLLMClient.return_value = self.mock_orchestrator_llm_client
        MockComplexityAssessor.return_value = self.mock_complexity_assessor_instance
        MockHeuristicDetector.return_value = self.mock_heuristic_detector_instance
        MockHybridProcess.return_value = self.mock_hybrid_process_instance

        mock_assessment_error_stats = LLMCallStats(model_name="assess/dummy", completion_tokens=0, prompt_tokens=0, call_duration_seconds=0.05)
        self.mock_complexity_assessor_instance.assess.return_value = (AssessmentDecision.ERROR, mock_assessment_error_stats)

        fallback_oneshot_answer = "Assessment failed, fallback oneshot."
        mock_fallback_stats = LLMCallStats(model_name="direct/fallback", completion_tokens=12, prompt_tokens=6, call_duration_seconds=0.25)
        self.mock_orchestrator_llm_client.call.return_value = (fallback_oneshot_answer, mock_fallback_stats)

        orchestrator = HybridOrchestrator(
            trigger_mode=HybridTriggerMode.ASSESS_FIRST_HYBRID,
            hybrid_config=self.default_hybrid_config,
            direct_oneshot_model_names=self.direct_oneshot_models,
            direct_oneshot_temperature=self.direct_oneshot_temp,
            assessment_model_names=self.assessment_models,
            assessment_temperature=self.assessment_temp,
            api_key=self.api_key
        )

        solution, summary_str = orchestrator.solve(self.problem_text)

        self.mock_complexity_assessor_instance.assess.assert_called_once_with(self.problem_text)
        self.mock_hybrid_process_instance.execute.assert_not_called()

        # Check the new LLMConfig structure
        calls = self.mock_orchestrator_llm_client.call.call_args_list
        self.assertEqual(len(calls), 1)
        call_args = calls[0]
        self.assertEqual(call_args[1]['prompt'], self.problem_text)
        self.assertEqual(call_args[1]['models'], self.direct_oneshot_models)
        config = call_args[1]['config']
        self.assertIsInstance(config, LLMConfig)
        self.assertEqual(config.temperature, self.direct_oneshot_temp)

        self.assertEqual(solution.final_answer, fallback_oneshot_answer)
        self.assertEqual(solution.fallback_call_stats, mock_fallback_stats)
        self.assertEqual(solution.assessment_stats, mock_assessment_error_stats)
        self.assertEqual(solution.assessment_decision, AssessmentDecision.ERROR)
        self.assertIn("Assessment Phase", summary_str)
        self.assertIn("Decision='ERROR'", summary_str) # Relies on HybridOrchestrator correctly formatting this
        self.assertIn("Orchestrator Fallback One-Shot Call", summary_str)


if __name__ == '__main__':
    unittest.main()
