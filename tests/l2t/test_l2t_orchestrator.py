import unittest
from unittest.mock import patch, MagicMock, ANY

from src.l2t.orchestrator import L2TOrchestrator
from src.l2t.dataclasses import L2TConfig, L2TResult, L2TGraph, L2TSolution, L2TModelConfigs # Added L2TModelConfigs
from src.l2t.enums import L2TTriggerMode
from src.aot.dataclasses import LLMCallStats
from src.aot.enums import AssessmentDecision
from src.l2t_orchestrator_utils.oneshot_executor import OneShotExecutor 
from src.heuristic_detector import HeuristicDetector
from src.llm_config import LLMConfig # Added LLMConfig
from typing import Optional

import logging
logging.disable(logging.CRITICAL)

class MockHeuristicDetector(HeuristicDetector):
    def __init__(self):
        super().__init__()
        self.should_trigger_complex_process_heuristically = MagicMock(return_value=True)

class TestL2TOrchestrator(unittest.TestCase):
    def setUp(self):
        self.problem_text = "Test L2T problem for orchestrator."
        self.l2t_config = L2TConfig(
            max_steps=3,
            classification_model_names=["test-l2t-classifier"],
            thought_generation_model_names=["test-l2t-generator"],
            initial_prompt_model_names=["test-l2t-initial-prompt-model"], # Added for oneshot_executor
        )
        self.api_key = "test_api_key_for_l2t"
        
        # Define LLMConfig objects for various L2T stages
        self.l2t_model_configs = L2TModelConfigs(
            initial_thought_config=LLMConfig(temperature=0.7),
            node_classification_config=LLMConfig(temperature=0.1),
            node_thought_generation_config=LLMConfig(temperature=0.8),
            orchestrator_oneshot_config=LLMConfig(temperature=0.6), # For direct one-shot from orchestrator
            summary_config=LLMConfig(temperature=0.5)
        )
        # These are now derived from l2t_config and l2t_model_configs
        self.direct_oneshot_model_names = self.l2t_config.initial_prompt_model_names # For orchestrator's direct oneshot
        self.assessment_model_names = self.l2t_config.classification_model_names # For complexity assessor

        self.mock_l2t_process_solution = L2TSolution()
        self.mock_l2t_process_solution.final_answer = "Final Answer from Mocked L2TProcess"
        self.mock_l2t_process_solution.l2t_result = L2TResult(
            succeeded=True, 
            final_answer="Final Answer from Mocked L2TProcess",
            reasoning_graph=L2TGraph(),
            total_llm_calls=3, total_prompt_tokens=300, total_completion_tokens=250, 
            total_llm_interaction_time_seconds=3.5, total_process_wall_clock_time_seconds=4.0
        )
        self.mock_l2t_process_summary = "Summary from Mocked L2TProcess"

        self.mock_direct_oneshot_llm_response = "Direct one-shot answer from L2T orchestrator"
        self.mock_direct_oneshot_stats = LLMCallStats(
            model_name=self.direct_oneshot_model_names[0],
            completion_tokens=60, prompt_tokens=110, call_duration_seconds=1.1
        )

        self.patch_llm_client = patch("src.l2t.orchestrator.LLMClient")
        self.patch_complexity_assessor = patch("src.l2t.orchestrator.ComplexityAssessor") 
        self.patch_l2t_process = patch("src.l2t.orchestrator.L2TProcess")
        self.patch_oneshot_executor = patch("src.l2t.orchestrator.OneShotExecutor")

        self.MockLLMClient = self.patch_llm_client.start()
        self.MockComplexityAssessorClass = self.patch_complexity_assessor.start() 
        self.MockL2TProcess = self.patch_l2t_process.start()
        self.MockOneShotExecutor = self.patch_oneshot_executor.start()
        
        self.mock_assessor_instance = self.MockComplexityAssessorClass.return_value 
        self.mock_assessor_instance.assess = MagicMock()

        self.mock_l2t_process_instance = self.MockL2TProcess.return_value
        self.mock_l2t_process_instance.execute = MagicMock()
        self.mock_l2t_process_instance.get_result = MagicMock(
            return_value=(self.mock_l2t_process_solution, self.mock_l2t_process_summary)
        )
        
        self.mock_oneshot_executor_instance = self.MockOneShotExecutor.return_value
        self.mock_oneshot_executor_instance.run_direct_oneshot.return_value = (
            self.mock_direct_oneshot_llm_response, self.mock_direct_oneshot_stats
        )
        
    def tearDown(self):
        self.patch_llm_client.stop()
        self.patch_complexity_assessor.stop()
        self.patch_l2t_process.stop()
        self.patch_oneshot_executor.stop()

    def _create_orchestrator(self, trigger_mode: L2TTriggerMode, use_heuristic_shortcut: bool = True, heuristic_detector: Optional[HeuristicDetector] = None, enable_rate_limiting: bool = False, enable_audit_logging: bool = False):
        return L2TOrchestrator(
            trigger_mode=trigger_mode,
            l2t_config=self.l2t_config,
            model_configs=self.l2t_model_configs, # Use L2TModelConfigs object
            api_key=self.api_key,
            use_heuristic_shortcut=use_heuristic_shortcut,
            heuristic_detector=heuristic_detector,
            enable_rate_limiting=enable_rate_limiting,
            enable_audit_logging=enable_audit_logging
        )

    def test_always_l2t_mode(self):
        orchestrator = self._create_orchestrator(L2TTriggerMode.ALWAYS_L2T)
        solution, summary = orchestrator.solve(self.problem_text)

        self.MockL2TProcess.assert_called_once_with(
            l2t_config=self.l2t_config,
            model_configs=self.l2t_model_configs, # Use L2TModelConfigs object
            api_key=self.api_key,
            enable_rate_limiting=False,
            enable_audit_logging=False
        )
        self.mock_l2t_process_instance.execute.assert_called_once_with(problem_description=self.problem_text, model_name=ANY)
        self.mock_l2t_process_instance.get_result.assert_called_once()
        
        self.assertEqual(solution.final_answer, self.mock_l2t_process_solution.final_answer)
        self.assertIn(self.mock_l2t_process_summary, summary) 
        self.mock_assessor_instance.assess.assert_not_called()
        self.mock_oneshot_executor_instance.run_direct_oneshot.assert_not_called()

    def test_never_l2t_mode(self):
        orchestrator = self._create_orchestrator(L2TTriggerMode.NEVER_L2T)
        solution, summary = orchestrator.solve(self.problem_text)

        self.MockOneShotExecutor.assert_called_once_with(
            llm_client=ANY, 
            direct_oneshot_model_names=self.direct_oneshot_model_names, # Use direct_oneshot_model_names
            llm_config=self.l2t_model_configs.orchestrator_oneshot_config # Use LLMConfig object
        )
        self.mock_oneshot_executor_instance.run_direct_oneshot.assert_called_once_with(self.problem_text)
        
        if orchestrator.l2t_process_instance: 
             self.mock_l2t_process_instance.execute.assert_not_called()
        self.mock_assessor_instance.assess.assert_not_called()
        
        self.assertEqual(solution.final_answer, self.mock_direct_oneshot_llm_response)
        self.assertIn("Orchestrator Main Model Call", summary)

    def test_assess_first_leads_to_l2t(self):
        self.mock_assessor_instance.assess.return_value = (
            AssessmentDecision.ADVANCED_REASONING, # ADVANCED_REASONING decision means use complex process (L2T here)
            LLMCallStats(model_name="assessor", completion_tokens=1, prompt_tokens=1, call_duration_seconds=0.1)
        )
        
        orchestrator = self._create_orchestrator(L2TTriggerMode.ASSESS_FIRST)
        solution, summary = orchestrator.solve(self.problem_text)

        self.MockComplexityAssessorClass.assert_called_once_with(
            llm_client=ANY,
            small_model_names=self.assessment_model_names, # Use assessment_model_names
            llm_config=self.l2t_model_configs.node_classification_config, # Use LLMConfig object for assessor
            use_heuristic_shortcut=True,
            heuristic_detector=ANY
        ) 
        self.mock_assessor_instance.assess.assert_called_once_with(self.problem_text)
        
        self.MockL2TProcess.assert_called_once() 
        self.mock_l2t_process_instance.execute.assert_called_once_with(problem_description=self.problem_text, model_name=ANY)
        self.mock_l2t_process_instance.get_result.assert_called_once()

        self.assertEqual(solution.final_answer, self.mock_l2t_process_solution.final_answer)
        self.assertIn(self.mock_l2t_process_summary, summary)
        self.mock_oneshot_executor_instance.run_direct_oneshot.assert_not_called()

    def test_assess_first_leads_to_oneshot(self):
        self.mock_assessor_instance.assess.return_value = (
            AssessmentDecision.ONE_SHOT, 
            LLMCallStats(model_name="assessor", completion_tokens=1, prompt_tokens=1, call_duration_seconds=0.1)
        )

        orchestrator = self._create_orchestrator(L2TTriggerMode.ASSESS_FIRST)
        solution, summary = orchestrator.solve(self.problem_text)

        self.MockComplexityAssessorClass.assert_called_once_with(
            llm_client=ANY,
            small_model_names=self.assessment_model_names, # Use assessment_model_names
            llm_config=self.l2t_model_configs.node_classification_config, # Use LLMConfig object for assessor
            use_heuristic_shortcut=True,
            heuristic_detector=ANY
        )
        self.mock_assessor_instance.assess.assert_called_once_with(self.problem_text)

        self.MockOneShotExecutor.assert_called_once_with(
            llm_client=ANY,
            direct_oneshot_model_names=self.direct_oneshot_model_names, # Use direct_oneshot_model_names
            llm_config=self.l2t_model_configs.orchestrator_oneshot_config # Use LLMConfig object
        )
        self.mock_oneshot_executor_instance.run_direct_oneshot.assert_called_once_with(self.problem_text)
        
        if orchestrator.l2t_process_instance:
            self.mock_l2t_process_instance.execute.assert_not_called()
        
        self.assertEqual(solution.final_answer, self.mock_direct_oneshot_llm_response)
        self.assertIn("Orchestrator Main Model Call", summary)
        self.assertNotIn("Delegated to L2TProcess", summary)

if __name__ == '__main__':
    unittest.main()
