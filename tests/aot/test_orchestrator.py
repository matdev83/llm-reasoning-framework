import unittest
from unittest.mock import patch, MagicMock, ANY

from src.aot.orchestrator import InteractiveAoTOrchestrator
from src.aot.dataclasses import AoTRunnerConfig, LLMCallStats, Solution, AoTResult
from src.aot.enums import AotTriggerMode, AssessmentDecision
from src.heuristic_detector import HeuristicDetector
from src.llm_config import LLMConfig # Added LLMConfig import
from typing import Optional, List

import logging
logging.disable(logging.CRITICAL)

class MockHeuristicDetector(HeuristicDetector):
    def __init__(self):
        super().__init__()
        self.should_trigger_complex_process_heuristically = MagicMock(return_value=True)

class TestInteractiveAoTOrchestrator(unittest.TestCase):
    def setUp(self):
        self.problem_text = "Test problem for orchestrator."
        self.aot_config = AoTRunnerConfig(
            main_model_names=["test-aot-main-model"],
            max_steps=5,
        )
        # Define temperatures and model names directly for orchestrator parameters
        self.direct_oneshot_temp: float = 0.5
        self.assessment_temp: float = 0.0
        self.direct_oneshot_model_names: List[str] = ["test-direct-oneshot-model"]
        self.assessment_model_names: List[str] = ["test-assessment-model"]

        # LLMConfig instances for internal processor/assessor mocks
        self.aot_main_llm_config = LLMConfig(temperature=0.7) # For AoTProcess direct call
        self.direct_oneshot_llm_config = LLMConfig(temperature=0.5) # For Orchestrator's direct one-shot
        self.assessment_llm_config_for_assessor = LLMConfig(temperature=0.0) # For ComplexityAssessor mock

        self.api_key = "test_api_key"

        self.mock_aot_process_solution = Solution()
        self.mock_aot_process_solution.final_answer = "Final Answer from Mocked AoTProcess"
        self.mock_aot_process_solution.reasoning_trace = ["Mocked AoTProcess Step 1"]
        self.mock_aot_process_solution.aot_result = MagicMock(spec=AoTResult) 
        self.mock_aot_process_solution.aot_result.succeeded = True
        self.mock_aot_process_solution.aot_result.final_answer = "Final Answer from Mocked AoTProcess"
        self.mock_aot_process_solution.aot_result.reasoning_trace = self.mock_aot_process_solution.reasoning_trace
        self.mock_aot_process_solution.aot_result.total_llm_calls=2
        self.mock_aot_process_solution.aot_result.total_prompt_tokens=200
        self.mock_aot_process_solution.aot_result.total_completion_tokens=150
        self.mock_aot_process_solution.aot_result.total_llm_interaction_time_seconds=2.5
        self.mock_aot_process_solution.aot_result.total_process_wall_clock_time_seconds=3.0
        
        self.mock_aot_process_summary = "Summary from Mocked AoTProcess"

        self.mock_direct_oneshot_llm_response = "Direct one-shot answer from orchestrator"
        self.mock_direct_oneshot_stats = LLMCallStats(
            model_name=self.aot_config.main_model_names[0],
            completion_tokens=50, prompt_tokens=100, call_duration_seconds=1.0
        )
        
        self.patch_llm_client = patch("src.llm_client.LLMClient")
        self.patch_complexity_assessor = patch("src.aot.orchestrator.ComplexityAssessor")
        self.patch_aot_process = patch("src.aot.orchestrator.AoTProcess")

        self.MockLLMClient = self.patch_llm_client.start()
        self.MockComplexityAssessorClass = self.patch_complexity_assessor.start()
        self.MockAoTProcess = self.patch_aot_process.start()
        
        self.mock_assessor_instance = self.MockComplexityAssessorClass.return_value
        self.mock_assessor_instance.assess = MagicMock()

        self.mock_aot_process_instance = self.MockAoTProcess.return_value
        self.mock_aot_process_instance.execute = MagicMock()
        self.mock_aot_process_instance.get_result = MagicMock(
            return_value=(self.mock_aot_process_solution, self.mock_aot_process_summary)
        )
        
        self.mock_orchestrator_llm_client_instance = self.MockLLMClient.return_value
        self.mock_orchestrator_llm_client_instance.call.return_value = (
            self.mock_direct_oneshot_llm_response, self.mock_direct_oneshot_stats
        )

    def tearDown(self):
        self.patch_llm_client.stop()
        self.patch_complexity_assessor.stop()
        self.patch_aot_process.stop()

    def _create_orchestrator(self, trigger_mode: AotTriggerMode, use_heuristic_shortcut: bool = True, heuristic_detector: Optional[HeuristicDetector] = None, enable_rate_limiting: bool = False, enable_audit_logging: bool = False):
        return InteractiveAoTOrchestrator(
            llm_client=self.mock_orchestrator_llm_client_instance, # Pass mock LLMClient
            trigger_mode=trigger_mode,
            aot_config=self.aot_config,
            direct_oneshot_llm_config=self.direct_oneshot_llm_config, # Pass LLMConfig
            assessment_llm_config=self.assessment_llm_config_for_assessor, # Pass LLMConfig
            aot_main_llm_config=self.aot_main_llm_config, # Pass LLMConfig
            direct_oneshot_model_names=self.direct_oneshot_model_names,
            assessment_model_names=self.assessment_model_names,
            use_heuristic_shortcut=use_heuristic_shortcut,
            heuristic_detector=heuristic_detector,
            enable_rate_limiting=enable_rate_limiting,
            enable_audit_logging=enable_audit_logging
        )

    def test_always_aot_mode(self):
        orchestrator = self._create_orchestrator(AotTriggerMode.ALWAYS_AOT)
        solution, summary = orchestrator.solve(self.problem_text)

        self.MockAoTProcess.assert_called_once_with(
            llm_client=ANY,
            aot_config=self.aot_config,
            aot_main_llm_config=self.aot_main_llm_config,
            direct_oneshot_llm_config=self.direct_oneshot_llm_config # Pass LLMConfig
        )
        self.mock_aot_process_instance.execute.assert_called_once_with(problem_description=self.problem_text, model_name=ANY)
        self.mock_aot_process_instance.get_result.assert_called_once()
        
        self.assertEqual(solution.final_answer, self.mock_aot_process_solution.final_answer)
        self.assertIn(self.mock_aot_process_summary, summary)
        self.mock_assessor_instance.assess.assert_not_called() 
        
        problem_solving_calls = [
            c for c in self.mock_orchestrator_llm_client_instance.call.call_args_list
            if c[0][0] == self.problem_text 
        ]
        self.assertEqual(len(problem_solving_calls), 0)


    def test_never_aot_mode(self):
        orchestrator = self._create_orchestrator(AotTriggerMode.NEVER_AOT)
        solution, summary = orchestrator.solve(self.problem_text)
        
        if orchestrator.aot_process_instance: 
            self.mock_aot_process_instance.execute.assert_not_called()
            self.mock_aot_process_instance.get_result.assert_not_called()

        self.mock_assessor_instance.assess.assert_not_called()
        self.mock_orchestrator_llm_client_instance.call.assert_called_once_with(
            prompt=self.problem_text, 
            models=self.direct_oneshot_model_names,
            config=ANY # Use ANY for config parameter
        )
        self.assertEqual(solution.final_answer, self.mock_direct_oneshot_llm_response)
        self.assertIn("Orchestrator Main Model Call", summary)


    def test_assess_first_leads_to_aot(self):
        self.mock_assessor_instance.assess.return_value = (
            AssessmentDecision.ADVANCED_REASONING, 
            LLMCallStats(model_name="assessor", completion_tokens=1, prompt_tokens=1, call_duration_seconds=0.1)
        )
        
        orchestrator = self._create_orchestrator(AotTriggerMode.ASSESS_FIRST)
        solution, summary = orchestrator.solve(self.problem_text)

        self.MockComplexityAssessorClass.assert_called_once_with(
            llm_client=ANY,
            small_model_names=self.assessment_model_names,
            llm_config=self.assessment_llm_config_for_assessor,
            use_heuristic_shortcut=True,
            heuristic_detector=ANY
        )
        self.mock_assessor_instance.assess.assert_called_once_with(self.problem_text)
        
        self.MockAoTProcess.assert_called_once() 
        self.mock_aot_process_instance.execute.assert_called_once_with(problem_description=self.problem_text, model_name=ANY)
        self.mock_aot_process_instance.get_result.assert_called_once()

        self.assertEqual(solution.final_answer, self.mock_aot_process_solution.final_answer)
        self.assertIn(self.mock_aot_process_summary, summary)
        
        problem_solving_calls_to_orchestrator_client = [
            c for c in self.mock_orchestrator_llm_client_instance.call.call_args_list
            if c[0][0] == self.problem_text and c[1].get('models') == self.aot_config.main_model_names
        ]
        self.assertEqual(len(problem_solving_calls_to_orchestrator_client), 0)


    def test_assess_first_leads_to_oneshot(self):
        self.mock_assessor_instance.assess.return_value = (
            AssessmentDecision.ONE_SHOT, 
            LLMCallStats(model_name="assessor", completion_tokens=1, prompt_tokens=1, call_duration_seconds=0.1)
        )

        orchestrator = self._create_orchestrator(AotTriggerMode.ASSESS_FIRST)
        solution, summary = orchestrator.solve(self.problem_text)

        self.MockComplexityAssessorClass.assert_called_once_with(
            llm_client=ANY,
            small_model_names=self.assessment_model_names,
            llm_config=self.assessment_llm_config_for_assessor,
            use_heuristic_shortcut=True,
            heuristic_detector=ANY
        )
        self.mock_assessor_instance.assess.assert_called_once_with(self.problem_text)

        if orchestrator.aot_process_instance:
            self.mock_aot_process_instance.execute.assert_not_called()
            self.mock_aot_process_instance.get_result.assert_not_called()
        
        self.mock_orchestrator_llm_client_instance.call.assert_called_once_with(
            prompt=self.problem_text, 
            models=self.direct_oneshot_model_names,
            config=ANY # Use ANY for config parameter
        )
        self.assertEqual(solution.final_answer, self.mock_direct_oneshot_llm_response)
        self.assertIn("Orchestrator Main Model Call", summary) 
        self.assertNotIn("Delegated to AoTProcess", summary)

    def test_assess_first_with_heuristic_detector_leads_to_aot(self):
        custom_heuristic_detector = MockHeuristicDetector()
        
        with patch.object(custom_heuristic_detector, 'should_trigger_complex_process_heuristically', return_value=True) as mock_should_trigger_complex_process_heuristically:
            def assess_side_effect(problem_text_param):
                use_heuristic = self.mock_assessor_instance.use_heuristic_shortcut_param
                detector_on_mock = self.mock_assessor_instance.heuristic_detector_param
                
                if use_heuristic and detector_on_mock and \
                   detector_on_mock.should_trigger_complex_process_heuristically(problem_text_param):
                    return AssessmentDecision.ADVANCED_REASONING, LLMCallStats(model_name="heuristic_path")
                return AssessmentDecision.ONE_SHOT, LLMCallStats(model_name="assess-model-direct")
            
            self.mock_assessor_instance.assess.side_effect = assess_side_effect
            
            def capture_assessor_init_args(*args, **kwargs):
                self.mock_assessor_instance.use_heuristic_shortcut_param = kwargs.get('use_heuristic_shortcut')
                self.mock_assessor_instance.heuristic_detector_param = kwargs.get('heuristic_detector')
                self.mock_assessor_instance.llm_config_param = kwargs.get('llm_config')
                self.mock_assessor_instance.small_model_names_param = kwargs.get('small_model_names')
                return self.mock_assessor_instance

            self.MockComplexityAssessorClass.side_effect = capture_assessor_init_args
            
            orchestrator = self._create_orchestrator(
                trigger_mode=AotTriggerMode.ASSESS_FIRST,
                use_heuristic_shortcut=True,
                heuristic_detector=custom_heuristic_detector
            )
            
            solution, summary = orchestrator.solve(self.problem_text)

            mock_should_trigger_complex_process_heuristically.assert_called_once_with(self.problem_text)
            self.mock_assessor_instance.assess.assert_called_once_with(self.problem_text)
            
            self.MockAoTProcess.assert_called_once() 
        self.mock_aot_process_instance.execute.assert_called_once_with(problem_description=self.problem_text, model_name=ANY)
        self.mock_aot_process_instance.get_result.assert_called_once()
        
        self.assertEqual(solution.final_answer, self.mock_aot_process_solution.final_answer)
        self.assertIn(self.mock_aot_process_summary, summary)

if __name__ == '__main__':
    unittest.main()
