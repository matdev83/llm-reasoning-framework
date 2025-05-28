import unittest
from unittest.mock import patch, MagicMock, ANY

from src.aot_orchestrator import InteractiveAoTOrchestrator
from src.aot_dataclasses import AoTRunnerConfig, LLMCallStats, Solution, AoTResult 
from src.aot_enums import AotTriggerMode, AssessmentDecision
from src.heuristic_detector import HeuristicDetector
from typing import Optional

import logging
logging.disable(logging.CRITICAL)

# MockComplexityAssessor class is removed. We'll use MagicMock for the class.

class MockHeuristicDetector(HeuristicDetector): # Can keep this if needed for testing heuristic path
    def __init__(self):
        super().__init__()
        self.should_trigger_complex_process_heuristically = MagicMock(return_value=True)

class TestInteractiveAoTOrchestrator(unittest.TestCase):
    def setUp(self):
        self.problem_text = "Test problem for orchestrator."
        self.aot_config = AoTRunnerConfig(
            main_model_names=["test-aot-main-model"],
            temperature=0.7,
            max_steps=5,
        )
        self.direct_oneshot_config = {
            "model_names": ["test-direct-oneshot-model"],
            "temperature": 0.5
        }
        self.assessment_config = {
            "model_names": ["test-assessment-model"],
            "temperature": 0.0
        }
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
            model_name=self.direct_oneshot_config["model_names"][0],
            completion_tokens=50, prompt_tokens=100, call_duration_seconds=1.0
        )
        
        self.patch_llm_client = patch("src.aot_orchestrator.LLMClient")
        self.patch_complexity_assessor = patch("src.aot_orchestrator.ComplexityAssessor") # Standard patch
        self.patch_aot_process = patch("src.aot_orchestrator.AoTProcess")

        self.MockLLMClient = self.patch_llm_client.start()
        self.MockComplexityAssessorClass = self.patch_complexity_assessor.start() # This is a MagicMock for the class
        self.MockAoTProcess = self.patch_aot_process.start()
        
        # Configure the instance returned by the ComplexityAssessor class mock
        self.mock_assessor_instance = self.MockComplexityAssessorClass.return_value
        self.mock_assessor_instance.assess = MagicMock() # Add assess mock to the instance

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
            trigger_mode=trigger_mode,
            aot_config=self.aot_config,
            direct_oneshot_model_names=self.direct_oneshot_config["model_names"],
            direct_oneshot_temperature=self.direct_oneshot_config["temperature"],
            assessment_model_names=self.assessment_config["model_names"],
            assessment_temperature=self.assessment_config["temperature"],
            api_key=self.api_key,
            use_heuristic_shortcut=use_heuristic_shortcut,
            heuristic_detector=heuristic_detector,
            enable_rate_limiting=enable_rate_limiting, 
            enable_audit_logging=enable_audit_logging
        )

    def test_always_aot_mode(self):
        orchestrator = self._create_orchestrator(AotTriggerMode.ALWAYS_AOT)
        solution, summary = orchestrator.solve(self.problem_text)

        self.MockAoTProcess.assert_called_once_with(
            aot_config=self.aot_config,
            direct_oneshot_model_names=self.direct_oneshot_config["model_names"],
            direct_oneshot_temperature=self.direct_oneshot_config["temperature"],
            api_key=self.api_key,
            enable_rate_limiting=False,
            enable_audit_logging=False
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
            models=self.direct_oneshot_config["model_names"], 
            temperature=self.direct_oneshot_config["temperature"]
        )
        self.assertEqual(solution.final_answer, self.mock_direct_oneshot_llm_response)
        self.assertIn("Orchestrator Main Model Call", summary)


    def test_assess_first_leads_to_aot(self):
        # Configure the mock assessor instance (which is self.MockComplexityAssessorClass.return_value)
        self.mock_assessor_instance.assess.return_value = (
            AssessmentDecision.AOT, 
            LLMCallStats(model_name="assessor", completion_tokens=1, prompt_tokens=1, call_duration_seconds=0.1) # Removed assessment_decision
        )
        
        orchestrator = self._create_orchestrator(AotTriggerMode.ASSESS_FIRST)
        solution, summary = orchestrator.solve(self.problem_text)

        self.MockComplexityAssessorClass.assert_called_once() 
        self.mock_assessor_instance.assess.assert_called_once_with(self.problem_text)
        
        self.MockAoTProcess.assert_called_once() 
        self.mock_aot_process_instance.execute.assert_called_once_with(problem_description=self.problem_text, model_name=ANY)
        self.mock_aot_process_instance.get_result.assert_called_once()

        self.assertEqual(solution.final_answer, self.mock_aot_process_solution.final_answer)
        self.assertIn(self.mock_aot_process_summary, summary)
        
        problem_solving_calls_to_orchestrator_client = [
            c for c in self.mock_orchestrator_llm_client_instance.call.call_args_list
            if c[0][0] == self.problem_text and c[1].get('models') == self.direct_oneshot_config["model_names"]
        ]
        self.assertEqual(len(problem_solving_calls_to_orchestrator_client), 0)


    def test_assess_first_leads_to_oneshot(self):
        self.mock_assessor_instance.assess.return_value = (
            AssessmentDecision.ONESHOT, 
            LLMCallStats(model_name="assessor", completion_tokens=1, prompt_tokens=1, call_duration_seconds=0.1) # Removed assessment_decision
        )

        orchestrator = self._create_orchestrator(AotTriggerMode.ASSESS_FIRST)
        solution, summary = orchestrator.solve(self.problem_text)

        self.MockComplexityAssessorClass.assert_called_once()
        self.mock_assessor_instance.assess.assert_called_once_with(self.problem_text)

        if orchestrator.aot_process_instance:
            self.mock_aot_process_instance.execute.assert_not_called()
            self.mock_aot_process_instance.get_result.assert_not_called()
        
        self.mock_orchestrator_llm_client_instance.call.assert_called_once_with(
            prompt=self.problem_text, 
            models=self.direct_oneshot_config["model_names"], 
            temperature=self.direct_oneshot_config["temperature"]
        )
        self.assertEqual(solution.final_answer, self.mock_direct_oneshot_llm_response)
        self.assertIn("Orchestrator Main Model Call", summary) 
        self.assertNotIn("Delegated to AoTProcess", summary)

    def test_assess_first_with_heuristic_detector_leads_to_aot(self):
        custom_heuristic_detector = MockHeuristicDetector()
        custom_heuristic_detector.should_trigger_complex_process_heuristically.return_value = True
        
        # Configure the mock assessor instance to use this heuristic detector behaviour
        # The orchestrator will pass the heuristic_detector to the ComplexityAssessor constructor.
        # Our mock_assessor_instance (self.MockComplexityAssessorClass.return_value) needs to simulate this.
        # We can store the detector on the mock_assessor_instance when it's "created" by the orchestrator.
        
        def assess_side_effect(problem_text_param):
            # This simulates the ComplexityAssessor's internal logic
            # Access the heuristic_detector that would have been set on it by the orchestrator
            use_heuristic = self.mock_assessor_instance.use_heuristic_shortcut_param # Param passed to constructor
            detector_on_mock = self.mock_assessor_instance.heuristic_detector_param # Param passed to constructor
            
            if use_heuristic and detector_on_mock and \
               detector_on_mock.should_trigger_complex_process_heuristically(problem_text_param):
                return AssessmentDecision.AOT, LLMCallStats(model_name="heuristic_path") # Removed assessment_decision
            return AssessmentDecision.ONESHOT, LLMCallStats(model_name="assess-model-direct") # Removed assessment_decision
        
        self.mock_assessor_instance.assess.side_effect = assess_side_effect
        
        # This function will be called when ComplexityAssessor is instantiated by the orchestrator.
        # We use it to capture the heuristic_detector passed by the orchestrator to the mock.
        def capture_assessor_init_args(*args, **kwargs):
            self.mock_assessor_instance.use_heuristic_shortcut_param = kwargs.get('use_heuristic_shortcut')
            self.mock_assessor_instance.heuristic_detector_param = kwargs.get('heuristic_detector')
            return self.mock_assessor_instance # Return the configured instance

        self.MockComplexityAssessorClass.side_effect = capture_assessor_init_args
        
        orchestrator = self._create_orchestrator(
            trigger_mode=AotTriggerMode.ASSESS_FIRST,
            use_heuristic_shortcut=True,
            heuristic_detector=custom_heuristic_detector
        )
        
        solution, summary = orchestrator.solve(self.problem_text)

        custom_heuristic_detector.should_trigger_complex_process_heuristically.assert_called_once_with(self.problem_text)
        self.mock_assessor_instance.assess.assert_called_once_with(self.problem_text)
        
        self.MockAoTProcess.assert_called_once() 
        self.mock_aot_process_instance.execute.assert_called_once_with(problem_description=self.problem_text, model_name=ANY)
        self.mock_aot_process_instance.get_result.assert_called_once()
        
        self.assertEqual(solution.final_answer, self.mock_aot_process_solution.final_answer)
        self.assertIn(self.mock_aot_process_summary, summary)

if __name__ == '__main__':
    unittest.main()
