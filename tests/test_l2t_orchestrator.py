import unittest
from unittest.mock import patch, MagicMock

from src.l2t_orchestrator import L2TOrchestrator
from src.l2t_dataclasses import L2TConfig, L2TResult, L2TGraph, L2TSolution
from src.l2t_enums import L2TTriggerMode
from src.aot_dataclasses import LLMCallStats
from src.aot_enums import AssessmentDecision # Import AssessmentDecision

# Suppress logging during tests
import logging
logging.disable(logging.CRITICAL)


# Mock for L2TProcessor to be used by the orchestrator tests
class MockL2TProcessor:
    def __init__(self, llm_client, config):
        self.llm_client = llm_client
        self.config = config
        self.run = MagicMock()

# Mock for ComplexityAssessor
class MockComplexityAssessor:
    def __init__(self, llm_client, small_model_names, temperature, use_heuristic_shortcut):
        self.llm_client = llm_client
        self.small_model_names = small_model_names
        self.temperature = temperature
        self.use_heuristic_shortcut = use_heuristic_shortcut
        self.assess = MagicMock()


class TestL2TOrchestrator(unittest.TestCase):
    def setUp(self):
        self.l2t_config = L2TConfig(
            max_steps=3,
            classification_model_names=["test-classifier"],
            thought_generation_model_names=["test-generator"],
        )
        self.api_key = "test_api_key_for_orchestrator"
        self.direct_oneshot_model_names = ["test-oneshot-model"]
        self.direct_oneshot_temperature = 0.7
        self.assessment_model_names = ["test-assessment-model"]
        self.assessment_temperature = 0.0

    def _create_orchestrator(self, trigger_mode: L2TTriggerMode, use_heuristic_shortcut: bool = True):
        return L2TOrchestrator(
            trigger_mode=trigger_mode,
            l2t_config=self.l2t_config,
            direct_oneshot_model_names=self.direct_oneshot_model_names,
            direct_oneshot_temperature=self.direct_oneshot_temperature,
            assessment_model_names=self.assessment_model_names,
            assessment_temperature=self.assessment_temperature,
            api_key=self.api_key,
            use_heuristic_shortcut=use_heuristic_shortcut
        )

    @patch("src.l2t_orchestrator.L2TProcessor")
    @patch("src.l2t_orchestrator.ComplexityAssessor")
    @patch("src.l2t_orchestrator.LLMClient")
    def test_solve_successful_run_always_l2t(self, MockLLMClient, MockComplexityAssessor, MockL2TProcessorClass):
        problem_text = "Test problem: orchestrator success always L2T."

        mock_successful_result = L2TResult(
            final_answer="This is the final answer from L2T.",
            reasoning_graph=L2TGraph(),
            total_llm_calls=5,
            total_completion_tokens=100,
            total_prompt_tokens=200,
            total_llm_interaction_time_seconds=2.5,
            total_process_wall_clock_time_seconds=3.0,
            succeeded=True,
            error_message=None,
        )
        
        mock_processor_instance = MockL2TProcessor(llm_client=MagicMock(), config=self.l2t_config)
        MockL2TProcessorClass.return_value = mock_processor_instance
        mock_processor_instance.run.return_value = mock_successful_result

        orchestrator = self._create_orchestrator(L2TTriggerMode.ALWAYS_L2T)
        
        solution, summary_str = orchestrator.solve(problem_text)

        MockL2TProcessorClass.assert_called_once_with(llm_client=orchestrator.llm_client, config=self.l2t_config)
        mock_processor_instance.run.assert_called_once_with(problem_text)
        MockComplexityAssessor.assert_not_called() # Should not be called in ALWAYS_L2T mode

        self.assertTrue(solution.succeeded)
        self.assertEqual(solution.final_answer, mock_successful_result.final_answer)
        self.assertEqual(solution.l2t_result, mock_successful_result)
        self.assertIn("OVERALL L2T ORCHESTRATOR SUMMARY", summary_str)
        self.assertIn("Trigger Mode: ALWAYS_L2T", summary_str)
        self.assertIn("L2T Succeeded (as per L2T summary): Yes", summary_str)
        self.assertIn(f"Final Answer:\n{mock_successful_result.final_answer}", summary_str)
        self.assertIsNone(solution.assessment_stats)
        self.assertIsNone(solution.main_call_stats)
        self.assertFalse(solution.l2t_failed_and_fell_back)


    @patch("src.l2t_orchestrator.L2TProcessor")
    @patch("src.l2t_orchestrator.ComplexityAssessor")
    @patch("src.l2t_orchestrator.LLMClient")
    def test_solve_failed_run_always_l2t_fallback(self, MockLLMClient, MockComplexityAssessor, MockL2TProcessorClass):
        problem_text = "Test problem: orchestrator failure always L2T."

        mock_failed_result = L2TResult(
            final_answer=None,
            reasoning_graph=L2TGraph(),
            total_llm_calls=2,
            total_completion_tokens=50,
            total_prompt_tokens=70,
            total_llm_interaction_time_seconds=1.0,
            total_process_wall_clock_time_seconds=1.2,
            succeeded=False,
            error_message="Max steps reached during L2T processing.",
        )
        mock_fallback_answer = "Fallback one-shot answer."
        mock_fallback_stats = LLMCallStats(model_name="fallback-model", prompt_tokens=10, completion_tokens=20, call_duration_seconds=0.5)

        mock_processor_instance = MockL2TProcessor(llm_client=MagicMock(), config=self.l2t_config)
        MockL2TProcessorClass.return_value = mock_processor_instance
        mock_processor_instance.run.return_value = mock_failed_result

        MockLLMClient.return_value.call.return_value = (mock_fallback_answer, mock_fallback_stats)

        orchestrator = self._create_orchestrator(L2TTriggerMode.ALWAYS_L2T)
        solution, summary_str = orchestrator.solve(problem_text)

        MockL2TProcessorClass.assert_called_once_with(llm_client=orchestrator.llm_client, config=self.l2t_config)
        mock_processor_instance.run.assert_called_once_with(problem_text)
        MockLLMClient.return_value.call.assert_called_once_with(
            prompt=problem_text, models=self.direct_oneshot_model_names, temperature=self.direct_oneshot_temperature
        )
        MockComplexityAssessor.assert_not_called()

        self.assertTrue(solution.succeeded) # Overall solution is successful if fallback provides an answer
        self.assertEqual(solution.final_answer, mock_fallback_answer)
        self.assertEqual(solution.l2t_result, mock_failed_result)
        self.assertTrue(solution.l2t_failed_and_fell_back)
        self.assertEqual(solution.fallback_call_stats, mock_fallback_stats)
        self.assertIn("OVERALL L2T ORCHESTRATOR SUMMARY", summary_str)
        self.assertIn("Trigger Mode: ALWAYS_L2T", summary_str)
        self.assertIn("L2T FAILED and Fell Back to One-Shot: Yes", summary_str)
        self.assertIn(f"Fallback One-Shot Call ({mock_fallback_stats.model_name})", summary_str)
        self.assertIn(f"Final Answer:\n{mock_fallback_answer}", summary_str)


    @patch("src.l2t_orchestrator.LLMClient")
    @patch("src.l2t_orchestrator.L2TProcessor")
    @patch("src.l2t_orchestrator.ComplexityAssessor")
    def test_solve_never_l2t(self, MockComplexityAssessor, MockL2TProcessor, MockLLMClient):
        problem_text = "Test problem: never L2T."
        mock_oneshot_answer = "Direct one-shot answer."
        mock_oneshot_stats = LLMCallStats(model_name="oneshot-model", prompt_tokens=15, completion_tokens=25, call_duration_seconds=0.8)

        MockLLMClient.return_value.call.return_value = (mock_oneshot_answer, mock_oneshot_stats)

        orchestrator = self._create_orchestrator(L2TTriggerMode.NEVER_L2T)
        solution, summary_str = orchestrator.solve(problem_text)

        MockLLMClient.return_value.call.assert_called_once_with(
            prompt=problem_text, models=self.direct_oneshot_model_names, temperature=self.direct_oneshot_temperature
        )
        # In NEVER_L2T mode, L2TProcessor should not be instantiated
        MockL2TProcessor.assert_not_called()
        MockComplexityAssessor.assert_not_called()

        self.assertTrue(solution.succeeded) # One-shot is considered successful if it returns an answer
        self.assertEqual(solution.final_answer, mock_oneshot_answer)
        self.assertEqual(solution.main_call_stats, mock_oneshot_stats)
        self.assertIsNone(solution.l2t_result)
        self.assertIn("OVERALL L2T ORCHESTRATOR SUMMARY", summary_str)
        self.assertIn("Trigger Mode: NEVER_L2T", summary_str)
        self.assertIn(f"Main Model Call (Direct ONESHOT path) ({mock_oneshot_stats.model_name})", summary_str)
        self.assertIn(f"Final Answer:\n{mock_oneshot_answer}", summary_str)


    @patch("src.l2t_orchestrator.L2TProcessor")
    @patch("src.l2t_orchestrator.ComplexityAssessor")
    @patch("src.l2t_orchestrator.LLMClient")
    def test_solve_assess_first_oneshot_decision(self, MockLLMClient, MockComplexityAssessorClass, MockL2TProcessorClass): # Changed MockL2TProcessor to MockL2TProcessorClass
        problem_text = "Test problem: assess first, then oneshot."
        mock_oneshot_answer = "Assessed one-shot answer."
        mock_oneshot_stats = LLMCallStats(model_name="assessed-oneshot-model", prompt_tokens=20, completion_tokens=30, call_duration_seconds=1.0)
        mock_assessment_stats = LLMCallStats(model_name="assess-model", prompt_tokens=5, completion_tokens=1, call_duration_seconds=0.2)

        mock_assessor_instance = MockComplexityAssessor(llm_client=MagicMock(), small_model_names=self.assessment_model_names, temperature=self.assessment_temperature, use_heuristic_shortcut=True)
        MockComplexityAssessorClass.return_value = mock_assessor_instance
        mock_assessor_instance.assess.return_value = (AssessmentDecision.ONESHOT, mock_assessment_stats)
        MockLLMClient.return_value.call.return_value = (mock_oneshot_answer, mock_oneshot_stats)

        orchestrator = self._create_orchestrator(L2TTriggerMode.ASSESS_FIRST)
        solution, summary_str = orchestrator.solve(problem_text)

        mock_assessor_instance.assess.assert_called_once_with(problem_text)
        MockLLMClient.return_value.call.assert_called_once_with(
            prompt=problem_text, models=self.direct_oneshot_model_names, temperature=self.direct_oneshot_temperature
        )
        # L2TProcessor is instantiated in ASSESS_FIRST mode, but its run method should not be called if ONESHOT
        MockL2TProcessorClass.return_value.run.assert_not_called()

        self.assertTrue(solution.succeeded)
        self.assertEqual(solution.final_answer, mock_oneshot_answer)
        self.assertEqual(solution.assessment_stats, mock_assessment_stats)
        self.assertEqual(solution.assessment_decision, AssessmentDecision.ONESHOT)
        self.assertEqual(solution.main_call_stats, mock_oneshot_stats)
        self.assertIsNone(solution.l2t_result)
        self.assertIn("OVERALL L2T ORCHESTRATOR SUMMARY", summary_str)
        self.assertIn("Trigger Mode: ASSESS_FIRST", summary_str)
        self.assertIn(f"Assessment ({mock_assessment_stats.model_name}): Decision=ONESHOT", summary_str)
        self.assertIn(f"Main Model Call (Direct ONESHOT path) ({mock_oneshot_stats.model_name})", summary_str)
        self.assertIn(f"Final Answer:\n{mock_oneshot_answer}", summary_str)


    @patch("src.l2t_orchestrator.L2TProcessor")
    @patch("src.l2t_orchestrator.ComplexityAssessor")
    @patch("src.l2t_orchestrator.LLMClient")
    def test_solve_assess_first_l2t_decision(self, MockLLMClient, MockComplexityAssessorClass, MockL2TProcessorClass):
        problem_text = "Test problem: assess first, then L2T."
        mock_l2t_result = L2TResult(
            final_answer="L2T answer after assessment.",
            reasoning_graph=L2TGraph(),
            total_llm_calls=3, total_completion_tokens=70, total_prompt_tokens=150,
            total_llm_interaction_time_seconds=1.8, total_process_wall_clock_time_seconds=2.2,
            succeeded=True
        )
        mock_assessment_stats = LLMCallStats(model_name="assess-model", prompt_tokens=5, completion_tokens=1, call_duration_seconds=0.2)

        mock_assessor_instance = MockComplexityAssessor(llm_client=MagicMock(), small_model_names=self.assessment_model_names, temperature=self.assessment_temperature, use_heuristic_shortcut=True)
        MockComplexityAssessorClass.return_value = mock_assessor_instance
        mock_assessor_instance.assess.return_value = (AssessmentDecision.AOT, mock_assessment_stats) # AOT decision

        mock_processor_instance = MockL2TProcessor(llm_client=MagicMock(), config=self.l2t_config)
        MockL2TProcessorClass.return_value = mock_processor_instance
        mock_processor_instance.run.return_value = mock_l2t_result

        orchestrator = self._create_orchestrator(L2TTriggerMode.ASSESS_FIRST)
        solution, summary_str = orchestrator.solve(problem_text)

        mock_assessor_instance.assess.assert_called_once_with(problem_text)
        MockL2TProcessorClass.assert_called_once_with(llm_client=orchestrator.llm_client, config=self.l2t_config)
        mock_processor_instance.run.assert_called_once_with(problem_text)
        MockLLMClient.return_value.call.assert_not_called() # No direct one-shot call

        self.assertTrue(solution.succeeded)
        self.assertEqual(solution.final_answer, mock_l2t_result.final_answer)
        self.assertEqual(solution.assessment_stats, mock_assessment_stats)
        self.assertEqual(solution.assessment_decision, AssessmentDecision.AOT)
        self.assertEqual(solution.l2t_result, mock_l2t_result)
        self.assertIn("OVERALL L2T ORCHESTRATOR SUMMARY", summary_str)
        self.assertIn("Trigger Mode: ASSESS_FIRST", summary_str)
        self.assertIn(f"Assessment ({mock_assessment_stats.model_name}): Decision=AOT", summary_str)
        self.assertIn("L2T Process Attempted: Yes", summary_str)
        self.assertIn("L2T Succeeded (as per L2T summary): Yes", summary_str)
        self.assertIn(f"Final Answer:\n{mock_l2t_result.final_answer}", summary_str)


    @patch("src.l2t_orchestrator.L2TProcessor")
    @patch("src.l2t_orchestrator.ComplexityAssessor")
    @patch("src.l2t_orchestrator.LLMClient")
    def test_solve_assess_first_l2t_decision_failed_fallback(self, MockLLMClient, MockComplexityAssessorClass, MockL2TProcessorClass):
        problem_text = "Test problem: assess first, L2T fails, then fallback."
        mock_failed_l2t_result = L2TResult(
            final_answer=None,
            reasoning_graph=L2TGraph(),
            total_llm_calls=1, total_completion_tokens=20, total_prompt_tokens=40,
            total_llm_interaction_time_seconds=0.5, total_process_wall_clock_time_seconds=0.8,
            succeeded=False, error_message="L2T failed internally."
        )
        mock_fallback_answer = "Fallback answer after L2T failure."
        mock_fallback_stats = LLMCallStats(model_name="fallback-model", prompt_tokens=12, completion_tokens=22, call_duration_seconds=0.6)
        mock_assessment_stats = LLMCallStats(model_name="assess-model", prompt_tokens=5, completion_tokens=1, call_duration_seconds=0.2)

        mock_assessor_instance = MockComplexityAssessor(llm_client=MagicMock(), small_model_names=self.assessment_model_names, temperature=self.assessment_temperature, use_heuristic_shortcut=True)
        MockComplexityAssessorClass.return_value = mock_assessor_instance
        mock_assessor_instance.assess.return_value = (AssessmentDecision.AOT, mock_assessment_stats) # AOT decision

        mock_processor_instance = MockL2TProcessor(llm_client=MagicMock(), config=self.l2t_config)
        MockL2TProcessorClass.return_value = mock_processor_instance
        mock_processor_instance.run.return_value = mock_failed_l2t_result
        MockLLMClient.return_value.call.return_value = (mock_fallback_answer, mock_fallback_stats)

        orchestrator = self._create_orchestrator(L2TTriggerMode.ASSESS_FIRST)
        solution, summary_str = orchestrator.solve(problem_text)

        mock_assessor_instance.assess.assert_called_once_with(problem_text)
        mock_processor_instance.run.assert_called_once_with(problem_text)
        MockLLMClient.return_value.call.assert_called_once_with(
            prompt=problem_text, models=self.direct_oneshot_model_names, temperature=self.direct_oneshot_temperature
        )

        self.assertTrue(solution.succeeded) # Overall solution is successful if fallback provides an answer
        self.assertEqual(solution.final_answer, mock_fallback_answer)
        self.assertEqual(solution.assessment_stats, mock_assessment_stats)
        self.assertEqual(solution.assessment_decision, AssessmentDecision.AOT)
        self.assertEqual(solution.l2t_result, mock_failed_l2t_result)
        self.assertTrue(solution.l2t_failed_and_fell_back)
        self.assertEqual(solution.fallback_call_stats, mock_fallback_stats)
        self.assertIn("OVERALL L2T ORCHESTRATOR SUMMARY", summary_str)
        self.assertIn("Trigger Mode: ASSESS_FIRST", summary_str)
        self.assertIn("L2T FAILED and Fell Back to One-Shot: Yes", summary_str)
        self.assertIn(f"Fallback One-Shot Call ({mock_fallback_stats.model_name})", summary_str)
        self.assertIn(f"Final Answer:\n{mock_fallback_answer}", summary_str)


    @patch("src.l2t_orchestrator.L2TProcessor")
    @patch("src.l2t_orchestrator.ComplexityAssessor")
    @patch("src.l2t_orchestrator.LLMClient")
    def test_solve_assess_first_assessment_error_fallback(self, MockLLMClient, MockComplexityAssessorClass, MockL2TProcessorClass): # Changed MockL2TProcessor to MockL2TProcessorClass
        problem_text = "Test problem: assessment error, then fallback."
        mock_fallback_answer = "Fallback answer after assessment error."
        mock_fallback_stats = LLMCallStats(model_name="fallback-model", prompt_tokens=13, completion_tokens=23, call_duration_seconds=0.7)
        mock_assessment_stats = LLMCallStats(model_name="assess-model", prompt_tokens=5, completion_tokens=1, call_duration_seconds=0.2)

        mock_assessor_instance = MockComplexityAssessor(llm_client=MagicMock(), small_model_names=self.assessment_model_names, temperature=self.assessment_temperature, use_heuristic_shortcut=True)
        MockComplexityAssessorClass.return_value = mock_assessor_instance
        mock_assessor_instance.assess.return_value = (AssessmentDecision.ERROR, mock_assessment_stats) # ERROR decision

        MockLLMClient.return_value.call.return_value = (mock_fallback_answer, mock_fallback_stats)

        orchestrator = self._create_orchestrator(L2TTriggerMode.ASSESS_FIRST)
        solution, summary_str = orchestrator.solve(problem_text)

        mock_assessor_instance.assess.assert_called_once_with(problem_text)
        # L2TProcessor is instantiated in ASSESS_FIRST mode, but its run method should not be called if assessment errors
        MockL2TProcessorClass.return_value.run.assert_not_called()
        MockLLMClient.return_value.call.assert_called_once_with(
            prompt=problem_text, models=self.direct_oneshot_model_names, temperature=self.direct_oneshot_temperature
        )

        self.assertTrue(solution.succeeded) # Overall solution is successful if fallback provides an answer
        self.assertEqual(solution.final_answer, mock_fallback_answer)
        self.assertEqual(solution.assessment_stats, mock_assessment_stats)
        self.assertEqual(solution.assessment_decision, AssessmentDecision.ERROR)
        self.assertTrue(solution.l2t_failed_and_fell_back)
        self.assertEqual(solution.fallback_call_stats, mock_fallback_stats)
        self.assertIn("OVERALL L2T ORCHESTRATOR SUMMARY", summary_str)
        self.assertIn("Trigger Mode: ASSESS_FIRST", summary_str)
        self.assertIn(f"Assessment ({mock_assessment_stats.model_name}): Decision=ERROR", summary_str)
        self.assertIn("Process led to Fallback One-Shot (due to Assessment Error, L2T not attempted): Yes", summary_str)
        self.assertIn(f"Fallback One-Shot Call ({mock_fallback_stats.model_name})", summary_str)
        self.assertIn(f"Final Answer:\n{mock_fallback_answer}", summary_str)


    @patch("src.l2t_orchestrator.L2TProcessor")
    @patch("src.l2t_orchestrator.ComplexityAssessor")
    @patch("src.l2t_orchestrator.LLMClient")
    def test_summary_generation_content(self, MockLLMClient, MockComplexityAssessorClass, MockL2TProcessorClass):
        problem_text = "Test problem for summary details."
        
        reasoning_graph_mock = L2TGraph()
        mock_detailed_l2t_result = L2TResult(
            final_answer="Detailed answer.",
            reasoning_graph=reasoning_graph_mock,
            total_llm_calls=3,
            total_completion_tokens=123,
            total_prompt_tokens=456,
            total_llm_interaction_time_seconds=1.23,
            total_process_wall_clock_time_seconds=2.34,
            succeeded=True
        )
        reasoning_graph_mock.add_node(MagicMock(id="root12345"), is_root=True)

        mock_assessment_stats = LLMCallStats(model_name="assess-model", prompt_tokens=5, completion_tokens=1, call_duration_seconds=0.2)
        mock_main_oneshot_stats = LLMCallStats(model_name="main-oneshot-model", prompt_tokens=10, completion_tokens=20, call_duration_seconds=0.5)
        mock_fallback_stats = LLMCallStats(model_name="fallback-model", prompt_tokens=15, completion_tokens=25, call_duration_seconds=0.8)

        # Scenario: Assess -> L2T (successful)
        mock_assessor_instance = MockComplexityAssessor(llm_client=MagicMock(), small_model_names=self.assessment_model_names, temperature=self.assessment_temperature, use_heuristic_shortcut=True)
        MockComplexityAssessorClass.return_value = mock_assessor_instance
        mock_assessor_instance.assess.return_value = (AssessmentDecision.AOT, mock_assessment_stats)

        mock_processor_instance = MockL2TProcessor(llm_client=MagicMock(), config=self.l2t_config)
        MockL2TProcessorClass.return_value = mock_processor_instance
        mock_processor_instance.run.return_value = mock_detailed_l2t_result

        orchestrator = self._create_orchestrator(L2TTriggerMode.ASSESS_FIRST)
        solution, summary_str = orchestrator.solve(problem_text)

        self.assertIn("OVERALL L2T ORCHESTRATOR SUMMARY", summary_str)
        self.assertIn(f"Trigger Mode: {L2TTriggerMode.ASSESS_FIRST.value.upper()}", summary_str)
        self.assertIn(f"Heuristic Shortcut Enabled: True", summary_str)
        self.assertIn(f"Assessment ({mock_assessment_stats.model_name}): Decision={AssessmentDecision.AOT.value}", summary_str)
        self.assertIn("L2T Process Attempted: Yes", summary_str)
        self.assertIn("L2T Succeeded (as per L2T summary): Yes", summary_str)
        self.assertIn(f"Final Answer:\n{mock_detailed_l2t_result.final_answer}", summary_str)
        
        # Check total tokens and time properties from L2TSolution
        expected_total_completion = mock_assessment_stats.completion_tokens + mock_detailed_l2t_result.total_completion_tokens
        expected_total_prompt = mock_assessment_stats.prompt_tokens + mock_detailed_l2t_result.total_prompt_tokens
        expected_total_llm_time = mock_assessment_stats.call_duration_seconds + mock_detailed_l2t_result.total_llm_interaction_time_seconds

        self.assertIn(f"Total Completion Tokens (All Calls): {expected_total_completion}", summary_str)
        self.assertIn(f"Total Prompt Tokens (All Calls): {expected_total_prompt}", summary_str)
        self.assertIn(f"Grand Total Tokens (All Calls): {expected_total_completion + expected_total_prompt}", summary_str)
        self.assertIn(f"Total LLM Interaction Time (All Calls): {expected_total_llm_time:.2f}s", summary_str)
        self.assertIn(f"Total Process Wall-Clock Time: {solution.total_wall_clock_time_seconds:.2f}s", summary_str)

        # Scenario: Never L2T (direct one-shot)
        MockLLMClient.return_value.call.reset_mock()
        orchestrator_never = self._create_orchestrator(L2TTriggerMode.NEVER_L2T)
        MockLLMClient.return_value.call.return_value = (mock_main_oneshot_stats.model_name, mock_main_oneshot_stats) # Mock for direct one-shot
        solution_never, summary_str_never = orchestrator_never.solve(problem_text)

        self.assertIn("OVERALL L2T ORCHESTRATOR SUMMARY", summary_str_never)
        self.assertIn("Trigger Mode: NEVER_L2T", summary_str_never)
        self.assertIn(f"Main Model Call (Direct ONESHOT path) ({mock_main_oneshot_stats.model_name})", summary_str_never)
        self.assertIn(f"Total Completion Tokens (All Calls): {mock_main_oneshot_stats.completion_tokens}", summary_str_never)
        self.assertIn(f"Total Prompt Tokens (All Calls): {mock_main_oneshot_stats.prompt_tokens}", summary_str_never)
        self.assertIn(f"Grand Total Tokens (All Calls): {mock_main_oneshot_stats.completion_tokens + mock_main_oneshot_stats.prompt_tokens}", summary_str_never)
        self.assertIn(f"Total LLM Interaction Time (All Calls): {mock_main_oneshot_stats.call_duration_seconds:.2f}s", summary_str_never)

        # Scenario: Assess -> L2T (failed) -> Fallback
        MockLLMClient.return_value.call.reset_mock()
        mock_assessor_instance.assess.reset_mock()
        mock_processor_instance.run.reset_mock()

        mock_assessor_instance.assess.return_value = (AssessmentDecision.AOT, mock_assessment_stats)
        mock_processor_instance.run.return_value = L2TResult(succeeded=False, error_message="L2T failed for summary test")
        MockLLMClient.return_value.call.return_value = ("Fallback summary answer", mock_fallback_stats)

        orchestrator_fallback = self._create_orchestrator(L2TTriggerMode.ASSESS_FIRST)
        solution_fallback, summary_str_fallback = orchestrator_fallback.solve(problem_text)

        self.assertIn("OVERALL L2T ORCHESTRATOR SUMMARY", summary_str_fallback)
        self.assertIn("L2T FAILED and Fell Back to One-Shot: Yes", summary_str_fallback)
        self.assertIn(f"Fallback One-Shot Call ({mock_fallback_stats.model_name})", summary_str_fallback)
        
        expected_total_completion_fallback = mock_assessment_stats.completion_tokens + mock_fallback_stats.completion_tokens
        expected_total_prompt_fallback = mock_assessment_stats.prompt_tokens + mock_fallback_stats.prompt_tokens
        expected_total_llm_time_fallback = mock_assessment_stats.call_duration_seconds + mock_fallback_stats.call_duration_seconds

        self.assertIn(f"Total Completion Tokens (All Calls): {expected_total_completion_fallback}", summary_str_fallback)
        self.assertIn(f"Total Prompt Tokens (All Calls): {expected_total_prompt_fallback}", summary_str_fallback)
        self.assertIn(f"Grand Total Tokens (All Calls): {expected_total_completion_fallback + expected_total_prompt_fallback}", summary_str_fallback)
        self.assertIn(f"Total LLM Interaction Time (All Calls): {expected_total_llm_time_fallback:.2f}s", summary_str_fallback)
