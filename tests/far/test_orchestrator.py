import pytest
from unittest.mock import MagicMock, patch, call

from src.llm_client import LLMClient
from src.llm_config import LLMConfig
from src.far.orchestrator import FaRProcess, FaROrchestrator
from src.far.processor import FaRProcessor
from src.far.dataclasses import FaRConfig, FaRResult, FaRSolution, LLMCallStats
from src.far.enums import FaRTriggerMode
from src.complexity_assessor import ComplexityAssessor, AssessmentDecision # Reusing AssessmentDecision

@pytest.fixture
def mock_llm_client_orchestrator():
    return MagicMock(spec=LLMClient)

@pytest.fixture
def far_base_config(): # Renamed to avoid conflict with processor test config name
    return FaRConfig(
        fact_model_names=["mock/fact-model-orch"],
        main_model_names=["mock/main-model-orch"]
    )

@pytest.fixture
def direct_oneshot_config():
    return LLMConfig(temperature=0.5, max_tokens=150)

@pytest.fixture
def direct_oneshot_models():
    return ["mock/direct-oneshot-model"]

# --- Tests for FaRProcess ---

@patch('src.far.orchestrator.FaRProcessor') # Mock the FaRProcessor within orchestrator module
def test_far_process_successful_execution(MockFaRProcessor, mock_llm_client_orchestrator, far_base_config, direct_oneshot_config, direct_oneshot_models):
    problem_desc = "Process this successfully."
    mock_processor_instance = MockFaRProcessor.return_value

    mock_far_result = FaRResult(
        succeeded=True,
        problem_description=problem_desc,
        elicited_facts="Facts A, B, C.",
        final_answer="Successful answer based on facts.",
        fact_call_stats=LLMCallStats(model_name="f-model", completion_tokens=10, prompt_tokens=5, call_duration_seconds=1.0),
        main_call_stats=LLMCallStats(model_name="m-model", completion_tokens=20, prompt_tokens=15, call_duration_seconds=1.5)
    )
    mock_processor_instance.run.return_value = mock_far_result

    far_process = FaRProcess(
        llm_client=mock_llm_client_orchestrator,
        far_config=far_base_config,
        direct_oneshot_llm_config=direct_oneshot_config,
        direct_oneshot_model_names=direct_oneshot_models
    )
    far_process.execute(problem_desc, model_name="test_model") # model_name is for API consistency
    solution, summary = far_process.get_result()

    assert solution is not None
    assert solution.final_answer == "Successful answer based on facts."
    assert solution.far_result == mock_far_result
    assert not solution.far_failed_and_fell_back
    assert solution.fallback_call_stats is None
    mock_processor_instance.run.assert_called_once_with(problem_desc)
    # Ensure direct one-shot fallback was NOT called by FaRProcess
    assert len([c for c in mock_llm_client_orchestrator.call.call_args_list if c[1].get('models') == direct_oneshot_models]) == 0


@patch('src.far.orchestrator.FaRProcessor')
def test_far_process_failure_and_fallback(MockFaRProcessor, mock_llm_client_orchestrator, far_base_config, direct_oneshot_config, direct_oneshot_models):
    problem_desc = "This will fail and fallback."
    mock_processor_instance = MockFaRProcessor.return_value

    mock_far_result_failed = FaRResult(
        succeeded=False,
        error_message="FaR processor failed badly.",
        elicited_facts="Partial facts." # Even if it fails, it might have some partial data
    )
    mock_processor_instance.run.return_value = mock_far_result_failed

    fallback_answer = "This is the fallback answer."
    fallback_stats = LLMCallStats(model_name=direct_oneshot_models[0], completion_tokens=30, prompt_tokens=25, call_duration_seconds=2.0)

    # Mock only the call that corresponds to the fallback
    mock_llm_client_orchestrator.call.return_value = (fallback_answer, fallback_stats)

    far_process = FaRProcess(
        llm_client=mock_llm_client_orchestrator,
        far_config=far_base_config,
        direct_oneshot_llm_config=direct_oneshot_config,
        direct_oneshot_model_names=direct_oneshot_models
    )
    far_process.execute(problem_desc, model_name="test_model")
    solution, summary = far_process.get_result()

    assert solution is not None
    assert solution.final_answer == fallback_answer
    assert solution.far_result == mock_far_result_failed
    assert solution.far_failed_and_fell_back
    assert solution.fallback_call_stats == fallback_stats
    assert "Partial facts" in solution.reasoning_trace[0]

    mock_processor_instance.run.assert_called_once_with(problem_desc)
    # Check that the fallback was called
    mock_llm_client_orchestrator.call.assert_called_once_with(
        prompt=problem_desc,
        models=direct_oneshot_models,
        config=direct_oneshot_config
    )

# --- Tests for FaROrchestrator ---

@pytest.fixture
def mock_complexity_assessor():
    return MagicMock(spec=ComplexityAssessor)

@patch('src.far.orchestrator.FaRProcess') # Mock FaRProcess used by FaROrchestrator
def test_far_orchestrator_always_far_mode(MockFaRProcessClass, mock_llm_client_orchestrator, far_base_config, direct_oneshot_config, direct_oneshot_models, mock_complexity_assessor):
    problem_desc = "Always use FaR for this."
    mock_far_process_instance = MockFaRProcessClass.return_value

    # Simulate FaRProcess.get_result()
    far_process_solution = FaRSolution(
        final_answer="FaR answer from ALWAYS_FAR",
        far_result=FaRResult(succeeded=True, final_answer="FaR answer from ALWAYS_FAR")
    )
    mock_far_process_instance.get_result.return_value = (far_process_solution, "FaR process summary")

    orchestrator = FaROrchestrator(
        llm_client=mock_llm_client_orchestrator,
        trigger_mode=FaRTriggerMode.ALWAYS_FAR,
        far_config=far_base_config,
        direct_oneshot_llm_config=direct_oneshot_config,
        direct_oneshot_model_names=direct_oneshot_models,
        assessment_llm_config=None, # Not used in this mode
        assessment_model_names=None # Not used
    )
    solution, summary = orchestrator.solve(problem_desc)

    assert solution.final_answer == "FaR answer from ALWAYS_FAR"
    MockFaRProcessClass.assert_called_once() # FaRProcess should be instantiated
    mock_far_process_instance.execute.assert_called_once_with(problem_description=problem_desc, model_name="default_far_model")
    mock_complexity_assessor.assess.assert_not_called()
    # Ensure orchestrator's direct one-shot was not called
    # This is tricky because FaRProcess might call it for its own fallback, so we check specific calls.
    # For this specific test, FaRProcess is mocked to succeed, so LLMClient shouldn't be called by FaRProcess fallback.
    # And Orchestrator itself shouldn't call it.
    mock_llm_client_orchestrator.call.assert_not_called()


def test_far_orchestrator_never_far_mode(mock_llm_client_orchestrator, far_base_config, direct_oneshot_config, direct_oneshot_models, mock_complexity_assessor):
    problem_desc = "Never use FaR for this."
    oneshot_answer = "Direct one-shot answer for NEVER_FAR"
    oneshot_stats = LLMCallStats(model_name=direct_oneshot_models[0], completion_tokens=10, prompt_tokens=5, call_duration_seconds=0.5)

    mock_llm_client_orchestrator.call.return_value = (oneshot_answer, oneshot_stats)

    orchestrator = FaROrchestrator(
        llm_client=mock_llm_client_orchestrator,
        trigger_mode=FaRTriggerMode.NEVER_FAR,
        far_config=far_base_config,
        direct_oneshot_llm_config=direct_oneshot_config,
        direct_oneshot_model_names=direct_oneshot_models
    )
    solution, summary = orchestrator.solve(problem_desc)

    assert solution.final_answer == oneshot_answer
    assert solution.main_call_stats == oneshot_stats # Check it's stored as main_call_stats
    assert solution.far_result is None
    mock_complexity_assessor.assess.assert_not_called()
    mock_llm_client_orchestrator.call.assert_called_once_with(
        prompt=problem_desc,
        models=direct_oneshot_models,
        config=direct_oneshot_config
    )

@patch('src.far.orchestrator.FaRProcess')
@patch('src.far.orchestrator.ComplexityAssessor')
def test_far_orchestrator_assess_first_use_far(MockComplexityAssessor, MockFaRProcessClass, mock_llm_client_orchestrator, far_base_config, direct_oneshot_config, direct_oneshot_models):
    problem_desc = "Assess this, then use FaR."

    mock_assessor_instance = MockComplexityAssessor.return_value
    mock_assessor_instance.assess.return_value = (
        AssessmentDecision.ADVANCED_REASONING,
        LLMCallStats(model_name="assess-model", completion_tokens=1, prompt_tokens=1, call_duration_seconds=0.2)
    )

    mock_far_process_instance = MockFaRProcessClass.return_value
    far_process_solution = FaRSolution(final_answer="FaR answer after assessment", far_result=FaRResult(succeeded=True, final_answer="FaR answer after assessment"))
    mock_far_process_instance.get_result.return_value = (far_process_solution, "FaR process summary")

    # Need to provide assessment configs for ComplexityAssessor instantiation
    assessment_config = LLMConfig(temperature=0.1)
    assessment_models = ["mock/assess-model"]

    orchestrator = FaROrchestrator(
        llm_client=mock_llm_client_orchestrator,
        trigger_mode=FaRTriggerMode.ASSESS_FIRST_FAR,
        far_config=far_base_config,
        direct_oneshot_llm_config=direct_oneshot_config,
        direct_oneshot_model_names=direct_oneshot_models,
        assessment_llm_config=assessment_config, # Provided
        assessment_model_names=assessment_models  # Provided
    )
    solution, summary = orchestrator.solve(problem_desc)

    assert solution.final_answer == "FaR answer after assessment"
    assert solution.assessment_stats is not None
    mock_assessor_instance.assess.assert_called_once_with(problem_desc)
    MockFaRProcessClass.assert_called_once()
    mock_far_process_instance.execute.assert_called_once()
    # Orchestrator's direct one-shot should not be called by itself
    # (FaRProcess fallback is part of the mocked FaRProcess execute)
    assert len([c for c in mock_llm_client_orchestrator.call.call_args_list if c[1].get('models') == direct_oneshot_models]) == 0


@patch('src.far.orchestrator.FaRProcess')
@patch('src.far.orchestrator.ComplexityAssessor')
def test_far_orchestrator_assess_first_use_oneshot(MockComplexityAssessor, MockFaRProcessClass, mock_llm_client_orchestrator, far_base_config, direct_oneshot_config, direct_oneshot_models):
    problem_desc = "Assess this, then use one-shot."

    mock_assessor_instance = MockComplexityAssessor.return_value
    mock_assessor_instance.assess.return_value = (
        AssessmentDecision.ONE_SHOT,
        LLMCallStats(model_name="assess-model", completion_tokens=1, prompt_tokens=1, call_duration_seconds=0.2)
    )

    oneshot_answer = "Direct one-shot answer after assessment."
    oneshot_stats = LLMCallStats(model_name=direct_oneshot_models[0], completion_tokens=10, prompt_tokens=5, call_duration_seconds=0.5)
    mock_llm_client_orchestrator.call.return_value = (oneshot_answer, oneshot_stats)

    assessment_config = LLMConfig(temperature=0.1)
    assessment_models = ["mock/assess-model"]

    orchestrator = FaROrchestrator(
        llm_client=mock_llm_client_orchestrator,
        trigger_mode=FaRTriggerMode.ASSESS_FIRST_FAR,
        far_config=far_base_config,
        direct_oneshot_llm_config=direct_oneshot_config,
        direct_oneshot_model_names=direct_oneshot_models,
        assessment_llm_config=assessment_config,
        assessment_model_names=assessment_models
    )
    solution, summary = orchestrator.solve(problem_desc)

    assert solution.final_answer == oneshot_answer
    assert solution.assessment_stats is not None
    assert solution.main_call_stats == oneshot_stats # Check it's stored as main_call_stats
    mock_assessor_instance.assess.assert_called_once_with(problem_desc)
    MockFaRProcessClass.assert_not_called() # FaRProcess should not be instantiated or called
    mock_llm_client_orchestrator.call.assert_called_once_with( # Orchestrator's direct one-shot
        prompt=problem_desc,
        models=direct_oneshot_models,
        config=direct_oneshot_config
    )
