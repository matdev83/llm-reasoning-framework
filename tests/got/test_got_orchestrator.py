import pytest
from unittest.mock import MagicMock, patch

from src.llm_client import LLMClient
from src.aot.enums import AssessmentDecision
from src.got.orchestrator import GoTOrchestrator, GoTProcess
from src.got.enums import GoTTriggerMode
from src.got.dataclasses import GoTConfig, GoTModelConfigs, GoTSolution, GoTResult
from src.llm_config import LLMConfig
from src.complexity_assessor import ComplexityAssessor # For mocking
from src.aot.dataclasses import LLMCallStats


@pytest.fixture
def mock_llm_client_for_orchestrator():
    client = MagicMock(spec=LLMClient)
    # Generic response for any call, specific tests can override
    mock_stats = LLMCallStats(
        model_name="mock_orch_model",
        call_duration_seconds=0.1,
        prompt_tokens=20,
        completion_tokens=30
    )
    client.call.return_value = ("Orchestrator mock response", mock_stats)
    return client

@pytest.fixture
def base_got_config(): # Renamed to avoid conflict if used in same file as processor tests
    return GoTConfig(max_iterations=2, max_thoughts=5) # Small values for testing

@pytest.fixture
def base_got_model_configs(): # Renamed
    return GoTModelConfigs()

@pytest.fixture
def direct_oneshot_config():
    return LLMConfig(temperature=0.1)

@pytest.fixture
def assessment_config():
    return LLMConfig(temperature=0.1)

def test_got_orchestrator_always_got_mode(mock_llm_client_for_orchestrator, base_got_config, base_got_model_configs, direct_oneshot_config):
    # Mock GoTProcess execution
    mock_got_process_instance = MagicMock(spec=GoTProcess)
    mock_solution = GoTSolution(
        final_answer="Success from ALWAYS_GOT",
        got_result=GoTResult(succeeded=True, final_answer="Success from ALWAYS_GOT")
    )
    mock_got_process_instance.get_result.return_value = (mock_solution, "GoTProcess summary")

    with patch('src.got.orchestrator.GoTProcess', return_value=mock_got_process_instance) as mock_GoTProcess_class:
        orchestrator = GoTOrchestrator(
            llm_client=mock_llm_client_for_orchestrator,
            trigger_mode=GoTTriggerMode.ALWAYS_GOT,
            got_config=base_got_config,
            got_model_configs=base_got_model_configs,
            direct_oneshot_llm_config=direct_oneshot_config,
            direct_oneshot_model_names=["oneshot/model"],
            assessment_llm_config=assessment_config, # Dummy for this mode
            assessment_model_names=["assess/model"]  # Dummy
        )
        solution, summary = orchestrator.solve("test problem")

        mock_GoTProcess_class.assert_called_once()
        mock_got_process_instance.execute.assert_called_once()
        assert solution.final_answer == "Success from ALWAYS_GOT"
        assert "GoTProcess summary" in summary
        assert solution.got_result is not None
        assert solution.got_result.succeeded

def test_got_orchestrator_never_got_mode(mock_llm_client_for_orchestrator, base_got_config, base_got_model_configs, direct_oneshot_config):
    # Ensure GoTProcess is NOT called, and direct one-shot is used.
    mock_stats_oneshot = LLMCallStats(model_name="direct_oneshot_model", call_duration_seconds=0.1, prompt_tokens=10, completion_tokens=20) # Corrected
    mock_llm_client_for_orchestrator.call.return_value = ("Direct one-shot answer", mock_stats_oneshot)

    orchestrator = GoTOrchestrator(
        llm_client=mock_llm_client_for_orchestrator,
        trigger_mode=GoTTriggerMode.NEVER_GOT,
        got_config=base_got_config,
        got_model_configs=base_got_model_configs,
        direct_oneshot_llm_config=direct_oneshot_config,
        direct_oneshot_model_names=["direct/model"],
            assessment_llm_config=assessment_config,
        assessment_model_names=["assess/model"]
    )

    # Patch GoTProcess to ensure it's not instantiated or called
    with patch('src.got.orchestrator.GoTProcess') as mock_GoTProcess_class:
        solution, summary = orchestrator.solve("test problem for never_got")

        mock_GoTProcess_class.assert_not_called()
        mock_llm_client_for_orchestrator.call.assert_called_once() # The orchestrator's direct call
        assert solution.final_answer == "Direct one-shot answer"
        assert solution.got_result is None
        assert solution.fallback_call_stats is not None # Stats from orchestrator's one-shot

@patch('src.complexity_assessor.ComplexityAssessor.assess')
def test_got_orchestrator_assess_first_got_chooses_got(mock_assess_method, mock_llm_client_for_orchestrator, base_got_config, base_got_model_configs, direct_oneshot_config, assessment_config):
    # Assessor decides ADVANCED_REASONING
    mock_assess_stats = LLMCallStats(model_name="assess_model", call_duration_seconds=0.05, prompt_tokens=5, completion_tokens=5) # Corrected (assuming 10 was total)
    mock_assess_method.return_value = (AssessmentDecision.ADVANCED_REASONING, mock_assess_stats)

    # Mock GoTProcess execution
    mock_got_process_instance = MagicMock(spec=GoTProcess)
    mock_solution_got = GoTSolution(
        final_answer="Success from GoT after assessment",
        got_result=GoTResult(succeeded=True, final_answer="Success from GoT after assessment")
    )
    mock_got_process_instance.get_result.return_value = (mock_solution_got, "GoTProcess advanced summary")

    with patch('src.got.orchestrator.GoTProcess', return_value=mock_got_process_instance) as mock_GoTProcess_class:
        orchestrator = GoTOrchestrator(
            llm_client=mock_llm_client_for_orchestrator,
            trigger_mode=GoTTriggerMode.ASSESS_FIRST_GOT,
            got_config=base_got_config,
            got_model_configs=base_got_model_configs,
            direct_oneshot_llm_config=direct_oneshot_config,
            direct_oneshot_model_names=["oneshot/model"],
            assessment_llm_config=assessment_config,
            assessment_model_names=["assess/model"]
        )
        solution, summary = orchestrator.solve("test problem assess advanced")

        mock_assess_method.assert_called_once()
        mock_GoTProcess_class.assert_called_once()
        mock_got_process_instance.execute.assert_called_once()
        assert solution.final_answer == "Success from GoT after assessment"
        assert solution.assessment_stats == mock_assess_stats
        assert solution.assessment_decision == AssessmentDecision.ADVANCED_REASONING

@patch('src.complexity_assessor.ComplexityAssessor.assess')
def test_got_orchestrator_assess_first_got_chooses_oneshot(mock_assess_method, mock_llm_client_for_orchestrator, base_got_config, base_got_model_configs, direct_oneshot_config, assessment_config):
    # Assessor decides ONE_SHOT
    mock_assess_stats = LLMCallStats(model_name="assess_model", call_duration_seconds=0.05, prompt_tokens=5, completion_tokens=5) # Corrected
    mock_assess_method.return_value = (AssessmentDecision.ONE_SHOT, mock_assess_stats)

    # Mock orchestrator's direct one-shot call
    mock_oneshot_stats = LLMCallStats(model_name="direct_oneshot_model", call_duration_seconds=0.1, prompt_tokens=10, completion_tokens=20) # Corrected
    # Important: the llm_client mock is used by both assessor and orchestrator's one-shot.
    # We need to make sure the side_effect list handles calls in order.
    # First call by assessor (mocked by mock_assess_method, doesn't use client.call directly in this test structure)
    # Second call by orchestrator for one-shot
    mock_llm_client_for_orchestrator.call.return_value = ("Assessed one-shot answer", mock_oneshot_stats)

    with patch('src.got.orchestrator.GoTProcess') as mock_GoTProcess_class: # Ensure GoTProcess not called
        orchestrator = GoTOrchestrator(
            llm_client=mock_llm_client_for_orchestrator,
            trigger_mode=GoTTriggerMode.ASSESS_FIRST_GOT,
            got_config=base_got_config,
            got_model_configs=base_got_model_configs,
            direct_oneshot_llm_config=direct_oneshot_config,
            direct_oneshot_model_names=["direct/model"],
            assessment_llm_config=assessment_config,
            assessment_model_names=["assess/model"]
        )
        solution, summary = orchestrator.solve("test problem assess oneshot")

        mock_assess_method.assert_called_once()
        mock_GoTProcess_class.assert_not_called() # GoTProcess should not be involved
        mock_llm_client_for_orchestrator.call.assert_called_once() # Orchestrator's one-shot
        assert solution.final_answer == "Assessed one-shot answer"
        assert solution.assessment_stats == mock_assess_stats
        assert solution.assessment_decision == AssessmentDecision.ONE_SHOT
        assert solution.fallback_call_stats == mock_oneshot_stats # Check orchestrator's one-shot stats

# TODO: Add tests for GoTProcess internal fallback if GoTProcessor fails within ALWAYS_GOT.
# TODO: Add tests for heuristic shortcut in ASSESS_FIRST_GOT.
