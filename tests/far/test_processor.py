import pytest
from unittest.mock import MagicMock, patch, call

from src.llm_client import LLMClient
from src.llm_config import LLMConfig
from src.far.processor import FaRProcessor
from src.far.dataclasses import FaRConfig, FaRResult, LLMCallStats

# Mock prompt file content
MOCK_FACT_PROMPT_TEMPLATE = "Fact prompt for {problem_description}"
MOCK_REFLECTION_PROMPT_TEMPLATE = "Reflection for {problem_description} with facts {elicited_facts}"

@pytest.fixture
def mock_llm_client():
    return MagicMock(spec=LLMClient)

@pytest.fixture
def far_config():
    return FaRConfig(
        fact_model_names=["mock/fact-model"],
        main_model_names=["mock/main-model"],
        fact_model_temperature=0.1,
        main_model_temperature=0.8,
        max_fact_tokens=50,
        max_main_tokens=100
    )

@patch('src.far.processor.FaRProcessor._load_prompt_template')
def test_far_processor_successful_run(mock_load_template, mock_llm_client, far_config):
    mock_load_template.side_effect = lambda template_name: MOCK_FACT_PROMPT_TEMPLATE if template_name == "far_fact_elicitation.txt" else MOCK_REFLECTION_PROMPT_TEMPLATE

    problem_desc = "What is the capital of France and its population?"
    expected_facts = "Capital: Paris, Population: ~2.1 million"
    expected_answer = "The capital of France is Paris, with a population of approximately 2.1 million."

    mock_llm_client.call.side_effect = [
        (expected_facts, LLMCallStats(model_name="mock/fact-model", completion_tokens=10, prompt_tokens=5, call_duration_seconds=1.0)),
        (expected_answer, LLMCallStats(model_name="mock/main-model", completion_tokens=20, prompt_tokens=15, call_duration_seconds=2.0))
    ]

    processor = FaRProcessor(llm_client=mock_llm_client, config=far_config)
    result = processor.run(problem_desc)

    assert result.succeeded
    assert result.problem_description == problem_desc
    assert result.elicited_facts == expected_facts
    assert result.final_answer == expected_answer
    assert result.error_message is None

    assert result.fact_call_stats is not None
    assert result.fact_call_stats.model_name == "mock/fact-model"
    assert result.main_call_stats is not None
    assert result.main_call_stats.model_name == "mock/main-model"

    assert result.total_completion_tokens == 30
    assert result.total_prompt_tokens == 20
    assert result.total_llm_interaction_time_seconds == 3.0

    # Check calls to LLM client
    expected_fact_prompt = MOCK_FACT_PROMPT_TEMPLATE.format(problem_description=problem_desc)
    expected_reflection_prompt = MOCK_REFLECTION_PROMPT_TEMPLATE.format(problem_description=problem_desc, elicited_facts=expected_facts)

    fact_llm_config_expected = LLMConfig(temperature=far_config.fact_model_temperature, max_tokens=far_config.max_fact_tokens)
    main_llm_config_expected = LLMConfig(temperature=far_config.main_model_temperature, max_tokens=far_config.max_main_tokens)

    calls = mock_llm_client.call.call_args_list
    assert len(calls) == 2
    # First call (fact elicitation)
    args, kwargs = calls[0]
    assert kwargs['prompt'] == expected_fact_prompt
    assert kwargs['models'] == far_config.fact_model_names
    assert kwargs['config'] == fact_llm_config_expected
    # Second call (reflection/answer)
    args, kwargs = calls[1]
    assert kwargs['prompt'] == expected_reflection_prompt
    assert kwargs['models'] == far_config.main_model_names
    assert kwargs['config'] == main_llm_config_expected

    # Check prompt loading calls
    mock_load_template.assert_any_call("far_fact_elicitation.txt")
    mock_load_template.assert_any_call("far_reflection_answer.txt")


@patch('src.far.processor.FaRProcessor._load_prompt_template')
def test_far_processor_fact_elicitation_failure(mock_load_template, mock_llm_client, far_config):
    mock_load_template.return_value = MOCK_FACT_PROMPT_TEMPLATE
    problem_desc = "Test problem"
    error_message_from_llm = "Error: Fact model unavailable"

    mock_llm_client.call.return_value = (
        error_message_from_llm,
        LLMCallStats(model_name="mock/fact-model", completion_tokens=0, prompt_tokens=5, call_duration_seconds=0.5)
    )

    processor = FaRProcessor(llm_client=mock_llm_client, config=far_config)
    result = processor.run(problem_desc)

    assert not result.succeeded
    assert result.elicited_facts is None
    assert result.final_answer is None
    assert result.error_message == f"Fact elicitation LLM call failed: {error_message_from_llm}"
    assert result.fact_call_stats is not None
    assert result.main_call_stats is None # Should not be called
    mock_llm_client.call.assert_called_once() # Only fact model should be called

@patch('src.far.processor.FaRProcessor._load_prompt_template')
def test_far_processor_reflection_failure(mock_load_template, mock_llm_client, far_config):
    mock_load_template.side_effect = lambda template_name: MOCK_FACT_PROMPT_TEMPLATE if template_name == "far_fact_elicitation.txt" else MOCK_REFLECTION_PROMPT_TEMPLATE
    problem_desc = "Test problem"
    elicited_facts = "Some facts were found."
    error_message_from_llm = "Error: Main model overload"

    mock_llm_client.call.side_effect = [
        (elicited_facts, LLMCallStats(model_name="mock/fact-model", completion_tokens=10, prompt_tokens=5, call_duration_seconds=1.0)),
        (error_message_from_llm, LLMCallStats(model_name="mock/main-model", completion_tokens=0, prompt_tokens=15, call_duration_seconds=0.5))
    ]

    processor = FaRProcessor(llm_client=mock_llm_client, config=far_config)
    result = processor.run(problem_desc)

    assert not result.succeeded
    assert result.elicited_facts == elicited_facts
    assert result.final_answer is None
    assert result.error_message == f"Reflection/Answer LLM call failed: {error_message_from_llm}"
    assert result.fact_call_stats is not None
    assert result.main_call_stats is not None
    assert mock_llm_client.call.call_count == 2

@patch('src.far.processor.FaRProcessor._load_prompt_template')
def test_far_processor_prompt_template_load_failure(mock_load_template, mock_llm_client, far_config):
    mock_load_template.side_effect = FileNotFoundError("Prompt file not found")
    problem_desc = "Test problem"

    processor = FaRProcessor(llm_client=mock_llm_client, config=far_config)

    result = processor.run(problem_desc)
    # The processor catches the exception and returns a failed result
    assert not result.succeeded
    assert "Exception in fact elicitation phase" in result.error_message
    assert "Prompt file not found" in result.error_message
    mock_llm_client.call.assert_not_called()


# Realistic question examples for testing prompt construction and data flow
@patch('src.far.processor.FaRProcessor._load_prompt_template')
@pytest.mark.parametrize("problem_desc, mock_facts, mock_answer", [
    (
        "Who won the last FIFA World Cup and what was the score in the final?",
        "Last FIFA World Cup winner: Argentina. Score: Argentina 3 - 3 France (Argentina won 4-2 on penalties).",
        "Argentina won the last FIFA World Cup, defeating France in the final. The score was 3-3 after extra time, and Argentina won 4-2 on penalties."
    ),
    (
        "What is the boiling point of water at the summit of Mount Everest in Celsius?",
        "Mount Everest height: approx 8848m. Boiling point of water decreases with altitude. At 8848m, boiling point is around 70-71°C.",
        "The boiling point of water at the summit of Mount Everest (approximately 8848 meters) is around 70-71 degrees Celsius, significantly lower than the 100°C at sea level due to reduced atmospheric pressure."
    )
])
def test_far_processor_realistic_questions(mock_load_template, mock_llm_client, far_config, problem_desc, mock_facts, mock_answer):
    mock_load_template.side_effect = lambda template_name: MOCK_FACT_PROMPT_TEMPLATE if template_name == "far_fact_elicitation.txt" else MOCK_REFLECTION_PROMPT_TEMPLATE

    mock_llm_client.call.side_effect = [
        (mock_facts, LLMCallStats(model_name="mock/fact-model", completion_tokens=10, prompt_tokens=5, call_duration_seconds=1.0)),
        (mock_answer, LLMCallStats(model_name="mock/main-model", completion_tokens=20, prompt_tokens=15, call_duration_seconds=2.0))
    ]

    processor = FaRProcessor(llm_client=mock_llm_client, config=far_config)
    result = processor.run(problem_desc)

    assert result.succeeded
    assert result.elicited_facts == mock_facts
    assert result.final_answer == mock_answer

    expected_fact_prompt = MOCK_FACT_PROMPT_TEMPLATE.format(problem_description=problem_desc)
    expected_reflection_prompt = MOCK_REFLECTION_PROMPT_TEMPLATE.format(problem_description=problem_desc, elicited_facts=mock_facts)

    calls = mock_llm_client.call.call_args_list
    assert calls[0][1]['prompt'] == expected_fact_prompt # prompt text for fact call
    assert calls[1][1]['prompt'] == expected_reflection_prompt # prompt text for reflection call

@patch('src.far.processor.FaRProcessor._load_prompt_template')
def test_far_processor_token_budget_limit(mock_load_template, mock_llm_client, far_config):
    """Test that FaR processor respects token budget limits."""
    mock_load_template.side_effect = lambda template_name: MOCK_FACT_PROMPT_TEMPLATE if template_name == "far_fact_elicitation.txt" else MOCK_REFLECTION_PROMPT_TEMPLATE
    
    # Set a low token budget
    far_config.max_reasoning_tokens = 50
    
    problem_desc = "Token budget test problem"
    elicited_facts = "Some facts"
    
    # Mock fact call to consume many tokens
    high_token_stats = LLMCallStats(
        model_name="mock/fact-model", 
        completion_tokens=60,  # Exceeds budget
        prompt_tokens=5, 
        call_duration_seconds=1.0
    )
    
    mock_llm_client.call.return_value = (elicited_facts, high_token_stats)
    
    processor = FaRProcessor(llm_client=mock_llm_client, config=far_config)
    result = processor.run(problem_desc)
    
    # Should fail due to token budget exceeded after fact elicitation
    assert not result.succeeded
    assert "Token limit" in result.error_message
    assert result.elicited_facts == elicited_facts
    assert result.final_answer is None
    assert result.reasoning_completion_tokens == 60
    assert result.fact_call_stats is not None
    assert result.main_call_stats is None  # Should not reach reflection phase
    mock_llm_client.call.assert_called_once()  # Only fact call should be made

@patch('src.far.processor.FaRProcessor._load_prompt_template')
def test_far_processor_time_budget_limit(mock_load_template, mock_llm_client, far_config):
    """Test that FaR processor respects time budget limits."""
    mock_load_template.side_effect = lambda template_name: MOCK_FACT_PROMPT_TEMPLATE if template_name == "far_fact_elicitation.txt" else MOCK_REFLECTION_PROMPT_TEMPLATE
    
    # Set a very short time budget
    far_config.max_time_seconds = 1
    
    problem_desc = "Time budget test problem"
    elicited_facts = "Some facts"
    
    # Mock fact call to take a long time
    slow_stats = LLMCallStats(
        model_name="mock/fact-model", 
        completion_tokens=10,
        prompt_tokens=5, 
        call_duration_seconds=1.5  # Exceeds budget
    )
    
    def slow_call(*args, **kwargs):
        import time
        time.sleep(1.2)  # Simulate slow processing
        return elicited_facts, slow_stats
    
    mock_llm_client.call.side_effect = slow_call
    
    processor = FaRProcessor(llm_client=mock_llm_client, config=far_config)
    result = processor.run(problem_desc)
    
    # Should fail due to time budget exceeded after fact elicitation
    assert not result.succeeded
    assert "Time limit" in result.error_message
    assert result.elicited_facts == elicited_facts
    assert result.final_answer is None
    assert result.reasoning_completion_tokens == 10
    assert result.fact_call_stats is not None
    assert result.main_call_stats is None  # Should not reach reflection phase
    mock_llm_client.call.assert_called_once()  # Only fact call should be made

@patch('src.far.processor.FaRProcessor._load_prompt_template')
def test_far_processor_reasoning_token_tracking(mock_load_template, mock_llm_client, far_config):
    """Test that FaR processor correctly tracks reasoning tokens."""
    mock_load_template.side_effect = lambda template_name: MOCK_FACT_PROMPT_TEMPLATE if template_name == "far_fact_elicitation.txt" else MOCK_REFLECTION_PROMPT_TEMPLATE
    
    problem_desc = "Token tracking test"
    elicited_facts = "Some facts"
    final_answer = "Some answer"
    
    # Mock calls with predictable token counts
    fact_stats = LLMCallStats(model_name="mock/fact-model", completion_tokens=25, prompt_tokens=10, call_duration_seconds=1.0)
    main_stats = LLMCallStats(model_name="mock/main-model", completion_tokens=35, prompt_tokens=15, call_duration_seconds=2.0)
    
    mock_llm_client.call.side_effect = [
        (elicited_facts, fact_stats),
        (final_answer, main_stats)
    ]
    
    processor = FaRProcessor(llm_client=mock_llm_client, config=far_config)
    result = processor.run(problem_desc)
    
    # Should track reasoning tokens correctly
    expected_reasoning_tokens = fact_stats.completion_tokens + main_stats.completion_tokens
    assert result.succeeded
    assert result.reasoning_completion_tokens == expected_reasoning_tokens
    assert result.total_completion_tokens == expected_reasoning_tokens
    assert result.total_prompt_tokens == fact_stats.prompt_tokens + main_stats.prompt_tokens
    assert result.fact_call_stats is not None
    assert result.main_call_stats is not None

@patch('src.far.processor.FaRProcessor._load_prompt_template')
def test_far_processor_successful_within_budget(mock_load_template, mock_llm_client, far_config):
    """Test that FaR processor completes successfully when within budget constraints."""
    mock_load_template.side_effect = lambda template_name: MOCK_FACT_PROMPT_TEMPLATE if template_name == "far_fact_elicitation.txt" else MOCK_REFLECTION_PROMPT_TEMPLATE
    
    # Set reasonable budgets
    far_config.max_reasoning_tokens = 100
    far_config.max_time_seconds = 10
    
    problem_desc = "Budget test problem"
    elicited_facts = "Some facts"
    final_answer = "Some answer"
    
    # Mock calls within budget
    fact_stats = LLMCallStats(model_name="mock/fact-model", completion_tokens=20, prompt_tokens=10, call_duration_seconds=1.0)
    main_stats = LLMCallStats(model_name="mock/main-model", completion_tokens=30, prompt_tokens=15, call_duration_seconds=2.0)
    
    mock_llm_client.call.side_effect = [
        (elicited_facts, fact_stats),
        (final_answer, main_stats)
    ]
    
    processor = FaRProcessor(llm_client=mock_llm_client, config=far_config)
    result = processor.run(problem_desc)
    
    # Should complete successfully
    assert result.succeeded
    assert result.elicited_facts == elicited_facts
    assert result.final_answer == final_answer
    assert result.reasoning_completion_tokens == 50  # 20 + 30
    assert result.reasoning_completion_tokens < far_config.max_reasoning_tokens
    assert result.total_process_wall_clock_time_seconds < far_config.max_time_seconds
    assert result.fact_call_stats is not None
    assert result.main_call_stats is not None
    assert mock_llm_client.call.call_count == 2  # Both calls should be made
