import pytest
from unittest.mock import MagicMock, patch

from src.llm_client import LLMClient
from src.aot.dataclasses import LLMCallStats
from src.got.processor import GoTProcessor
from src.got.dataclasses import GoTConfig, GoTModelConfigs, GoTGraph, GoTThought, GoTThoughtStatus, GoTResult
from src.llm_config import LLMConfig

@pytest.fixture
def mock_llm_client(monkeypatch): # Removed self, monkeypatch is a pytest fixture
    client = MagicMock(spec=LLMClient)
    # Ensure 'call' can be called and returns a tuple (content, stats_obj)
    # The content will be varied per test. stats_obj can be a generic mock.

    def mock_call(prompt, models, config):
        # Default mock response, tests can override this using client.call.return_value
        mock_stats = LLMCallStats(
            model_name=models[0] if models else "mock_model",
            call_duration_seconds=0.1,
            prompt_tokens=len(prompt.split()), # Simple token estimation
            completion_tokens=50, # Default completion tokens
            total_tokens=len(prompt.split()) + 50
        )
        return "Default mock response", mock_stats

    client.call = MagicMock(side_effect=mock_call) # Use side_effect for more flexibility
    return client

@pytest.fixture
def got_config(): # Removed self
    return GoTConfig(
        max_thoughts=10, # Smaller limits for testing
        max_iterations=3,
        min_score_for_expansion=0.5,
        pruning_threshold_score=0.2,
        solution_found_score_threshold=0.9,
        max_time_seconds=30 # Short time limit for tests
    )

@pytest.fixture
def got_model_configs(): # Removed self
    return GoTModelConfigs(
        thought_generation_config=LLMConfig(temperature=0.1),
        scoring_config=LLMConfig(temperature=0.1),
        aggregation_config=LLMConfig(temperature=0.1),
        refinement_config=LLMConfig(temperature=0.1)
    )

def test_got_processor_initialization(mock_llm_client, got_config, got_model_configs): # Removed self
    processor = GoTProcessor(
        llm_client=mock_llm_client,
        config=got_config,
        model_configs=got_model_configs
    )
    assert processor is not None
    assert processor.config == got_config

def test_initial_thought_generation(mock_llm_client, got_config, got_model_configs): # Removed self
    processor = GoTProcessor(mock_llm_client, got_config, got_model_configs)
    problem_description = "Test problem for initial thoughts."

    # Mock the LLM response for initial thought generation
    initial_thought_response = "Thought: Initial idea 1\nThought: Initial idea 2"
    score_response = "Score: 0.8\nJustification: Good start"

    # Set up side effects for multiple calls if _generate_initial_thoughts makes >1 call (e.g. gen then score)
    mock_stats_gen = LLMCallStats(model_name="gen_model", call_duration_seconds=0.1, prompt_tokens=10, completion_tokens=20)
    mock_stats_score = LLMCallStats(model_name="score_model", call_duration_seconds=0.05, prompt_tokens=5, completion_tokens=10)
    default_fallback_stats = LLMCallStats(model_name="fallback_model", call_duration_seconds=0.02, prompt_tokens=2, completion_tokens=2)

    responses_for_initial_gen = [
        (initial_thought_response, mock_stats_gen),
        (score_response, mock_stats_score),
        (score_response, mock_stats_score)
    ]

    def mock_call_for_initial_gen(*args, **kwargs):
        if responses_for_initial_gen:
            return responses_for_initial_gen.pop(0)
        # Fallback for any subsequent calls during iterations not specifically tested here
        prompt_text = kwargs.get('prompt', '')
        if "Score:" in prompt_text:
            return "Score: 0.1\nJustification: Default fallback score", default_fallback_stats
        return "Default fallback thought", default_fallback_stats

    mock_llm_client.call.side_effect = mock_call_for_initial_gen

    result = processor.run(problem_description)

    assert result.succeeded or not result.error_message # Allow success or graceful non-success if no answer
    assert result.final_graph is not None
    # Check if initial thoughts were added and scored (approx count, depends on mocking)
    initial_active_thoughts = [t for t in result.final_graph.thoughts.values() if t.generation_step == 0]
    assert len(initial_active_thoughts) == 2
    for thought in initial_active_thoughts:
        assert thought.score > 0 # Assuming score_response gives a positive score

    # Verify LLM was called multiple times (gen + 2 scores)
    assert mock_llm_client.call.call_count >= 3
    assert result.total_llm_calls >= 3

def test_thought_expansion(mock_llm_client, got_config, got_model_configs): # Removed self
    processor = GoTProcessor(mock_llm_client, got_config, got_model_configs)
    problem_description = "Test problem for expansion."

    # Initial setup: one initial thought
    initial_thought_content = "Initial thought to expand."
    initial_thought_id = processor._generate_thought_id()
    initial_thought = GoTThought(id=initial_thought_id, content=initial_thought_content, score=0.7, generation_step=0)

    # Mock responses
    # 1. Initial generation (skipped by direct graph setup for this test, but run() calls it)
    initial_gen_resp = f"Thought: {initial_thought_content}"
    # 2. Score for initial thought
    initial_score_resp = "Score: 0.7\nJustification: Good initial thought"
    # 3. Expansion response
    expansion_resp = "NewThought: Expanded idea A\nNewThought: Expanded idea B"
    # 4. Score for expanded idea A
    score_expanded_A_resp = "Score: 0.6\nJustification: Ok expansion"
    # 5. Score for expanded idea B
    score_expanded_B_resp = "Score: 0.65\nJustification: Better expansion"

    mock_stats_initial = LLMCallStats(model_name="initial_model", call_duration_seconds=0.1, prompt_tokens=10, completion_tokens=10)
    mock_stats_expansion_content = LLMCallStats(model_name="expansion_model", call_duration_seconds=0.1, prompt_tokens=10, completion_tokens=10)
    mock_stats_expansion_score = LLMCallStats(model_name="score_expansion_model", call_duration_seconds=0.05, prompt_tokens=5, completion_tokens=5)
    default_fallback_stats_expansion = LLMCallStats(model_name="fallback_expansion_model", call_duration_seconds=0.02, prompt_tokens=2, completion_tokens=2)

    responses_for_expansion = [
        (initial_gen_resp, mock_stats_initial),
        (initial_score_resp, mock_stats_initial),
        (expansion_resp, mock_stats_expansion_content),
        (score_expanded_A_resp, mock_stats_expansion_score),
        (score_expanded_B_resp, mock_stats_expansion_score)
    ]

    def mock_call_for_expansion(*args, **kwargs):
        if responses_for_expansion:
            return responses_for_expansion.pop(0)
        # Fallback for any subsequent calls
        prompt_text = kwargs.get('prompt', '')
        if "Score:" in prompt_text:
            return "Score: 0.1\nJustification: Default fallback score for expansion", default_fallback_stats_expansion
        return "Default fallback expansion thought", default_fallback_stats_expansion

    mock_llm_client.call.side_effect = mock_call_for_expansion

    # processor.run will handle the full flow
    result = processor.run(problem_description)

    assert result.succeeded or not result.error_message
    assert result.final_graph is not None

    parent_thought_in_graph = result.final_graph.get_thought(initial_thought_id)
    # Note: ID will be different as it's generated inside run(). We need to find it.
    found_initial_thought = None
    for t_id, thought_in_graph in result.final_graph.thoughts.items():
        if thought_in_graph.content == initial_thought_content:
            found_initial_thought = thought_in_graph
            break

    assert found_initial_thought is not None, "Initial thought not found in graph"
    assert len(found_initial_thought.children_ids) > 0

    children_found = 0
    for child_id in found_initial_thought.children_ids:
        child_thought = result.final_graph.get_thought(child_id)
        assert child_thought is not None
        assert child_thought.score > 0
        children_found +=1
    assert children_found == 2 # Expecting two expanded thoughts

# TODO: Add tests for aggregation, refinement, pruning, max_iterations, max_thoughts, solution finding.

def test_solution_candidate_identification(mock_llm_client, got_config, got_model_configs): # Removed self
    processor = GoTProcessor(mock_llm_client, got_config, got_model_configs)
    problem = "problem that leads to a solution"

    # Mock LLM calls:
    # 1. Initial thought generation
    # 2. Score for initial thought (make it high to become a solution candidate)
    mock_stats_solution = LLMCallStats(model_name="model", call_duration_seconds=0.1, prompt_tokens=10, completion_tokens=10) # Corrected
    mock_llm_client.call.side_effect = [
        ("Thought: This is the perfect solution", mock_stats_solution), # Initial thought
        (f"Score: {got_config.solution_found_score_threshold + 0.05}\nJustification: Perfect!", mock_stats_solution) # Score
    ]

    result = processor.run(problem)
    assert result.succeeded
    assert result.final_answer == "This is the perfect solution"
    assert len(result.solution_candidates) == 1
    assert result.solution_candidates[0].content == "This is the perfect solution"
    assert result.solution_candidates[0].status == GoTThoughtStatus.SOLUTION_CANDIDATE

def test_max_iterations_limit(mock_llm_client, got_config, got_model_configs): # Removed self
    # Ensure process stops after max_iterations
    processor = GoTProcessor(mock_llm_client, got_config, got_model_configs)
    problem = "iteration limit test"

    # Setup LLM to always generate new, scorable thoughts to keep iterations going
    def iterative_llm_response(*args, **kwargs):
        prompt_text = kwargs['prompt'] # Corrected: access prompt via kwargs
        stats = LLMCallStats(model_name="iter_model", call_duration_seconds=0.01, prompt_tokens=5, completion_tokens=5) # Corrected LLMCallStats
        if "Initial thought" in prompt_text or "NewThought:" in prompt_text or "Thought:" in prompt_text : # generation
            return "Thought: Another thought", stats
        elif "Score:" in prompt_text: # scoring
            return "Score: 0.6\nJustification: Keep going", stats
        return "Default response", stats # Fallback for other prompt types if any

    mock_llm_client.call.side_effect = iterative_llm_response

    # processor.config.max_iterations is already set by got_config fixture (e.g., to 3)
    result = processor.run(problem)

    # The number of LLM calls can be complex due to scoring each new thought.
    # Check that the number of generation steps in graph doesn't exceed max_iterations
    max_gen_step_found = 0
    if result.final_graph:
        for thought in result.final_graph.thoughts.values():
            if thought.generation_step > max_gen_step_found:
                max_gen_step_found = thought.generation_step

    # generation_step is 0-indexed for initial, then 1-indexed for iterations
    # So, max_gen_step_found should be <= config.max_iterations
    # Iterations run from 0 to max_iterations-1. Iteration number passed to expand is current_iter + 1.
    # So generation_step can go up to max_iterations.
    assert max_gen_step_found <= got_config.max_iterations
    # This assertion primarily checks that it doesn't run away indefinitely.
    # Exact number of iterations might be tricky if other limits are hit first.
    # For this test, ensure other limits (max_thoughts, time) are high enough not to interfere.
    # got_config.max_thoughts is 10, max_iterations is 3. Each iter might add 1-2 thoughts.
    # This should hit max_iterations.

# Add more tests for pruning, aggregation, refinement, error handling, etc.
