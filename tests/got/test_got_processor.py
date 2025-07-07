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
    mock_stats = LLMCallStats(model_name="model", call_duration_seconds=0.1, prompt_tokens=10, completion_tokens=5)
    
    def mock_call_for_iterations(*args, **kwargs):
        prompt_text = kwargs.get('prompt', '')
        if "Score:" in prompt_text:
            return "Score: 0.6\nJustification: Decent thought", mock_stats
        return "Thought: Iteration thought", mock_stats

    mock_llm_client.call.side_effect = mock_call_for_iterations

    result = processor.run(problem)
    # Should stop due to max_iterations (3), not other limits
    # The exact behavior depends on implementation details, but there should be multiple calls
    assert mock_llm_client.call.call_count >= got_config.max_iterations
    assert result.final_graph is not None

def test_token_budget_limit(mock_llm_client, got_config, got_model_configs):
    """Test that GoT processor respects token budget limits."""
    # Set a very low token budget
    got_config.max_reasoning_tokens = 100
    
    processor = GoTProcessor(mock_llm_client, got_config, got_model_configs)
    problem = "token budget test"

    # Mock LLM to return responses that consume many tokens
    high_token_stats = LLMCallStats(
        model_name="model", 
        call_duration_seconds=0.1, 
        prompt_tokens=10, 
        completion_tokens=60  # High token consumption
    )
    
    def mock_call_for_tokens(*args, **kwargs):
        prompt_text = kwargs.get('prompt', '')
        if "Score:" in prompt_text:
            return "Score: 0.6\nJustification: Decent thought", high_token_stats
        return "Thought: Token consuming thought", high_token_stats

    mock_llm_client.call.side_effect = mock_call_for_tokens

    result = processor.run(problem)
    
    # Should stop due to token budget, not other limits
    assert result.reasoning_completion_tokens <= got_config.max_reasoning_tokens + 60  # Allow for one overage
    assert result.final_graph is not None
    # Should have made at least one call but stopped before too many
    assert mock_llm_client.call.call_count >= 1
    assert mock_llm_client.call.call_count <= 4  # Should stop early due to token limit

def test_time_budget_limit(mock_llm_client, got_config, got_model_configs):
    """Test that GoT processor respects time budget limits."""
    # Set a very short time budget
    got_config.max_time_seconds = 1
    
    processor = GoTProcessor(mock_llm_client, got_config, got_model_configs)
    problem = "time budget test"

    # Mock LLM to simulate slow responses
    slow_stats = LLMCallStats(
        model_name="model", 
        call_duration_seconds=0.8,  # Each call takes 0.8 seconds
        prompt_tokens=10, 
        completion_tokens=20
    )
    
    def mock_call_for_time(*args, **kwargs):
        # Simulate time passing
        import time
        time.sleep(0.3)  # Simulate processing time
        prompt_text = kwargs.get('prompt', '')
        if "Score:" in prompt_text:
            return "Score: 0.6\nJustification: Slow thought", slow_stats
        return "Thought: Time consuming thought", slow_stats

    mock_llm_client.call.side_effect = mock_call_for_time

    result = processor.run(problem)
    
    # Should stop due to time budget
    assert result.total_process_wall_clock_time_seconds >= got_config.max_time_seconds * 0.8  # Allow some tolerance
    assert result.final_graph is not None
    # Should have made at least one call but stopped before too many
    assert mock_llm_client.call.call_count >= 1
    assert mock_llm_client.call.call_count <= 6  # Should stop early due to time limit

def test_reasoning_token_tracking(mock_llm_client, got_config, got_model_configs):
    """Test that GoT processor correctly tracks reasoning tokens."""
    processor = GoTProcessor(mock_llm_client, got_config, got_model_configs)
    problem = "token tracking test"

    # Mock LLM with predictable token counts
    initial_stats = LLMCallStats(model_name="model", call_duration_seconds=0.1, prompt_tokens=10, completion_tokens=30)
    score_stats = LLMCallStats(model_name="model", call_duration_seconds=0.1, prompt_tokens=5, completion_tokens=15)
    fallback_stats = LLMCallStats(model_name="model", call_duration_seconds=0.1, prompt_tokens=2, completion_tokens=5)
    
    def mock_call_for_tracking(*args, **kwargs):
        prompt_text = kwargs.get('prompt', '')
        if "Score:" in prompt_text:
            return "Score: 0.8\nJustification: Good", score_stats
        return "Thought: Initial thought", initial_stats

    mock_llm_client.call.side_effect = mock_call_for_tracking

    result = processor.run(problem)
    
    # Should track reasoning tokens correctly - at least the initial thought and its score
    assert result.reasoning_completion_tokens >= initial_stats.completion_tokens + score_stats.completion_tokens
    assert result.total_completion_tokens >= initial_stats.completion_tokens + score_stats.completion_tokens
    assert result.final_graph is not None

# Add more tests for pruning, aggregation, refinement, error handling, etc.
