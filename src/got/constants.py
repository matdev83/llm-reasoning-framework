# Placeholder for initial thought generation prompt
# This prompt should guide the LLM to break down the problem or provide initial hypotheses.
# It will be formatted with {problem_description}.
INITIAL_THOUGHT_PROMPT_TEMPLATE = """
Given the problem:
{problem_description}

Generate one or more initial thoughts or hypotheses to start exploring a solution.
Each thought should be a concise statement.
Present your thoughts clearly, one per line.
Example:
Thought: [Your first thought here]
Thought: [Your second thought here]
"""

# Placeholder for expanding a thought
# This prompt should guide the LLM to generate new thoughts based on a parent thought.
# It will be formatted with {problem_description}, {parent_thought_content}, and {max_new_thoughts}.
EXPAND_THOUGHT_PROMPT_TEMPLATE = """
Problem: {problem_description}

Current thought under consideration:
{parent_thought_content}

Based on this current thought, generate up to {max_new_thoughts} new, distinct, and relevant thoughts that build upon or explore different aspects of the current thought.
Each new thought should be a concise statement.
Present your new thoughts clearly, one per line.
NewThought: [Your new thought here]
"""

# Placeholder for aggregating multiple thoughts
# This prompt should guide the LLM to synthesize or summarize a list of thoughts.
# It will be formatted with {problem_description} and {thoughts_to_aggregate} (a string of thoughts, each on a new line).
AGGREGATE_THOUGHTS_PROMPT_TEMPLATE = """
Problem: {problem_description}

The following thoughts have been generated:
{thoughts_to_aggregate}

Synthesize these thoughts into a single, more comprehensive thought or summary.
The aggregated thought should capture the essence of the input thoughts.
AggregatedThought: [Your aggregated thought here]
"""

# Placeholder for refining a thought
# This prompt should guide the LLM to critique and improve an existing thought.
# It will be formatted with {problem_description} and {thought_to_refine}.
REFINE_THOUGHT_PROMPT_TEMPLATE = """
Problem: {problem_description}

Current thought to refine:
{thought_to_refine}

Critique this thought. What are its weaknesses? How can it be improved or made more precise?
Then, provide a refined version of this thought.
RefinedThought: [Your refined thought here]
"""

# Placeholder for scoring a thought
# This prompt should guide the LLM to evaluate a thought's relevance/quality.
# It will be formatted with {problem_description} and {thought_to_score}.
# The LLM should output a score (e.g., 0.0 to 1.0) and a brief justification.
SCORE_THOUGHT_PROMPT_TEMPLATE = """
Problem: {problem_description}

Thought to score:
{thought_to_score}

Evaluate this thought based on its relevance to solving the problem, its clarity, and its potential to lead to a solution.
Provide a score between 0.0 (not useful) and 1.0 (very useful).
Also, provide a brief justification for your score.
Score: [score from 0.0 to 1.0]
Justification: [brief justification]
"""

# Response parsing cues
NEW_THOUGHT_PREFIX = "NewThought:"
AGGREGATED_THOUGHT_PREFIX = "AggregatedThought:"
REFINED_THOUGHT_PREFIX = "RefinedThought:"
SCORE_PREFIX = "Score:"
JUSTIFICATION_PREFIX = "Justification:"
INITIAL_THOUGHT_PREFIX = "Thought:"
