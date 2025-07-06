# Default model names and parameters for the Hybrid reasoning process

# These are placeholders and should be replaced with actual desired default model names
DEFAULT_HYBRID_REASONING_MODEL_NAMES = ["google/gemini-pro"] # Example
DEFAULT_HYBRID_RESPONSE_MODEL_NAMES = ["anthropic/claude-3-sonnet-20240229"] # Example

DEFAULT_HYBRID_REASONING_TEMPERATURE = 0.1
DEFAULT_HYBRID_RESPONSE_TEMPERATURE = 0.7

# Default prompt templates (can be overridden by CLI arguments)
# Note: {problem_description} and {reasoning_complete_token} are placeholders for HybridConfig
# {extracted_reasoning} is also a placeholder for HybridConfig

# DeepSeek-R1 optimized prompt (uses THINKING tags)
DEFAULT_DEEPSEEK_R1_REASONING_PROMPT_TEMPLATE = "Problem: {problem_description}\n\nPlease think through this step-by-step using <THINKING> tags before providing your answer.\n\n<THINKING>\n"
DEFAULT_DEEPSEEK_R1_RESPONSE_PROMPT_TEMPLATE = "Original Problem: {problem_description}\n\nReasoning Process:\n{extracted_reasoning}\n\nBased on the above reasoning, provide the final solution:"

# Generic reasoning prompt (backward compatible)
DEFAULT_HYBRID_REASONING_PROMPT_TEMPLATE = "Problem: {problem_description}\n\nThink step-by-step to reach the solution. After you have finished your reasoning, output the token sequence: {reasoning_complete_token}\n\nReasoning:"
DEFAULT_HYBRID_REASONING_COMPLETE_TOKEN = "<REASONING_COMPLETE>"
DEFAULT_HYBRID_RESPONSE_PROMPT_TEMPLATE = "Original Problem: {problem_description}\n\nExtracted Reasoning:\n<thinking>{extracted_reasoning}</thinking>\n\nBased on the original problem and the extracted reasoning, provide the final solution."

# OpenAI o1 optimized prompt
DEFAULT_OPENAI_O1_REASONING_PROMPT_TEMPLATE = "Problem: {problem_description}\n\nPlease reason through this problem step by step, showing your work before arriving at the final answer."
DEFAULT_OPENAI_O1_RESPONSE_PROMPT_TEMPLATE = "Original Problem: {problem_description}\n\nReasoning:\n{extracted_reasoning}\n\nFinal Answer:"

# Gemini thinking mode prompt
DEFAULT_GEMINI_THINKING_PROMPT_TEMPLATE = "Problem: {problem_description}\n\nPlease use thinking mode to analyze this problem systematically before providing your answer."
DEFAULT_GEMINI_THINKING_RESPONSE_PROMPT_TEMPLATE = "Problem: {problem_description}\n\nAnalysis:\n{extracted_reasoning}\n\nSolution:"

# Model-specific reasoning formats
REASONING_FORMAT_MAPPINGS = {
    'deepseek-r1': {
        'reasoning_template': DEFAULT_DEEPSEEK_R1_REASONING_PROMPT_TEMPLATE,
        'response_template': DEFAULT_DEEPSEEK_R1_RESPONSE_PROMPT_TEMPLATE,
        'completion_token': None  # Uses THINKING tags instead
    },
    'openai-o1': {
        'reasoning_template': DEFAULT_OPENAI_O1_REASONING_PROMPT_TEMPLATE,
        'response_template': DEFAULT_OPENAI_O1_RESPONSE_PROMPT_TEMPLATE,
        'completion_token': None  # Uses hidden reasoning
    },
    'gemini-thinking': {
        'reasoning_template': DEFAULT_GEMINI_THINKING_PROMPT_TEMPLATE,
        'response_template': DEFAULT_GEMINI_THINKING_RESPONSE_PROMPT_TEMPLATE,
        'completion_token': None  # Uses thinking mode
    }
}

DEFAULT_HYBRID_MAX_REASONING_TOKENS = 2000
DEFAULT_HYBRID_MAX_RESPONSE_TOKENS = 2000

# Default model names for assessment if Hybrid uses ASSESS_FIRST
DEFAULT_HYBRID_ASSESSMENT_MODEL_NAMES = ["google/gemini-pro"] # Example, often a smaller/faster model
DEFAULT_HYBRID_ASSESSMENT_TEMPERATURE = 0.2

# Default direct one-shot model names (for fallback within HybridProcess or HybridOrchestrator)
# Often, this might be the same as the response model or another capable model.
DEFAULT_HYBRID_ONESHOT_MODEL_NAMES = ["anthropic/claude-3-sonnet-20240229"] # Example
DEFAULT_HYBRID_ONESHOT_TEMPERATURE = 0.7
