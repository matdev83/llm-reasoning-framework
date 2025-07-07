# Default model names for FaR
DEFAULT_FAR_FACT_MODEL_NAMES = ["perplexity/sonar-small-online"]
# Using deepseek/deepseek-chat as r1-0528:free is not on OpenRouter based on current info
# For user's specific request: perplexity/sonar, deepseek/deepseek-r1-0528:free
# The CLI runner will allow overriding these defaults.
REQUESTED_FAR_FACT_MODEL_NAMES = ["perplexity/sonar-small-online"] # Defaulting to available sonar
REQUESTED_FAR_MAIN_MODEL_NAMES = ["deepseek/deepseek-chat"] # Defaulting to available deepseek

# Default temperatures
DEFAULT_FAR_FACT_TEMPERATURE = 0.3
DEFAULT_FAR_MAIN_TEMPERATURE = 0.7
DEFAULT_FAR_ASSESSMENT_TEMPERATURE = 0.3 # For complexity assessment if used
DEFAULT_FAR_ONESHOT_TEMPERATURE = 0.7 # For orchestrator's one-shot/fallback

# Default token limits
DEFAULT_FAR_MAX_FACT_TOKENS = 1000
DEFAULT_FAR_MAX_MAIN_TOKENS = 2000

# Default models for assessment and orchestrator's one-shot (can reuse AoT/general defaults)
# These are often smaller/cheaper or general purpose models.
DEFAULT_FAR_ASSESSMENT_MODEL_NAMES = ["openai/gpt-3.5-turbo"] # Example, align with other assessors
DEFAULT_FAR_ONESHOT_MODEL_NAMES = ["openai/gpt-3.5-turbo"]    # Example, align with other one-shot fallbacks
