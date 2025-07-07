# Default model names for FaR
DEFAULT_FAR_FACT_MODEL_NAMES = ["openrouter/cypher-alpha:free"]
# Using openrouter/cypher-alpha:free as it's confirmed working
REQUESTED_FAR_FACT_MODEL_NAMES = ["openrouter/cypher-alpha:free"]
REQUESTED_FAR_MAIN_MODEL_NAMES = ["openrouter/cypher-alpha:free"]

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
DEFAULT_FAR_ASSESSMENT_MODEL_NAMES = ["openrouter/cypher-alpha:free"]
DEFAULT_FAR_ONESHOT_MODEL_NAMES = ["openrouter/cypher-alpha:free"]
