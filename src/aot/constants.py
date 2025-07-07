# --- Configuration & Constants ---
DEFAULT_MAIN_MODEL_NAMES = ["tngtech/deepseek-r1t-chimera:free", "deepseek/deepseek-prover-v2:free"]
DEFAULT_SMALL_MODEL_NAMES = ["meta-llama/llama-3.3-8b-instruct:free", "nousresearch/hermes-2-pro-llama-3-8b:free"]
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MAX_STEPS = 12
DEFAULT_MAX_TIME_SECONDS = 60
DEFAULT_NO_PROGRESS_LIMIT = 2
DEFAULT_MAIN_TEMPERATURE = 0.2
DEFAULT_ASSESSMENT_TEMPERATURE = 0.1
MIN_PREDICTED_STEP_TOKENS_FALLBACK = 10
MIN_PREDICTED_STEP_DURATION_FALLBACK = 1.0 # seconds

HTTP_REFERER = "http://localhost"
APP_TITLE = "llm-reasoning-framework"
