# LLM AOT Process

This project is designed for **Answer-then-Think (AoT) processing with Large Language Models (LLMs)**. It provides a flexible framework to orchestrate complex reasoning tasks by breaking them down into iterative steps, managing LLM interactions, and dynamically adapting based on problem complexity and resource constraints.

## Key Features & Flows

The core of this project revolves around the `InteractiveAoTOrchestrator`, which manages the overall problem-solving flow based on a chosen trigger mode:

*   **`NEVER_AOT` (One-Shot)**: The problem is sent directly to the main LLM for a single, immediate answer. This is suitable for simple problems that do not require iterative reasoning.
*   **`ALWAYS_AOT` (Forced AoT)**: The system always initiates the iterative AoT reasoning process. This is ideal for complex problems where a step-by-step approach is known to be beneficial. If the AoT process fails (e.g., hits limits or parsing issues), it can fall back to a one-shot call.
*   **`ASSESS_FIRST` (Adaptive AoT)**: A smaller, faster LLM (the `ComplexityAssessor`) first evaluates the problem's complexity. This mode also incorporates a **local heuristic analysis** to potentially bypass the LLM call for assessment.

### Local Heuristic Analysis

When `ASSESS_FIRST` mode is active, a local, deterministic heuristic function (`should_use_aot_heuristically` in `complexity_assessor.py`) is used by default to quickly identify problems that are *highly likely* to require an AoT process. If this heuristic triggers, the system immediately proceeds with AoT without making an LLM call for assessment, saving time and tokens.

This heuristic checks for specific keywords and patterns in the problem prompt that strongly indicate a need for detailed, multi-step reasoning (e.g., "design architecture for", "explain in great detail", "step-by-step", "prove that").

To **disable** this heuristic and always use the assessment LLM for complexity evaluation (even for problems that might trigger the heuristic), use the `--disable-heuristic` CLI flag.

### AoT Reasoning Process (`AoTProcessor`)

When the AoT process is triggered, the `AoTProcessor` takes over. It iteratively:
1.  Constructs a prompt using `PromptGenerator`, incorporating the problem statement and the history of previous reasoning steps.
2.  Sends the prompt to the main LLM via `LLMClient`.
3.  Parses the LLM's response using `ResponseParser` to extract the current reasoning step, intermediate answers, and potential final answers.
4.  Dynamically manages resource limits:
    *   **Max Steps**: Stops after a configured number of reasoning steps.
    *   **Max Reasoning Tokens**: Halts if the cumulative completion tokens for reasoning exceed a specified budget.
    *   **Max Time**: Stops if the overall wall-clock time for the process exceeds a limit.
    *   **No Progress Limit**: Terminates if the LLM repeatedly provides the same "current answer" for a set number of steps, indicating a potential loop or lack of progress.
5.  If a final answer is not found within the iterative steps, an explicit final call to the LLM is made to synthesize the solution from the accumulated reasoning trace.

## Installation

You can install this package using pip:

```bash
pip install .
```

## Usage

The project is run via a command-line interface (`cli_runner.py`). You need to set your `OPENROUTER_API_KEY` as an environment variable before running.

```bash
export OPENROUTER_API_KEY="your_api_key_here" # On macOS/Linux
set OPENROUTER_API_KEY="your_api_key_here" # On Windows (Command Prompt)
$env:OPENROUTER_API_KEY="your_api_key_here" # On Windows (PowerShell)

# Example: Run with default settings, assessing first (heuristic enabled by default)
python -m src.cli_runner --problem "Explain the concept of quantum entanglement in simple terms."

# Example: Force AoT for a problem from a file
python -m src.cli_runner --aot-mode always --problem-filename conf/example_user_prompts/problem1.txt --max-steps 20 --max-time 120

# Example: Run a direct one-shot call with specific models
python -m src.cli_runner --aot-mode never --problem "What is the capital of France?" --main-models "gpt-3.5-turbo"

# Example: Disable the heuristic, forcing LLM assessment
python -m src.cli_runner --aot-mode assess --problem "Design a scalable microservices architecture." --disable-heuristic
```

### CLI Parameters

Here are the available command-line arguments:

*   **`--problem` / `-p`**: (Mutually exclusive with `--problem-filename`) The problem or question to solve.
*   **`--problem-filename`**: (Mutually exclusive with `--problem`) Path to a file containing the problem.
*   **`--aot-mode`**: AoT trigger mode.
    *   Choices: `always`, `assess`, `never`
    *   Default: `assess` (`ASSESS_FIRST`)
    *   `always`: Force AoT process.
    *   `assess`: Use a small LLM to decide if AoT or ONESHOT.
    *   `never`: Force ONESHOT (direct answer).
*   **`--main-models`**: Space-separated list of main LLM(s) for AoT/ONESHOT.
    *   Default: `tngtech/deepseek-r1t-chimera:free deepseek/deepseek-prover-v2:free`
*   **`--main-temp`**: Temperature for main LLM(s).
    *   Default: `0.2`
*   **`--assess-models`**: Space-separated list of small LLM(s) for assessment (used in `assess` mode).
    *   Default: `meta-llama/llama-3.3-8b-instruct:free nousresearch/hermes-2-pro-llama-3-8b:free`
*   **`--assess-temp`**: Temperature for assessment LLM(s).
    *   Default: `0.1`
*   **`--max-steps`**: Max AoT reasoning steps.
    *   Default: `12`
*   **`--max-reasoning-tokens`**: Max completion tokens for AoT reasoning phase. Enforced dynamically.
    *   Default: `None` (no limit)
*   **`--max-time`**: Overall max time for an AoT run (seconds), also used for predictive step limiting.
    *   Default: `60` seconds
*   **`--no-progress-limit`**: Stop AoT if no progress (same "current answer") for this many steps.
    *   Default: `2`
*   **`--pass-remaining-steps-pct`**: Percentage (0-100) of original `max_steps` at which to inform LLM about dynamically remaining steps.
    *   Choices: `0` to `100`
    *   Default: `None`
*   **`--disable-heuristic`**: Flag to disable the local heuristic analysis for complexity assessment. When present, the assessment LLM will always be used.

### Configuration Files

The `conf/` directory contains important configuration and example files:

*   **`conf/prompts/`**: This directory holds the text files that define the various prompt templates used by the LLMs throughout the AoT process. These include:
    *   `aot_intro.txt`: Initial prompt for the AoT reasoning steps.
    *   `aot_final_answer.txt`: Prompt used to synthesize the final answer from the reasoning trace.
    *   `assessment_system_prompt.txt`: System prompt for the complexity assessment LLM.
    *   `assessment_user_prompt.txt`: User prompt for the complexity assessment LLM.
*   **`conf/example_user_prompts/`**: Contains example problem statements (`problem1.txt`, `problem2.txt`) that can be used with the `--problem-filename` CLI argument.

## Development

To set up a development environment:

1.  Clone the repository:
    ```bash
    git clone https://github.com/matdev83/llm-aot-process.git
    cd llm-aot-process
    ```
2.  Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -e .
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
