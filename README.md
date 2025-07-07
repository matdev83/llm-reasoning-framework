# Modular LLM Reasoning Framework

This project provides a flexible and modular framework for orchestrating diverse Large Language Model (LLM) reasoning strategies. It enables the breakdown of complex tasks into manageable steps, manages LLM interactions, and dynamically adapts problem-solving approaches based on complexity and resource constraints.

This framework is designed to support various iterative and adaptive reasoning patterns, including Algorithm of Thoughts (AoT), Learn to Think (L2T), and a novel Hybrid approach. AoT, for instance, is a prompting strategy that aims to reduce hallucinations by first providing an answer and then explaining the reasoning. This approach can be particularly useful for complex problems, revealing the model's initial instincts versus its rationalizations. For more details on AoT, refer to the [Order Matters in Hallucination](https://arxiv.org/html/2408.05093v1) paper by Zikai Xie. L2T, on the other hand, focuses on generating and classifying thoughts to build a reasoning graph.

## Key Features & Flows

The core of this project revolves around orchestrators (e.g., `InteractiveAoTOrchestrator`, `L2TOrchestrator`), which manage the overall problem-solving flow based on a chosen strategy or trigger mode:

*   **`NEVER_REASONING` (One-Shot Processing)**: The problem is sent directly to the main LLM for a single, immediate answer. This is suitable for simple problems that do not require multi-step reasoning.
*   **`ALWAYS_REASONING` (Forced Iterative Reasoning)**: The system always initiates an iterative reasoning process (e.g., AoT, L2T). This is ideal for complex problems where a step-by-step approach is known to be beneficial. If the iterative process fails (e.g., hits limits or parsing issues), it can fall back to a one-shot call.
*   **`ASSESS_FIRST` (Adaptive Reasoning)**: A smaller, faster LLM (the `ComplexityAssessor`) first evaluates the problem's complexity to determine the most suitable approach (one-shot or an advanced reasoning process). This mode also incorporates a **local heuristic analysis** to potentially bypass the LLM call for assessment.

### Local Heuristic Analysis

When `ASSESS_FIRST` mode is active, a local, deterministic heuristic function (`should_trigger_complex_process_heuristically` in `complexity_assessor.py`) is used by default to quickly identify problems that are *highly likely* to require an advanced reasoning process. If this heuristic triggers, the system immediately proceeds with the advanced reasoning process (e.g., AoT, L2T) without making an LLM call for assessment, saving time and tokens.

This heuristic checks for specific keywords and patterns in the problem prompt that strongly indicate a need for detailed, multi-step reasoning (e.g., "design architecture for", "explain in great detail", "step-by-step", "prove that").

To **disable** this heuristic and always use the assessment LLM for complexity evaluation (even for problems that might trigger the heuristic), use the `--disable-heuristic` CLI flag.

### Iterative Reasoning Process (e.g., `src/aot/processor.py`, `src/l2t/processor.py`)

When an iterative reasoning process is triggered (e.g., the AoT or L2T process), a dedicated processor like `src.aot.processor.AoTProcessor` or `src.l2t.processor.L2TProcessor` takes over. It iteratively:
1.  Constructs a prompt using `src.prompt_generator.PromptGenerator` (or `src.l2t.prompt_generator.L2TPromptGenerator`), incorporating the problem statement and the history of previous reasoning steps.
2.  Sends the prompt to the main LLM via `src.llm_client.LLMClient`.
3.  Parses the LLM's response using `src.response_parser.ResponseParser` (or `src.l2t.response_parser.L2TResponseParser`) to extract the current reasoning step, intermediate answers, and potential final answers.
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

# Example: Force an advanced reasoning process (e.g., AoT) for a problem from a file
python -m src.cli_runner --reasoning-mode always-reasoning --problem-filename conf/example_user_prompts/problem1.txt --max-steps 20 --max-time 120

# Example: Run a direct one-shot call with specific models
python -m src.cli_runner --reasoning-mode never-reasoning --problem "What is the capital of France?" --main-models "gpt-3.5-turbo"

# Example: Disable the heuristic, forcing LLM assessment
python -m src.cli_runner --reasoning-mode assess-first --problem "Design a scalable microservices architecture." --disable-heuristic

# Example: Directly run AoTProcess
python -m src.cli_runner --processing-mode aot_direct --problem "Design a simple REST API for a blog." --aot-max-steps 5

# Example: Directly run L2TProcess
python -m src.cli_runner --processing-mode l2t_direct --problem "How does quantum computing work?" --l2t-max-steps 10

# Example: Directly run HybridProcess
python -m src.cli_runner --processing-mode hybrid_direct --problem "Explain the concept of recursion with a Python example." --max-steps 5
```

### Hybrid Reasoning Process

The Hybrid Reasoning Process is a novel approach that combines elements of different reasoning strategies to tackle complex problems. Unlike traditional Chain-of-Thought (CoT) variants that primarily focus on sequential thought generation, the Hybrid process dynamically adapts its strategy based on the problem's evolving context and intermediate results. It is designed to be more flexible and robust, integrating various sub-processes and decision points to navigate intricate problem spaces.

This process is not a simple CoT variant; instead, it leverages a combination of:
-   **Dynamic Sub-problem Decomposition**: Breaking down the main problem into smaller, manageable sub-problems as needed.
-   **Adaptive Strategy Selection**: Choosing the most appropriate sub-strategy (e.g., direct answer, iterative refinement, knowledge retrieval) for each sub-problem.
-   **Contextual Integration**: Continuously integrating new information and insights gained from solving sub-problems back into the overall reasoning context.
-   **Iterative Refinement**: Refining answers through multiple passes, incorporating feedback or new data.

The Hybrid process aims to mimic a more human-like problem-solving approach, where different cognitive tools are employed as the situation demands, rather than adhering to a rigid, pre-defined sequence of thoughts.

**How it works:**
The `src.hybrid.orchestrator.HybridOrchestrator` manages the overall flow, while `src.hybrid.processor.HybridProcessor` implements the core logic. The processor iteratively:
1.  Analyzes the current problem state and available information.
2.  Determines the next best action or sub-strategy to apply.
3.  Executes the chosen sub-strategy, which might involve:
    *   Making direct LLM calls for specific questions.
    *   Initiating a mini-iterative process for a complex sub-problem.
    *   Retrieving relevant information from internal or external sources.
4.  Integrates the results back into the main reasoning context.
5.  Continues until a satisfactory final answer is reached or resource limits are met.

**Usage via CLI:**
You can directly invoke the Hybrid process using the `--processing-mode hybrid_direct` flag:
```bash
python -m src.cli_runner --processing-mode hybrid_direct --problem "Design a secure authentication system for a web application, considering common vulnerabilities." --main-models "gpt-4o" --max-steps 7
```

**Usage via API:**
You can integrate the Hybrid process into your Python application by importing and instantiating `HybridOrchestrator` or `HybridProcess`:
```python
from src.hybrid.orchestrator import HybridOrchestrator
from src.hybrid.dataclasses import HybridOrchestratorConfig
from src.llm_config import LLMConfig

# Configure LLM models
main_llm_config = LLMConfig(
    model="gpt-4o",
    temperature=0.2,
    max_tokens=1000
)

# Configure the Hybrid Orchestrator
hybrid_config = HybridOrchestratorConfig(
    main_llm_config=main_llm_config,
    max_steps=5,
    max_time=60,
    # ... other configurations
)

# Instantiate and run the orchestrator
orchestrator = HybridOrchestrator(config=hybrid_config)
problem_statement = "Explain the concept of quantum entanglement in simple terms and provide a real-world analogy."
result = orchestrator.execute(problem_statement)

if result.final_answer:
    print("Final Answer:", result.final_answer)
else:
    print("Hybrid process did not yield a final answer.")
```

### CLI Parameters

Here are the available command-line arguments:

*   **`--problem` / `-p`**: (Mutually exclusive with `--problem-filename`) The problem or question to solve.
*   **`--problem-filename`**: (Mutually exclusive with `--problem`) Path to a file containing the problem.
*   **`--processing-mode`**: Reasoning strategy trigger mode.
    *   Choices: `always-reasoning`, `assess-first`, `never-reasoning`, `aot_direct`, `l2t_direct`, `l2t`, `hybrid_direct`
    *   Default: `assess-first` (`ASSESS_FIRST`)
    *   `always-reasoning`: Force an iterative reasoning process (e.g., AoT, L2T, Hybrid).
    *   `assess-first`: Use a small LLM to decide between one-shot or an advanced reasoning process.
    *   `never-reasoning`: Force one-shot processing (direct answer).
    *   `aot_direct`: Directly run the AoTProcess.
    *   `l2t_direct`: Directly run the L2TProcess.
    *   `l2t`: Use the L2TOrchestrator (which internally uses L2TProcess).
    *   `hybrid_direct`: Directly run the HybridProcess.
*   **`--main-models`**: Space-separated list of main LLM(s) for one-shot or advanced reasoning processes.
    *   Default: `tngtech/deepseek-r1t-chimera:free deepseek/deepseek-prover-v2:free`
*   **`--main-temp`**: Temperature for main LLM(s).
    *   Default: `0.2`
*   **`--assess-models`**: Space-separated list of small LLM(s) for assessment (used in `assess-first` mode).
    *   Default: `meta-llama/llama-3.3-8b-instruct:free nousresearch/hermes-2-pro-llama-3-8b:free`
*   **`--assess-temp`**: Temperature for assessment LLM(s).
    *   Default: `0.1`
*   **`--max-steps`**: Max steps for iterative reasoning processes (e.g., AoT, L2T).
    *   Default: `12`
*   **`--max-reasoning-tokens`**: Max completion tokens for iterative reasoning phases. Enforced dynamically.
    *   Default: `None` (no limit)
*   **`--max-time`**: Overall max time for an iterative reasoning run (seconds), also used for predictive step limiting.
    *   Default: `60` seconds
*   **`--no-progress-limit`**: Stop iterative reasoning if no progress (same "current answer") for this many steps.
    *   Default: `2`
*   **`--pass-remaining-steps-pct`**: Percentage (0-100) of original `max_steps` at which to inform LLM about dynamically remaining steps.
    *   Choices: `0` to `100`
    *   Default: `None`
*   **`--disable-heuristic`**: Flag to disable the local heuristic analysis for complexity assessment. When present, the assessment LLM will always be used.

### Configuration Files

The `conf/` directory contains important configuration and example files:

*   **`conf/prompts/`**: This directory holds the text files that define the various prompt templates used by the LLMs throughout the reasoning processes (AoT, L2T, etc.). These include:
    *   `aot_intro.txt`: Initial prompt for the AoT reasoning steps.
    *   `aot_final_answer.txt`: Prompt used to synthesize the final answer from the AoT reasoning trace.
    *   `assessment_system_prompt.txt`: System prompt for the complexity assessment LLM.
    *   `assessment_user_prompt.txt`: User prompt for the complexity assessment LLM.
    *   `l2t_initial.txt`: Initial prompt for the L2T process.
    *   `l2t_node_classification.txt`: Prompt for classifying nodes in the L2T graph.
    *   `l2t_thought_generation.txt`: Prompt for generating new thoughts/nodes in the L2T graph.
*   **`conf/example_user_prompts/`**: Contains example problem statements (`problem1.txt`, `problem2.txt`) that can be used with the `--problem-filename` CLI argument.

## Development

To set up a development environment:

1.  Clone the repository:
    ```bash
    git clone https://github.com/matdev83/llm-reasoning-framework.git
    cd llm-reasoning-framework
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

## LLM Call Auditing

All outgoing calls to Large Language Models (LLMs) made by this project are meticulously tracked using the [llm-accounting](https://github.com/matdev83/llm-accounting) library. This integration provides a comprehensive audit trail of LLM interactions.

The audit data, which includes details such as the model used, prompt content, token counts (prompt and completion), call duration, and any associated costs (calculated by `llm-accounting`), is stored in an SQLite database. By default, this database is named `accounting.sqlite` and is created in the data/ directory of the project when the first LLM call is made.
