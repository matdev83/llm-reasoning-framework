# Modular LLM Reasoning Framework

This project provides a flexible and modular framework for orchestrating diverse Large Language Model (LLM) reasoning strategies. It enables the breakdown of complex tasks into manageable steps, manages LLM interactions, and dynamically adapts problem-solving approaches based on complexity and resource constraints.

This framework is designed to support various iterative and adaptive reasoning patterns, including Algorithm of Thoughts (AoT), Learn to Think (L2T), Graph of Thoughts (GoT), and a novel Hybrid approach. AoT is an iterative step-by-step reasoning process that breaks down complex problems into manageable sequential steps, with each step building upon previous ones and maintaining a current answer state. L2T focuses on generating and classifying thoughts to build a reasoning graph. GoT creates a graph-based exploration of thoughts with scoring and refinement capabilities. The Hybrid approach separates reasoning from response generation using specialized models for each stage.

## Key Features & Flows

The core of this project revolves around orchestrators (e.g., `InteractiveAoTOrchestrator`, `L2TOrchestrator`, `HybridOrchestrator`, `GoTOrchestrator`), which manage the overall problem-solving flow based on a chosen strategy or trigger mode:

*   **`NEVER_REASONING` (One-Shot Processing)**: The problem is sent directly to the main LLM for a single, immediate answer. This is suitable for simple problems that do not require multi-step reasoning.
*   **`ALWAYS_REASONING` (Forced Iterative Reasoning)**: The system always initiates an iterative reasoning process (e.g., AoT, L2T). This is ideal for complex problems where a step-by-step approach is known to be beneficial. If the iterative process fails (e.g., hits limits or parsing issues), it can fall back to a one-shot call.
*   **`ASSESS_FIRST` (Adaptive Reasoning)**: A smaller, faster LLM (the `ComplexityAssessor`) first evaluates the problem's complexity to determine the most suitable approach (one-shot or an advanced reasoning process). This mode also incorporates a **local heuristic analysis** to potentially bypass the LLM call for assessment.

### Local Heuristic Analysis

When `ASSESS_FIRST` mode is active, a local, deterministic heuristic function (`should_trigger_complex_process_heuristically` in `src.heuristic_detector.main_detector.MainHeuristicDetector`) is used by default to quickly identify problems that are *highly likely* to require an advanced reasoning process. If this heuristic triggers, the system immediately proceeds with the advanced reasoning process (e.g., AoT, L2T) without making an LLM call for assessment, saving time and tokens.

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

# Example: Force AoT reasoning process for a problem from a file
python -m src.cli_runner --processing-mode aot-always --problem-filename conf/tests/prompts/example_user_prompts/problem1.txt --aot-max-steps 20 --aot-max-time 120

# Example: Run AoT with never mode (direct one-shot)
python -m src.cli_runner --processing-mode aot-never --problem "What is the capital of France?" --aot-main-models "openai/gpt-3.5-turbo"

# Example: Disable the heuristic for AoT assessment, forcing LLM assessment
python -m src.cli_runner --processing-mode aot-assess-first --problem "Design a scalable microservices architecture." --aot-disable-heuristic

# Example: Directly run AoTProcessor
python -m src.cli_runner --processing-mode aot-direct --problem "Design a simple REST API for a blog." --aot-max-steps 5

# Example: Directly run L2TProcessor
python -m src.cli_runner --processing-mode l2t-direct --problem "How does quantum computing work?" --l2t-max-steps 10

# Example: Directly run HybridProcessor
python -m src.cli_runner --processing-mode hybrid-direct --problem "Explain the concept of recursion with a Python example." --hybrid-reasoning-models "google/gemini-pro" --hybrid-response-models "anthropic/claude-3-sonnet-20240229"

# Example: Run GoT (Graph of Thoughts) reasoning process
python -m src.cli_runner --processing-mode got-always --problem "Solve this complex optimization problem step by step." --got-max-thoughts 30 --got-max-iterations 8

# Example: Directly run GoTProcessor
python -m src.cli_runner --processing-mode got-direct --problem "Analyze this multi-faceted problem from different angles." --got-thought-gen-models "openai/gpt-4o" --got-scoring-models "openai/gpt-4o-mini"
```

### AoT (Algorithm of Thoughts) Reasoning Process

The AoT (Algorithm of Thoughts) Reasoning Process is an iterative step-by-step reasoning approach that systematically breaks down complex problems into manageable sequential steps. Unlike single-shot approaches, AoT builds reasoning incrementally, with each step contributing to a growing understanding of the problem.

The process leverages:
-   **Incremental Reasoning**: Each step builds upon previous steps, creating a coherent reasoning chain.
-   **Current Answer Tracking**: Maintains a "current answer" state that evolves as reasoning progresses.
-   **Dynamic Resource Management**: Automatically adjusts based on token limits, time constraints, and progress monitoring.
-   **Progress Detection**: Identifies when reasoning has stalled or reached completion.
-   **Adaptive Step Limiting**: Dynamically adjusts the number of remaining steps based on resource consumption.

**How it works:**
The `src.aot.orchestrator.InteractiveAoTOrchestrator` manages the overall flow, while `src.aot.processor.AoTProcessor` implements the core iterative logic. The AoT process operates through:

1.  **Step-by-Step Reasoning**: Each iteration asks for the next unique reasoning step, avoiding repetition.
2.  **Current Answer Evolution**: Tracks how the current answer changes with each reasoning step.
3.  **Resource Monitoring**: Continuously monitors token usage, time consumption, and step count.
4.  **Progress Assessment**: Detects when the same current answer appears multiple times (indicating stagnation).
5.  **Final Answer Synthesis**: If no final answer emerges during reasoning, makes an explicit final call to synthesize the solution.

**Usage via CLI:**
```bash
# AoT with assessment mode
python -m src.cli_runner --processing-mode aot-assess-first --problem "Design a distributed system architecture."

# Direct AoT processor
python -m src.cli_runner --processing-mode aot-direct --problem "Solve this optimization problem step by step." --aot-max-steps 15 --aot-max-time 90
```

### L2T (Learn to Think) Reasoning Process

The L2T (Learn to Think) Reasoning Process is a graph-based reasoning approach that builds a dynamic tree of thoughts, where each thought is classified and can lead to new thoughts, backtracking, or final answers. Unlike linear approaches, L2T creates a branching exploration of the problem space with intelligent pruning and expansion.

The process leverages:
-   **Graph-based Reasoning**: Thoughts are organized in a graph structure with parent-child relationships.
-   **Thought Classification**: Each thought is classified as CONTINUE, TERMINATE_BRANCH, FINAL_ANSWER, or BACKTRACK.
-   **Dynamic Exploration**: The process maintains active thoughts (v_pres) and historical thoughts (v_hist).
-   **Intelligent Backtracking**: Can backtrack to parent nodes when a reasoning path becomes unfruitful.
-   **Multi-model Architecture**: Uses separate models for initial thoughts, classification, and thought generation.

**How it works:**
The `src.l2t.orchestrator.L2TOrchestrator` manages the overall flow, while `src.l2t.processor.L2TProcessor` implements the core graph logic. The L2T process operates through:

1.  **Initial Thought Generation**: Creates a root node with the initial reasoning approach.
2.  **Iterative Processing**: Processes each active thought in the current generation.
3.  **Thought Classification**: Classifies each thought to determine next actions:
    - **CONTINUE**: Generate a new child thought
    - **TERMINATE_BRANCH**: End this reasoning path
    - **FINAL_ANSWER**: This thought contains the solution
    - **BACKTRACK**: Return to parent node for alternative exploration
4.  **Graph Management**: Maintains active thoughts (v_pres) and processed thoughts (v_hist).
5.  **Resource Monitoring**: Tracks steps, nodes, and time limits to prevent infinite expansion.

**Usage via CLI:**
```bash
# L2T orchestrator mode
python -m src.cli_runner --processing-mode l2t --problem "Analyze this complex decision-making scenario."

# Direct L2T processor
python -m src.cli_runner --processing-mode l2t-direct --problem "Explore multiple solution approaches." --l2t-max-steps 15 --l2t-max-total-nodes 40
```

### Hybrid Reasoning Process

The Hybrid Reasoning Process is a novel two-stage approach that separates reasoning from response generation to improve answer quality. Unlike traditional single-model approaches, the Hybrid process uses specialized models for different phases of problem-solving.

The process leverages a combination of:
-   **Specialized Model Roles**: One model focuses on reasoning/thinking, another on clear response generation.
-   **Reasoning Extraction**: Advanced reasoning extraction techniques that work with various model formats (DeepSeek-R1, OpenAI o1, Gemini Thinking, etc.).
-   **Flexible Configuration**: Each stage can be independently configured with different models, temperatures, and token limits.
-   **Fallback Mechanisms**: If the two-stage process fails, it can fall back to a traditional one-shot approach.

The Hybrid process aims to combine the deep reasoning capabilities of thinking-optimized models with the clear communication abilities of response-optimized models, resulting in both thorough analysis and well-structured final answers.

**How it works:**
The `src.hybrid.orchestrator.HybridOrchestrator` manages the overall flow, while `src.hybrid.processor.HybridProcessor` implements the core logic. The Hybrid process operates in two distinct stages:

1.  **Reasoning Stage**: A reasoning model (e.g., a model optimized for thinking/reasoning) analyzes the problem and generates detailed reasoning steps without providing a final answer.
2.  **Response Stage**: A response model takes the problem description and the extracted reasoning from stage 1, then generates a clear, well-structured final answer based on that reasoning.

This two-stage approach allows for:
- **Specialized Model Usage**: Different models can be optimized for different tasks (reasoning vs. response generation)
- **Reasoning Extraction**: The reasoning from stage 1 can be extracted and used to inform the final response
- **Flexible Configuration**: Each stage can use different models, temperatures, and token limits

**Usage via CLI:**
You can directly invoke the Hybrid process using the `--processing-mode hybrid_direct` flag:
```bash
python -m src.cli_runner --processing-mode hybrid_direct --problem "Design a secure authentication system for a web application, considering common vulnerabilities." --hybrid-reasoning-models "google/gemini-pro" --hybrid-response-models "anthropic/claude-3-sonnet-20240229"
```

**Usage via API:**
You can integrate the Hybrid process into your Python application by importing and instantiating `HybridOrchestrator` or `HybridProcess`:
```python
from src.hybrid.orchestrator import HybridOrchestrator, HybridProcess
from src.hybrid.dataclasses import HybridConfig
from src.hybrid.enums import HybridTriggerMode

# Configure the Hybrid process with two models
hybrid_config = HybridConfig(
    reasoning_model_name="google/gemini-pro",
    reasoning_model_temperature=0.1,
    response_model_name="anthropic/claude-3-sonnet-20240229",
    response_model_temperature=0.7,
    max_reasoning_tokens=2000,
    max_response_tokens=2000
)

# Option 1: Use HybridProcess directly
hybrid_process = HybridProcess(
    hybrid_config=hybrid_config,
    direct_oneshot_model_names=["anthropic/claude-3-sonnet-20240229"],
    direct_oneshot_temperature=0.7,
    api_key="your_api_key",
    enable_rate_limiting=True,
    enable_audit_logging=True
)

problem_statement = "Explain the concept of quantum entanglement in simple terms and provide a real-world analogy."
hybrid_process.execute(problem_description=problem_statement, model_name="hybrid_model")
solution, summary = hybrid_process.get_result()

if solution and solution.final_answer:
    print("Final Answer:", solution.final_answer)
else:
    print("Hybrid process did not yield a final answer.")

# Option 2: Use HybridOrchestrator (supports assessment modes)
hybrid_orchestrator = HybridOrchestrator(
    trigger_mode=HybridTriggerMode.ALWAYS_HYBRID,
    hybrid_config=hybrid_config,
    direct_oneshot_model_names=["anthropic/claude-3-sonnet-20240229"],
    direct_oneshot_temperature=0.7,
    api_key="your_api_key"
)

solution, summary = hybrid_orchestrator.solve(problem_statement)
if solution and solution.final_answer:
    print("Final Answer:", solution.final_answer)
```

### GoT (Graph of Thoughts) Reasoning Process

The GoT (Graph of Thoughts) Reasoning Process is an advanced reasoning strategy that builds a graph of interconnected thoughts to explore complex problems from multiple angles. Unlike linear reasoning approaches, GoT creates a network of thoughts where each thought can be scored, refined, and used to generate new related thoughts.

The process leverages:
-   **Graph-based Exploration**: Thoughts are organized in a graph structure allowing for non-linear reasoning paths.
-   **Thought Scoring**: Each thought is evaluated and scored to determine its quality and relevance.
-   **Iterative Refinement**: High-scoring thoughts can be refined and improved through multiple iterations.
-   **Thought Aggregation**: Multiple related thoughts can be combined to form more comprehensive insights.
-   **Adaptive Pruning**: Low-scoring thoughts can be removed to focus computational resources on promising paths.

**How it works:**
The `src.got.orchestrator.GoTOrchestrator` manages the overall flow, while `src.got.processor.GoTProcessor` implements the core logic. The GoT process operates through iterative cycles of:

1.  **Thought Generation**: Generate new thoughts from existing ones or from the original problem.
2.  **Thought Scoring**: Evaluate each thought for quality, relevance, and potential.
3.  **Thought Selection**: Choose the most promising thoughts for further development.
4.  **Thought Transformation**: Refine, aggregate, or expand selected thoughts.
5.  **Solution Identification**: Identify when a thought or combination of thoughts constitutes a solution.

**Usage via CLI:**
```bash
# GoT with assessment mode
python -m src.cli_runner --processing-mode got-assess-first --problem "Design a sustainable city transportation system."

# Direct GoT processor
python -m src.cli_runner --processing-mode got-direct --problem "Optimize resource allocation across multiple constraints." --got-max-thoughts 50 --got-max-iterations 10
```

### CLI Parameters

Here are the available command-line arguments:

*   **`--problem` / `-p`**: (Mutually exclusive with `--problem-filename`) The problem or question to solve.
*   **`--problem-filename`**: (Mutually exclusive with `--problem`) Path to a file containing the problem.
*   **`--processing-mode`**: Reasoning strategy trigger mode.
    *   Choices: `aot-always`, `aot-assess-first`, `aot-never`, `l2t`, `hybrid-always`, `hybrid-assess-first`, `hybrid-never`, `got-always`, `got-assess-first`, `got-never`, `aot-direct`, `l2t-direct`, `hybrid-direct`, `got-direct`
    *   Default: `aot-assess-first`
    *   **AoT Orchestrator Modes:**
        *   `aot-always`: Always use AoT iterative reasoning process.
        *   `aot-assess-first`: Use assessment LLM to decide between one-shot or AoT process.
        *   `aot-never`: Force one-shot processing (direct answer).
    *   **L2T Orchestrator Mode:**
        *   `l2t`: Use the L2TOrchestrator (which internally uses L2TProcess).
    *   **Hybrid Orchestrator Modes:**
        *   `hybrid-always`: Always use Hybrid two-stage reasoning process.
        *   `hybrid-assess-first`: Use assessment LLM to decide between one-shot or Hybrid process.
        *   `hybrid-never`: Force one-shot processing (direct answer).
    *   **GoT (Graph of Thoughts) Orchestrator Modes:**
        *   `got-always`: Always use GoT graph-based reasoning process.
        *   `got-assess-first`: Use assessment LLM to decide between one-shot or GoT process.
        *   `got-never`: Force one-shot processing (direct answer).
    *   **Direct Process Modes:**
        *   `aot-direct`: Directly run the AoTProcessor.
        *   `l2t-direct`: Directly run the L2TProcessor.
        *   `hybrid-direct`: Directly run the HybridProcessor. Requires `--hybrid-reasoning-models` and `--hybrid-response-models` parameters.
        *   `got-direct`: Directly run the GoTProcessor.
**Note**: Each reasoning strategy (AoT, L2T, Hybrid, GoT) has its own specific model parameters. The generic `--main-models` and `--assess-models` parameters do not exist in the CLI. Instead, use strategy-specific parameters like:

- **AoT**: `--aot-main-models`, `--aot-assess-models`, `--aot-main-temp`, `--aot-assess-temp`
- **L2T**: `--l2t-classification-models`, `--l2t-thought-gen-models`, `--l2t-initial-prompt-models`
- **Hybrid**: `--hybrid-reasoning-models`, `--hybrid-response-models`, `--hybrid-assess-models`
- **GoT**: `--got-thought-gen-models`, `--got-scoring-models`, `--got-aggregation-models`, `--got-refinement-models`
**Note**: The generic parameters like `--max-steps`, `--max-reasoning-tokens`, `--max-time`, etc. do not exist in the CLI. Each reasoning strategy has its own specific parameters.

#### AoT Process Specific Parameters

*   **`--aot-main-models`**: Space-separated list of main LLM(s) for AoT reasoning.
    *   Default: `tngtech/deepseek-r1t-chimera:free deepseek/deepseek-prover-v2:free`
*   **`--aot-main-temp`**: Temperature for AoT main LLM(s).
    *   Default: `0.2`
*   **`--aot-assess-models`**: Space-separated list of small LLM(s) for AoT assessment.
    *   Default: `meta-llama/llama-3.3-8b-instruct:free nousresearch/hermes-2-pro-llama-3-8b:free`
*   **`--aot-assess-temp`**: Temperature for AoT assessment LLM(s).
    *   Default: `0.1`
*   **`--aot-max-steps`**: Max AoT reasoning steps.
    *   Default: `12`
*   **`--aot-max-reasoning-tokens`**: Max completion tokens for AoT reasoning phase.
    *   Default: `None` (no limit)
*   **`--aot-max-time`**: Overall max time for an AoT run (seconds).
    *   Default: `60` seconds
*   **`--aot-no-progress-limit`**: Stop AoT if no progress for this many steps.
    *   Default: `2`
*   **`--aot-pass-remaining-steps-pct`**: Percentage (0-100) of original max_steps at which to inform LLM about dynamically remaining steps in AoT.
    *   Default: `None`
*   **`--aot-disable-heuristic`**: Flag to disable the local heuristic analysis for AoT complexity assessment.

#### L2T Process Specific Parameters

*   **`--l2t-classification-models`**: L2T classification model(s).
*   **`--l2t-thought-gen-models`**: L2T thought generation model(s).
*   **`--l2t-initial-prompt-models`**: L2T initial prompt model(s).
*   **`--l2t-classification-temp`**: L2T classification temperature.
*   **`--l2t-thought-gen-temp`**: L2T thought generation temperature.
*   **`--l2t-initial-prompt-temp`**: L2T initial prompt temperature.
*   **`--l2t-max-steps`**: L2T max steps.
*   **`--l2t-max-total-nodes`**: L2T max total nodes.
*   **`--l2t-max-time-seconds`**: L2T max time (seconds).
*   **`--l2t-x-fmt`**: L2T default format constraints string.
*   **`--l2t-x-eva`**: L2T default evaluation criteria string.

#### Hybrid Process Specific Parameters

When using `--processing-mode hybrid_direct`, the following parameters are required:

*   **`--hybrid-reasoning-models`**: Space-separated list of models for the reasoning stage.
    *   Default: `google/gemini-pro`
*   **`--hybrid-response-models`**: Space-separated list of models for the response generation stage.
    *   Default: `anthropic/claude-3-sonnet-20240229`
*   **`--hybrid-reasoning-temp`**: Temperature for reasoning models.
    *   Default: `0.1`
*   **`--hybrid-response-temp`**: Temperature for response models.
    *   Default: `0.7`
*   **`--hybrid-max-reasoning-tokens`**: Max tokens for reasoning stage.
    *   Default: `2000`
*   **`--hybrid-max-response-tokens`**: Max tokens for response stage.
    *   Default: `2000`

Note: The Hybrid process does not use iterative steps like AoT or L2T, so `--max-steps` is not applicable.

#### GoT Process Specific Parameters

*   **`--got-thought-gen-models`**: LLM(s) for GoT thought generation.
    *   Default: `openai/gpt-4o-mini`
*   **`--got-scoring-models`**: LLM(s) for GoT thought scoring.
    *   Default: `openai/gpt-3.5-turbo`
*   **`--got-aggregation-models`**: LLM(s) for GoT thought aggregation.
    *   Default: `openai/gpt-4o-mini`
*   **`--got-refinement-models`**: LLM(s) for GoT thought refinement.
    *   Default: `openai/gpt-4o-mini`
*   **`--got-max-thoughts`**: Max total thoughts in the graph.
    *   Default: `50`
*   **`--got-max-iterations`**: Max iterations of generation/transformation.
    *   Default: `10`
*   **`--got-min-score-for-expansion`**: Minimum score to consider a thought for expansion.
    *   Default: `0.5`
*   **`--got-pruning-threshold-score`**: Thoughts below this score might be pruned.
    *   Default: `0.2`
*   **`--got-max-children-per-thought`**: Max new thoughts to generate from one parent.
    *   Default: `3`
*   **`--got-max-parents-for-aggregation`**: Max parents to consider for aggregation.
    *   Default: `5`
*   **`--got-solution-found-score-threshold`**: If a thought reaches this score, it might be a solution.
    *   Default: `0.9`
*   **`--got-max-time-seconds`**: Max time for the GoT process.
    *   Default: `300` seconds
*   **`--got-thought-gen-temp`**: Temperature for thought generation.
    *   Default: `0.7`
*   **`--got-scoring-temp`**: Temperature for scoring.
    *   Default: `0.2`
*   **`--got-aggregation-temp`**: Temperature for aggregation.
    *   Default: `0.7`
*   **`--got-refinement-temp`**: Temperature for refinement.
    *   Default: `0.7`
*   **`--got-orchestrator-oneshot-temp`**: Temperature for orchestrator's one-shot/fallback.
    *   Default: `0.7`
*   **`--got-assess-models`**: Assessment LLM(s) for GoT (if assess_first).
*   **`--got-assess-temp`**: Temperature for GoT assessment LLM(s).
*   **`--got-orchestrator-oneshot-models`**: Fallback one-shot LLM(s) for GoT Orchestrator.
*   **`--got-disable-aggregation`**: Disable aggregation step.
*   **`--got-disable-refinement`**: Disable refinement step.
*   **`--got-disable-pruning`**: Disable pruning step.
*   **`--got-disable-heuristic`**: Disable local heuristic for GoT complexity assessment.

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
