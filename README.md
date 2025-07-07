# Modular LLM Reasoning Framework

This project provides a flexible and modular framework for orchestrating diverse Large Language Model (LLM) reasoning strategies. It enables the breakdown of complex tasks into manageable steps, manages LLM interactions, and dynamically adapts problem-solving approaches based on complexity and resource constraints.

This framework is designed to support various iterative and adaptive reasoning patterns, including Answer On Thought (AoT), Learn to Think (L2T), Graph of Thoughts (GoT), a novel Hybrid approach, and Fact-and-Reflection (FaR). 

**Academic Foundations:**
- **Answer on Thought (AoT)**: Based on "Answer on Thought: Enhancing LLM Reasoning Through Iterative Answer Generation" by Ding et al. (2024) - arXiv:2412.11021
- **Learn to Think (L2T)**: Based on "Learn to Think: Bootstrapping LLM Reasoning Capability Through Graph Representation Learning" by Zhang et al. (2024) - arXiv:2412.11940  
- **Graph of Thoughts (GoT)**: Based on "Graph of Thoughts: Solving Elaborate Problems with Large Language Models" by Besta et al. (2023) - arXiv:2308.09687
- **Fact-and-Reflection (FaR)**: Based on "Fact-and-Reflection (FaR) Improves Confidence Calibration of Large Language Models" by Tian et al. (2023) - arXiv:2402.17124
- **Hybrid Thinking**: A novel approach developed specifically for this framework that separates reasoning from response generation using specialized models for each stage.

AoT is an iterative step-by-step reasoning process that breaks down complex problems into manageable sequential steps, with each step building upon previous ones and maintaining a current answer state. L2T focuses on generating and classifying thoughts to build a reasoning graph. GoT creates a graph-based exploration of thoughts with scoring and refinement capabilities. The Hybrid approach separates reasoning from response generation using specialized models for each stage. FaR first elicits relevant facts using one model and then uses these facts along with the original problem to generate a reflected answer with another model.

## Key Features & Flows

The core of this project revolves around orchestrators (e.g., `InteractiveAoTOrchestrator`, `L2TOrchestrator`, `HybridOrchestrator`, `GoTOrchestrator`, `FaROrchestrator`), which manage the overall problem-solving flow based on a chosen strategy or trigger mode:

*   **`NEVER_REASONING` (One-Shot Processing)**: The problem is sent directly to the main LLM for a single, immediate answer. This is suitable for simple problems that do not require multi-step reasoning. (e.g., `aot-never`, `hybrid-never`, `got-never`, `far-never`)
*   **`ALWAYS_REASONING` (Forced Iterative/Multi-step Reasoning)**: The system always initiates an iterative or multi-step reasoning process (e.g., AoT, L2T, Hybrid, GoT, FaR). This is ideal for complex problems where a structured approach is known to be beneficial. If the process fails, it can fall back to a one-shot call. (e.g., `aot-always`, `l2t`, `hybrid-always`, `got-always`, `far-always`)
*   **`ASSESS_FIRST` (Adaptive Reasoning)**: A smaller, faster LLM (the `ComplexityAssessor`) first evaluates the problem's complexity to determine the most suitable approach (one-shot or an advanced reasoning process). This mode also incorporates a **local heuristic analysis** to potentially bypass the LLM call for assessment. (e.g., `aot-assess-first`, `hybrid-assess-first`, `got-assess-first`, `far-assess-first`)

### Local Heuristic Analysis

When `ASSESS_FIRST` mode is active, a local, deterministic heuristic function (`should_trigger_complex_process_heuristically` in `src.heuristic_detector.main_detector.MainHeuristicDetector`) is used by default to quickly identify problems that are *highly likely* to require an advanced reasoning process. If this heuristic triggers, the system immediately proceeds with the advanced reasoning process (e.g., AoT, L2T, Hybrid, GoT, FaR) without making an LLM call for assessment, saving time and tokens.

This heuristic checks for specific keywords and patterns in the problem prompt that strongly indicate a need for detailed, multi-step reasoning (e.g., "design architecture for", "explain in great detail", "step-by-step", "prove that").

To **disable** this heuristic and always use the assessment LLM for complexity evaluation (even for problems that might trigger the heuristic), use the `--disable-heuristic` CLI flag (e.g., `--aot-disable-heuristic`, `--hybrid-disable-heuristic`, etc.).

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

# Example: Run FaR (Fact-and-Reflection) reasoning process
python -m src.cli_runner --processing-mode far-always --problem "What were the key events of the French Revolution and their significance?"

# Example: Directly run FaRProcessor
python -m src.cli_runner --processing-mode far-direct --problem "Describe the process of photosynthesis and its importance." --far-fact-models "openrouter/cypher-alpha:free" --far-main-models "openrouter/cypher-alpha:free"
```

### AoT (Answer On Thought) Reasoning Process
(Content remains the same)

...

### L2T (Learn to Think) Reasoning Process
(Content remains the same)

...

### Hybrid Reasoning Process
(Content remains the same)

...

### GoT (Graph of Thoughts) Reasoning Process
(Content remains the same)

...

### FaR (Fact-and-Reflection) Reasoning Process

The FaR (Fact-and-Reflection) Reasoning Process is a two-stage approach designed to improve the accuracy and calibration of LLM responses, particularly for questions requiring factual knowledge.

The process leverages:
-   **Fact Elicitation**: An initial LLM call (the "fact model") is made to extract relevant facts, claims, and pieces of information from the problem description. This step aims to ground the subsequent reasoning in verified or explicitly stated information.
-   **Reflection and Answer Generation**: A second LLM call (the "main model") takes the original problem description *and* the elicited facts. This model then reflects on this combined information to generate a final, comprehensive answer.
-   **Specialized Model Roles**: Allows for using different models optimized for fact retrieval versus reasoning and answer synthesis. For example, a model with web access might be used for fact elicitation.

**How it works:**
The `src.far.orchestrator.FaROrchestrator` manages the overall flow, while `src.far.processor.FaRProcessor` implements the core logic. The FaR process operates in two distinct stages:

1.  **Fact Elicitation Stage**: The fact model is prompted to extract key facts relevant to the input problem.
2.  **Reflection and Answer Generation Stage**: The main model receives the original problem and the facts from stage 1. It's instructed to reflect on these inputs and produce the final answer.

This two-stage process aims to reduce hallucination and improve the factual grounding of the generated responses.

**Usage via CLI:**
```bash
# FaR with assessment mode
python -m src.cli_runner --processing-mode far-assess-first --problem "What is the current world record for the 100m sprint and who holds it?"

# Direct FaR process
python -m src.cli_runner --processing-mode far-direct --problem "Describe the process of photosynthesis and its importance."
```

### CLI Parameters

Here are the available command-line arguments:

*   **`--problem` / `-p`**: (Mutually exclusive with `--problem-filename`) The problem or question to solve.
*   **`--problem-filename`**: (Mutually exclusive with `--problem`) Path to a file containing the problem.
*   **`--processing-mode`**: Reasoning strategy trigger mode.
    *   Choices: `aot-always`, `aot-assess-first`, `aot-never`, `l2t`, `hybrid-always`, `hybrid-assess-first`, `hybrid-never`, `got-always`, `got-assess-first`, `got-never`, `far-always`, `far-assess-first`, `far-never`, `aot-direct`, `l2t-direct`, `hybrid-direct`, `got-direct`, `far-direct`
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
    *   **FaR (Fact-and-Reflection) Orchestrator Modes:**
        *   `far-always`: Always use FaR two-stage reasoning process.
        *   `far-assess-first`: Use assessment LLM to decide between one-shot or FaR process.
        *   `far-never`: Force one-shot processing (direct answer).
    *   **Direct Process Modes:**
        *   `aot-direct`: Directly run the AoTProcessor.
        *   `l2t-direct`: Directly run the L2TProcessor.
        *   `hybrid-direct`: Directly run the HybridProcessor.
        *   `got-direct`: Directly run the GoTProcessor.
        *   `far-direct`: Directly run the FaRProcess (which internally uses FaRProcessor).
**Note**: Each reasoning strategy (AoT, L2T, Hybrid, GoT, FaR) has its own specific model parameters. The generic `--main-models` and `--assess-models` parameters do not exist in the CLI. Instead, use strategy-specific parameters like:

- **AoT**: `--aot-main-models`, `--aot-assess-models`, `--aot-main-temp`, `--aot-assess-temp`
- **L2T**: `--l2t-classification-models`, `--l2t-thought-gen-models`, `--l2t-initial-prompt-models`
- **Hybrid**: `--hybrid-reasoning-models`, `--hybrid-response-models`, `--hybrid-assess-models`
- **GoT**: `--got-thought-gen-models`, `--got-scoring-models`, `--got-aggregation-models`, `--got-refinement-models`
- **FaR**: `--far-fact-models`, `--far-main-models`, `--far-assess-models`

**Note**: The generic parameters like `--max-steps`, `--max-reasoning-tokens`, `--max-time`, etc. do not exist in the CLI. Each reasoning strategy has its own specific parameters.

#### AoT Process Specific Parameters
(Content remains the same)
...

#### L2T Process Specific Parameters
(Content remains the same)
...

#### Hybrid Process Specific Parameters
(Content remains the same)
...

#### GoT Process Specific Parameters
(Content remains the same)
...

#### FaR Process Specific Parameters

*   **`--far-fact-models`**: Space-separated list of LLM(s) for FaR fact elicitation.
    *   Default: `openrouter/cypher-alpha:free`
*   **`--far-fact-temp`**: Temperature for FaR fact LLM(s).
    *   Default: `0.3`
*   **`--far-main-models`**: Space-separated list of main LLM(s) for FaR reflection and answer generation.
    *   Default: `openrouter/cypher-alpha:free`
*   **`--far-main-temp`**: Temperature for FaR main LLM(s).
    *   Default: `0.7`
*   **`--far-max-fact-tokens`**: Max tokens for FaR fact elicitation stage.
    *   Default: `1000`
*   **`--far-max-main-tokens`**: Max tokens for FaR main response stage.
    *   Default: `2000`
*   **`--far-assess-models`**: Assessment LLM(s) for FaR (if `far-assess-first` is used).
    *   Default: `openrouter/cypher-alpha:free`
*   **`--far-assess-temp`**: Temperature for FaR assessment LLM(s).
    *   Default: `0.3`
*   **`--far-oneshot-models`**: Fallback/direct one-shot LLM(s) used by FaR Orchestrator.
    *   Default: `openrouter/cypher-alpha:free`
*   **`--far-oneshot-temp`**: Temperature for FaR Orchestrator's one-shot calls.
    *   Default: `0.7`
*   **`--far-disable-heuristic`**: Flag to disable the local heuristic analysis for FaR complexity assessment when using `far-assess-first`.

### Configuration Files

The `conf/` directory contains important configuration and example files:

*   **`conf/prompts/`**: This directory holds the text files that define the various prompt templates used by the LLMs throughout the reasoning processes. These include:
    *   `aot_intro.txt`, `aot_final_answer.txt`, `aot_reflection.txt`, `aot_refinement.txt`
    *   `assessment_system_prompt.txt`, `assessment_user_prompt.txt`
    *   `l2t_initial.txt`, `l2t_node_classification.txt`, `l2t_thought_generation.txt`
    *   `far_fact_elicitation.txt`, `far_reflection_answer.txt`
*   **`conf/example_user_prompts/`**: Contains example problem statements (`problem1.txt`, `problem2.txt`) that can be used with the `--problem-filename` CLI argument.

## Development
(Content remains the same)
...

## License
(Content remains the same)
...

## LLM Call Auditing
(Content remains the same)
...
