# Guidelines for LLM Agents: Modular LLM Reasoning Framework

This document provides essential guidelines and an overview of the Modular LLM Reasoning Framework's structure, designed to help coding agents understand, navigate, and effectively interact with the codebase. It outlines key components and their roles, enabling agents to perform tasks such as code analysis, feature development, bug fixing, and documentation updates.

## Top-Level Directory Structure

* `.gitignore`: Specifies files and directories to be ignored by Git.
* `conftest.py`: Configuration file for pytest, defining fixtures and hooks.
* `LICENSE`: Contains the project's licensing information (MIT License).
* `pyproject.toml`, `setup.py`, `requirements.txt`: Python project configuration, dependency management, and packaging metadata.
* `README.md`: Comprehensive project overview, key features, installation instructions, usage examples, CLI parameters, and development guidelines.
* `CHANGELOG.md`: A chronological overview of the key development milestones, tasks, and significant refactors in this project.
* `conf/`: Contains configuration files, prompt templates, and example user prompts.
* `context7-mcp/`: Contains MCP server configuration and tools for Context7 integration.
* `data/`: Stores runtime data, notably the `accounting.sqlite` database for LLM call auditing.
* `docs/`: Project documentation, including `l2t_paper.md`.
* `AGENTS.md`: This document, providing guidelines and structural overview for LLM agents.
* `src/`: The core source code of the framework.
* `update_changelog.py`: A utility script to automate updates to the `CHANGELOG.md` file.
* `tests/`: Contains unit and integration tests for the project components.

## `src/` - Core Source Code

This directory houses the main logic and components of the LLM Reasoning Framework.

* `__init__.py`: Initializes the `src` package.
* `cli_runner.py`: The command-line interface (CLI) entry point for running the framework. It parses arguments and orchestrates the chosen reasoning process.
* `llm_client.py`: Manages all interactions with Large Language Models (LLMs), handling API calls and responses.
* `complexity_assessor.py`: Determines the complexity of a given problem, deciding whether to use a one-shot approach or an advanced iterative reasoning process. It leverages the `heuristic_detector`.
* `heuristic_detector.py`: Provides a top-level entry point for the heuristic detection mechanism.
* `prompt_generator.py`: A generic component responsible for constructing prompts sent to LLMs.
* `response_parser.py`: A generic component for parsing and extracting information from LLM responses.
* `reasoning_process.py`: An abstract base class defining common interfaces for different reasoning processes (e.g., AoT, L2T, Hybrid).

### `src/llm_config/` - LLM Configuration

Modules for managing LLM configurations, including model selection, API keys, and rate limits.

* `__init__.py`: Initializes the `llm_config` package.
* `llm_config.py`: Defines the structure and management of LLM configurations.

### `src/aot/` - Answer On Thought (AoT) Implementation

Modules specifically designed for the Answer On Thought reasoning strategy, which implements the proper answer-first, reflection-based methodology.

* `__init__.py`: Initializes the `aot` package.
* `orchestrator.py`: Orchestrates the overall AoT problem-solving flow, managing the sequence of steps and interactions.
* `processor.py`: Implements the Answer On Thought process with initial answer generation, reflection, and iterative refinement phases.
* `constants.py`: Defines constants used within the AoT implementation.
* `dataclasses.py`: Contains data classes for structuring AoT-specific data.
* `enums.py`: Defines enumerations relevant to the AoT process.

### `src/l2t/` - Learn to Think (L2T) Implementation

Modules specifically designed for the Learn to Think reasoning strategy.

* `__init__.py`: Initializes the `l2t` package.
* `orchestrator.py`: Orchestrates the overall L2T problem-solving flow.
* `processor.py`: Implements the iterative reasoning steps for L2T, focusing on thought generation and classification.
* `prompt_generator.py`: Specialized prompt generator for L2T, adapting prompts based on the reasoning graph.
* `response_parser.py`: Specialized response parser for L2T, extracting thoughts and classifications.
* `constants.py`: Defines constants used within the L2T implementation.
* `dataclasses.py`: Contains data classes for structuring L2T-specific data.
* `enums.py`: Defines enumerations relevant to the L2T process.

#### `src/l2t_orchestrator_utils/`

Utility modules supporting the L2T orchestrator.

* `__init__.py`: Initializes the `l2t_orchestrator_utils` package.
* `oneshot_executor.py`: Handles one-shot LLM calls within the L2T context, potentially as a fallback or for specific sub-tasks.
* `summary_generator.py`: Generates summaries of the L2T reasoning trace or specific nodes.

#### `src/l2t_processor_utils/`

Utility modules supporting the L2T processor.

* `__init__.py`: Initializes the `l2t_processor_utils` package.
* `node_processor.py`: Manages the processing and classification of nodes within the L2T reasoning graph.

### `src/hybrid/` - Hybrid Reasoning Implementation

Modules specifically designed for the Hybrid reasoning strategy.

* `__init__.py`: Initializes the `hybrid` package.
* `orchestrator.py`: Orchestrates the overall Hybrid problem-solving flow.
* `processor.py`: Implements the core logic for the Hybrid reasoning process.
* `constants.py`: Defines constants used within the Hybrid implementation.
* `dataclasses.py`: Contains data classes for structuring Hybrid-specific data.
* `enums.py`: Defines enumerations relevant to the Hybrid process.

### `src/far/` - Fact-and-Reflection (FaR) Implementation

Modules specifically designed for the Fact-and-Reflection reasoning strategy.

* `__init__.py`: Initializes the `far` package.
* `orchestrator.py`: Orchestrates the overall FaR problem-solving flow, managing assessment and fallback mechanisms.
* `processor.py`: Implements the two-stage FaR process with fact elicitation and reflection phases.
* `constants.py`: Defines constants used within the FaR implementation.
* `dataclasses.py`: Contains data classes for structuring FaR-specific data.
* `enums.py`: Defines enumerations relevant to the FaR process.

### `src/heuristic_detector/` - Local Heuristic Analysis

Modules for performing local, deterministic heuristic analysis to quickly assess problem complexity without an LLM call.

* `__init__.py`: Initializes the `heuristic_detector` package.
* `main_detector.py`: The primary logic for the heuristic detection mechanism.
* `complex_conditional_keywords.py`, `creative_writing_complex.py`, `data_algo_tasks.py`, `design_architecture_keywords.py`, `explicit_decomposition_keywords.py`, `in_depth_explanation_phrases.py`, `math_logic_proof_keywords.py`, `multi_part_complex.py`, `patterns.py`, `problem_solving_keywords.py`, `simulation_modeling_keywords.py`, `specific_complex_coding_keywords.py`: These modules define specific keywords, phrases, and patterns used by the heuristic to identify problems likely requiring complex reasoning.

## `conf/` - Configuration and Prompts

This directory holds various configuration files and prompt templates.

* `conf/prompts/`: Contains text files defining the prompt templates used by different LLM interactions.
  * `aot_intro.txt`: Initial prompt for AoT reasoning steps.
  * `aot_final_answer.txt`: Prompt for synthesizing the final answer in AoT.
  * `assessment_system_prompt.txt`: System prompt for the complexity assessment LLM.
  * `assessment_user_prompt.txt`: User prompt for the complexity assessment LLM.
  * `l2t_initial.txt`: Initial prompt for the L2T process.
  * `l2t_node_classification.txt`: Prompt for classifying nodes in the L2T graph.
  * `l2t_thought_generation.txt`: Prompt for generating new thoughts/nodes in the L2T graph.
  * `far_fact_elicitation.txt`: Prompt for extracting facts in the FaR process.
  * `far_reflection_answer.txt`: Prompt for generating reflected answers in the FaR process.
* `conf/tests/`: Contains configuration and prompts specifically used for testing purposes, particularly for heuristic evaluation.
  * `prompts/example_user_prompts/`: Provides example problem statements for testing and demonstration.
    * `problem1.txt`, `problem2.txt`.
  * `prompts/hard_reasoning_problems/`: Contains prompts for hard reasoning problems, along with their solutions, used for testing and demonstration.
    * `problem_1.txt` through `problem_10.txt`
    * `problem_1_solution.txt` through `problem_10_solution.txt`
  * `prompts/heuristic/`: Contains test prompts for heuristic evaluation, categorized by complexity.
    * `complex/`: Prompts expected to trigger complex reasoning.
      * `prompt_01.txt` through `prompt_20.txt`
    * `oneshot/`: Prompts expected to be handled by one-shot processing.
      * `prompt_01.txt` through `prompt_20.txt`

## `data/` - Runtime Data

Not to be edited by agents.

## `docs/` - Documentation

* `l2t_paper.md`: Documentation or notes related to the "Learn to Think" (L2T) concept.

## `tests/` - Test Suite

This directory contains the project's test suite, organized to mirror the `src/` directory structure.

* `__init__.py`: Initializes the `tests` package.
* `aot/`: Tests for the AoT implementation.
  * `test_orchestrator.py`: Tests for the AoT orchestrator.
* `far/`: Tests for the FaR implementation.
  * `test_orchestrator.py`: Tests for the FaR orchestrator.
  * `test_processor.py`: Tests for the FaR processor.
* `l2t/`: Tests for the L2T implementation.
  * `test_l2t_orchestrator.py`: Tests for the L2T orchestrator.
  * `test_processor.py`: Tests for the L2T processor.
  * `test_prompt_generator.py`: Tests for the L2T prompt generator.
  * `test_response_parser.py`: Tests for the L2T response parser.
* `l2t_processor/`: Contains specific test cases for the L2T processor, covering various scenarios.
  * `test_backtrack_logic.py`: Tests the backtracking logic in the L2T processor.
  * `test_initial_thought_generation_failure_llm_error.py`: Tests error handling during initial thought generation due to LLM errors.
  * `test_initial_thought_generation_failure_parsing.py`: Tests error handling during initial thought generation due to parsing issues.
  * `test_max_steps_reached.py`: Tests behavior when the maximum number of steps is reached.
  * `test_successful_path_with_final_answer.py`: Tests a successful reasoning path culminating in a final answer.
* `integration/`: Contains integration tests for the framework components.
  * `test_far_integration.py`: Integration tests for the FaR reasoning process.
* `test_heuristic_detector.py`: Tests for the local heuristic analysis.
* `test_llm_accounting.py`: Tests for the integration with the `llm-accounting` library.
* `test_llm_config.py`: Tests for the LLM configuration management.
* `test_l2t_result_mutability.py`: Tests the mutability of L2T results.

## How Agents Can Use This Document

This document serves as a comprehensive map for LLM agents to understand the project's architecture and locate relevant code. Agents should refer to this guide for:

* **Code Navigation**: Quickly identify the purpose of directories and files to locate specific functionalities.
* **Contextual Understanding**: Grasp the role of each module and how it fits into the overall framework.
* **Task Execution**: Efficiently perform tasks by understanding where changes need to be made or where information can be retrieved.
* **Dependency Awareness**: Understand relationships between different components to avoid introducing breaking changes.

## Agent Capabilities and Interaction Points

LLM agents are expected to interact with various parts of this framework. Key areas for agent interaction include:

* **`src/cli_runner.py`**: The primary entry point for executing the framework. Agents can analyze this to understand how different reasoning processes are initiated and controlled.
* **`src/llm_client.py`**: Manages all LLM interactions. Agents can inspect this module to understand API call patterns, error handling, and response structures.
* **`src/complexity_assessor.py` and `src/heuristic_detector/`**: These modules are crucial for understanding how the framework determines problem complexity and selects reasoning strategies. Agents can analyze these to improve their own problem assessment capabilities.
* **`src/aot/`, `src/l2t/`, `src/hybrid/`, `src/far/`**: These directories contain the core implementations of different reasoning strategies. Agents working on specific reasoning processes should focus on these modules.
* **`conf/prompts/`**: This directory is vital for understanding the prompts used by the LLMs. Agents can analyze and suggest improvements to these prompts to enhance reasoning performance.
* **`tests/`**: Agents should leverage the test suite to understand expected behavior, write new tests for new features, or debug existing functionalities.

## Best Design Principles for LLM Agents

When performing development tasks within this codebase, LLM agents are expected to adhere to the following design principles to ensure code quality, maintainability, and robustness:

* **Modular, Layered Architecture**: Agents should strive to maintain and enhance the existing modular and layered architecture. This means keeping components loosely coupled, clearly separating concerns, and ensuring that each layer has a well-defined responsibility.
* **Pythonic Standards Employed**: All new code and modifications should follow Python's idiomatic conventions and best practices (e.g., PEP 8 for style, clear variable names, docstrings). Agents should prioritize readability and maintainability.
* **Test-Driven Development (TDD)**: Agents must adhere to TDD principles. Before introducing any changes or new functions, corresponding tests covering the introduced changes must be written. This ensures that new functionalities work as expected and prevents regressions.
* **Software Design Principles**: Agents should apply fundamental software design principles:
  * **SOLID**:
    * **S**ingle Responsibility Principle: Each module, class, or function should have only one reason to change.
    * **O**pen/Closed Principle: Software entities should be open for extension, but closed for modification.
    * **L**iskov Substitution Principle: Objects in a program should be replaceable with instances of their subtypes without altering the correctness of that program.
    * **I**nterface Segregation Principle: Clients should not be forced to depend on interfaces they do not use.
    * **D**ependency Inversion Principle: Depend upon abstractions, not concretions.
  * **KISS (Keep It Simple, Stupid)**: Favor simplicity and clarity over complexity. Avoid unnecessary abstractions or convoluted logic.
  * **DRY (Don't Repeat Yourself)**: Avoid duplicating code or logic. Strive for reusable components and functions.
