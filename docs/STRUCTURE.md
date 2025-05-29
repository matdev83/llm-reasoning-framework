# Project Structure: Modular LLM Reasoning Framework

This document outlines the directory and file structure of the Modular LLM Reasoning Framework, providing an overview of its components and their roles within the system.

## Top-Level Directory Structure

*   `.gitignore`: Specifies files and directories to be ignored by Git.
*   `conftest.py`: Configuration file for pytest, defining fixtures and hooks.
*   `LICENSE`: Contains the project's licensing information (MIT License).
*   `pyproject.toml`, `setup.py`, `requirements.txt`: Python project configuration, dependency management, and packaging metadata.
*   `README.md`: Comprehensive project overview, key features, installation instructions, usage examples, CLI parameters, and development guidelines.
*   `test_regex.py`: A utility script, likely for testing regular expressions used within the project.
*   `conf/`: Contains configuration files, prompt templates, and example user prompts.
*   `data/`: Stores runtime data, notably the `accounting.sqlite` database for LLM call auditing.
*   `docs/`: Project documentation, including this file (`PROJECT_STRUCTURE.md`) and `l2t_paper.md`.
*   `src/`: The core source code of the framework.
*   `tests/`: Contains unit and integration tests for the project components.

## `src/` - Core Source Code

This directory houses the main logic and components of the LLM Reasoning Framework.

*   `__init__.py`: Initializes the `src` package.
*   `cli_runner.py`: The command-line interface (CLI) entry point for running the framework. It parses arguments and orchestrates the chosen reasoning process.
*   `llm_client.py`: Manages all interactions with Large Language Models (LLMs), handling API calls and responses.
*   `complexity_assessor.py`: Determines the complexity of a given problem, deciding whether to use a one-shot approach or an advanced iterative reasoning process. It leverages the `heuristic_detector`.
*   `heuristic_detector.py`: Provides a top-level entry point for the heuristic detection mechanism.
*   `prompt_generator.py`: A generic component responsible for constructing prompts sent to LLMs.
*   `response_parser.py`: A generic component for parsing and extracting information from LLM responses.
*   `reasoning_process.py`: Likely an abstract base class or a utility module defining common interfaces or functionalities for different reasoning processes (e.g., AoT, L2T).

### `src/aot/` - Algorithm of Thoughts (AoT) Implementation

Modules specifically designed for the Algorithm of Thoughts reasoning strategy.

*   `__init__.py`: Initializes the `aot` package.
*   `orchestrator.py`: Orchestrates the overall AoT problem-solving flow, managing the sequence of steps and interactions.
*   `processor.py`: Implements the iterative reasoning steps for AoT, including prompt construction, LLM interaction, and response parsing.
*   `constants.py`: Defines constants used within the AoT implementation.
*   `dataclasses.py`: Contains data classes for structuring AoT-specific data.
*   `enums.py`: Defines enumerations relevant to the AoT process.

### `src/l2t/` - Learn to Think (L2T) Implementation

Modules specifically designed for the Learn to Think reasoning strategy.

*   `__init__.py`: Initializes the `l2t` package.
*   `orchestrator.py`: Orchestrates the overall L2T problem-solving flow.
*   `processor.py`: Implements the iterative reasoning steps for L2T, focusing on thought generation and classification.
*   `prompt_generator.py`: Specialized prompt generator for L2T, adapting prompts based on the reasoning graph.
*   `response_parser.py`: Specialized response parser for L2T, extracting thoughts and classifications.
*   `constants.py`: Defines constants used within the L2T implementation.
*   `dataclasses.py`: Contains data classes for structuring L2T-specific data.
*   `enums.py`: Defines enumerations relevant to the L2T process.

#### `src/l2t_orchestrator_utils/`

Utility modules supporting the L2T orchestrator.

*   `__init__.py`: Initializes the `l2t_orchestrator_utils` package.
*   `oneshot_executor.py`: Handles one-shot LLM calls within the L2T context, potentially as a fallback or for specific sub-tasks.
*   `summary_generator.py`: Generates summaries of the L2T reasoning trace or specific nodes.

#### `src/l2t_processor_utils/`

Utility modules supporting the L2T processor.

*   `__init__.py`: Initializes the `l2t_processor_utils` package.
*   `node_processor.py`: Manages the processing and classification of nodes within the L2T reasoning graph.

### `src/heuristic_detector/` - Local Heuristic Analysis

Modules for performing local, deterministic heuristic analysis to quickly assess problem complexity without an LLM call.

*   `__init__.py`: Initializes the `heuristic_detector` package.
*   `main_detector.py`: The primary logic for the heuristic detection mechanism.
*   `complex_conditional_keywords.py`, `creative_writing_complex.py`, `data_algo_tasks.py`, `design_architecture_keywords.py`, `explicit_decomposition_keywords.py`, `in_depth_explanation_phrases.py`, `math_logic_proof_keywords.py`, `multi_part_complex.py`, `patterns.py`, `problem_solving_keywords.py`, `simulation_modeling_keywords.py`, `specific_complex_coding_keywords.py`: These modules define specific keywords, phrases, and patterns used by the heuristic to identify problems likely requiring complex reasoning.

## `conf/` - Configuration and Prompts

This directory holds various configuration files and prompt templates.

*   `conf/prompts/`: Contains text files defining the prompt templates used by different LLM interactions.
    *   `aot_intro.txt`: Initial prompt for AoT reasoning steps.
    *   `aot_final_answer.txt`: Prompt for synthesizing the final answer in AoT.
    *   `assessment_system_prompt.txt`: System prompt for the complexity assessment LLM.
    *   `assessment_user_prompt.txt`: User prompt for the complexity assessment LLM.
    *   `l2t_initial.txt`: Initial prompt for the L2T process.
    *   `l2t_node_classification.txt`: Prompt for classifying nodes in the L2T graph.
    *   `l2t_thought_generation.txt`: Prompt for generating new thoughts/nodes in the L2T graph.
*   `conf/example_user_prompts/`: Provides example problem statements for testing and demonstration.
    *   `problem1.txt`, `problem2.txt`.
*   `conf/tests/`: Contains configuration and prompts specifically used for testing purposes, particularly for heuristic evaluation.
    *   `prompts/heuristic/`: Contains test prompts for heuristic evaluation, categorized by complexity.
        *   `complex/`: Prompts expected to trigger complex reasoning.
            *   `prompt_01.txt` through `prompt_20.txt`
        *   `oneshot/`: Prompts expected to be handled by one-shot processing.
            *   `prompt_01.txt` through `prompt_20.txt`

## `data/` - Runtime Data

*   `accounting.sqlite`: An SQLite database created by the `llm-accounting` library to store a detailed audit trail of all LLM calls, including model used, token counts, duration, and cost.

## `docs/` - Documentation

*   `PROJECT_STRUCTURE.md`: This document, detailing the project's file and directory structure.
*   `l2t_paper.md`: Likely contains documentation or notes related to the "Learn to Think" concept.

## `tests/` - Test Suite

This directory contains the project's test suite, organized to mirror the `src/` directory structure.

*   `__init__.py`: Initializes the `tests` package.
*   `aot/`: Tests for the AoT implementation.
    *   `test_orchestrator.py`: Tests for the AoT orchestrator.
*   `l2t/`: Tests for the L2T implementation.
    *   `test_l2t_orchestrator.py`: Tests for the L2T orchestrator.
    *   `test_processor.py`: Tests for the L2T processor.
    *   `test_prompt_generator.py`: Tests for the L2T prompt generator.
    *   `test_response_parser.py`: Tests for the L2T response parser.
*   `l2t_processor/`: Contains specific test cases for the L2T processor, covering various scenarios.
    *   `test_initial_thought_generation_failure_llm_error.py`: Tests error handling during initial thought generation due to LLM errors.
    *   `test_initial_thought_generation_failure_parsing.py`: Tests error handling during initial thought generation due to parsing issues.
    *   `test_max_steps_reached.py`: Tests behavior when the maximum number of steps is reached.
    *   `test_successful_path_with_final_answer.py`: Tests a successful reasoning path culminating in a final answer.
*   `test_heuristic_detector.py`: Tests for the local heuristic analysis.
*   `test_llm_accounting.py`: Tests for the integration with the `llm-accounting` library.
*   `test_l2t_result_mutability.py`: Tests the mutability of L2T results.
