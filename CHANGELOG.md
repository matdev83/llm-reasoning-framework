# Project History

This document provides a chronological overview of the key development milestones, tasks, and significant refactors in this project.

## Commit History
### 550e45a - 2025-06-01 - Docs: Update STRUCTURE.md to reflect latest project changes
**Changes:**
- Modified ocs/STRUCTURE.md


### 2d6b2cf - 2025-06-01 - Refactor: Restructure prompt files and update dependencies. Moved prompt files to conf/tests/prompts/ and added requests and pytest to pyproject.toml.
**Changes:**
- Modified gitignore
- Modified onf/example_user_prompts/problem1.txt
- Modified onf/example_user_prompts/problem2.txt
- Modified onf/problem_1.txt
- Modified onf/problem_10.txt
- Modified onf/problem_2.txt
- Modified onf/problem_3.txt
- Modified onf/problem_4.txt
- Modified onf/problem_5.txt
- Modified onf/problem_6.txt
- Modified onf/problem_7.txt
- Modified onf/problem_8.txt
- Modified onf/problem_9.txt
- Modified ard_reasoning_problems/problem_1.txt
- Modified ard_reasoning_problems/problem_10.txt
- Modified ard_reasoning_problems/problem_10_solution.txt
- Modified ard_reasoning_problems/problem_1_solution.txt
- Modified ard_reasoning_problems/problem_2.txt
- Modified ard_reasoning_problems/problem_2_solution.txt
- Modified ard_reasoning_problems/problem_3.txt
- Modified ard_reasoning_problems/problem_3_solution.txt
- Modified ard_reasoning_problems/problem_4.txt
- Modified ard_reasoning_problems/problem_4_solution.txt
- Modified ard_reasoning_problems/problem_5.txt
- Modified ard_reasoning_problems/problem_5_solution.txt
- Modified ard_reasoning_problems/problem_6.txt
- Modified ard_reasoning_problems/problem_6_solution.txt
- Modified ard_reasoning_problems/problem_7.txt
- Modified ard_reasoning_problems/problem_7_solution.txt
- Modified ard_reasoning_problems/problem_8.txt
- Modified ard_reasoning_problems/problem_8_solution.txt
- Modified ard_reasoning_problems/problem_9.txt
- Modified ard_reasoning_problems/problem_9_solution.txt
- Modified yproject.toml
- Modified equirements.txt
- Modified onf/tests/prompts/example_user_prompts/
- Modified onf/tests/prompts/hard_reasoning_problems/


### b06d6a2 - 2025-05-31 - Refactor: Integrate llm-accounting directly and remove dummy classes
**Changes:**
- Modified src/llm_client.py
- Modified tests/test_llm_accounting.py



### adcf919 - 2025-05-31 - Refactor: Centralize LLM configuration with LLMConfig dataclass
This commit introduces a new `LLMConfig` dataclass to standardize and centralize the configuration parameters for Large Language Model (LLM) calls across the project.

Key changes include:
- Moved `llm_config.py` content into `llm_config/__init__.py` to establish `llm_config` as a Python package.
- Defined `LLMConfig` dataclass with common LLM parameters (temperature, top_p, max_tokens, etc.) and a `to_payload_dict` method for API integration.
- Updated `AoT` and `L2T` orchestrators and processors to utilize `LLMConfig` objects for LLM calls and component initialization, replacing direct parameter passing (e.g., `temperature`).
- Introduced `L2TModelConfigs` to group `LLMConfig` objects for different stages within the L2T process, enhancing clarity and maintainability.
- Modified test files to reflect the new `LLMConfig` usage, ensuring consistency and proper testing of LLM parameter handling.

This refactoring improves code organization, reduces redundancy, and simplifies the management of LLM parameters throughout the application.

**Changes:**
- Deleted src/llm_config.py
- Modified src/llm_config/__init__.py
- Modified tests/aot/test_orchestrator.py
- Modified tests/l2t/test_l2t_orchestrator.py
- Modified tests/l2t_processor/test_backtrack_logic.py
- Modified tests/l2t_processor/test_initial_thought_generation_failure_llm_error.py
- Modified tests/l2t_processor/test_initial_thought_generation_failure_parsing.py
- Modified tests/l2t_processor/test_max_steps_reached.py
- Modified tests/l2t_processor/test_successful_path_with_final_answer.py
- Modified tests/test_llm_accounting.py
- Added conftest.py

### 4c79b22 - 2025-05-29 - docs: Restructure changelog and update project documentation
**Changes:**
- Modified CHANGELOG.md
- Deleted docs/PROJECT_STRUCTURE.md
- Added docs/STRUCTURE.md


### 98d4048 - 2025-05-29 - feat: Add CHANGELOG and update project structure documentation
**Changes:**
- Added CHANGELOG.md
- Deleted docs/PROJECT_STRUCTURE.md
- Added docs/STRUCTURE.md

### feaa760 - 2025-05-28 - feat: Implement L2T backtracking logic and refine processing flow
- Add to_v_pres method in L2TGraph to enable re-exploration of parent nodes during backtracking.
- Enhance NodeProcessor to handle BACKTRACK category by re-adding parent nodes to the unprocessed queue.
- Refine L2TProcessor's termination conditions and add debug logging for better process visibility.
- Add new test test_backtrack_logic.py to validate the backtracking functionality.

**Changes:**
- Modified src/l2t/dataclasses.py
- Modified src/l2t/processor.py
- Modified src/l2t_processor_utils/node_processor.py
- Added tests/l2t_processor/test_backtrack_logic.py

### f2209ad - 2025-05-28 - Refactor: Restructure AoT and L2T modules into subpackages
This commit introduces a significant refactoring of the project's module structure.
The Algorithm of Thoughts (AoT) and Learn to Think (L2T) related modules have been
moved into dedicated subpackages (src/aot/ and src/l2t/ respectively) to improve
code organization, maintainability, and clarity.

Key changes include:
- Moved AoT and L2T core components (e.g., processors, orchestrators, dataclasses, enums, constants)
  into src/aot/ and src/l2t/ subdirectories.
- Updated all import paths across the codebase to reflect the new module structure.
- Modified src/cli_runner.py to support direct execution of AoT and L2T processes
  via new --processing-mode options (ot_direct, l2t_direct).
- Updated README.md and docs/PROJECT_STRUCTURE.md to document the new module
  structure and CLI usage.
- Adapted existing test files to align with the refactored module paths and updated
  processor instantiation patterns.

**Changes:**
- Modified README.md
- Added conftest.py
- Modified docs/PROJECT_STRUCTURE.md
- Added src/aot/__init__.py
- Renamed src/aot_constants.py to src/aot/constants.py
- Renamed src/aot_dataclasses.py to src/aot/dataclasses.py
- Renamed src/aot_enums.py to src/aot/enums.py
- Renamed src/aot_orchestrator.py to src/aot/orchestrator.py
- Renamed src/aot_processor.py to src/aot/processor.py
- Modified src/cli_runner.py
- Modified src/complexity_assessor.py
- Added src/l2t/__init__.py
- Renamed src/l2t_constants.py to src/l2t/constants.py
- Renamed src/l2t_dataclasses.py to src/l2t/dataclasses.py
- Renamed src/l2t_enums.py to src/l2t/enums.py
- Renamed src/l2t_orchestrator.py to src/l2t/orchestrator.py
- Renamed src/l2t_processor.py to src/l2t/processor.py
- Renamed src/l2t_prompt_generator.py to src/l2t/prompt_generator.py
- Renamed src/l2t_response_parser.py to src/l2t/response_parser.py
- Modified src/l2t_orchestrator_utils/oneshot_executor.py
- Modified src/l2t_orchestrator_utils/summary_generator.py
- Modified src/l2t_processor_utils/node_processor.py
- Modified src/llm_client.py
- Modified src/prompt_generator.py
- Modified src/response_parser.py
- Renamed tests/test_aot_orchestrator.py to tests/aot/test_orchestrator.py
- Renamed tests/test_l2t_orchestrator.py to tests/l2t/test_l2t_orchestrator.py
- Renamed tests/test_l2t_processor.py to tests/l2t/test_processor.py
- Renamed tests/test_l2t_prompt_generator.py to tests/l2t/test_prompt_generator.py
- Renamed tests/test_l2t_response_parser.py to tests/l2t/test_response_parser.py
- Modified tests/l2t_processor/test_initial_thought_generation_failure_llm_error.py
- Modified tests/l2t_processor/test_initial_thought_generation_failure_parsing.py
- Modified tests/l2t_processor/test_max_steps_reached.py
- Modified tests/l2t_processor/test_successful_path_with_final_answer.py
- Added tests/test_l2t_result_mutability.py
- Modified tests/test_llm_accounting.py

### 65308a0 - 2025-05-28 - feat: Update AOT and L2T modules, add project structure documentation
**Changes:**
- Modified README.md
- Modified conf/prompts/assessment_system_prompt.txt
- Modified conf/prompts/assessment_user_prompt.txt
- Added docs/PROJECT_STRUCTURE.md
- Modified src/aot_dataclasses.py
- Modified src/aot_enums.py
- Modified src/aot_orchestrator.py
- Modified src/complexity_assessor.py
- Modified src/l2t_orchestrator.py
- Modified src/l2t_orchestrator_utils/summary_generator.py
- Modified tests/test_aot_orchestrator.py
- Modified tests/test_l2t_orchestrator.py

### 07d68f4 - 2025-05-28 - Refactor: Improve modularity of reasoning processes
I've introduced a common `ReasoningProcess` interface to abstract different Chain-of-Thought (CoT) process variants.

Key changes:
- Defined `src/reasoning_process.py` with an abstract `ReasoningProcess` base class, requiring `execute` and `get_result` methods.
- Refactored `AoTOrchestrator` into `AoTProcess` (in `src/aot_orchestrator.py`) which implements `ReasoningProcess`. `InteractiveAoTOrchestrator` now uses `AoTProcess`.
- Refactored `L2TOrchestrator` into `L2TProcess` (in `src/l2t_orchestrator.py`) which implements `ReasoningProcess`. `L2TOrchestrator` now uses `L2TProcess`.
- Updated `cli_runner.py` to support direct instantiation and execution of `AoTProcess` and `L2TProcess` via new `--processing-mode` options (`aot_direct`, `l2t_direct`). Existing orchestrator-based modes continue to function, now leveraging the new process classes internally.
- I updated all relevant tests in `tests/test_aot_orchestrator.py` and `tests/test_l2t_orchestrator.py` to reflect these architectural changes. This included adjusting mocks, verifying calls to the new process methods, and fixing issues uncovered during testing.
- I fixed several minor bugs in the orchestrators and supporting utility classes that were identified during test execution.

This new architecture allows for easier introduction of new reasoning processes in a clean, modular, and extensible way, with strong encapsulation of implementation details. All existing tests have been updated and are passing.

**Changes:**
- Modified src/aot_orchestrator.py
- Modified src/cli_runner.py
- Modified src/l2t_orchestrator.py
- Modified src/l2t_orchestrator_utils/summary_generator.py
- Added src/reasoning_process.py
- Modified tests/test_aot_orchestrator.py
- Modified tests/test_l2t_orchestrator.py

### ba88776 - 2025-05-28 - feat: Integrate LLM accounting and refactor L2T process for modularity
This commit introduces significant architectural improvements to the Learn-to-Think (L2T) process, focusing on enhanced modularity, better LLM accounting, and improved prompt engineering.

Key changes include:
- LLM Accounting: Integrated rate limiting and audit logging capabilities into `LLMClient`, with configurable options exposed via CLI arguments and orchestrator parameters. This provides better control and visibility over LLM usage.
- L2T Modularity: Refactored core L2T logic into dedicated utility modules (`oneshot_executor`, `summary_generator`, `node_processor`). This simplifies `L2TOrchestrator` and `L2TProcessor`, making the codebase more maintainable and extensible.
- Prompt Budgeting: Added support for passing "remaining steps" hints to LLMs via prompt templates, enabling more efficient reasoning within a defined budget.
- Test Refactoring: Split the monolithic `test_l2t_processor.py` into smaller, scenario-specific test files for improved test suite organization and maintainability.

**Changes:**
- Modified conf/prompts/l2t_node_classification.txt
- Modified conf/prompts/l2t_thought_generation.txt
- Added data/audit_log.sqlite
- Modified src/aot_orchestrator.py
- Modified src/cli_runner.py
- Modified src/l2t_dataclasses.py
- Modified src/l2t_orchestrator.py
- Added src/l2t_orchestrator_utils/__init__.py
- Added src/l2t_orchestrator_utils/oneshot_executor.py
- Added src/l2t_orchestrator_utils/summary_generator.py
- Modified src/l2t_processor.py
- Added src/l2t_processor_utils/__init__.py
- Added src/l2t_processor_utils/node_processor.py
- Modified src/l2t_prompt_generator.py
- Modified src/llm_client.py
- Added tests/l2t_processor/test_initial_thought_generation_failure_llm_error.py
- Added tests/l2t_processor/test_initial_thought_generation_failure_parsing.py
- Added tests/l2t_processor/test_max_steps_reached.py
- Added tests/l2t_processor/test_successful_path_with_final_answer.py
- Modified tests/test_l2t_orchestrator.py
- Modified tests/test_l2t_processor.py
- Modified tests/test_llm_accounting.py

### 60adfcc - 2025-05-28 - feat: Update llm-accounting package and integrate audit logging
- Updated llm-accounting to the latest version.
- Integrated AuditLogger in LLMClient for logging prompts and responses.
- Fixed tests in test_llm_accounting.py to align with new AuditLogger API.
- Added data/audit_log.sqlite to .gitignore.

**Changes:**
- Modified .gitignore
- Deleted data/audit_log.sqlite
- Modified src/llm_client.py
- Modified tests/test_llm_accounting.py

### 2c85999 - 2025-05-27 - Make heiristic based detection more modular and robust. Create a separate set of prompts for testing of the heuristic detector.
**Changes:**
- Added conf/tests/prompts/heuristic/complex/prompt_01.txt
- Added conf/tests/prompts/heuristic/complex/prompt_02.txt
- Added conf/tests/prompts/heuristic/complex/prompt_03.txt
- Added conf/tests/prompts/heuristic/complex/prompt_04.txt
- Added conf/tests/prompts/heuristic/complex/prompt_05.txt
- Added conf/tests/prompts/heuristic/complex/prompt_06.txt
- Added conf/tests/prompts/heuristic/complex/prompt_07.txt
- Added conf/tests/prompts/heuristic/complex/prompt_08.txt
- Added conf/tests/prompts/heuristic/complex/prompt_09.txt
- Added conf/tests/prompts/heuristic/complex/prompt_10.txt
- Added conf/tests/prompts/heuristic/complex/prompt_11.txt
- Added conf/tests/prompts/heuristic/complex/prompt_12.txt
- Added conf/tests/prompts/heuristic/complex/prompt_13.txt
- Added conf/tests/prompts/heuristic/complex/prompt_14.txt
- Added conf/tests/prompts/heuristic/complex/prompt_15.txt
- Added conf/tests/prompts/heuristic/complex/prompt_16.txt
- Added conf/tests/prompts/heuristic/complex/prompt_17.txt
- Added conf/tests/prompts/heuristic/complex/prompt_18.txt
- Added conf/tests/prompts/heuristic/complex/prompt_19.txt
- Added conf/tests/prompts/heuristic/complex/prompt_20.txt
- Added conf/tests/prompts/heuristic/oneshot/prompt_01.txt
- Added conf/tests/prompts/heuristic/oneshot/prompt_02.txt
- Added conf/tests/prompts/heuristic/oneshot/prompt_03.txt
- Added conf/tests/prompts/heuristic/oneshot/prompt_04.txt
- Added conf/tests/prompts/heuristic/oneshot/prompt_05.txt
- Added conf/tests/prompts/heuristic/oneshot/prompt_06.txt
- Added conf/tests/prompts/heuristic/oneshot/prompt_07.txt
- Added conf/tests/prompts/heuristic/oneshot/prompt_08.txt
- Added conf/tests/prompts/heuristic/oneshot/prompt_09.txt
- Added conf/tests/prompts/heuristic/oneshot/prompt_10.txt
- Added conf/tests/prompts/heuristic/oneshot/prompt_11.txt
- Added conf/tests/prompts/heuristic/oneshot/prompt_12.txt
- Added conf/tests/prompts/heuristic/oneshot/prompt_13.txt
- Added conf/tests/prompts/heuristic/oneshot/prompt_14.txt
- Added conf/tests/prompts/heuristic/oneshot/prompt_15.txt
- Added conf/tests/prompts/heuristic/oneshot/prompt_16.txt
- Added conf/tests/prompts/heuristic/oneshot/prompt_17.txt
- Added conf/tests/prompts/heuristic/oneshot/prompt_18.txt
- Added conf/tests/prompts/heuristic/oneshot/prompt_19.txt
- Added conf/tests/prompts/heuristic/oneshot/prompt_20.txt
- Modified src/aot_orchestrator.py
- Modified src/complexity_assessor.py
- Modified src/heuristic_detector.py
- Added src/heuristic_detector/__init__.py
- Added src/heuristic_detector/complex_conditional_keywords.py
- Added src/heuristic_detector/creative_writing_complex.py
- Added src/heuristic_detector/data_algo_tasks.py
- Added src/heuristic_detector/design_architecture_keywords.py
- Added src/heuristic_detector/explicit_decomposition_keywords.py
- Added src/heuristic_detector/in_depth_explanation_phrases.py
- Added src/heuristic_detector/main_detector.py
- Added src/heuristic_detector/math_logic_proof_keywords.py
- Added src/heuristic_detector/multi_part_complex.py
- Added src/heuristic_detector/patterns.py
- Added src/heuristic_detector/problem_solving_keywords.py
- Added src/heuristic_detector/simulation_modeling_keywords.py
- Added src/heuristic_detector/specific_complex_coding_keywords.py
- Modified src/l2t_orchestrator.py
- Added test_regex.py
- Added tests/test_aot_orchestrator.py
- Added tests/test_heuristic_detector.py
- Modified tests/test_l2t_orchestrator.py

### 2f9e6f9 - 2025-05-27 - feat: Implement L2T heuristic bypass and refactor heuristic detector
- Introduced L2TTriggerMode enum for L2T orchestration control.
- Added L2TSolution dataclass for comprehensive L2T orchestration results.
- Refactored heuristic detection logic into a new HeuristicDetector class in src/heuristic_detector.py.
- Updated ComplexityAssessor to use the new HeuristicDetector.
- Modified L2TOrchestrator to incorporate assessment and heuristic bypass, allowing for conditional one-shot or L2T process execution.
- Ensured both AoT and L2T processes now share the same heuristic detection function via ComplexityAssessor.
- Updated test_l2t_orchestrator.py to reflect new L2TOrchestrator signature and behavior, and fixed related test assertions.

**Changes:**
- Modified src/complexity_assessor.py
- Added src/heuristic_detector.py
- Modified src/l2t_dataclasses.py
- Added src/l2t_enums.py
- Modified src/l2t_orchestrator.py
- Modified tests/test_l2t_orchestrator.py

### f524d7b - 2025-05-27 - Fix linter errors in tests
**Changes:**
- Modified tests/test_l2t_orchestrator.py
- Modified tests/test_l2t_processor.py

### f042ae8 - 2025-05-27 - Committing changes as requested
**Changes:**
- Modified requirements.txt
- Modified src/cli_runner.py
- Modified src/l2t_dataclasses.py
- Modified src/l2t_orchestrator.py
- Modified src/l2t_processor.py
- Modified tests/test_l2t_orchestrator.py
- Modified tests/test_l2t_processor.py
- Modified tests/test_l2t_prompt_generator.py
- Modified tests/test_llm_accounting.py

### 50dd3be - 2025-05-27 - I've implemented a new reasoning process to help you with your code. This new process is based on the Learn-to-Think (L2T) paper, but for now, I've focused on the core graph-based reasoning flow.
Here's a summary of the changes I made:
- I introduced new data structures for this process, which you can find in `src/l2t_dataclasses.py`, along with some constants in `src/l2t_constants.py`.
- I've set up prompt generation for the different stages of this new process. You can see the generator in `src/l2t_prompt_generator.py` and the templates in `conf/prompts/l2t_*.txt`.
- I've also implemented response parsing in `src/l2t_response_parser.py` to handle the outputs for each stage.
- The core logic for executing this new process step-by-step is managed by a new component in `src/l2t_processor.py`. This includes constructing a graph, classifying nodes, and generating new thoughts.
- To set up and run this new process, I've developed an orchestrator in `src/l2t_orchestrator.py`.
- I've updated `cli_runner.py` so you can use this new 'l2t' processing mode. There are also new command-line arguments for configuring it.
- I've added some basic unit tests for these new components in `tests/test_l2t_*.py` to verify the core logic.

This means you can now run this new reasoning process as an alternative to the existing method by using a command-line flag.

**Changes:**
- Added conf/prompts/l2t_initial.txt
- Added conf/prompts/l2t_node_classification.txt
- Added conf/prompts/l2t_thought_generation.txt
- Modified src/cli_runner.py
- Added src/l2t_constants.py
- Added src/l2t_dataclasses.py
- Added src/l2t_orchestrator.py
- Added src/l2t_processor.py
- Added src/l2t_prompt_generator.py
- Added src/l2t_response_parser.py
- Added tests/__init__.py
- Added tests/test_l2t_orchestrator.py
- Added tests/test_l2t_processor.py
- Added tests/test_l2t_prompt_generator.py
- Added tests/test_l2t_response_parser.py

### 28d0159 - 2025-05-27 - Update docs
**Changes:**
- Added docs/l2t_paper.md

### ae9c1eb - 2025-05-27 - Remove stale docs
**Changes:**
- Deleted docs/l2t_paper.pdf
- Deleted docs/paper_text.txt

### b9631e8 - 2025-05-26 - Update docs
**Changes:**
- Added docs/paper_text.txt

### de14f0c - 2025-05-26 - Upload paper
**Changes:**
- Added docs/l2t_paper.pdf

### ea83151 - 2025-05-26 - Fix README.md
**Changes:**
- Modified README.md

### 636d12f - 2025-05-26 - Update README.md
**Changes:**
- Modified README.md

### ba4702b - 2025-05-26 - Update README.md
**Changes:**
- Modified README.md

### 43c02d3 - 2025-05-26 - Update README.md
**Changes:**
- Modified README.md

### cc34e32 - 2025-05-26 - Fix README.md
**Changes:**
- Modified README.md

### 9c117e0 - 2025-05-26 - Fix readme
**Changes:**
- Modified README.md

### 598af0d - 2025-05-26 - Fix readme
**Changes:**
- Modified README.md

### 028402c - 2025-05-26 - Merge branch 'llm-accounting-integration'
**Changes:**
- No explicit file changes listed, likely a merge commit.

### ba67cf5 - 2025-05-26 - Committing local changes before merge
**Changes:**
- Modified .gitignore
- Modified README.md
- Modified requirements.txt
- Modified setup.py
- Modified src/llm_client.py
- Added tests/test_llm_accounting.py

### b86568e - 2025-05-26 - Fix usage of llm-accounting
**Changes:**
- Modified .gitignore
- Modified requirements.txt
- Modified src/llm_client.py

### d89af4c - 2025-05-26 - Integrate llm-accounting for LLM call auditing
This commit integrates the llm-accounting library to track and log all outgoing requests to remote LLMs.

Key changes:
- Added `llm-accounting` to project dependencies.
- Initialized `llm-accounting` in the `LLMClient` using the default SQLite backend (llm_accounting.db).
- Wrapped LLM API calls in `LLMClient.call` to log request and response details, including model name, prompt, tokens, duration, and cost.
- Implemented a comprehensive test suite in `tests/test_llm_accounting.py` to verify the logging functionality, covering successful calls, API errors, network errors, and model failover scenarios.
- Updated `README.md` to document the new LLM call auditing feature.

This integration provides a persistent audit trail of LLM interactions, which is valuable for cost tracking, debugging, monitoring usage patterns, and performance analysis.

**Changes:**
- Modified README.md
- Modified requirements.txt
- Modified setup.py
- Modified src/llm_client.py
- Added tests/test_llm_accounting.py

### 75a9304 - 2025-05-26 - Fix README.md
**Changes:**
- Modified README.md

### 1df405c - 2025-05-26 - Init
**Changes:**
- Added .gitignore
- Added LICENSE
- Added README.md
- Added conf/example_user_prompts/problem1.txt
- Added conf/example_user_prompts/problem2.txt
- Added conf/prompts/aot_final_answer.txt
- Added conf/prompts/aot_intro.txt
- Added conf/prompts/assessment_system_prompt.txt
- Added conf/prompts/assessment_user_prompt.txt
- Added pyproject.toml
- Added requirements.txt
- Added setup.py
- Added src/__init__.py
- Added src/aot_constants.py
- Added src/aot_dataclasses.py
- Added src/aot_enums.py
- Added src/aot_orchestrator.py
- Added src/aot_processor.py
- Added src/cli_runner.py
- Added src/complexity_assessor.py
- Added src/llm_client.py
- Added src/prompt_generator.py
- Added src/response_parser.py
