import sys
import os
import argparse
import logging
from typing import Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.aot_enums import AotTriggerMode
from src.aot_dataclasses import AoTRunnerConfig
from src.aot_orchestrator import InteractiveAoTOrchestrator, AoTProcess # Import AoTProcess
from src.l2t_enums import L2TTriggerMode 
from src.aot_constants import (
    DEFAULT_MAIN_MODEL_NAMES as DEFAULT_AOT_MAIN_MODEL_NAMES, 
    DEFAULT_SMALL_MODEL_NAMES as DEFAULT_AOT_ASSESSMENT_MODEL_NAMES, 
    DEFAULT_MAX_STEPS as DEFAULT_AOT_MAX_STEPS, 
    DEFAULT_MAX_TIME_SECONDS as DEFAULT_AOT_MAX_TIME_SECONDS, 
    DEFAULT_NO_PROGRESS_LIMIT as DEFAULT_AOT_NO_PROGRESS_LIMIT, 
    DEFAULT_MAIN_TEMPERATURE as DEFAULT_AOT_MAIN_TEMPERATURE, 
    DEFAULT_ASSESSMENT_TEMPERATURE as DEFAULT_AOT_ASSESSMENT_TEMPERATURE 
)

from src.l2t_orchestrator import L2TOrchestrator, L2TProcess # Import L2TProcess
from src.l2t_dataclasses import L2TConfig, L2TSolution # L2TSolution for direct L2TProcess
from src.l2t_constants import (
    DEFAULT_L2T_CLASSIFICATION_MODEL_NAMES,
    DEFAULT_L2T_THOUGHT_GENERATION_MODEL_NAMES,
    DEFAULT_L2T_INITIAL_PROMPT_MODEL_NAMES,
    DEFAULT_L2T_CLASSIFICATION_TEMPERATURE,
    DEFAULT_L2T_THOUGHT_GENERATION_TEMPERATURE,
    DEFAULT_L2T_INITIAL_PROMPT_TEMPERATURE,
    DEFAULT_L2T_MAX_STEPS,
    DEFAULT_L2T_MAX_TOTAL_NODES,
    DEFAULT_L2T_MAX_TIME_SECONDS,
    DEFAULT_L2T_X_FMT_DEFAULT,
    DEFAULT_L2T_X_EVA_DEFAULT,
)

# Helper to define new direct processing modes
DIRECT_AOT_MODE = "aot_direct"
DIRECT_L2T_MODE = "l2t_direct"

def main():
    parser = argparse.ArgumentParser(
        description="CLI Runner for Algorithm of Thought (AoT) and Learn-to-Think (L2T) processes.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    problem_group = parser.add_mutually_exclusive_group(required=True)
    problem_group.add_argument("--problem", "-p", type=str, help="Problem/question to solve.")
    problem_group.add_argument("--problem-filename", type=str, help="File containing the problem.")

    # General arguments
    all_modes = [mode.value for mode in AotTriggerMode] + ["l2t", DIRECT_AOT_MODE, DIRECT_L2T_MODE]
    parser.add_argument(
        "--processing-mode", "--mode", dest="processing_mode", type=str,
        choices=all_modes,
        default=AotTriggerMode.ASSESS_FIRST.value,
        help=(f"Processing mode (default: {AotTriggerMode.ASSESS_FIRST.value}).\n"
              f" '{AotTriggerMode.ALWAYS_AOT.value}': Use InteractiveAoTOrchestrator (forces AoT path).\n"
              f" '{AotTriggerMode.ASSESS_FIRST.value}': Use InteractiveAoTOrchestrator (assess, then AoT or ONESHOT).\n"
              f" '{AotTriggerMode.NEVER_AOT.value}': Use InteractiveAoTOrchestrator (forces ONESHOT).\n"
              f" '{DIRECT_AOT_MODE}': Directly run AoTProcess.\n"
              f" 'l2t': Use L2TOrchestrator (which internally uses L2TProcess).\n"
              f" '{DIRECT_L2T_MODE}': Directly run L2TProcess.")
    )

    parser.add_argument("--enable-rate-limiting", action="store_true",
                        help="Enable rate limiting for LLM calls.")
    parser.add_argument("--enable-audit-logging", action="store_true",
                        help="Enable audit logging for LLM prompts and responses.")

    # AoT Configuration Group
    aot_group = parser.add_argument_group('AoT Process Configuration (used if processing-mode is AoT-related or aot_direct)')
    aot_group.add_argument("--aot-main-models", type=str, nargs='+', default=DEFAULT_AOT_MAIN_MODEL_NAMES,
                           help=f"Main LLM(s) for AoT/ONESHOT. Default: {' '.join(DEFAULT_AOT_MAIN_MODEL_NAMES)}")
    aot_group.add_argument("--aot-main-temp", type=float, default=DEFAULT_AOT_MAIN_TEMPERATURE,
                           help=f"Temperature for AoT main LLM(s). Default: {DEFAULT_AOT_MAIN_TEMPERATURE}")
    aot_group.add_argument("--aot-assess-models", type=str, nargs='+', default=DEFAULT_AOT_ASSESSMENT_MODEL_NAMES,
                           help=f"Small LLM(s) for AoT assessment. Default: {' '.join(DEFAULT_AOT_ASSESSMENT_MODEL_NAMES)}")
    aot_group.add_argument("--aot-assess-temp", type=float, default=DEFAULT_AOT_ASSESSMENT_TEMPERATURE,
                           help=f"Temperature for AoT assessment LLM(s). Default: {DEFAULT_AOT_ASSESSMENT_TEMPERATURE}")
    aot_group.add_argument("--aot-max-steps", type=int, default=DEFAULT_AOT_MAX_STEPS,
                           help=f"Max AoT reasoning steps. Default: {DEFAULT_AOT_MAX_STEPS}.")
    aot_group.add_argument("--aot-max-reasoning-tokens", type=int, default=None,
                           help="Max completion tokens for AoT reasoning phase. Enforced dynamically.")
    aot_group.add_argument("--aot-max-time", type=int, default=DEFAULT_AOT_MAX_TIME_SECONDS,
                           help=f"Overall max time for an AoT run (seconds). Default: {DEFAULT_AOT_MAX_TIME_SECONDS}s")
    aot_group.add_argument("--aot-no-progress-limit", type=int, default=DEFAULT_AOT_NO_PROGRESS_LIMIT,
                           help=f"Stop AoT if no progress for this many steps. Default: {DEFAULT_AOT_NO_PROGRESS_LIMIT}")
    aot_group.add_argument("--aot-pass-remaining-steps-pct", type=int, default=None, metavar="PCT", choices=range(0, 101),
                           help="Percentage (0-100) of original max_steps at which to inform LLM about dynamically remaining steps in AoT. Default: None.")
    aot_group.add_argument("--aot-disable-heuristic", action="store_true",
                           help="Disable the local heuristic analysis for AoT complexity assessment, always using the LLM for assessment.")

    # L2T Configuration Group
    l2t_group = parser.add_argument_group('L2T Process Configuration (used if processing-mode is l2t or l2t_direct)')
    l2t_group.add_argument("--l2t-classification-models", type=str, nargs='+', default=DEFAULT_L2T_CLASSIFICATION_MODEL_NAMES,
                           help=f"L2T classification model(s). Default: {' '.join(DEFAULT_L2T_CLASSIFICATION_MODEL_NAMES)}")
    l2t_group.add_argument("--l2t-thought-gen-models", type=str, nargs='+', default=DEFAULT_L2T_THOUGHT_GENERATION_MODEL_NAMES,
                           help=f"L2T thought generation model(s). Default: {' '.join(DEFAULT_L2T_THOUGHT_GENERATION_MODEL_NAMES)}")
    l2t_group.add_argument("--l2t-initial-prompt-models", type=str, nargs='+', default=DEFAULT_L2T_INITIAL_PROMPT_MODEL_NAMES,
                           help=f"L2T initial prompt model(s). Default: {' '.join(DEFAULT_L2T_INITIAL_PROMPT_MODEL_NAMES)}")
    l2t_group.add_argument("--l2t-classification-temp", type=float, default=DEFAULT_L2T_CLASSIFICATION_TEMPERATURE,
                           help=f"L2T classification temperature. Default: {DEFAULT_L2T_CLASSIFICATION_TEMPERATURE}")
    l2t_group.add_argument("--l2t-thought-gen-temp", type=float, default=DEFAULT_L2T_THOUGHT_GENERATION_TEMPERATURE,
                           help=f"L2T thought generation temperature. Default: {DEFAULT_L2T_THOUGHT_GENERATION_TEMPERATURE}")
    l2t_group.add_argument("--l2t-initial-prompt-temp", type=float, default=DEFAULT_L2T_INITIAL_PROMPT_TEMPERATURE,
                           help=f"L2T initial prompt temperature. Default: {DEFAULT_L2T_INITIAL_PROMPT_TEMPERATURE}")
    l2t_group.add_argument("--l2t-max-steps", type=int, default=DEFAULT_L2T_MAX_STEPS,
                           help=f"L2T max steps. Default: {DEFAULT_L2T_MAX_STEPS}")
    l2t_group.add_argument("--l2t-max-total-nodes", type=int, default=DEFAULT_L2T_MAX_TOTAL_NODES,
                           help=f"L2T max total nodes. Default: {DEFAULT_L2T_MAX_TOTAL_NODES}")
    l2t_group.add_argument("--l2t-max-time-seconds", type=int, default=DEFAULT_L2T_MAX_TIME_SECONDS,
                           help=f"L2T max time (seconds). Default: {DEFAULT_L2T_MAX_TIME_SECONDS}")
    l2t_group.add_argument("--l2t-x-fmt", dest="l2t_x_fmt_default", type=str, default=DEFAULT_L2T_X_FMT_DEFAULT,
                           help="L2T default format constraints string.")
    l2t_group.add_argument("--l2t-x-eva", dest="l2t_x_eva_default", type=str, default=DEFAULT_L2T_X_EVA_DEFAULT,
                           help="L2T default evaluation criteria string.")
    
    args = parser.parse_args()

    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level_val = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(
        level=log_level_val,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        stream=sys.stderr
    )

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key: 
        logging.critical("OPENROUTER_API_KEY environment variable not set.")
        sys.exit(1)
    
    problem_text: str
    if args.problem_filename:
        try:
            with open(args.problem_filename, 'r', encoding='utf-8') as f: problem_text = f.read()
            logging.info(f"Successfully read problem from file: {args.problem_filename}")
        except Exception as e: 
            logging.critical(f"Error reading problem file '{args.problem_filename}': {e}")
            sys.exit(1)
    else: 
        problem_text = args.problem
        if not problem_text: # Should be caught by parser group, but defensive check
            logging.critical("No problem text provided.")
            sys.exit(1)

    current_processing_mode = args.processing_mode
    solution = None
    overall_summary_str = ""

    # Prepare AoT configuration (used by InteractiveAoTOrchestrator and direct AoTProcess)
    aot_pass_remaining_steps_float: Optional[float] = None
    if args.aot_pass_remaining_steps_pct is not None:
        aot_pass_remaining_steps_float = args.aot_pass_remaining_steps_pct / 100.0
    
    aot_runner_config = AoTRunnerConfig(
        main_model_names=args.aot_main_models,
        temperature=args.aot_main_temp,
        max_steps=args.aot_max_steps,
        max_reasoning_tokens=args.aot_max_reasoning_tokens,
        max_time_seconds=args.aot_max_time,
        no_progress_limit=args.aot_no_progress_limit,
        pass_remaining_steps_pct=aot_pass_remaining_steps_float
    )

    # Prepare L2T configuration (used by L2TOrchestrator and direct L2TProcess)
    l2t_config = L2TConfig(
        classification_model_names=args.l2t_classification_models,
        thought_generation_model_names=args.l2t_thought_gen_models,
        initial_prompt_model_names=args.l2t_initial_prompt_models,
        classification_temperature=args.l2t_classification_temp,
        thought_generation_temperature=args.l2t_thought_gen_temp,
        initial_prompt_temperature=args.l2t_initial_prompt_temp,
        max_steps=args.l2t_max_steps,
        max_total_nodes=args.l2t_max_total_nodes,
        max_time_seconds=args.l2t_max_time_seconds,
        x_fmt_default=args.l2t_x_fmt_default,
        x_eva_default=args.l2t_x_eva_default,
    )

    if current_processing_mode == DIRECT_AOT_MODE:
        logging.info(f"Direct AoTProcess mode selected.")
        # Direct AoTProcess instantiation and execution
        aot_process = AoTProcess(
            aot_config=aot_runner_config,
            direct_oneshot_model_names=args.aot_main_models, # For fallback within AoTProcess
            direct_oneshot_temperature=args.aot_main_temp, # For fallback within AoTProcess
            api_key=api_key,
            enable_rate_limiting=args.enable_rate_limiting,
            enable_audit_logging=args.enable_audit_logging
        )
        # model_name for execute is for interface compatibility, actual models are in aot_config
        aot_process.execute(problem_text, model_name="N/A_direct_aot") 
        solution, overall_summary_str = aot_process.get_result()
        
        print("\nDirect AoTProcess Execution Summary:")
        print(overall_summary_str if overall_summary_str else "No summary returned.")
        
        # Assuming solution is of type Solution from aot_dataclasses for direct AoT
        if solution and solution.final_answer:
            logging.info("Direct AoTProcess completed successfully.")
        else:
            logging.error("Direct AoTProcess did not produce a final answer or solution object.")
            sys.exit(1)

    elif current_processing_mode == DIRECT_L2T_MODE:
        logging.info(f"Direct L2TProcess mode selected.")
        # Direct L2TProcess instantiation and execution
        l2t_process = L2TProcess(
            l2t_config=l2t_config,
            direct_oneshot_model_names=args.l2t_initial_prompt_models, # Fallback for L2TProcess
            direct_oneshot_temperature=args.l2t_initial_prompt_temperature, # Fallback for L2TProcess
            api_key=api_key,
            enable_rate_limiting=args.enable_rate_limiting,
            enable_audit_logging=args.enable_audit_logging
        )
        # model_name for execute is for interface compatibility
        l2t_process.execute(problem_text, model_name="N/A_direct_l2t")
        solution, overall_summary_str = l2t_process.get_result() # Expects L2TSolution, str

        print("\nDirect L2TProcess Execution Summary:")
        print(overall_summary_str if overall_summary_str else "No summary returned.")

        if solution and solution.final_answer: # Assuming solution is L2TSolution
             logging.info("Direct L2TProcess completed successfully.")
        else:
            logging.error("Direct L2TProcess did not produce a final answer or solution object.")
            # Error message might be in solution.l2t_result.error_message if it's an L2TSolution
            if solution and solution.l2t_result and solution.l2t_result.error_message:
                logging.error(f"L2TProcess error: {solution.l2t_result.error_message}")
            sys.exit(1)

    elif current_processing_mode == "l2t":
        logging.info("L2TOrchestrator mode selected.")
        l2t_orchestrator = L2TOrchestrator(
            trigger_mode=L2TTriggerMode.ALWAYS_L2T, # L2TOrchestrator handles this
            l2t_config=l2t_config,
            # Parameters for L2TOrchestrator's own fallback/assessment if it had one,
            # or for passing to L2TProcess for its fallback.
            direct_oneshot_model_names=args.l2t_initial_prompt_models, 
            direct_oneshot_temperature=args.l2t_initial_prompt_temperature,
            # Assessment params for L2TOrchestrator if it were to use ASSESS_FIRST for L2T
            assessment_model_names=args.l2t_classification_models, 
            assessment_temperature=args.l2t_classification_temperature, 
            api_key=api_key,
            use_heuristic_shortcut=True, 
            heuristic_detector=None, 
            enable_rate_limiting=args.enable_rate_limiting,
            enable_audit_logging=args.enable_audit_logging
        )
        solution, overall_summary_str = l2t_orchestrator.solve(problem_text)
        
        print("\nL2T Orchestrator Process Summary:")
        print(overall_summary_str)

        # solution here is L2TSolution
        if solution and solution.l2t_result and not solution.l2t_result.succeeded:
            logging.error(f"L2T process (via orchestrator) did not succeed. Error: {solution.l2t_result.error_message if solution.l2t_result.error_message else 'Unknown error'}")
            sys.exit(1)
        elif solution and solution.final_answer:
            logging.info("L2T process (via orchestrator) completed successfully.")
        else:
            logging.error("L2T process (via orchestrator) did not produce a final answer.")
            sys.exit(1)


    else: # AoT modes handled by InteractiveAoTOrchestrator (assess, always_aot, never_aot)
        try:
            aot_mode_enum_val = AotTriggerMode(current_processing_mode)
        except ValueError:
            logging.critical(f"Invalid AoT mode string '{current_processing_mode}' for AoT Orchestrator path. Exiting.")
            sys.exit(1)

        logging.info(f"InteractiveAoTOrchestrator mode selected: {aot_mode_enum_val}")
        
        aot_orchestrator = InteractiveAoTOrchestrator(
            trigger_mode=aot_mode_enum_val,
            aot_config=aot_runner_config, # This config is used by AoTProcess within the orchestrator
            direct_oneshot_model_names=args.aot_main_models, 
            direct_oneshot_temperature=args.aot_main_temp,
            assessment_model_names=args.aot_assess_models,
            assessment_temperature=args.aot_assess_temp,
            api_key=api_key,
            use_heuristic_shortcut=not args.aot_disable_heuristic,
            enable_rate_limiting=args.enable_rate_limiting,
            enable_audit_logging=args.enable_audit_logging
        )
        
        solution, overall_summary_str = aot_orchestrator.solve(problem_text) 

        print("\nInteractive AoT Orchestrator Summary:")
        print(overall_summary_str) 
        
        # solution here is Solution from aot_dataclasses
        if solution and solution.aot_summary_output: # This is summary from AoTProcessor via AoTProcess
            print(solution.aot_summary_output) 
        
        if solution and solution.final_answer:
            logging.info("Interactive AoT Orchestrator process completed.")
        else:
            logging.warning("Interactive AoT Orchestrator process did not produce a final answer.")
            # Consider if sys.exit(1) is appropriate based on desired behavior
            if aot_mode_enum_val != AotTriggerMode.NEVER_AOT: # Don't exit if it was just a oneshot
                 sys.exit(1)


if __name__ == "__main__":
    main()
