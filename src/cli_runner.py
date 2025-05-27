import sys
import os
import argparse
import logging
from typing import Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.aot_enums import AotTriggerMode
from src.aot_dataclasses import AoTRunnerConfig
from src.aot_orchestrator import InteractiveAoTOrchestrator
from src.aot_constants import (
    DEFAULT_MAIN_MODEL_NAMES as DEFAULT_AOT_MAIN_MODEL_NAMES, # Renamed for clarity
    DEFAULT_SMALL_MODEL_NAMES as DEFAULT_AOT_ASSESSMENT_MODEL_NAMES, # Renamed
    DEFAULT_MAX_STEPS as DEFAULT_AOT_MAX_STEPS, # Renamed
    DEFAULT_MAX_TIME_SECONDS as DEFAULT_AOT_MAX_TIME_SECONDS, # Renamed
    DEFAULT_NO_PROGRESS_LIMIT as DEFAULT_AOT_NO_PROGRESS_LIMIT, # Renamed
    DEFAULT_MAIN_TEMPERATURE as DEFAULT_AOT_MAIN_TEMPERATURE, # Renamed
    DEFAULT_ASSESSMENT_TEMPERATURE as DEFAULT_AOT_ASSESSMENT_TEMPERATURE # Renamed
)

from src.l2t_orchestrator import L2TOrchestrator
from src.l2t_dataclasses import L2TConfig
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

def main():
    parser = argparse.ArgumentParser(
        description="CLI Runner for Algorithm of Thought (AoT) and Learn-to-Think (L2T) processes.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    problem_group = parser.add_mutually_exclusive_group(required=True)
    problem_group.add_argument("--problem", "-p", type=str, help="Problem/question to solve.")
    problem_group.add_argument("--problem-filename", type=str, help="File containing the problem.")

    # General arguments
    parser.add_argument(
        "--processing-mode", "--mode", dest="processing_mode", type=str,
        choices=[mode.value for mode in AotTriggerMode] + ["l2t"], # Added 'l2t'
        default=AotTriggerMode.ASSESS_FIRST.value,
        help=(f"Processing mode (default: {AotTriggerMode.ASSESS_FIRST.value}).\n"
              f" '{AotTriggerMode.ALWAYS_AOT.value}': Force AoT process.\n"
              f" '{AotTriggerMode.ASSESS_FIRST.value}': Use small LLM to decide if AoT or ONESHOT.\n"
              f" '{AotTriggerMode.NEVER_AOT.value}': Force ONESHOT (direct answer).\n"
              f" 'l2t': Use Learn-to-Think process.")
    )

    # AoT Configuration Group
    aot_group = parser.add_argument_group('AoT Process Configuration (used if processing-mode is AoT-related)')
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
    l2t_group = parser.add_argument_group('L2T Process Configuration (used if processing-mode is l2t)')
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
                           help="L2T default format constraints string. Use with caution if string contains shell special characters.")
    l2t_group.add_argument("--l2t-x-eva", dest="l2t_x_eva_default", type=str, default=DEFAULT_L2T_X_EVA_DEFAULT,
                           help="L2T default evaluation criteria string. Use with caution if string contains shell special characters.")
    
    args = parser.parse_args()

    # Setup logging
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level_val = getattr(logging, log_level_str, logging.INFO)
    if not isinstance(log_level_val, int):
        print(f"Warning: Invalid LOG_LEVEL '{log_level_str}'. Defaulting to INFO.", file=sys.stderr)
        log_level_val = logging.INFO
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
            with open(args.problem_filename, 'r', encoding='utf-8') as f: 
                problem_text = f.read()
            logging.info(f"Successfully read problem from file: {args.problem_filename}")
        except Exception as e: 
            logging.critical(f"Error reading problem file '{args.problem_filename}': {e}")
            sys.exit(1)
    else: 
        problem_text = args.problem
        if not problem_text:
            logging.critical("No problem text provided either directly or via file.")
            sys.exit(1)

    # Main logic based on processing mode
    current_processing_mode = args.processing_mode

    if current_processing_mode == "l2t":
        logging.info("L2T mode selected.")
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
        l2t_orchestrator = L2TOrchestrator(l2t_config=l2t_config, api_key=api_key)
        l2t_result, l2t_summary_str = l2t_orchestrator.solve(problem_text)
        
        print("\nL2T Process Summary:")
        print(l2t_summary_str)

        if not l2t_result.succeeded:
            logging.error(f"L2T process did not succeed. Error: {l2t_result.error_message}")
            sys.exit(1)
        else:
            logging.info("L2T process completed successfully.")

    else: # AoT modes (assess, always, never)
        try:
            aot_mode_enum_val = AotTriggerMode(current_processing_mode)
        except ValueError:
            logging.critical(f"Invalid AoT mode string '{current_processing_mode}' for AoT path. Exiting.")
            sys.exit(1)

        logging.info(f"AoT mode selected: {aot_mode_enum_val}")
        
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
        
        aot_orchestrator = InteractiveAoTOrchestrator(
            trigger_mode=aot_mode_enum_val,
            aot_config=aot_runner_config,
            direct_oneshot_model_names=args.aot_main_models, 
            direct_oneshot_temperature=args.aot_main_temp,
            assessment_model_names=args.aot_assess_models,
            assessment_temperature=args.aot_assess_temp,
            api_key=api_key,
            use_heuristic_shortcut=not args.aot_disable_heuristic 
        )
        
        solution, overall_summary_str = aot_orchestrator.solve(problem_text) 

        print(overall_summary_str) 

        if solution.aot_summary_output: 
            print(solution.aot_summary_output) 
        
        if not solution.final_answer and aot_mode_enum_val != AotTriggerMode.NEVER_AOT:
            logging.warning("AoT process did not produce a final answer.")
            # Consider if sys.exit(1) is appropriate here based on desired behavior for AoT non-success

if __name__ == "__main__":
    main()
