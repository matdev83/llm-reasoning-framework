import os
import sys
import argparse
import logging
from typing import Optional

from src.aot_enums import AotTriggerMode
from src.aot_dataclasses import AoTRunnerConfig
from src.aot_constants import (
    DEFAULT_MAIN_MODEL_NAMES, DEFAULT_SMALL_MODEL_NAMES,
    DEFAULT_MAX_STEPS, DEFAULT_MAX_TIME_SECONDS, DEFAULT_NO_PROGRESS_LIMIT,
    DEFAULT_MAIN_TEMPERATURE, DEFAULT_ASSESSMENT_TEMPERATURE
)
from src.aot_orchestrator import InteractiveAoTOrchestrator

def main():
    parser = argparse.ArgumentParser(
        description="Interactive Algorithm of Thought (AoT) Solver with model failover and dynamic resource management.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    problem_group = parser.add_mutually_exclusive_group(required=True)
    problem_group.add_argument("--problem", "-p", type=str, help="Problem/question to solve.")
    problem_group.add_argument("--problem-filename", type=str, help="File containing the problem.")

    parser.add_argument(
        "--aot-mode", type=AotTriggerMode, choices=list(AotTriggerMode), default=AotTriggerMode.ASSESS_FIRST,
        help=f"AoT trigger mode (default: {AotTriggerMode.ASSESS_FIRST.value}).\n"
             f" '{AotTriggerMode.ALWAYS_AOT.value}': Force AoT process.\n"
             f" '{AotTriggerMode.ASSESS_FIRST.value}': Use small LLM to decide if AoT or ONESHOT.\n"
             f" '{AotTriggerMode.NEVER_AOT.value}': Force ONESHOT (direct answer)."
    )
    parser.add_argument("--main-models", dest="main_model_names", type=str, nargs='+', default=DEFAULT_MAIN_MODEL_NAMES,
                        help=f"Main LLM(s) for AoT/ONESHOT. Default: {' '.join(DEFAULT_MAIN_MODEL_NAMES)}")
    parser.add_argument("--main-temp", dest="main_temperature", type=float, default=DEFAULT_MAIN_TEMPERATURE,
                        help=f"Temperature for main LLM(s). Default: {DEFAULT_MAIN_TEMPERATURE}")
    parser.add_argument("--assess-models", dest="assessment_model_names", type=str, nargs='+', default=DEFAULT_SMALL_MODEL_NAMES,
                        help=f"Small LLM(s) for assessment. Default: {' '.join(DEFAULT_SMALL_MODEL_NAMES)}")
    parser.add_argument("--assess-temp", dest="assessment_temperature", type=float, default=DEFAULT_ASSESSMENT_TEMPERATURE,
                        help=f"Temperature for assessment LLM(s). Default: {DEFAULT_ASSESSMENT_TEMPERATURE}")

    aot_group = parser.add_argument_group('AoT Process Configuration (used if AoT is triggered)')
    aot_group.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                           help=f"Max AoT reasoning steps. Default: {DEFAULT_MAX_STEPS}.")
    aot_group.add_argument("--max-reasoning-tokens", type=int, default=None,
                           help="Max completion tokens for AoT reasoning phase. Enforced dynamically.")
    aot_group.add_argument("--max-time", type=int, default=DEFAULT_MAX_TIME_SECONDS,
                           help=f"Overall max time for an AoT run (seconds), also used for predictive step limiting. Default: {DEFAULT_MAX_TIME_SECONDS}s")
    aot_group.add_argument("--no-progress-limit", type=int, default=DEFAULT_NO_PROGRESS_LIMIT,
                           help=f"Stop AoT if no progress for this many steps. Default: {DEFAULT_NO_PROGRESS_LIMIT}")
    aot_group.add_argument(
        "--pass-remaining-steps-pct",
        type=int, # Changed to int for argparse choices
        default=None,
        metavar="PCT",
        choices=range(0, 101), # Standard percentage range
        help="Percentage (0-100) of original max_steps at which to inform LLM about dynamically remaining steps. Default: None."
    )
    parser.add_argument(
        "--disable-heuristic",
        action="store_true",
        help="Disable the local heuristic analysis for complexity assessment, always using the LLM for assessment."
    )
    
    args = parser.parse_args()

    # Setup logging
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level_val = getattr(logging, log_level_str, logging.INFO)
    if not isinstance(log_level_val, int): # Check if getattr returned a valid level name
        # Use print to stderr for this pre-logging configuration warning
        print(f"Warning: Invalid LOG_LEVEL '{log_level_str}'. Defaulting to INFO.", file=sys.stderr)
        log_level_val = logging.INFO

    logging.basicConfig(
        level=log_level_val,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        stream=sys.stderr # Direct logs to stderr
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
        if not problem_text: # Should be caught by argparse required=True, but defensive check
            logging.critical("No problem text provided either directly or via file.")
            sys.exit(1)

    pass_remaining_steps_float: Optional[float] = None
    if args.pass_remaining_steps_pct is not None:
        pass_remaining_steps_float = args.pass_remaining_steps_pct / 100.0
    
    aot_runner_config = AoTRunnerConfig(
        main_model_names=args.main_model_names,
        temperature=args.main_temperature,
        max_steps=args.max_steps,
        max_reasoning_tokens=args.max_reasoning_tokens,
        max_time_seconds=args.max_time,
        no_progress_limit=args.no_progress_limit,
        pass_remaining_steps_pct=pass_remaining_steps_float
    )
    
    orchestrator = InteractiveAoTOrchestrator(
        trigger_mode=args.aot_mode,
        aot_config=aot_runner_config,
        direct_oneshot_model_names=args.main_model_names, # Using main models for oneshot too
        direct_oneshot_temperature=args.main_temperature,
        assessment_model_names=args.assessment_model_names,
        assessment_temperature=args.assessment_temperature,
        api_key=api_key,
        use_heuristic_shortcut=not args.disable_heuristic # Pass the new parameter
    )
    
    solution, overall_summary_str = orchestrator.solve(problem_text) # Unpack the tuple

    print(overall_summary_str) # Print the overall summary

    if solution.aot_summary_output: # Check if AoT summary exists
        print(solution.aot_summary_output) # Print the AoT summary

if __name__ == "__main__":
    main()
