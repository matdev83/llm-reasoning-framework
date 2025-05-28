import os
from pathlib import Path
from typing import List, Optional
from src.aot.constants import APP_TITLE

_PROMPT_DIR = Path("conf/prompts")

def _read_prompt_file(filename: str) -> str:
    file_path = _PROMPT_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

class PromptGenerator:
    AOT_INTRO = _read_prompt_file("aot_intro.txt")

    @staticmethod
    def construct_aot_step_prompt(
        problem: str,
        history: List[str],
        current_step_number_1_indexed: int,
        current_effective_max_steps: int,
        original_max_steps_config: int,
        pass_remaining_steps_pct: Optional[float]
    ) -> str:
        remaining_steps_info = ""
        if pass_remaining_steps_pct is not None and original_max_steps_config > 0:
            threshold_trigger_step = pass_remaining_steps_pct * original_max_steps_config
            if current_step_number_1_indexed >= threshold_trigger_step:
                num_steps_left_inclusive = current_effective_max_steps - current_step_number_1_indexed + 1
                if num_steps_left_inclusive >= 1:
                    plural_s = "s" if num_steps_left_inclusive > 1 else ""
                    remaining_steps_info = (
                        f"\nIMPORTANT ADVISORY: You are currently on reasoning step {current_step_number_1_indexed} "
                        f"out of a maximum of {current_effective_max_steps} reasoning steps effectively allowed for this phase. "
                        f"You have {num_steps_left_inclusive} step{plural_s} (including this current one) "
                        f"to complete the detailed reasoning before a final answer must be formulated. "
                        f"Please aim to conclude your reasoning within this limit."
                    )
        if current_step_number_1_indexed == 1:
            prompt = f"{_read_prompt_file('aot_intro.txt')}\nProblem: {problem}{remaining_steps_info}\nWhat is the first step?"
        else:
            so_far = "\n".join(history)
            prompt = (
                f"{_read_prompt_file('aot_intro.txt')}\nProblem: {problem}\nSo far:\n{so_far}{remaining_steps_info}\n"
                f"What is the next step? Output only one new, unique step and the current answer. "
                f"Do not output the final answer yet."
            )
        return prompt
    @staticmethod
    def construct_aot_final_prompt(problem: str, history: List[str]) -> str:
        so_far = "\n".join(history)
        final_answer_prompt_template = _read_prompt_file("aot_final_answer.txt")
        return final_answer_prompt_template.format(problem_placeholder=problem, so_far_placeholder=so_far)

    @staticmethod
    def construct_assessment_prompt(user_problem_text: str) -> str:
        assessment_prompt_template = _read_prompt_file("assessment_system_prompt.txt")
        return assessment_prompt_template.format(user_problem_text_placeholder=user_problem_text)
