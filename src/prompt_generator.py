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
    def construct_aot_initial_prompt(problem: str) -> str:
        return f"{_read_prompt_file('aot_intro.txt')}\n\nProblem: {problem}"
    
    @staticmethod
    def construct_aot_reflection_prompt(problem: str, history: List[str]) -> str:
        so_far = "\n".join(history)
        return f"Problem: {problem}\n\nYour reasoning so far:\n{so_far}\n\n{_read_prompt_file('aot_reflection.txt')}"
    
    @staticmethod
    def construct_aot_refinement_prompt(problem: str, history: List[str]) -> str:
        so_far = "\n".join(history)
        return f"Problem: {problem}\n\nYour reasoning so far:\n{so_far}\n\n{_read_prompt_file('aot_refinement.txt')}"
    
    @staticmethod
    def construct_aot_step_prompt(
        problem: str,
        history: List[str],
        current_step_number_1_indexed: int,
        current_effective_max_steps: int,
        original_max_steps_config: int,
        pass_remaining_steps_pct: Optional[float]
    ) -> str:
        # Legacy method for backward compatibility - now delegates to new AoT methods
        if current_step_number_1_indexed == 1:
            return PromptGenerator.construct_aot_initial_prompt(problem)
        elif current_step_number_1_indexed % 2 == 0:  # Even steps are reflections
            return PromptGenerator.construct_aot_reflection_prompt(problem, history)
        else:  # Odd steps (after 1) are refinements
            return PromptGenerator.construct_aot_refinement_prompt(problem, history)
    @staticmethod
    def construct_aot_final_prompt(problem: str, history: List[str]) -> str:
        so_far = "\n".join(history)
        final_answer_prompt_template = _read_prompt_file("aot_final_answer.txt")
        return final_answer_prompt_template.format(problem_placeholder=problem, so_far_placeholder=so_far)

    @staticmethod
    def construct_assessment_prompt(user_problem_text: str) -> str:
        assessment_prompt_template = _read_prompt_file("assessment_system_prompt.txt")
        return assessment_prompt_template.format(user_problem_text_placeholder=user_problem_text)
