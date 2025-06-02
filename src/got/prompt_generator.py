from typing import List
from . import constants
from .dataclasses import GoTThought # Assuming GoTThought is in dataclasses

class GoTPromptGenerator:
    @staticmethod
    def construct_initial_thought_prompt(problem_description: str) -> str:
        return constants.INITIAL_THOUGHT_PROMPT_TEMPLATE.format(
            problem_description=problem_description
        )

    @staticmethod
    def construct_expand_thought_prompt(
        problem_description: str, parent_thought: GoTThought, max_new_thoughts: int
    ) -> str:
        return constants.EXPAND_THOUGHT_PROMPT_TEMPLATE.format(
            problem_description=problem_description,
            parent_thought_content=parent_thought.content,
            max_new_thoughts=max_new_thoughts,
        )

    @staticmethod
    def construct_aggregate_thoughts_prompt(
        problem_description: str, thoughts_to_aggregate: List[GoTThought]
    ) -> str:
        formatted_thoughts = "\n".join(
            [f"Thought: {t.content}" for t in thoughts_to_aggregate]
        )
        return constants.AGGREGATE_THOUGHTS_PROMPT_TEMPLATE.format(
            problem_description=problem_description,
            thoughts_to_aggregate=formatted_thoughts,
        )

    @staticmethod
    def construct_refine_thought_prompt(
        problem_description: str, thought_to_refine: GoTThought
    ) -> str:
        return constants.REFINE_THOUGHT_PROMPT_TEMPLATE.format(
            problem_description=problem_description,
            thought_to_refine=thought_to_refine.content,
        )

    @staticmethod
    def construct_score_thought_prompt(
        problem_description: str, thought_to_score: GoTThought
    ) -> str:
        return constants.SCORE_THOUGHT_PROMPT_TEMPLATE.format(
            problem_description=problem_description,
            thought_to_score=thought_to_score.content,
        )
