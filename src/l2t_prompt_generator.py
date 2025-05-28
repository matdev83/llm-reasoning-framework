import os
from typing import Optional

from src.l2t_dataclasses import L2TConfig

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "conf", "prompts")
_L2T_INITIAL_PROMPT_FILE = os.path.join(_PROMPT_DIR, "l2t_initial.txt")
_L2T_NODE_CLASSIFICATION_PROMPT_FILE = os.path.join(
    _PROMPT_DIR, "l2t_node_classification.txt"
)
_L2T_THOUGHT_GENERATION_PROMPT_FILE = os.path.join(
    _PROMPT_DIR, "l2t_thought_generation.txt"
)


def _read_prompt_template(file_path: str) -> str:
    """Helper function to read prompt template from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # Fallback for environments where the relative path might be tricky (e.g. some testing setups)
        # Try path relative to current working directory if direct path fails
        alt_path = os.path.join("conf", "prompts", os.path.basename(file_path))
        try:
            with open(alt_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
             raise FileNotFoundError(f"Prompt template file not found at {file_path} or {alt_path}")


class L2TPromptGenerator:
    def __init__(self, l2t_config: Optional[L2TConfig] = None):
        self.l2t_config = l2t_config if l2t_config else L2TConfig()
        self._initial_prompt_template = _read_prompt_template(
            _L2T_INITIAL_PROMPT_FILE
        )
        self._node_classification_prompt_template = _read_prompt_template(
            _L2T_NODE_CLASSIFICATION_PROMPT_FILE
        )
        self._thought_generation_prompt_template = _read_prompt_template(
            _L2T_THOUGHT_GENERATION_PROMPT_FILE
        )

    def construct_l2t_initial_prompt(
        self, problem_text: str, x_fmt: Optional[str] = None, x_eva: Optional[str] = None
    ) -> str:
        """
        Constructs the initial prompt for the L2T process.
        """
        fmt = x_fmt if x_fmt is not None else self.l2t_config.x_fmt_default
        eva = x_eva if x_eva is not None else self.l2t_config.x_eva_default

        return (
            self._initial_prompt_template.replace("{{problem_text}}", problem_text)
            .replace("{{x_fmt}}", fmt)
            .replace("{{x_eva}}", eva)
        )

    def construct_l2t_node_classification_prompt(
        self,
        graph_context: str,
        node_to_classify_content: str,
        x_eva: Optional[str] = None,
        remaining_steps_hint: Optional[int] = None, # New parameter
    ) -> str:
        """
        Constructs the prompt for classifying a thought node.
        """
        eva = x_eva if x_eva is not None else self.l2t_config.x_eva_default
        
        prompt = (
            self._node_classification_prompt_template.replace(
                "{{graph_context}}", graph_context
            )
            .replace("{{node_to_classify_content}}", node_to_classify_content)
            .replace("{{x_eva}}", eva)
        )
        
        if remaining_steps_hint is not None:
            prompt = prompt.replace(
                "{{remaining_steps_hint}}",
                f"You have approximately {remaining_steps_hint} reasoning steps remaining. Please try to converge to a final answer."
            )
        else:
            prompt = prompt.replace("{{remaining_steps_hint}}", "") # Remove placeholder if no hint

        return prompt

    def construct_l2t_thought_generation_prompt(
        self,
        graph_context: str,
        parent_node_content: str,
        x_fmt: Optional[str] = None,
        x_eva: Optional[str] = None,
        remaining_steps_hint: Optional[int] = None, # New parameter
    ) -> str:
        """
        Constructs the prompt for generating new thoughts.
        """
        fmt = x_fmt if x_fmt is not None else self.l2t_config.x_fmt_default
        eva = x_eva if x_eva is not None else self.l2t_config.x_eva_default
        
        prompt = (
            self._thought_generation_prompt_template.replace(
                "{{graph_context}}", graph_context
            )
            .replace("{{parent_node_content}}", parent_node_content)
            .replace("{{x_fmt}}", fmt)
            .replace("{{x_eva}}", eva)
        )

        if remaining_steps_hint is not None:
            prompt = prompt.replace(
                "{{remaining_steps_hint}}",
                f"You have approximately {remaining_steps_hint} reasoning steps remaining. Please try to converge to a final answer."
            )
        else:
            prompt = prompt.replace("{{remaining_steps_hint}}", "") # Remove placeholder if no hint

        return prompt
