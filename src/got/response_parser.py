import re
from typing import List, Tuple, Optional, Dict, Any
from . import constants

class GoTResponseParser:
    @staticmethod
    def parse_initial_thoughts(response_content: str) -> List[str]:
        thoughts = []
        for line in response_content.splitlines():
            line = line.strip()
            if line.startswith(constants.INITIAL_THOUGHT_PREFIX):
                thought_content = line[len(constants.INITIAL_THOUGHT_PREFIX):].strip()
                if thought_content:
                    thoughts.append(thought_content)
        return thoughts

    @staticmethod
    def parse_expanded_thoughts(response_content: str) -> List[str]:
        new_thoughts = []
        for line in response_content.splitlines():
            line = line.strip()
            if line.startswith(constants.NEW_THOUGHT_PREFIX):
                thought_content = line[len(constants.NEW_THOUGHT_PREFIX):].strip()
                if thought_content:
                    new_thoughts.append(thought_content)
        return new_thoughts

    @staticmethod
    def parse_aggregated_thought(response_content: str) -> Optional[str]:
        for line in response_content.splitlines():
            line = line.strip()
            if line.startswith(constants.AGGREGATED_THOUGHT_PREFIX):
                return line[len(constants.AGGREGATED_THOUGHT_PREFIX):].strip()
        return None

    @staticmethod
    def parse_refined_thought(response_content: str) -> Optional[str]:
        for line in response_content.splitlines():
            line = line.strip()
            if line.startswith(constants.REFINED_THOUGHT_PREFIX):
                return line[len(constants.REFINED_THOUGHT_PREFIX):].strip()
        return None

    @staticmethod
    def parse_scored_thought(response_content: str) -> Tuple[Optional[float], Optional[str]]:
        score_str: Optional[str] = None
        justification: Optional[str] = None

        for line in response_content.splitlines():
            line = line.strip()
            if line.startswith(constants.SCORE_PREFIX):
                score_str = line[len(constants.SCORE_PREFIX):].strip()
            elif line.startswith(constants.JUSTIFICATION_PREFIX):
                justification = line[len(constants.JUSTIFICATION_PREFIX):].strip()

        score: Optional[float] = None
        if score_str:
            try:
                score = float(score_str)
            except ValueError:
                # Log error or handle parsing failure
                pass
        return score, justification
