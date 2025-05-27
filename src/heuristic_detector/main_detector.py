import logging
import re
from .patterns import HeuristicPatterns
from .explicit_decomposition_keywords import EXPLICIT_DECOMPOSITION_KEYWORDS
from .design_architecture_keywords import DESIGN_ARCHITECTURE_KEYWORDS
from .in_depth_explanation_phrases import IN_DEPTH_EXPLANATION_PHRASES
from .specific_complex_coding_keywords import SPECIFIC_COMPLEX_CODING_KEYWORDS
from .data_algo_tasks import DATA_ALGO_TASKS
from .multi_part_complex import MULTI_PART_COMPLEX
from .complex_conditional_keywords import COMPLEX_CONDITIONAL_KEYWORDS
from .problem_solving_keywords import PROBLEM_SOLVING_KEYWORDS
from .math_logic_proof_keywords import MATH_LOGIC_PROOF_KEYWORDS
from .creative_writing_complex import CREATIVE_WRITING_COMPLEX
from .simulation_modeling_keywords import SIMULATION_MODELING_KEYWORDS

logger = logging.getLogger(__name__)

class MainHeuristicDetector:
    @staticmethod
    def should_trigger_complex_process_heuristically(prompt_text: str) -> bool:
        """
        Analyzes the prompt text using deterministic heuristics to decide if it's
        VERY HIGHLY LIKELY to require a complex, multi-step reasoning process
        (like AoT or L2T).
        """
        print(f"Checking prompt: {prompt_text[:100]}...")

        if any(re.search(keyword, prompt_text, re.IGNORECASE) for keyword in EXPLICIT_DECOMPOSITION_KEYWORDS):
            print(f"Heuristic matched: explicit_decomposition_keywords for '{prompt_text[:50]}...'")
            logger.debug(f"Heuristic matched: explicit_decomposition_keywords for '{prompt_text[:50]}...'")
            return True

        if any(re.search(keyword, prompt_text, re.IGNORECASE) for keyword in DESIGN_ARCHITECTURE_KEYWORDS):
            print(f"Heuristic matched: design_architecture_keywords for '{prompt_text[:50]}...'")
            logger.debug(f"Heuristic matched: design_architecture_keywords for '{prompt_text[:50]}...'")
            return True
        if HeuristicPatterns._check_architect_pattern(prompt_text):
            print(f"Heuristic matched: _check_architect_pattern for '{prompt_text[:50]}...'")
            logger.debug(f"Heuristic matched: _check_architect_pattern for '{prompt_text[:50]}...'")
            return True

        if any(re.search(phrase, prompt_text, re.IGNORECASE) for phrase in IN_DEPTH_EXPLANATION_PHRASES):
            print(f"Heuristic matched: in_depth_explanation_phrases for '{prompt_text[:50]}...'")
            logger.debug(f"Heuristic matched: in_depth_explanation_phrases for '{prompt_text[:50]}...'")
            return True

        if HeuristicPatterns._check_complex_implementation_pattern(prompt_text):
            print(f"Heuristic matched: _check_complex_implementation_pattern for '{prompt_text[:50]}...'")
            logger.debug(f"Heuristic matched: _check_complex_implementation_pattern for '{prompt_text[:50]}...'")
            return True

        if any(re.search(keyword, prompt_text, re.IGNORECASE) for keyword in SPECIFIC_COMPLEX_CODING_KEYWORDS):
            print(f"Heuristic matched: specific_complex_coding_keywords for '{prompt_text[:50]}...'")
            logger.debug(f"Heuristic matched: specific_complex_coding_keywords for '{prompt_text[:50]}...'")
            return True

        if any(re.search(keyword, prompt_text, re.IGNORECASE) for keyword in DATA_ALGO_TASKS):
            print(f"Heuristic matched: data_algo_tasks for '{prompt_text[:50]}...'")
            logger.debug(f"Heuristic matched: data_algo_tasks for '{prompt_text[:50]}...'")
            return True

        if any(re.search(keyword, prompt_text, re.IGNORECASE) for keyword in MULTI_PART_COMPLEX):
            print(f"Heuristic matched: multi_part_complex for '{prompt_text[:50]}...'")
            logger.debug(f"Heuristic matched: multi_part_complex for '{prompt_text[:50]}...'")
            return True

        if any(re.search(keyword, prompt_text, re.IGNORECASE) for keyword in COMPLEX_CONDITIONAL_KEYWORDS):
            print(f"Heuristic matched: complex_conditional_keywords for '{prompt_text[:50]}...'")
            logger.debug(f"Heuristic matched: complex_conditional_keywords for '{prompt_text[:50]}...'")
            return True

        if any(re.search(keyword, prompt_text, re.IGNORECASE) for keyword in PROBLEM_SOLVING_KEYWORDS):
            print(f"Heuristic matched: problem_solving_keywords for '{prompt_text[:50]}...'")
            logger.debug(f"Heuristic matched: problem_solving_keywords for '{prompt_text[:50]}...'")
            return True

        if any(re.search(keyword, prompt_text, re.IGNORECASE) for keyword in MATH_LOGIC_PROOF_KEYWORDS):
            print(f"Heuristic matched: math_logic_proof_keywords for '{prompt_text[:50]}...'")
            logger.debug(f"Heuristic matched: math_logic_proof_keywords for '{prompt_text[:50]}...'")
            return True

        if any(re.search(pattern, prompt_text, re.IGNORECASE) for pattern in CREATIVE_WRITING_COMPLEX):
            print(f"Heuristic matched: creative_writing_complex for '{prompt_text[:50]}...'")
            logger.debug(f"Heuristic matched: creative_writing_complex for '{prompt_text[:50]}...'")
            return True

        if any(re.search(keyword, prompt_text, re.IGNORECASE) for keyword in SIMULATION_MODELING_KEYWORDS):
            print(f"Heuristic matched: simulation_modeling_keywords for '{prompt_text[:50]}...'")
            logger.debug(f"Heuristic matched: simulation_modeling_keywords for '{prompt_text[:50]}...'")
            return True

        return False
