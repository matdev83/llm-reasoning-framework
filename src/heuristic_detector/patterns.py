import re

class HeuristicPatterns:
    @staticmethod
    def _check_architect_pattern(prompt_text: str) -> bool:
        """
        Checks for "architect a [system/solution] for [complex subject] that [handles complexity]".
        The subject itself needs to be somewhat descriptive.
        """
        pattern = r"\barchitect (?:a|an|the)? (?:solution|system|platform|application|framework|infrastructure)(?: for)?(?:[\w\s\-':,.()]+?)? (?:that|which|to|for)?(?:.*(?:handles|supports|addresses|scales to|processes|manages|considering|requirements|achieve|capable of))?\b"
        match = re.search(pattern, prompt_text, re.IGNORECASE)
        if match:
            return True
        return False

    @staticmethod
    def _check_complex_implementation_pattern(prompt_text: str) -> bool:
        """
        Checks for "implement [item] that [has multiple/complex features/requirements]".
        Looks for conjunctions, lists of features, or keywords indicating complexity in requirements.
        """
        pattern = r"\b(?:implement|develop|create|build) (?:an?|the)? (?:[\w\s\-()':,.]+?)?(?: that|which|to|for)?(?:.*(?:(?:and|or|,){2,}|handles (?:multiple|complex|various)|interacts with (?:multiple|several)|processes large|requires (?:complex|advanced|detailed|multiple|integration of|coordination between)|supports|featuring|covers|addressing issues of|considering))?\b"
        match = re.search(pattern, prompt_text, re.IGNORECASE)
        if match:
            return True
        return False
