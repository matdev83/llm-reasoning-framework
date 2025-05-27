from .main_detector import MainHeuristicDetector

class HeuristicDetector(MainHeuristicDetector):
    """
    Provides deterministic heuristic checks to identify if a problem prompt
    is highly likely to require a complex, multi-step reasoning process (like AoT or L2T).
    """
    pass
