from enum import Enum

class L2TTriggerMode(Enum):
    """
    Defines how the L2T process is triggered.
    - ALWAYS_L2T: Always run the L2T process.
    - NEVER_L2T: Never run the L2T process, always use a direct one-shot LLM call.
    - ASSESS_FIRST: Use an assessor model (with optional heuristic shortcut)
                    to decide between a one-shot LLM call and the L2T process.
    """
    ALWAYS_L2T = "always_l2t"
    NEVER_L2T = "never_l2t"
    ASSESS_FIRST = "assess_first"
