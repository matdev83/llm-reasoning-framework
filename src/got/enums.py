from enum import Enum

class GoTTriggerMode(Enum):
    ALWAYS_GOT = "always_got"
    ASSESS_FIRST_GOT = "assess_first_got"
    NEVER_GOT = "never_got" # Results in a direct one-shot call by the orchestrator
