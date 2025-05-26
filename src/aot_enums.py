from enum import Enum, auto

class AotTriggerMode(Enum):
    ALWAYS_AOT = "always"
    ASSESS_FIRST = "assess"
    NEVER_AOT = "never"

    def __str__(self):
        return self.value

class AssessmentDecision(Enum):
    ONESHOT = "ONESHOT"
    AOT = "AOT"
    ERROR = "ERROR"
