from enum import Enum, auto

class AotTriggerMode(Enum):
    ALWAYS_AOT = "always"
    ASSESS_FIRST = "assess"
    NEVER_AOT = "never"

    def __str__(self):
        return self.value

class AssessmentDecision(Enum):
    ONE_SHOT = "ONE_SHOT"
    ADVANCED_REASONING = "ADVANCED_REASONING"
    ERROR = "ERROR"
