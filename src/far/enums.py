from enum import Enum

class FaRTriggerMode(Enum):
    ALWAYS_FAR = "always_far"
    ASSESS_FIRST_FAR = "assess_first_far"
    NEVER_FAR = "never_far"

# If we need specific assessment decisions for FaR, different from the generic ones.
# For now, we can reuse AssessmentDecision from src.aot.enums
# class FaRAssessmentDecision(Enum):
#     USE_FAR = "use_far"
#     USE_ONESHOT = "use_oneshot"
#     ERROR = "error"
