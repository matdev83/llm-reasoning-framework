from enum import Enum

class HybridTriggerMode(Enum):
    NEVER_HYBRID = "NEVER_HYBRID"
    ALWAYS_HYBRID = "ALWAYS_HYBRID"
    ASSESS_FIRST_HYBRID = "ASSESS_FIRST_HYBRID" # Made value unique
