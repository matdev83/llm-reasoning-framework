import re

EXPLICIT_DECOMPOSITION_KEYWORDS = [
    r"\bstep(?:-| )by(?:-| )step\b",
    r"\b(?:detailed|outline the|detailing)? steps?\b",
    r"\bbreak down (?:the)? (?:process|task|problem|steps)\b",
    r"\bwalk me through\b",
    r"\bstepwise (?:solution|guide|approach|instructions)\b",
    r"\b(?:provide|create|develop|formulate|outline|devise|draft) (?:a)? (?:detailed|comprehensive|strategic)? (?:plan|strategy|roadmap|blueprint|procedure|method|approach|framework|research proposal)\b",
    r"\b(?:give|provide|write|list|generate) (?:detailed|full|comprehensive)? instructions\b",
    r"\bshow (?:all|your)? work\b",
    r"\b(?:explain|describe|detail) (?:your)? (?:reasoning|thinking process|logical steps|how you arrived at this)\b"
]
