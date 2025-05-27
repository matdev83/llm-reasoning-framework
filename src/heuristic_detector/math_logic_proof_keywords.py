import re

MATH_LOGIC_PROOF_KEYWORDS = [
    r"\bprove (?:that|the following|the correctness of|this (?:mathematical)? (?:statement|assertion|claim|identity|inequality)|the proposition)?\b",
    r"\bderive (?:the (?:complete|general)? formula|a set of equations|the (?:statistical)? properties|the relationship between|an expression for|the general solution)?\b"
]
