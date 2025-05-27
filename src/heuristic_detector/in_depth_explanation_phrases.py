import re

IN_DEPTH_EXPLANATION_PHRASES = [
    r"\bexplain in (?:great|full|thorough)? detail\b",
    r"\b(?:provide|give|offer) (?:a)? comprehensive (?:explanation|overview|analysis|breakdown)\b",
    r"\banalyze (?:the|its|their)? (?:causes and effects|pros and cons|trade-offs|implications|impact|benefits and drawbacks|root cause(?:s)?|ethical implications)\b",
    r"\bperform (?:a)? detailed (?:analysis|review|examination|investigation|audit)\b",
    r"\b(?:thoroughly|critically|deeply)? (?:examine|discuss|evaluate|review|investigate|analyze|scrutinize)\b",
    r"\b(?:explore|delve into|unpack|dissect) (?:the)? (?:complexities|nuances|intricacies|underlying principles|full scope)\b"
]
