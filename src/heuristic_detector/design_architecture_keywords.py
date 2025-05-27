import re

DESIGN_ARCHITECTURE_KEYWORDS = [
    r"design .* architecture",
    r"\b(?:propose|define|detail|outline|create|develop) (?:an)? architecture\b",
    r"\b(?:create|develop|write|draft) a system design document\b",
    r"\b(?:detail|define|specify) (?:the)? (?:high-level|low-level|detailed|overall|complete)? design\b",
    r"\b(?:recommend|suggest|propose) (?:a|an)? (?:suitable|appropriate)? architecture\b",
    r"\b(?:discuss|explain|apply|implement|choose|select) (?:appropriate|suitable|relevant)? (?:design patterns|architectural patterns)\b",
    r"\bexplain (?:the)? (?:architecture|system design|internals|detailed workings)\b",
    r"\b(?:what would be|propose|design) (?:a)? (?:robust|scalable|efficient|secure|comprehensive|resilient)? solution architecture\b"
]
