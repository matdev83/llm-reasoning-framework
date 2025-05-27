import re

COMPLEX_CONDITIONAL_KEYWORDS = [
    r"\b(?:what are the|analyze the|discuss the|detail the|assess the|evaluate the|predict the|forecast the) (?:potential|likely|possible|significant|major|far-reaching|cascading) (?:failures|implications|effects|consequences|ramifications|risks and benefits|ethical considerations|downstream impacts) (?:if|should|when|of (?:a)? scenario where|in (?:the)? event that)\b",
    r"\banalyze (?:the)? (?:full|overall|potential|detailed|comprehensive|systemic) impact (?:of|on).*",
    r"\b(?:devise|develop|create|outline|propose) (?:a)? comprehensive (?:contingency plan|disaster recovery strategy|mitigation plan|risk management strategy|response plan|policy proposal)\b",
]
