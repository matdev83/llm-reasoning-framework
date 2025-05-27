import re

SPECIFIC_COMPLEX_CODING_KEYWORDS = [
    r"\b(?:write|create|develop|build) (?:code|a script|a program|software|an application)?(?: to|for|that)? (?:integrate|automate (?:a)? multi-step (?:process|workflow|pipeline)|build (?:a)? full-stack application|parse (?:and|to)? transform complex data)\b",
    r"\bdevelop (?:an)? algorithm (?:for)? (?:solving (?:an?)? (?:optimization problem|complex problem)|pathfinding in (?:a)? (?:complex|large)? graph|optimizing)\b",
    r"\brefactor (?:this|a|an|the)? (?:large|complex|legacy)? codebase (?:to|for|in order to|into a)? (?:improve performance|achieve better modularity|separate concerns|enhance security|address critical vulnerabilities|migrate to (?:a)? new architecture|reduce technical debt)\b",
    r"\bdebug (?:a|an|the|this)? (?:complex|multi-threaded|distributed|concurrent|large-scale|performance-critical)? (?:code|system|application|bug|issue|problem|failure)\b",
    r"\b(?:using|incorporating)? design patterns and principles\b",
    r"\bintegrate (?:a|this|an)? (?:third-party|external)? API (?:into|with)? (?:our|an|the)? existing (?:platform|system|application)\b",
    r"\b(?:create|set up|implement|design|configure) (?:a)? (?:secure and resilient)? (?:ci/cd pipeline|continuous integration (?:and|&) (?:deployment|delivery) setup)\b",
    r"\b(?:containerize|orchestrate) (?:a)? (?:multi-container application|this legacy system|an application with multiple services)\b",
    r"\bimplement (?:the)? (?:design|architectural)? pattern\b",
    r"\b(?:write|develop|implement|create|design) (?:a)? (?:blockchain|distributed ledger)? (?:application|solution|contract)\b"
]
