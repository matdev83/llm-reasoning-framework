import re

DATA_ALGO_TASKS = [
    r"\bdesign (?:a)? custom data structure\b",
    r"\bdevelop (?:an)? (?:efficient|optimized|advanced|novel)? algorithm (?:to)? (?:solve|find the optimal strategy|perform (?:[\w\s]+?)? on (?:streaming|large-scale|real-time|high-dimensional|noisy|heterogeneous|complex)? data|optimizing)\b",
    r"\bsolve (?:the)? (?:problem|challenge) (?:using)? (?:a|an)? (?:[\w\s]+?)? algorithm\b"
]
