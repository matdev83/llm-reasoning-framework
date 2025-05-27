import re

MULTI_PART_COMPLEX = [
    r"\bcompare and contrast\b",
    r"\bevaluate (?:the)? (?:pros and cons|advantages and disadvantages|benefits and drawbacks|strengths and weaknesses)\b",
    r"\bdiscuss (?:[\w\s()\-.'’]+?)(?:,? (?:then|and)? [\w\s()\-.'’]+?)+\b",
    r"\bexplain (?:how|why)? (?:[\w\s()\-.'’]+?) (?:relates to|differs from|impacts|interacts with|influences|complements|contradicts)? (?:[\w\s()\-.'’]+?)\b",
    r"\b(?:firstly|first|to begin with)(?:.*)(?:secondly|second|next|then)(?:.*)(?:thirdly|third|furthermore|additionally|also|in addition|subsequently|finally|lastly)\b"
]
