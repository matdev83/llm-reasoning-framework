import re
import logging
from typing import Optional, Dict, List, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class ReasoningFormat(Enum):
    """Supported reasoning formats by different AI models"""
    DEEPSEEK_R1 = "deepseek_r1"  # <THINKING>...</THINKING> or <think>...</think>
    OPENAI_O1 = "openai_o1"  # Hidden reasoning with completion tokens
    CUSTOM_TOKEN = "custom_token"  # Custom token like <REASONING_COMPLETE>
    GEMINI_THINKING = "gemini_thinking"  # Google's thinking mode format
    CLAUDE_THINKING = "claude_thinking"  # Anthropic's thinking format
    QWQ_THINKING = "qwq_thinking"  # Alibaba QwQ format
    GENERIC_COT = "generic_cot"  # Generic chain-of-thought patterns

class ReasoningExtractor:
    """
    Flexible reasoning extractor that can handle multiple AI model formats.
    
    This class supports various reasoning token formats used by different AI models:
    - DeepSeek-R1: <THINKING>...</THINKING> or <think>...</think>
    - OpenAI o1: Uses completion tokens or hidden reasoning
    - Gemini: Flash Thinking mode with structured output
    - Claude: Thinking mode with step-by-step reasoning
    - QwQ: Self-dialogue and reflection patterns
    - Generic: Common chain-of-thought patterns
    """
    
    def __init__(self):
        self.format_patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[ReasoningFormat, List[Dict]]:
        """Initialize regex patterns for different reasoning formats"""
        return {
            ReasoningFormat.DEEPSEEK_R1: [
                {
                    'pattern': r'<THINKING>(.*?)</THINKING>',
                    'flags': re.DOTALL | re.IGNORECASE,
                    'description': 'DeepSeek-R1 THINKING tags (uppercase)'
                },
                {
                    'pattern': r'<think>(.*?)</think>',
                    'flags': re.DOTALL | re.IGNORECASE,
                    'description': 'DeepSeek-R1 think tags (lowercase)'
                },
                {
                    'pattern': r'<thinking>(.*?)</thinking>',
                    'flags': re.DOTALL | re.IGNORECASE,
                    'description': 'DeepSeek-R1 thinking tags (mixed case)'
                }
            ],
            ReasoningFormat.OPENAI_O1: [
                {
                    'pattern': r'<reasoning>(.*?)</reasoning>',
                    'flags': re.DOTALL | re.IGNORECASE,
                    'description': 'OpenAI o1 reasoning tags'
                },
                {
                    'pattern': r'<internal_thought>(.*?)</internal_thought>',
                    'flags': re.DOTALL | re.IGNORECASE,
                    'description': 'OpenAI o1 internal thought tags'
                }
            ],
            ReasoningFormat.GEMINI_THINKING: [
                {
                    'pattern': r'<thinking>(.*?)</thinking>',
                    'flags': re.DOTALL | re.IGNORECASE,
                    'description': 'Gemini Flash Thinking mode'
                },
                {
                    'pattern': r'<analysis>(.*?)</analysis>',
                    'flags': re.DOTALL | re.IGNORECASE,
                    'description': 'Gemini analysis blocks'
                }
            ],
            ReasoningFormat.CLAUDE_THINKING: [
                {
                    'pattern': r'<thinking>(.*?)</thinking>',
                    'flags': re.DOTALL | re.IGNORECASE,
                    'description': 'Claude thinking mode'
                },
                {
                    'pattern': r'<reflection>(.*?)</reflection>',
                    'flags': re.DOTALL | re.IGNORECASE,
                    'description': 'Claude reflection blocks'
                }
            ],
            ReasoningFormat.QWQ_THINKING: [
                {
                    'pattern': r'<thinking>(.*?)</thinking>',
                    'flags': re.DOTALL | re.IGNORECASE,
                    'description': 'QwQ thinking blocks'
                },
                {
                    'pattern': r'Let me think about this step by step\.(.*?)(?=\n\n|\Z)',
                    'flags': re.DOTALL | re.IGNORECASE,
                    'description': 'QwQ step-by-step reasoning'
                }
            ],
            ReasoningFormat.GENERIC_COT: [
                {
                    'pattern': r'(?:Let me think|I need to think|Thinking step by step|Step by step)(.*?)(?=\n\n[A-Z]|\Z)',
                    'flags': re.DOTALL | re.IGNORECASE,
                    'description': 'Generic chain-of-thought patterns'
                },
                {
                    'pattern': r'<cot>(.*?)</cot>',
                    'flags': re.DOTALL | re.IGNORECASE,
                    'description': 'Generic COT tags'
                }
            ]
        }
    
    def extract_reasoning(self, text: str, format_hint: Optional[ReasoningFormat] = None, 
                         custom_token: Optional[str] = None) -> Tuple[str, str, ReasoningFormat]:
        """
        Extract reasoning from text using various formats.
        
        Args:
            text: The full model output text
            format_hint: Hint about which format to try first
            custom_token: Custom completion token to split on
            
        Returns:
            Tuple of (reasoning_text, remaining_text, detected_format)
        """
        logger.debug(f"Extracting reasoning from text of length {len(text)}")
        
        # Handle custom token first if provided
        if custom_token:
            reasoning, remaining = self._extract_with_custom_token(text, custom_token)
            if reasoning:
                logger.info(f"Successfully extracted reasoning using custom token: {custom_token}")
                return reasoning, remaining, ReasoningFormat.CUSTOM_TOKEN
        
        # Try format hint first if provided
        if format_hint and format_hint in self.format_patterns:
            reasoning, remaining = self._try_format(text, format_hint)
            if reasoning:
                logger.info(f"Successfully extracted reasoning using format hint: {format_hint.value}")
                return reasoning, remaining, format_hint
        
        # Try all formats in order of likelihood
        format_priority = [
            ReasoningFormat.DEEPSEEK_R1,
            ReasoningFormat.GEMINI_THINKING,
            ReasoningFormat.CLAUDE_THINKING,
            ReasoningFormat.OPENAI_O1,
            ReasoningFormat.QWQ_THINKING,
            ReasoningFormat.GENERIC_COT
        ]
        
        for format_type in format_priority:
            if format_type == format_hint:
                continue  # Already tried
                
            reasoning, remaining = self._try_format(text, format_type)
            if reasoning:
                logger.info(f"Successfully extracted reasoning using format: {format_type.value}")
                return reasoning, remaining, format_type
        
        # If no structured reasoning found, try to extract implicit reasoning
        reasoning, remaining = self._extract_implicit_reasoning(text)
        if reasoning:
            logger.info("Extracted implicit reasoning from text")
            return reasoning, remaining, ReasoningFormat.GENERIC_COT
        
        logger.warning("No reasoning pattern detected in text")
        return "", text, ReasoningFormat.GENERIC_COT
    
    def _extract_with_custom_token(self, text: str, custom_token: str) -> Tuple[str, str]:
        """Extract reasoning using a custom completion token"""
        # Handle DeepSeek-R1 reverse pattern: <REASONING_COMPLETE> followed by reasoning
        if custom_token in text:
            if text.strip().startswith(custom_token):
                # DeepSeek-R1 pattern: token first, then reasoning
                after_token = text.split(custom_token, 1)[1].strip()
                return after_token, ""
            else:
                # Standard pattern: reasoning first, then token
                parts = text.split(custom_token, 1)
                if len(parts) == 2:
                    return parts[0].strip(), parts[1].strip()
        
        return "", text
    
    def _try_format(self, text: str, format_type: ReasoningFormat) -> Tuple[str, str]:
        """Try to extract reasoning using a specific format"""
        if format_type not in self.format_patterns:
            return "", text
        
        patterns = self.format_patterns[format_type]
        
        for pattern_info in patterns:
            pattern = pattern_info['pattern']
            flags = pattern_info['flags']
            
            match = re.search(pattern, text, flags)
            if match:
                reasoning = match.group(1).strip()
                if reasoning:  # Only return if we found actual content
                    # Remove the matched reasoning from the text
                    remaining = text[:match.start()] + text[match.end():]
                    remaining = remaining.strip()
                    
                    logger.debug(f"Pattern matched: {pattern_info['description']}")
                    return reasoning, remaining
        
        return "", text
    
    def _extract_implicit_reasoning(self, text: str) -> Tuple[str, str]:
        """
        Extract implicit reasoning from text that doesn't use structured tags.
        Looks for common reasoning patterns and separates them from final answers.
        """
        # Look for common reasoning indicators
        reasoning_indicators = [
            r"Let me think about this",
            r"I need to consider",
            r"First, let me analyze",
            r"To solve this problem",
            r"Step \d+:",
            r"Here's my approach",
            r"Let me break this down",
            r"I'll work through this"
        ]
        
        # Find the first reasoning indicator
        for indicator in reasoning_indicators:
            match = re.search(indicator, text, re.IGNORECASE)
            if match:
                # Look for a natural break point after reasoning
                reasoning_part = text[match.start():]
                
                # Try to find where reasoning ends (common patterns)
                end_patterns = [
                    r"\n\n(?:Therefore|So|In conclusion|Final answer|Answer:)",
                    r"\n\n[A-Z][a-z]+ (?:answer|solution|result)",
                    r"\n\n\d+\.",  # Numbered final answer
                ]
                
                for end_pattern in end_patterns:
                    end_match = re.search(end_pattern, reasoning_part, re.IGNORECASE)
                    if end_match:
                        reasoning = reasoning_part[:end_match.start()].strip()
                        remaining = text[:match.start()] + reasoning_part[end_match.start():]
                        return reasoning, remaining.strip()
                
                # If no clear end found, use a heuristic (first 2/3 of text)
                split_point = int(len(reasoning_part) * 0.67)
                reasoning = reasoning_part[:split_point].strip()
                remaining = text[:match.start()] + reasoning_part[split_point:]
                return reasoning, remaining.strip()
        
        return "", text
    
    def detect_format(self, text: str) -> Optional[ReasoningFormat]:
        """
        Detect which reasoning format is used in the text.
        
        Args:
            text: The text to analyze
            
        Returns:
            The detected format or None if no format is detected
        """
        for format_type, patterns in self.format_patterns.items():
            for pattern_info in patterns:
                if re.search(pattern_info['pattern'], text, pattern_info['flags']):
                    return format_type
        return None
    
    def get_supported_formats(self) -> List[ReasoningFormat]:
        """Get list of all supported reasoning formats"""
        return list(self.format_patterns.keys())
    
    def get_format_description(self, format_type: ReasoningFormat) -> str:
        """Get description of a reasoning format"""
        format_descriptions = {
            ReasoningFormat.DEEPSEEK_R1: "DeepSeek-R1 thinking tags: <THINKING>...</THINKING> or <think>...</think>",
            ReasoningFormat.OPENAI_O1: "OpenAI o1 reasoning tags and internal thought patterns",
            ReasoningFormat.CUSTOM_TOKEN: "Custom completion token specified by user",
            ReasoningFormat.GEMINI_THINKING: "Google Gemini Flash Thinking mode with analysis blocks",
            ReasoningFormat.CLAUDE_THINKING: "Anthropic Claude thinking mode with reflection",
            ReasoningFormat.QWQ_THINKING: "Alibaba QwQ self-dialogue and step-by-step reasoning",
            ReasoningFormat.GENERIC_COT: "Generic chain-of-thought patterns and implicit reasoning"
        }
        return format_descriptions.get(format_type, "Unknown format") 