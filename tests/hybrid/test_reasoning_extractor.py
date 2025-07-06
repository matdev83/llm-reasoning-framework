import pytest
from src.hybrid.reasoning_extractor import ReasoningExtractor, ReasoningFormat

class TestReasoningExtractor:
    """Test the ReasoningExtractor class with various reasoning formats"""
    
    def setup_method(self):
        self.extractor = ReasoningExtractor()
    
    def test_deepseek_r1_thinking_tags_uppercase(self):
        """Test DeepSeek-R1 format with uppercase THINKING tags"""
        text = """<THINKING>
Some reasoning content here.
Multiple lines of thought process.
Working through the problem systematically.
</THINKING>

The final answer after reasoning."""
        
        reasoning, remaining, format_type = self.extractor.extract_reasoning(text)
        
        assert format_type == ReasoningFormat.DEEPSEEK_R1
        assert len(reasoning.strip()) > 0  # Should extract non-empty reasoning
        assert len(reasoning.split()) >= 3  # Should have multiple words/tokens
        assert "<THINKING>" not in reasoning  # Tags should be removed
        assert "</THINKING>" not in reasoning  # Tags should be removed
        assert len(remaining.strip()) > 0  # Should have remaining content
        assert "<THINKING>" not in remaining  # Reasoning should be removed from remaining
    
    def test_deepseek_r1_thinking_tags_lowercase(self):
        """Test DeepSeek-R1 format with lowercase think tags"""
        text = """<think>
Mathematical reasoning content.
Breaking down the problem systematically.
Analyzing step by step.
</think>

Final answer content here."""
        
        reasoning, remaining, format_type = self.extractor.extract_reasoning(text)
        
        assert format_type == ReasoningFormat.DEEPSEEK_R1
        assert len(reasoning.strip()) > 0  # Should extract non-empty reasoning
        assert len(reasoning.split()) >= 3  # Should have multiple words/tokens
        assert "<think>" not in reasoning  # Tags should be removed
        assert "</think>" not in reasoning  # Tags should be removed
        assert len(remaining.strip()) > 0  # Should have remaining content
        assert "<think>" not in remaining  # Reasoning should be removed from remaining
    
    def test_deepseek_r1_thinking_tags_mixed_case(self):
        """Test DeepSeek-R1 format with mixed case thinking tags"""
        text = """<thinking>
Step by step analysis:
1. Problem breakdown
2. Variable identification  
3. Formula application
</thinking>

Final answer content."""
        
        reasoning, remaining, format_type = self.extractor.extract_reasoning(text)
        
        assert format_type == ReasoningFormat.DEEPSEEK_R1
        assert len(reasoning.strip()) > 0  # Should extract non-empty reasoning
        assert len(reasoning.split()) >= 3  # Should have multiple words/tokens
        assert "<thinking>" not in reasoning  # Tags should be removed
        assert "</thinking>" not in reasoning  # Tags should be removed
        assert len(remaining.strip()) > 0  # Should have remaining content
        assert "<thinking>" not in remaining  # Reasoning should be removed from remaining
    
    def test_openai_o1_reasoning_tags(self):
        """Test OpenAI o1 format with reasoning tags"""
        text = """<reasoning>
Systematic approach here.
Information analysis content.
Logical reasoning steps.
</reasoning>

Final answer content."""
        
        reasoning, remaining, format_type = self.extractor.extract_reasoning(text)
        
        assert format_type == ReasoningFormat.OPENAI_O1
        assert len(reasoning.strip()) > 0  # Should extract non-empty reasoning
        assert len(reasoning.split()) >= 3  # Should have multiple words/tokens
        assert "<reasoning>" not in reasoning  # Tags should be removed
        assert "</reasoning>" not in reasoning  # Tags should be removed
        assert len(remaining.strip()) > 0  # Should have remaining content
        assert "<reasoning>" not in remaining  # Reasoning should be removed from remaining
    
    def test_gemini_thinking_mode(self):
        """Test Gemini Flash Thinking mode format"""
        text = """<analysis>
Multi-angle analysis content.
Mathematical aspect consideration.
Practical implications review.
</analysis>

Comprehensive analysis conclusion."""
        
        reasoning, remaining, format_type = self.extractor.extract_reasoning(text)
        
        assert format_type == ReasoningFormat.GEMINI_THINKING
        assert len(reasoning.strip()) > 0  # Should extract non-empty reasoning
        assert len(reasoning.split()) >= 3  # Should have multiple words/tokens
        assert "<analysis>" not in reasoning  # Tags should be removed
        assert "</analysis>" not in reasoning  # Tags should be removed
        assert len(remaining.strip()) > 0  # Should have remaining content
        assert "<analysis>" not in remaining  # Reasoning should be removed from remaining
    
    def test_claude_thinking_mode(self):
        """Test Claude thinking mode format"""
        text = """<reflection>
Careful consideration content.
Key factors analysis.
Interaction assessment.
</reflection>

Final response content."""
        
        reasoning, remaining, format_type = self.extractor.extract_reasoning(text)
        
        assert format_type == ReasoningFormat.CLAUDE_THINKING
        assert len(reasoning.strip()) > 0  # Should extract non-empty reasoning
        assert len(reasoning.split()) >= 3  # Should have multiple words/tokens
        assert "<reflection>" not in reasoning  # Tags should be removed
        assert "</reflection>" not in reasoning  # Tags should be removed
        assert len(remaining.strip()) > 0  # Should have remaining content
        assert "<reflection>" not in remaining  # Reasoning should be removed from remaining
    
    def test_qwq_thinking_format(self):
        """Test QwQ self-dialogue format"""
        text = """Let me think about this step by step.
Core question understanding follows.
Various approaches consideration needed.
Method effectiveness evaluation required.

Analysis conclusion content."""
        
        reasoning, remaining, format_type = self.extractor.extract_reasoning(text)
        
        assert format_type == ReasoningFormat.QWQ_THINKING
        assert len(reasoning.strip()) > 0  # Should extract non-empty reasoning
        assert len(reasoning.split()) >= 5  # Should have multiple words/tokens
        assert len(remaining.strip()) > 0  # Should have remaining content
        # QwQ format doesn't use tags, so reasoning should be extracted based on patterns
    
    def test_generic_cot_format(self):
        """Test generic chain-of-thought patterns"""
        text = """<cot>
Systematic breakdown content:
- Point A: Initial observation
- Point B: Secondary analysis  
- Point C: Final synthesis
</cot>

Evidence-based conclusion."""
        
        reasoning, remaining, format_type = self.extractor.extract_reasoning(text)
        
        assert format_type == ReasoningFormat.GENERIC_COT
        assert len(reasoning.strip()) > 0  # Should extract non-empty reasoning
        assert len(reasoning.split()) >= 5  # Should have multiple words/tokens
        assert "<cot>" not in reasoning  # Tags should be removed
        assert "</cot>" not in reasoning  # Tags should be removed
        assert len(remaining.strip()) > 0  # Should have remaining content
        assert "<cot>" not in remaining  # Reasoning should be removed from remaining
    
    def test_custom_token_extraction(self):
        """Test extraction using custom completion token"""
        text = """Careful thinking process.
Step 1: Problem understanding
Step 2: Solution identification
Step 3: Best approach selection
<REASONING_COMPLETE>
Final answer content."""
        
        reasoning, remaining, format_type = self.extractor.extract_reasoning(
            text, custom_token="<REASONING_COMPLETE>"
        )
        
        assert format_type == ReasoningFormat.CUSTOM_TOKEN
        assert len(reasoning.strip()) > 0  # Should extract non-empty reasoning
        assert len(reasoning.split()) >= 5  # Should have multiple words/tokens
        assert "<REASONING_COMPLETE>" not in reasoning  # Token should be removed
        assert len(remaining.strip()) > 0  # Should have remaining content
        assert "<REASONING_COMPLETE>" not in remaining  # Token should be removed from remaining
    
    def test_format_hint_priority(self):
        """Test that format hints are tried first"""
        # Text that could match multiple formats
        text = """<thinking>
This could be DeepSeek or Gemini format.
Let me work through this systematically.
</thinking>

The answer is clear."""
        
        # Without hint, should detect DeepSeek (higher priority)
        reasoning1, _, format1 = self.extractor.extract_reasoning(text)
        assert format1 == ReasoningFormat.DEEPSEEK_R1
        
        # With Gemini hint, should use Gemini format
        reasoning2, _, format2 = self.extractor.extract_reasoning(
            text, format_hint=ReasoningFormat.GEMINI_THINKING
        )
        assert format2 == ReasoningFormat.GEMINI_THINKING
    
    def test_implicit_reasoning_extraction(self):
        """Test extraction of implicit reasoning without structured tags"""
        text = """Let me think about this problem carefully.
First, I need to consider the basic principles involved.
Second, I should look at the specific constraints.
Third, I'll apply the appropriate methodology.

Therefore, the solution is to implement approach X."""
        
        reasoning, remaining, format_type = self.extractor.extract_reasoning(text)
        
        assert format_type == ReasoningFormat.GENERIC_COT
        assert len(reasoning.strip()) > 0  # Should extract non-empty reasoning
        assert len(reasoning.split()) >= 5  # Should have multiple words/tokens
        assert len(remaining.strip()) > 0  # Should have remaining content
    
    def test_no_reasoning_found(self):
        """Test behavior when no reasoning pattern is detected"""
        text = """This is just a direct answer without any reasoning structure.
There are no thinking tags or reasoning indicators here.
Just a plain response."""
        
        reasoning, remaining, format_type = self.extractor.extract_reasoning(text)
        
        # Should return empty reasoning and original text as remaining
        assert reasoning == ""
        assert remaining == text
        assert format_type == ReasoningFormat.GENERIC_COT
    
    def test_detect_format_method(self):
        """Test the detect_format method"""
        deepseek_text = "<THINKING>Some reasoning</THINKING>Answer"
        openai_text = "<reasoning>Some reasoning</reasoning>Answer"
        gemini_text = "<analysis>Some reasoning</analysis>Answer"
        
        assert self.extractor.detect_format(deepseek_text) == ReasoningFormat.DEEPSEEK_R1
        assert self.extractor.detect_format(openai_text) == ReasoningFormat.OPENAI_O1
        assert self.extractor.detect_format(gemini_text) == ReasoningFormat.GEMINI_THINKING
        assert self.extractor.detect_format("No reasoning here") is None
    
    def test_get_supported_formats(self):
        """Test getting list of supported formats"""
        formats = self.extractor.get_supported_formats()
        
        assert ReasoningFormat.DEEPSEEK_R1 in formats
        assert ReasoningFormat.OPENAI_O1 in formats
        assert ReasoningFormat.GEMINI_THINKING in formats
        assert ReasoningFormat.CLAUDE_THINKING in formats
        assert ReasoningFormat.QWQ_THINKING in formats
        assert ReasoningFormat.GENERIC_COT in formats
        assert len(formats) >= 6
    
    def test_get_format_description(self):
        """Test getting format descriptions"""
        desc = self.extractor.get_format_description(ReasoningFormat.DEEPSEEK_R1)
        assert "DeepSeek-R1" in desc
        assert "THINKING" in desc
        
        desc = self.extractor.get_format_description(ReasoningFormat.OPENAI_O1)
        assert "OpenAI o1" in desc
        
        desc = self.extractor.get_format_description(ReasoningFormat.GEMINI_THINKING)
        assert "Gemini" in desc
    
    def test_complex_nested_reasoning(self):
        """Test extraction from complex text with multiple potential patterns"""
        text = """I need to think about this problem.

<THINKING>
This is the main reasoning section.
Let me work through this step by step:

1. First, analyze the problem structure
2. Then, consider the constraints
3. Finally, develop a solution approach

Wait, let me reconsider step 2. The constraints are actually...
Actually, I think I need to revise my approach entirely.

New approach:
- Start with the end goal
- Work backwards to identify requirements
- Build a systematic solution
</THINKING>

Based on my thorough analysis above, the solution is to use method Y because it addresses all the key requirements while maintaining efficiency."""
        
        reasoning, remaining, format_type = self.extractor.extract_reasoning(text)
        
        assert format_type == ReasoningFormat.DEEPSEEK_R1
        assert "This is the main reasoning section" in reasoning
        assert "Let me work through this step by step" in reasoning
        assert "Wait, let me reconsider" in reasoning
        assert "New approach:" in reasoning
        assert "Based on my thorough analysis above" in remaining
        assert len(reasoning) > 100  # Should capture substantial reasoning
    
    def test_empty_reasoning_tags(self):
        """Test handling of empty reasoning tags"""
        text = """<thinking></thinking>The answer is 42."""
        
        reasoning, remaining, format_type = self.extractor.extract_reasoning(text)
        
        # Should fall back to original method since reasoning is empty
        assert reasoning == ""
        assert "The answer is 42." in remaining
    
    def test_malformed_tags(self):
        """Test handling of malformed or incomplete tags"""
        text = """<thinking>
This reasoning section is never properly closed...
The answer is somewhere here."""
        
        reasoning, remaining, format_type = self.extractor.extract_reasoning(text)
        
        # Should handle gracefully, possibly falling back to implicit reasoning
        assert format_type in [ReasoningFormat.DEEPSEEK_R1, ReasoningFormat.GENERIC_COT]
    
    def test_multiple_reasoning_blocks(self):
        """Test handling text with multiple reasoning blocks"""
        text = """<thinking>
First reasoning block with initial thoughts.
</thinking>

Some intermediate text here.

<thinking>
Second reasoning block with additional analysis.
</thinking>

Final answer based on all reasoning above."""
        
        reasoning, remaining, format_type = self.extractor.extract_reasoning(text)
        
        assert format_type == ReasoningFormat.DEEPSEEK_R1
        # Should extract the first reasoning block
        assert "First reasoning block" in reasoning
        # Remaining should include everything else
        assert "Some intermediate text" in remaining
        assert "Second reasoning block" in remaining
        assert "Final answer" in remaining 