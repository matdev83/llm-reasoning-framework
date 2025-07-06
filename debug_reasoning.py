#!/usr/bin/env python3
"""
Debug script to see what DeepSeek-R1 is actually outputting
"""

import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm_client import LLMClient
from llm_config import LLMConfig

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s [%(name)s] %(message)s')

def test_deepseek_output():
    """Test what DeepSeek-R1 actually outputs"""
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        return
    
    llm_client = LLMClient(api_key=api_key)
    
    # Test prompt
    prompt = """Problem: What is 2+2?

Think step-by-step to solve this problem. When you finish your reasoning, output exactly: <REASONING_COMPLETE>

Reasoning:"""
    
    print("🔍 Testing DeepSeek-R1 output...")
    print("📝 Prompt:")
    print(prompt)
    print("\n" + "="*50 + "\n")
    
    # Test 1: Without stop token
    print("🧪 TEST 1: Without stop token")
    try:
        config = LLMConfig(
            temperature=0.1,
            max_tokens=800,
            stop=None  # No stop token
        )
        
        output, stats = llm_client.call(
            prompt=prompt,
            models=["deepseek/deepseek-r1-0528:free"],
            config=config
        )
        
        print("✅ SUCCESS!")
        print("📊 Stats:")
        print(f"   Model: {stats.model_name}")
        print(f"   Tokens: {stats.completion_tokens}")
        print(f"   Duration: {stats.call_duration_seconds:.2f}s")
        print("\n🔍 Raw Output:")
        print("="*50)
        print(repr(output))
        print("="*50)
        print(output)
        print("="*50)
        
        # Test reasoning extraction
        print("\n🧠 Testing reasoning extraction...")
        
        # Check if it contains DeepSeek-R1 thinking tags
        if "<THINKING>" in output.upper():
            print("✅ Found <THINKING> tags")
        elif "<think>" in output.lower():
            print("✅ Found <think> tags")
        else:
            print("❌ No DeepSeek thinking tags found")
            
        # Check if it contains the completion token
        if "<REASONING_COMPLETE>" in output:
            print("✅ Found <REASONING_COMPLETE> token")
        else:
            print("❌ No <REASONING_COMPLETE> token found")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Test 2: With stop token
    print("🧪 TEST 2: With stop token")
    try:
        config = LLMConfig(
            temperature=0.1,
            max_tokens=800,
            stop=["<REASONING_COMPLETE>"]
        )
        
        output, stats = llm_client.call(
            prompt=prompt,
            models=["deepseek/deepseek-r1-0528:free"],
            config=config
        )
        
        print("✅ SUCCESS!")
        print("📊 Stats:")
        print(f"   Model: {stats.model_name}")
        print(f"   Tokens: {stats.completion_tokens}")
        print(f"   Duration: {stats.call_duration_seconds:.2f}s")
        print("\n🔍 Raw Output:")
        print("="*50)
        print(repr(output))
        print("="*50)
        print(output)
        print("="*50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Test 3: Different prompt style for DeepSeek
    print("🧪 TEST 3: DeepSeek-specific prompt")
    try:
        deepseek_prompt = """<THINKING>
Let me solve this step by step:

Problem: What is 2+2?

I need to add 2 and 2 together.
2 + 2 = 4

The answer is 4.
</THINKING>

Looking at this problem, I need to add 2 and 2 together."""
        
        config = LLMConfig(
            temperature=0.1,
            max_tokens=400,
            stop=None
        )
        
        output, stats = llm_client.call(
            prompt=deepseek_prompt,
            models=["deepseek/deepseek-r1-0528:free"],
            config=config
        )
        
        print("✅ SUCCESS!")
        print("📊 Stats:")
        print(f"   Model: {stats.model_name}")
        print(f"   Tokens: {stats.completion_tokens}")
        print(f"   Duration: {stats.call_duration_seconds:.2f}s")
        print("\n🔍 Raw Output:")
        print("="*50)
        print(repr(output))
        print("="*50)
        print(output)
        print("="*50)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_deepseek_output() 