#!/usr/bin/env python3
"""
Test the hybrid thinking model with your own problems.

Usage:
    python test_my_problem.py "Your problem here"
    
Requirements:
    - Set OPENROUTER_API_KEY environment variable
    - Activate virtual environment (.venv\\Scripts\\activate)
"""

import sys
import os
import logging
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hybrid.processor import HybridProcessor
from hybrid.dataclasses import HybridConfig, ReasoningConfig
from llm_client import LLMClient
from hybrid.orchestrator import HybridOrchestrator
from llm_config import LLMConfig

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s [%(name)s] %(message)s')

def test_hybrid_thinking_debug(problem_text, api_key=None):
    """Test the hybrid thinking model with detailed debug output"""
    
    if not api_key:
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            print("❌ Error: OPENROUTER_API_KEY environment variable not set")
            print("   Get one from https://openrouter.ai/")
            return False
    
    # Configure the hybrid thinking model with new OpenRouter reasoning support
    # Note: DeepSeek-R1 models don't support effort levels, only basic reasoning
    reasoning_config = ReasoningConfig(
        enabled=True,
        effort="medium",  # This will be filtered out for DeepSeek models
        exclude=False     # Include reasoning tokens in response
    )
    
    config = HybridConfig(
        reasoning_model_name="deepseek/deepseek-r1-0528:free",  # Free reasoning model
        response_model_name="openrouter/cypher-alpha:free",     # Free response model
        reasoning_model_temperature=0.1,
        response_model_temperature=0.3,
        reasoning_complete_token="<REASONING_COMPLETE>",
        reasoning_prompt_template="Problem: {problem_description}\n\nThink step-by-step to solve this problem. When you finish your reasoning, output exactly: {reasoning_complete_token}\n\nReasoning:",
        response_prompt_template="Original Problem: {problem_description}\n\nReasoning from first model:\n{extracted_reasoning}\n\nBased on the problem and the reasoning above, provide a clear final answer:",
        max_reasoning_tokens=800,
        max_response_tokens=400,
        reasoning_config=reasoning_config,
        use_streaming=True,  # Enable streaming for token optimization
        model_specific_headers={
            "deepseek": {
                "HTTP-Referer": "https://your-app.com",
                "X-Title": "Hybrid Reasoning Test"
            }
        }
    )
    
    # Show effective reasoning config for the model
    effective_config = config.get_effective_reasoning_config(config.reasoning_model_name)
    print(f"🔧 Model: {config.reasoning_model_name}")
    print(f"📋 Effective reasoning config: {effective_config.to_openrouter_dict() if effective_config else 'None (no reasoning support)'}")
    print()
    
    # Create LLM client and processor
    llm_client = LLMClient(api_key=api_key)
    processor = HybridProcessor(llm_client=llm_client, config=config)
    
    print(f"🤔 Testing problem: {problem_text}")
    print("🔄 Running hybrid thinking model with OpenRouter reasoning...")
    print()
    
    # Run the hybrid thinking process
    result = processor.run(problem_text)
    
    # Display results
    if result.succeeded:
        print("✅ SUCCESS!")
        print()
        print("🧠 REASONING EXTRACTED:")
        print("-" * 50)
        print(repr(result.extracted_reasoning))  # Use repr to see whitespace/empty strings
        print("-" * 50)
        print(result.extracted_reasoning)
        print()
        print("💡 FINAL ANSWER:")
        print("-" * 50)
        print(result.final_answer)
        print()
        print("📊 STATS:")
        print(f"   Reasoning tokens: {result.reasoning_call_stats.completion_tokens}")
        print(f"   Response tokens: {result.response_call_stats.completion_tokens}")
        print(f"   Total time: {result.reasoning_call_stats.call_duration_seconds + result.response_call_stats.call_duration_seconds:.2f}s")
        print(f"   Detected format: {result.detected_reasoning_format}")
        return True
    else:
        print("❌ FAILED!")
        print(f"Error: {result.error_message}")
        return False

def test_hybrid_thinking(problem_text, api_key=None):
    """Test the hybrid thinking model with default reasoning configuration"""
    
    if not api_key:
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            print("❌ Error: OPENROUTER_API_KEY environment variable not set")
            print("   Get one from https://openrouter.ai/")
            return False
    
    # Configure the hybrid thinking model - no explicit reasoning config
    # This will use model-specific defaults automatically
    config = HybridConfig(
        reasoning_model_name="deepseek/deepseek-r1-0528:free",  # Free reasoning model
        response_model_name="openrouter/cypher-alpha:free",     # Free response model
        reasoning_model_temperature=0.1,
        response_model_temperature=0.3,
        reasoning_complete_token="<REASONING_COMPLETE>",
        reasoning_prompt_template="Problem: {problem_description}\n\nThink step-by-step to solve this problem. When you finish your reasoning, output exactly: {reasoning_complete_token}\n\nReasoning:",
        response_prompt_template="Original Problem: {problem_description}\n\nReasoning from first model:\n{extracted_reasoning}\n\nBased on the problem and the reasoning above, provide a clear final answer:",
        max_reasoning_tokens=800,
        max_response_tokens=400,
        reasoning_config=None,  # Use model-specific defaults
        use_streaming=True,
        model_specific_headers={
            "deepseek": {
                "HTTP-Referer": "https://your-app.com",
                "X-Title": "Hybrid Reasoning Test"
            }
        }
    )
    
    # Show the effective reasoning config (should be model defaults)
    effective_config = config.get_effective_reasoning_config(config.reasoning_model_name)
    print(f"🔧 Model: {config.reasoning_model_name}")
    print(f"📋 Using default reasoning config: {effective_config.to_openrouter_dict() if effective_config else 'None'}")
    print()
    
    # Create LLM client and processor
    llm_client = LLMClient(api_key=api_key)
    processor = HybridProcessor(llm_client=llm_client, config=config)
    
    print(f"🤔 Testing problem: {problem_text}")
    print("🔄 Running hybrid thinking model with default configuration...")
    print()
    
    # Run the hybrid thinking process
    result = processor.run(problem_text)
    
    # Display results
    if result.succeeded:
        print("✅ SUCCESS!")
        print()
        print("🧠 REASONING EXTRACTED:")
        print("-" * 50)
        reasoning_preview = result.extracted_reasoning[:200] + "..." if len(result.extracted_reasoning) > 200 else result.extracted_reasoning
        print(reasoning_preview)
        print()
        print("💡 FINAL ANSWER:")
        print("-" * 50)
        print(result.final_answer)
        print()
        print("📊 STATS:")
        print(f"   Reasoning tokens: {result.reasoning_call_stats.completion_tokens}")
        print(f"   Response tokens: {result.response_call_stats.completion_tokens}")
        print(f"   Total time: {result.reasoning_call_stats.call_duration_seconds + result.response_call_stats.call_duration_seconds:.2f}s")
        print(f"   Detected format: {result.detected_reasoning_format}")
        return True
    else:
        print("❌ FAILED!")
        print(f"Error: {result.error_message}")
        return False

def test_model_default_reasoning_configs():
    """Test model-specific default reasoning configurations"""
    
    print("🎯 Testing model-specific default reasoning configurations:")
    print("=" * 70)
    
    # Test models with their expected defaults
    test_models = [
        {
            "name": "OpenAI o3-mini",
            "model": "openai/o3-mini",
            "expected_default": {"enabled": True, "effort": "high", "exclude": False}
        },
        {
            "name": "Grok Beta",
            "model": "grok/grok-beta",
            "expected_default": {"enabled": True, "effort": "high", "exclude": False}
        },
        {
            "name": "Gemini Thinking",
            "model": "google/gemini-2.5-flash-preview:thinking",
            "expected_default": {"enabled": True, "max_tokens": 32000, "exclude": False}
        },
        {
            "name": "Anthropic Claude",
            "model": "anthropic/claude-3.7-sonnet",
            "expected_default": {"enabled": True, "max_tokens": 8000, "exclude": False}
        },
        {
            "name": "DeepSeek-R1",
            "model": "deepseek/deepseek-r1-0528:free",
            "expected_default": {"enabled": True, "exclude": False}
        },
        {
            "name": "Non-reasoning model",
            "model": "openrouter/cypher-alpha:free",
            "expected_default": None
        }
    ]
    
    for test_model in test_models:
        # Create config without reasoning_config to test defaults
        config = HybridConfig(
            reasoning_model_name=test_model["model"],
            response_model_name="openrouter/cypher-alpha:free",
            reasoning_config=None  # No explicit config - should use defaults
        )
        
        # Get the default config
        default_config = config.get_model_default_reasoning_config(test_model["model"])
        effective_config = config.get_effective_reasoning_config(test_model["model"])
        
        print(f"\n🤖 {test_model['name']}")
        print(f"   Model: {test_model['model']}")
        print(f"   Default config: {default_config.to_openrouter_dict() if default_config else 'None'}")
        print(f"   Effective config: {effective_config.to_openrouter_dict() if effective_config else 'None'}")
        
        # Verify the defaults match expectations
        expected = test_model["expected_default"]
        if effective_config:
            actual = effective_config.to_openrouter_dict()
            if actual == expected:
                print("   ✅ Default configuration is correct")
            else:
                print(f"   ❌ Unexpected default: expected {expected}, got {actual}")
        elif expected is None:
            print("   ✅ Correctly identified as non-reasoning model")
        else:
            print(f"   ❌ Expected default config but got None")
    
    print("\n" + "=" * 70)
    print("📋 Default Configuration Summary:")
    print("   • OpenAI/Grok models: effort='high' (best reasoning quality)")
    print("   • Gemini Thinking: max_tokens=32,000 (maximum capability)")
    print("   • Anthropic Claude: max_tokens=8,000 (high effort equivalent)")
    print("   • DeepSeek-R1: enabled=True (basic reasoning only)")
    print("   • Other models: No reasoning support")

def test_different_reasoning_models():
    """Test different reasoning models to show model-specific configuration"""
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        return
    
    # Test configurations for different model types
    test_configs = [
        {
            "name": "DeepSeek-R1 (Basic reasoning only)",
            "model": "deepseek/deepseek-r1-0528:free",
            "reasoning_config": ReasoningConfig(enabled=True, effort="high", exclude=False),
            "expected_effective": {"enabled": True, "exclude": False}  # effort will be filtered out
        },
        {
            "name": "OpenAI o3-mini (Effort levels supported)",
            "model": "openai/o3-mini",
            "reasoning_config": ReasoningConfig(enabled=True, effort="high", exclude=False),
            "expected_effective": {"enabled": True, "effort": "high", "exclude": False}
        },
        {
            "name": "Anthropic Claude (Max tokens supported)",
            "model": "anthropic/claude-3.7-sonnet",
            "reasoning_config": ReasoningConfig(enabled=True, max_tokens=2000, exclude=False),
            "expected_effective": {"enabled": True, "max_tokens": 2000, "exclude": False}
        },
        {
            "name": "Non-reasoning model",
            "model": "openrouter/cypher-alpha:free",
            "reasoning_config": ReasoningConfig(enabled=True, effort="high", exclude=False),
            "expected_effective": None  # No reasoning support
        }
    ]
    
    print("🧪 Testing model-specific reasoning configuration filtering:")
    print("=" * 70)
    
    for test_config in test_configs:
        config = HybridConfig(
            reasoning_model_name=test_config["model"],
            response_model_name="openrouter/cypher-alpha:free",
            reasoning_config=test_config["reasoning_config"]
        )
        
        effective_config = config.get_effective_reasoning_config(test_config["model"])
        
        print(f"\n📱 {test_config['name']}")
        print(f"   Model: {test_config['model']}")
        print(f"   Original config: {test_config['reasoning_config'].to_openrouter_dict()}")
        print(f"   Effective config: {effective_config.to_openrouter_dict() if effective_config else 'None'}")
        
        # Verify the filtering worked as expected
        if effective_config:
            actual = effective_config.to_openrouter_dict()
            expected = test_config["expected_effective"]
            if actual == expected:
                print("   ✅ Filtering worked correctly")
            else:
                print(f"   ❌ Unexpected filtering: expected {expected}, got {actual}")
        elif test_config["expected_effective"] is None:
            print("   ✅ Correctly identified as non-reasoning model")
        else:
            print("   ❌ Expected reasoning support but got None")
    
    print("\n" + "=" * 70)

def test_token_limits():
    """Test model-specific token limits"""
    print("\n=== Testing Model-Specific Token Limits ===")
    
    # Test Gemini Thinking model
    gemini_config = HybridConfig(
        reasoning_model_name="google/gemini-2.5-flash-preview:thinking",
        response_model_name="google/gemini-2.5-flash-preview"
    )
    
    gemini_limits = gemini_config.get_effective_token_limits()
    print(f"Gemini Thinking limits: {gemini_limits}")
    
    # Test Claude model
    claude_config = HybridConfig(
        reasoning_model_name="anthropic/claude-3.5-sonnet-20241022",
        response_model_name="anthropic/claude-3.5-sonnet-20241022"
    )
    
    claude_limits = claude_config.get_effective_token_limits()
    print(f"Claude limits: {claude_limits}")
    
    # Test DeepSeek model
    deepseek_config = HybridConfig(
        reasoning_model_name="deepseek/deepseek-r1:nitro",
        response_model_name="deepseek/deepseek-r1:nitro"
    )
    
    deepseek_limits = deepseek_config.get_effective_token_limits()
    print(f"DeepSeek limits: {deepseek_limits}")
    
    # Test OpenAI model
    openai_config = HybridConfig(
        reasoning_model_name="openai/o1-preview",
        response_model_name="openai/gpt-4o"
    )
    
    openai_limits = openai_config.get_effective_token_limits()
    print(f"OpenAI limits: {openai_limits}")
    
    # Verify Gemini gets higher allocations
    assert gemini_limits["max_reasoning_tokens"] > claude_limits["max_reasoning_tokens"], \
        "Gemini should get higher reasoning token allocation"
    assert gemini_limits["max_response_tokens"] > claude_limits["max_response_tokens"], \
        "Gemini should get higher response token allocation"
    
    print("✅ All token limit tests passed!")

def test_reasoning_with_gemini():
    """Test reasoning extraction with Gemini model and increased token limits"""
    print("\n=== Testing Gemini Reasoning with Increased Token Limits ===")
    
    # Create configuration with Gemini thinking model
    config = HybridConfig(
        reasoning_model_name="google/gemini-2.5-flash-preview:thinking",
        response_model_name="google/gemini-2.5-flash-preview",
        reasoning_config=ReasoningConfig(
            enabled=True,
            max_tokens=32000,  # High reasoning token allocation
            exclude=False
        ),
        use_streaming=True
    )
    
    # Check effective token limits
    token_limits = config.get_effective_token_limits()
    print(f"Effective token limits: {token_limits}")
    
    # Verify we have adequate space for both reasoning and response
    expected_reasoning = 40000  # Should be 40K for Gemini
    expected_response = 8000    # Should be 8K for Gemini
    
    assert token_limits["max_reasoning_tokens"] == expected_reasoning, \
        f"Expected {expected_reasoning} reasoning tokens, got {token_limits['max_reasoning_tokens']}"
    assert token_limits["max_response_tokens"] == expected_response, \
        f"Expected {expected_response} response tokens, got {token_limits['max_response_tokens']}"
    
    print(f"✅ Gemini model configured with {token_limits['max_reasoning_tokens']} reasoning tokens and {token_limits['max_response_tokens']} response tokens")
    
    # Test with actual reasoning (if API key available)
    try:
        llm_client = LLMClient()
        orchestrator = HybridOrchestrator(llm_client, config)
        
        problem = "What is the most efficient sorting algorithm for large datasets and why?"
        
        print(f"\nTesting reasoning extraction with problem: {problem}")
        result = orchestrator.run(problem)
        
        if result.hybrid_result and result.hybrid_result.succeeded:
            print("✅ Reasoning extraction successful!")
            print(f"Reasoning format: {result.hybrid_result.detected_reasoning_format}")
            
            if result.hybrid_result.reasoning_call_stats:
                reasoning_tokens = result.hybrid_result.reasoning_call_stats.completion_tokens
                print(f"Reasoning tokens used: {reasoning_tokens}")
                
                # Check if we have adequate token space
                if reasoning_tokens > 0:
                    utilization = reasoning_tokens / token_limits["max_reasoning_tokens"]
                    print(f"Token utilization: {utilization:.1%}")
                    
                    if utilization > 0.8:
                        print("⚠️  High token utilization - consider increasing limits")
                    else:
                        print("✅ Good token utilization")
            
            if result.hybrid_result.response_call_stats:
                response_tokens = result.hybrid_result.response_call_stats.completion_tokens
                print(f"Response tokens used: {response_tokens}")
        else:
            print("❌ Reasoning extraction failed")
            if result.hybrid_result:
                print(f"Error: {result.hybrid_result.error_message}")
    
    except Exception as e:
        print(f"⚠️  Could not test with actual API (probably no API key): {e}")
        print("✅ Configuration test passed anyway")

def test_token_optimization():
    """Test token optimization across different models"""
    print("\n=== Testing Token Optimization Across Models ===")
    
    # Test configurations for different model types
    test_configs = [
        {
            "name": "Gemini Thinking (High Capacity)",
            "reasoning_model": "google/gemini-2.5-flash-preview:thinking",
            "response_model": "google/gemini-2.5-flash-preview",
            "expected_reasoning_tokens": 40000,
            "expected_response_tokens": 8000,
            "reasoning_config": ReasoningConfig(enabled=True, max_tokens=32000, exclude=False)
        },
        {
            "name": "Claude (Balanced)",
            "reasoning_model": "anthropic/claude-3.5-sonnet-20241022",
            "response_model": "anthropic/claude-3.5-sonnet-20241022",
            "expected_reasoning_tokens": 12000,
            "expected_response_tokens": 4000,
            "reasoning_config": ReasoningConfig(enabled=True, max_tokens=8000, exclude=False)
        },
        {
            "name": "OpenAI o-series (Efficient)",
            "reasoning_model": "openai/o1-preview",
            "response_model": "openai/gpt-4o",
            "expected_reasoning_tokens": 8000,
            "expected_response_tokens": 3000,
            "reasoning_config": ReasoningConfig(enabled=True, effort="high", exclude=False)
        },
        {
            "name": "DeepSeek-R1 (Basic)",
            "reasoning_model": "deepseek/deepseek-r1:nitro",
            "response_model": "deepseek/deepseek-r1:nitro",
            "expected_reasoning_tokens": 2000,
            "expected_response_tokens": 1500,
            "reasoning_config": ReasoningConfig(enabled=True, exclude=False)
        }
    ]
    
    print("🧪 Testing model-specific token allocation optimization:")
    print("=" * 80)
    
    for test_config in test_configs:
        print(f"\n🤖 {test_config['name']}")
        print(f"   Reasoning Model: {test_config['reasoning_model']}")
        print(f"   Response Model: {test_config['response_model']}")
        
        # Create configuration
        config = HybridConfig(
            reasoning_model_name=test_config["reasoning_model"],
            response_model_name=test_config["response_model"],
            reasoning_config=test_config["reasoning_config"],
            use_streaming=True
        )
        
        # Check token limits
        token_limits = config.get_effective_token_limits()
        effective_reasoning = config.get_effective_reasoning_config(test_config["reasoning_model"])
        
        print(f"   Token Limits: {token_limits['max_reasoning_tokens']} reasoning + {token_limits['max_response_tokens']} response")
        print(f"   Reasoning Config: {effective_reasoning.to_openrouter_dict() if effective_reasoning else 'None'}")
        
        # Verify token allocations
        reasoning_ok = token_limits["max_reasoning_tokens"] == test_config["expected_reasoning_tokens"]
        response_ok = token_limits["max_response_tokens"] == test_config["expected_response_tokens"]
        
        if reasoning_ok and response_ok:
            print("   ✅ Token allocation is optimal")
        else:
            print(f"   ❌ Unexpected allocation: expected {test_config['expected_reasoning_tokens']}+{test_config['expected_response_tokens']}")
        
        # Calculate token efficiency
        total_allocated = token_limits["max_reasoning_tokens"] + token_limits["max_response_tokens"]
        reasoning_ratio = token_limits["max_reasoning_tokens"] / total_allocated
        
        print(f"   📊 Token Distribution: {reasoning_ratio:.1%} reasoning, {1-reasoning_ratio:.1%} response")
        
        # Show reasoning capability
        if effective_reasoning:
            reasoning_dict = effective_reasoning.to_openrouter_dict()
            if "max_tokens" in reasoning_dict:
                reasoning_capacity = reasoning_dict["max_tokens"]
                overhead = token_limits["max_reasoning_tokens"] - reasoning_capacity
                print(f"   🧠 Reasoning Capacity: {reasoning_capacity:,} tokens (+ {overhead:,} overhead)")
            elif "effort" in reasoning_dict:
                print(f"   🧠 Reasoning Effort: {reasoning_dict['effort']} (dynamic allocation)")
            else:
                print(f"   🧠 Reasoning Mode: Basic (no token control)")
        
        print(f"   💡 Use Case: {get_use_case_description(test_config['name'])}")
    
    print("\n" + "=" * 80)
    print("📋 Token Optimization Summary:")
    print("   • Gemini Thinking: 40K reasoning + 8K response = 48K total")
    print("   • Claude: 12K reasoning + 4K response = 16K total")
    print("   • OpenAI o-series: 8K reasoning + 3K response = 11K total")
    print("   • DeepSeek-R1: 2K reasoning + 1.5K response = 3.5K total")
    print("\n💡 Higher allocations for models with larger context windows and reasoning capabilities")

def get_use_case_description(model_name: str) -> str:
    """Get use case description for a model"""
    if "Gemini" in model_name:
        return "Complex reasoning requiring extensive thought processes"
    elif "Claude" in model_name:
        return "Balanced reasoning and response generation"
    elif "OpenAI" in model_name:
        return "Efficient reasoning with controlled token usage"
    elif "DeepSeek" in model_name:
        return "Basic reasoning tasks with minimal token overhead"
    else:
        return "General purpose reasoning"

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_my_problem.py \"Your problem here\"")
        print("  python test_my_problem.py --test-models    # Test model-specific configurations")
        print("  python test_my_problem.py --test-defaults  # Test model-specific default configs")
        print("  python test_my_problem.py --test-token-limits  # Test model-specific token limits")
        print("  python test_my_problem.py --test-gemini  # Test reasoning with Gemini model and increased token limits")
        print("  python test_my_problem.py --test-optimization  # Test token optimization across all models")
        print()
        print("Examples:")
        print("  python test_my_problem.py \"What is 2+2?\"")
        print("  python test_my_problem.py \"Explain quantum computing\"")
        print("  python test_my_problem.py --test-models")
        print("  python test_my_problem.py --test-defaults")
        print("  python test_my_problem.py --test-token-limits")
        print("  python test_my_problem.py --test-gemini")
        print("  python test_my_problem.py --test-optimization")
        sys.exit(1)
    
    if sys.argv[1] == "--test-models":
        test_different_reasoning_models()
        return
    
    if sys.argv[1] == "--test-defaults":
        test_model_default_reasoning_configs()
        return
    
    if sys.argv[1] == "--test-token-limits":
        test_token_limits()
        return
    
    if sys.argv[1] == "--test-gemini":
        test_reasoning_with_gemini()
        return
    
    if sys.argv[1] == "--test-optimization":
        test_token_optimization()
        return
    
    problem_text = sys.argv[1]
    
    # Check for debug flag
    debug_mode = "--debug" in sys.argv
    
    if debug_mode:
        success = test_hybrid_thinking_debug(problem_text)
    else:
        success = test_hybrid_thinking(problem_text)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 