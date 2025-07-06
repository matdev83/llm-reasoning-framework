# Hybrid Thinking Model - Reasoning Format Support

The hybrid thinking model now supports multiple reasoning token formats used by different AI models. This ensures compatibility with various reasoning models and their different output formats.

## Supported Reasoning Formats

### 1. DeepSeek-R1 Format (`deepseek_r1`)
- **Tags**: `<THINKING>...</THINKING>`, `<think>...</think>`, `<thinking>...</thinking>`
- **Description**: DeepSeek-R1 models use thinking tags to separate reasoning from final output
- **Example**:
  ```
  <THINKING>
  Let me analyze this step by step...
  The key insight is...
  </THINKING>
  
  Based on my analysis, the answer is...
  ```

### 2. OpenAI o1 Format (`openai_o1`)
- **Tags**: `<reasoning>...</reasoning>`
- **Description**: OpenAI o1 models use reasoning tags for their thought process
- **Example**:
  ```
  <reasoning>
  I need to approach this systematically...
  First, let me consider...
  </reasoning>
  
  The solution is...
  ```

### 3. Gemini Thinking Mode (`gemini_thinking`)
- **Tags**: `<analysis>...</analysis>`, `<thinking>...</thinking>`
- **Description**: Google Gemini Flash Thinking mode uses analysis blocks
- **Example**:
  ```
  <analysis>
  Let me analyze this from multiple angles...
  Mathematical considerations...
  </analysis>
  
  Therefore, the conclusion is...
  ```

### 4. Claude Thinking Mode (`claude_thinking`)
- **Tags**: `<reflection>...</reflection>`, `<thinking>...</thinking>`
- **Description**: Anthropic Claude thinking mode with reflection
- **Example**:
  ```
  <reflection>
  I should carefully consider all aspects...
  What are the key factors?
  </reflection>
  
  After reflection, my response is...
  ```

### 5. QwQ Thinking Format (`qwq_thinking`)
- **Patterns**: Self-dialogue patterns, "Let me think about this step by step"
- **Description**: Alibaba QwQ self-dialogue and step-by-step reasoning
- **Example**:
  ```
  Let me think about this step by step.
  First, I need to understand...
  Then, I should consider...
  
  Therefore, the answer is...
  ```

### 6. Generic Chain-of-Thought (`generic_cot`)
- **Tags**: `<cot>...</cot>`
- **Patterns**: Common reasoning indicators like "Let me think", "I need to consider"
- **Description**: Generic chain-of-thought patterns and implicit reasoning
- **Example**:
  ```
  <cot>
  Breaking this down systematically:
  - Point A: Initial observation
  - Point B: Analysis
  </cot>
  
  Conclusion: The evidence supports...
  ```

### 7. Custom Token Format (`custom_token`)
- **Token**: User-specified completion token (e.g., `<REASONING_COMPLETE>`)
- **Description**: Custom completion token specified by user
- **Example**:
  ```
  I need to think through this carefully.
  Step 1: Understand the problem
  Step 2: Identify solutions
  <REASONING_COMPLETE>
  The final answer is...
  ```

## How It Works

### Automatic Format Detection
The `ReasoningExtractor` automatically detects which format is being used:

1. **Format Hints**: Based on model names (e.g., "deepseek-r1" → DeepSeek format)
2. **Pattern Matching**: Tries formats in order of likelihood
3. **Fallback**: Falls back to generic chain-of-thought patterns

### Format Priority
When no format hint is provided, formats are tried in this order:
1. DeepSeek-R1 (most common)
2. Gemini Thinking
3. Claude Thinking
4. OpenAI o1
5. QwQ Thinking
6. Generic CoT (fallback)

### Early Cancellation
For models that support it, stop sequences are configured to cancel reasoning generation early:
- Saves tokens and reduces latency
- Configured automatically based on detected format
- Custom tokens can be specified for stop sequences

## Configuration

### Model-Based Format Hints
The system automatically detects format hints from model names:

```python
# DeepSeek models
"deepseek/deepseek-r1-distill-qwen-7b" → ReasoningFormat.DEEPSEEK_R1

# OpenAI models  
"openai/o1-preview" → ReasoningFormat.OPENAI_O1

# Gemini models
"google/gemini-2.0-flash-thinking" → ReasoningFormat.GEMINI_THINKING
```

### Custom Token Configuration
You can specify custom completion tokens:

```python
config = HybridConfig(
    reasoning_complete_token="<MY_CUSTOM_TOKEN>",
    # ... other config
)
```

## Testing

### Robust Test Design
Tests are designed to be robust and not fragile:
- ✅ **Structural checks**: Verify tag removal, non-empty content
- ✅ **Token counting**: Ensure reasonable amount of reasoning extracted
- ✅ **Format detection**: Verify correct format is identified
- ❌ **Specific content**: Avoid checking for exact phrases that LLMs might not produce

### Test Categories
1. **Unit Tests**: Test individual reasoning extraction formats
2. **Integration Tests**: Test with real AI models and API calls
3. **Method Signature Tests**: Ensure correct interface usage

## Usage Examples

### Basic Usage
```python
from src.hybrid.reasoning_extractor import ReasoningExtractor

extractor = ReasoningExtractor()
reasoning, remaining, format_type = extractor.extract_reasoning(model_output)
```

### With Format Hint
```python
from src.hybrid.reasoning_extractor import ReasoningFormat

reasoning, remaining, format_type = extractor.extract_reasoning(
    model_output, 
    format_hint=ReasoningFormat.DEEPSEEK_R1
)
```

### With Custom Token
```python
reasoning, remaining, format_type = extractor.extract_reasoning(
    model_output,
    custom_token="<REASONING_COMPLETE>"
)
```

## Benefits

1. **Universal Compatibility**: Works with any reasoning model format
2. **Automatic Detection**: No manual configuration needed
3. **Token Efficiency**: Early cancellation saves tokens
4. **Robust Extraction**: Handles malformed or partial outputs gracefully
5. **Extensible**: Easy to add new formats as they emerge

## Future Formats

The system is designed to be easily extensible. New reasoning formats can be added by:
1. Adding new `ReasoningFormat` enum value
2. Defining regex patterns in `_initialize_patterns()`
3. Adding format hint detection in `_get_format_hint()`
4. Creating tests for the new format

This ensures the hybrid thinking model stays compatible with emerging AI reasoning models and their unique output formats.

# Hybrid Reasoning Process

The Hybrid Reasoning Process combines the reasoning capabilities of advanced AI models with optimized token management and model-specific configurations.

## Key Features

### 1. OpenRouter Reasoning Token Support
- **Universal Compatibility**: Works with all OpenRouter reasoning models
- **Model-Specific Configuration**: Automatic parameter filtering based on model capabilities
- **Streaming Optimization**: Real-time token processing for cost efficiency
- **Smart Defaults**: Zero-configuration setup with optimal model-specific settings

### 2. Model-Specific Token Limits
The system automatically adjusts token allocations based on **actual OpenRouter API output limits**, not context windows. These limits ensure requests don't exceed the real constraints:

#### OpenAI o-series Models (100K output limit)
- **Reasoning Tokens**: 32,000 (highest capacity for complex reasoning)
- **Response Tokens**: 8,000 (ample space for detailed responses)

#### OpenAI GPT-4o Models (16K output limit)
- **Reasoning Tokens**: 12,000 (balanced allocation)
- **Response Tokens**: 4,000 (adequate response space)

#### Anthropic Claude Models (8K output limit)
- **Reasoning Tokens**: 6,000 (conservative allocation)
- **Response Tokens**: 2,000 (adequate response space)

#### Gemini Models (8K-16K output limit)
- **Thinking Models**: 12,000 reasoning + 4,000 response
- **Regular Models**: 6,000 reasoning + 2,000 response

#### Qwen Models (Prompt-Based Activation)
- **Large Models (32B+)**: 12,000 reasoning + 4,000 response
- **Small Models (<32B)**: 6,000 reasoning + 2,000 response
- **Special Feature**: Uses `/think` and `/no_think` slash commands instead of API headers

#### DeepSeek-R1 Models (8K output limit)
- **Reasoning Tokens**: 6,000 (conservative allocation)
- **Response Tokens**: 2,000 (adequate response space)

### 3. Reasoning Activation Methods

#### API Header-Based (Most Models)
Models like OpenAI, Claude, Gemini, and DeepSeek use API headers for reasoning control:
```python
reasoning_config = ReasoningConfig(
    enabled=True,
    effort="high",        # For OpenAI models
    max_tokens=8000,      # For Claude/Gemini models
    exclude=False
)
```

#### Prompt-Based (Qwen Models)
Qwen models use special slash commands in the prompt text:
```python
# Automatic activation for Qwen models
config = HybridConfig(
    reasoning_model_name="qwen/qwen3-32b",
    reasoning_config=ReasoningConfig(enabled=True)
)

# Prompts are automatically modified:
# "Solve this problem" → "Solve this problem /think"
# "Solve this problem" → "Solve this problem /no_think" (when disabled)
```

### 4. Model Support Matrix

| Model Family | Reasoning Method | Effort Levels | Max Tokens | Prompt Commands |
|--------------|------------------|---------------|------------|-----------------|
| **OpenAI o-series** | API Headers | ✅ (low/medium/high) | ❌ | ❌ |
| **OpenAI GPT-4** | API Headers | ✅ (low/medium/high) | ❌ | ❌ |
| **Grok** | API Headers | ✅ (low/medium/high) | ❌ | ❌ |
| **Anthropic Claude** | API Headers | ❌ | ✅ (1-8000) | ❌ |
| **Gemini Thinking** | API Headers | ❌ | ✅ (1-8000) | ❌ |
| **Qwen/QwQ** | **Prompt Commands** | ❌ | ❌ | ✅ (/think, /no_think) |
| **DeepSeek-R1** | API Headers | ❌ | ❌ | ❌ |

## Configuration Examples

### Basic Usage (Auto-Detection)
```python
from hybrid.orchestrator import HybridOrchestrator
from hybrid.dataclasses import HybridConfig

# Zero configuration - uses optimal defaults
config = HybridConfig(
    reasoning_model_name="qwen/qwen3-32b",  # Automatically uses /think commands
    response_model_name="anthropic/claude-3.5-sonnet-20241022"
)

orchestrator = HybridOrchestrator(config)
result = await orchestrator.run("Complex reasoning problem")
```

### Advanced Configuration
```python
from hybrid.dataclasses import HybridConfig, ReasoningConfig

# Custom reasoning configuration
config = HybridConfig(
    reasoning_model_name="openai/o1-preview",
    response_model_name="openai/gpt-4o",
    reasoning_config=ReasoningConfig(
        enabled=True,
        effort="high",        # OpenAI-specific
        exclude=False
    ),
    use_streaming=True  # Enable for cost optimization
)
```

### Model-Specific Examples

#### OpenAI Models
```python
config = HybridConfig(
    reasoning_model_name="openai/o1-preview",
    response_model_name="openai/gpt-4o",
    reasoning_config=ReasoningConfig(effort="high")  # Uses effort levels
)
```

#### Claude Models
```python
config = HybridConfig(
    reasoning_model_name="anthropic/claude-3.5-sonnet-20241022",
    response_model_name="anthropic/claude-3.5-sonnet-20241022",
    reasoning_config=ReasoningConfig(max_tokens=4000)  # Uses token limits
)
```

#### Qwen Models
```python
config = HybridConfig(
    reasoning_model_name="qwen/qwen3-32b",
    response_model_name="qwen/qwen3-32b",
    reasoning_config=ReasoningConfig(enabled=True)  # Uses /think commands automatically
)
```

## Testing

The system includes comprehensive tests for all model types:

```bash
# Test model-specific configurations
python test_my_problem.py --test-defaults

# Test token limits
python test_my_problem.py --test-token-limits

# Test Qwen prompt activation
python test_my_problem.py --test-qwen-activation

# Test token optimization
python test_my_problem.py --test-optimization
```

## Key Benefits

1. **Zero Configuration**: Works out-of-the-box with optimal settings for each model
2. **Cost Optimization**: Streaming support reduces token usage and costs
3. **Universal Compatibility**: Supports all major reasoning model families
4. **Automatic Detection**: Model capabilities detected automatically
5. **Prompt Enhancement**: Qwen models get automatic `/think` command injection
6. **Realistic Limits**: Uses actual API constraints, not theoretical maximums
7. **Robust Extraction**: Handles multiple reasoning token formats automatically

## Important Notes

### ⚠️ **Critical: Output Limits vs Context Windows**
- **Context Window**: Total tokens the model can process (input + output)
- **Output Limit**: Maximum tokens the model can generate (reasoning + response)
- **Our limits are based on OUTPUT constraints, not context windows**
- **Example**: GPT-4o has 128K context but only 16K output limit

### 🔧 **Automatic Validation**
- Token allocations are automatically validated against real API limits
- Requests that would exceed limits are prevented before API calls
- Graceful degradation when limits are approached

### 📊 **Performance Optimization**
- Model-specific defaults provide optimal reasoning quality
- Conservative limits prevent API failures
- Streaming support for real-time token processing

## Configuration Reference

### ReasoningConfig Parameters
- `enabled`: Enable reasoning (default: True)
- `effort`: Effort level for OpenAI/Grok models ("low", "medium", "high")
- `max_tokens`: Direct token allocation for Anthropic/Gemini models
- `exclude`: Exclude reasoning from response (default: False)

### HybridConfig Parameters
- `reasoning_model_name`: Model for reasoning phase
- `response_model_name`: Model for response generation
- `reasoning_config`: Custom reasoning configuration (optional)
- `max_reasoning_tokens`: Manual override (not recommended)
- `max_response_tokens`: Manual override (not recommended)

## Troubleshooting

### Common Issues
1. **Token Limit Exceeded**: Check if custom limits exceed model constraints
2. **Poor Reasoning Quality**: Try higher effort levels or more reasoning tokens
3. **Truncated Responses**: Increase response token allocation
4. **API Failures**: Ensure total tokens don't exceed output limits

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test token limits
config = HybridConfig(reasoning_model_name="your-model")
limits = config.get_effective_token_limits()
print(f"Token limits: {limits}")
```

This ensures optimal performance while maintaining adequate token space for both reasoning and response generation across all supported models. 