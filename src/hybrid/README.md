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
The system automatically adjusts token allocations based on the reasoning model's capabilities and typical usage patterns:

#### Gemini Thinking Models
- **Context**: 1M tokens available
- **Reasoning Tokens**: 40,000 (accommodates 32K reasoning + overhead)
- **Response Tokens**: 8,000 (generous space for detailed responses)
- **Use Case**: Complex reasoning tasks requiring extensive thought processes

#### Anthropic Claude Models
- **Context**: 200K tokens available
- **Reasoning Tokens**: 12,000 (accommodates 8K reasoning + overhead)
- **Response Tokens**: 4,000 (good space for comprehensive responses)
- **Use Case**: Balanced reasoning and response generation

#### OpenAI o-series Models
- **Context**: Variable (depends on model)
- **Reasoning Tokens**: 8,000 (effort-based reasoning)
- **Response Tokens**: 3,000 (standard response space)
- **Use Case**: Efficient reasoning with controlled token usage

#### DeepSeek-R1 and Other Models
- **Reasoning Tokens**: 2,000 (conservative for basic reasoning)
- **Response Tokens**: 1,500 (standard response space)
- **Use Case**: Basic reasoning tasks with minimal token overhead

### 3. Supported Models

#### OpenAI o-series & Grok
- **Parameters**: `effort` levels ("low", "medium", "high")
- **Default**: `effort="high"` (best reasoning quality)
- **Models**: `openai/o1-preview`, `openai/o3-mini`, `x-ai/grok-*`

#### Anthropic Claude & Gemini Thinking
- **Parameters**: `max_tokens` parameter
- **Defaults**: 
  - Gemini: `max_tokens=32000` (maximum reasoning capability)
  - Claude: `max_tokens=8000` (high effort equivalent)
- **Models**: `anthropic/claude-*`, `google/gemini-*:thinking`

#### DeepSeek-R1
- **Parameters**: Basic reasoning only (no effort/max_tokens control)
- **Default**: `enabled=True`
- **Models**: `deepseek/deepseek-r1*`

## Configuration

### Basic Usage (Automatic Defaults)
```python
from hybrid.dataclasses import HybridConfig

# Zero configuration - uses optimal model-specific defaults
config = HybridConfig(
    reasoning_model_name="google/gemini-2.5-flash-preview:thinking",
    response_model_name="google/gemini-2.5-flash-preview"
)

# Automatic configuration:
# - Reasoning tokens: 32,000 (via reasoning_config.max_tokens)
# - Token limits: 40,000 reasoning + 8,000 response
# - Streaming: enabled
```

### Advanced Configuration
```python
from hybrid.dataclasses import HybridConfig, ReasoningConfig

# Custom reasoning configuration
config = HybridConfig(
    reasoning_model_name="google/gemini-2.5-flash-preview:thinking",
    response_model_name="google/gemini-2.5-flash-preview",
    reasoning_config=ReasoningConfig(
        enabled=True,
        max_tokens=16000,  # Custom reasoning token limit
        exclude=False
    ),
    # Custom token limits (overrides model-specific defaults)
    max_reasoning_tokens=20000,
    max_response_tokens=5000,
    use_streaming=True
)
```

### Token Limit Behavior
The system prioritizes token limits in this order:
1. **Custom values**: If you explicitly set `max_reasoning_tokens` or `max_response_tokens`
2. **Model-specific defaults**: Automatic allocation based on model capabilities
3. **Fallback defaults**: 1,500 tokens each (for unknown models)

## Usage Examples

### High-Reasoning Tasks (Gemini)
```python
# Best for complex reasoning requiring extensive thought
config = HybridConfig(
    reasoning_model_name="google/gemini-2.5-flash-preview:thinking",
    response_model_name="google/gemini-2.5-flash-preview"
)
# Automatically gets: 40K reasoning + 8K response tokens
```

### Balanced Tasks (Claude)
```python
# Good balance of reasoning and response quality
config = HybridConfig(
    reasoning_model_name="anthropic/claude-3.5-sonnet-20241022",
    response_model_name="anthropic/claude-3.5-sonnet-20241022"
)
# Automatically gets: 12K reasoning + 4K response tokens
```

### Efficient Tasks (OpenAI)
```python
# Efficient reasoning with controlled costs
config = HybridConfig(
    reasoning_model_name="openai/o1-preview",
    response_model_name="openai/gpt-4o"
)
# Automatically gets: 8K reasoning + 3K response tokens
```

## Token Optimization

### Why Higher Limits for Gemini?
- **32K Reasoning Tokens**: Gemini's default reasoning allocation
- **40K Total Limit**: Ensures adequate space for reasoning + API overhead
- **8K Response Tokens**: Prevents truncation of detailed responses
- **1M Context**: Gemini's large context window supports these allocations

### Streaming Benefits
- **Real-time Processing**: Tokens processed as they arrive
- **Cost Optimization**: Early termination possible for token-efficient workflows
- **Better UX**: Faster perceived response times

## Error Handling

The system gracefully handles various scenarios:
- **Model Not Found**: Falls back to basic configuration
- **Token Limit Exceeded**: Automatic truncation with warnings
- **API Errors**: Detailed error messages with context
- **Invalid Configuration**: Parameter validation and filtering

## Monitoring

Track token usage and performance:
```python
result = orchestrator.run("Your problem here")

if result.hybrid_result.succeeded:
    reasoning_stats = result.hybrid_result.reasoning_call_stats
    response_stats = result.hybrid_result.response_call_stats
    
    print(f"Reasoning tokens: {reasoning_stats.completion_tokens}")
    print(f"Response tokens: {response_stats.completion_tokens}")
    print(f"Total duration: {reasoning_stats.call_duration_seconds + response_stats.call_duration_seconds:.2f}s")
```

This ensures optimal performance while maintaining adequate token space for both reasoning and response generation across all supported models. 