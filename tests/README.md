# Test Suite Documentation

This project includes a comprehensive test suite with different types of tests to ensure the hybrid thinking model functionality works correctly.

## Test Types

### 1. Unit Tests (Default)
- **Location**: `tests/hybrid/`, `tests/l2t/`, `tests/got/`, etc.
- **Purpose**: Test individual components in isolation with mocking
- **Run with**: `python run_tests.py` or `pytest`
- **Speed**: Fast (< 1 second)

### 2. Integration Tests
- **Location**: `tests/integration/`
- **Purpose**: Test end-to-end functionality with real API calls
- **Requirements**: `OPENROUTER_API_KEY` environment variable
- **Models Used**: 
  - Reasoning: `deepseek/deepseek-r1-0528:free`
  - Response: `openrouter/cypher-alpha:free`
- **Run with**: `python run_tests.py --integration`
- **Speed**: Slow (30+ seconds, makes real API calls)

### 3. Method Signature Tests
- **Location**: `tests/hybrid/test_hybrid_method_signatures.py`
- **Purpose**: Catch method signature issues that pure mocking tests miss
- **Run with**: `python run_tests.py --functionality`
- **Speed**: Fast (< 1 second)

## Running Tests

### Quick Commands

```bash
# Run unit tests only (default)
python run_tests.py

# Run integration tests (requires API key)
python run_tests.py --integration

# Run all tests
python run_tests.py --all

# Run hybrid-related tests only
python run_tests.py --hybrid

# Run with verbose output
python run_tests.py --verbose

# Run with coverage report
python run_tests.py --coverage
```

### Environment Setup

For integration tests, set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

### Test Configuration

Tests are configured via `pytest.ini`:
- Integration tests are **excluded by default**
- Use `-m integration` to run only integration tests
- Use `-m "not integration"` to explicitly exclude them

## Test Structure

### What the Tests Validate

1. **Method Signatures**: Ensures `LLMClient.call()` is called with correct parameters
2. **Stop Sequences**: Verifies early cancellation tokens are configured for token savings
3. **Reasoning Extraction**: Tests that reasoning tokens are properly extracted and injected
4. **Error Handling**: Validates graceful failure and fallback mechanisms
5. **End-to-End Flow**: Integration tests verify the complete hybrid thinking model works

### Why This Test Suite Was Created

The original test suite failed to detect a critical bug where `HybridProcessor` was calling `LLMClient.call()` with incorrect method signatures:

**Before (Broken)**:
```python
self.llm_client.call(prompt, models, temperature=temp, max_tokens=tokens)
```

**After (Fixed)**:
```python
config = LLMConfig(temperature=temp, max_tokens=tokens, stop=[stop_token])
self.llm_client.call(prompt, models, config)
```

The new test suite includes:
- **Method signature validation** to catch such issues
- **Integration tests** that make real API calls to verify end-to-end functionality
- **Functionality tests** that would have caught the original bug

## Test Results

Current test status:
- ✅ 19/19 hybrid unit tests passing
- ✅ 4/4 method signature tests passing  
- ✅ Integration tests available (require API key)
- ✅ CLI integration tests included

## Adding New Tests

When adding new functionality:

1. **Add unit tests** for individual components
2. **Add method signature tests** if new LLM calls are introduced
3. **Add integration tests** for new end-to-end workflows
4. **Update this README** with any new test categories

## Troubleshooting

### Common Issues

1. **"No module named pytest"**: Activate virtual environment first
   ```bash
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

2. **Integration tests skipped**: Set `OPENROUTER_API_KEY` environment variable

3. **Tests fail with method signature errors**: This indicates a real bug that needs fixing

### Getting Help

If tests fail:
1. Check the error message for specific issues
2. Run with `--verbose` for more details
3. Check that virtual environment is activated
4. Verify API keys are set for integration tests 