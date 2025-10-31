#!/usr/bin/env python3
"""
Minimal reproduction of the bug in llm.default_plugins.openai_models.not_nulls
"""

def not_nulls(data) -> dict:
    """
    This is the buggy implementation from llm/default_plugins/openai_models.py:915
    """
    return {key: value for key, value in data if value is not None}

# Test case that should work but crashes
test_data = {'temperature': 0.7, 'max_tokens': None, 'top_p': 0.9}
print(f"Testing not_nulls with: {test_data}")

try:
    result = not_nulls(test_data)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Even simpler case that also fails
print("\nTesting with simpler case: {'a': 1}")
try:
    result = not_nulls({'a': 1})
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Empty dict is the only case that works
print("\nTesting with empty dict: {}")
try:
    result = not_nulls({})
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()