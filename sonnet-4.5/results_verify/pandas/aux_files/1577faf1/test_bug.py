import pandas.api.types as pat
import re

# Test with invalid regex patterns
test_patterns = [')', '?', '*', '(', '[']

print("Testing invalid regex patterns with is_re_compilable:")
for pattern in test_patterns:
    print(f"\nPattern: {repr(pattern)}")
    try:
        result = pat.is_re_compilable(pattern)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Exception raised: {type(e).__name__}: {e}")

# Also test the expected behavior
print("\n\nTesting with valid patterns:")
print(f"'.*': {pat.is_re_compilable('.*')}")
print(f"1 (integer): {pat.is_re_compilable(1)}")
print(f"None: {pat.is_re_compilable(None)}")
