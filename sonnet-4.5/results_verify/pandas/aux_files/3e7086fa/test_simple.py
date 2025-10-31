import pandas.api.types as pat
import traceback

# Test the specific cases from the bug report
test_cases = [')', '?', '*', '+', '(', '[', '\\']

for test_input in test_cases:
    print(f"\nTesting input: '{test_input}'")
    try:
        result = pat.is_re_compilable(test_input)
        print(f"  Result: {result} (type: {type(result).__name__})")
    except Exception as e:
        print(f"  ERROR: Raised {type(e).__name__}: {e}")