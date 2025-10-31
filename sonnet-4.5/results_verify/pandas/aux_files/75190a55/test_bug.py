import re
from pandas.core.dtypes.inference import is_re_compilable

# Test the specific failing case
pattern = '['

print(f"Testing pattern: '{pattern}'")
try:
    result = is_re_compilable(pattern)
    print(f"Result: {result}")
except re.PatternError as e:
    print(f"FAILED: is_re_compilable raised re.PatternError: {e}")
except Exception as e:
    print(f"FAILED with unexpected exception: {type(e).__name__}: {e}")

# Test other invalid patterns mentioned
invalid_patterns = ['[', '(', ')', '?', '*', '+', '\\']
for p in invalid_patterns:
    print(f"\nTesting pattern: '{p}'")
    try:
        result = is_re_compilable(p)
        print(f"Result: {result}")
    except re.PatternError as e:
        print(f"FAILED: is_re_compilable raised re.PatternError: {e}")
    except Exception as e:
        print(f"FAILED with unexpected exception: {type(e).__name__}: {e}")