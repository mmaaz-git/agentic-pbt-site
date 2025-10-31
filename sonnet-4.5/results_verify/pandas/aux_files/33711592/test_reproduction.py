from pandas.core.dtypes.inference import is_re_compilable
import re

# Test the exact cases mentioned in the bug report
test_cases = ["(", ")", "?", "*", "["]

for pattern in test_cases:
    print(f"Testing pattern: {pattern!r}")
    try:
        result = is_re_compilable(pattern)
        print(f"  Result: {result}")
    except re.error as e:
        print(f"  ERROR: re.error: {e}")
    except re.PatternError as e:
        print(f"  ERROR: re.PatternError: {e}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

    # Compare with what re.compile does directly
    print(f"  Direct re.compile test:")
    try:
        re.compile(pattern)
        print(f"    re.compile succeeded")
    except (TypeError, re.error) as e:
        print(f"    re.compile failed with {type(e).__name__}: {e}")