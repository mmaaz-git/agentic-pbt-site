import pandas.api.types as pat
import traceback

invalid_regex_patterns = ['[', '?', '*', '(unclosed']

for pattern in invalid_regex_patterns:
    print(f"Testing pattern: '{pattern}'")
    try:
        result = pat.is_re_compilable(pattern)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Exception raised: {type(e).__name__}: {e}")
        traceback.print_exc()
    print()