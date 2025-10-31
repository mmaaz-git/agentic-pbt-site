import pandas.api.types as types

# Test with invalid regex patterns that should cause crashes
test_patterns = ['(', ')', '[', '?', '*', '+']

for pattern in test_patterns:
    print(f"Testing pattern: '{pattern}'")
    try:
        result = types.is_re_compilable(pattern)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
    print()