import pandas.api.types as pat

# Test cases that should return False but instead raise exceptions
invalid_regex_patterns = ['[', '?', '*', '(unclosed', ')', '(', '[]', '(*)', '+', '++']

print("Testing pandas.api.types.is_re_compilable with invalid regex patterns:")
print("=" * 60)

for pattern in invalid_regex_patterns:
    print(f"\nTesting pattern: {repr(pattern)}")
    try:
        result = pat.is_re_compilable(pattern)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")