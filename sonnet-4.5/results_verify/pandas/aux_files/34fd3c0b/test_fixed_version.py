import re

def is_re_compilable_fixed(obj) -> bool:
    """Fixed version of is_re_compilable."""
    try:
        re.compile(obj)
    except (TypeError, re.error):  # re.error is the base class for re.PatternError
        return False
    else:
        return True

# Test with invalid regex patterns
invalid_patterns = ['[', '?', '*', '(unclosed', ')', '((']
print("Testing fixed version with invalid patterns:")
for pattern in invalid_patterns:
    result = is_re_compilable_fixed(pattern)
    print(f"  is_re_compilable_fixed('{pattern}'): {result}")

# Test with valid patterns
valid_patterns = [".*", "[a-z]+", "(test)", "\\d+", "^start$", ""]
print("\nTesting fixed version with valid patterns:")
for pattern in valid_patterns:
    result = is_re_compilable_fixed(pattern)
    print(f"  is_re_compilable_fixed('{pattern}'): {result}")

# Test with non-strings
non_strings = [1, None, [], {}, 3.14]
print("\nTesting fixed version with non-strings:")
for obj in non_strings:
    result = is_re_compilable_fixed(obj)
    print(f"  is_re_compilable_fixed({obj}): {result}")