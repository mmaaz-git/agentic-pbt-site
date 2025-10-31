import re

def is_re_compilable_fixed(obj) -> bool:
    """
    Fixed version of is_re_compilable that catches both TypeError and re.error
    """
    try:
        re.compile(obj)
    except (TypeError, re.error):  # The fix: also catch re.error
        return False
    else:
        return True

# Test the fixed version
print("Testing fixed version:")
test_cases = [
    (".*", True),          # Valid regex
    ("hello", True),       # Valid regex
    ("[a-z]+", True),      # Valid regex
    ("(", False),          # Invalid regex - should return False
    (")", False),          # Invalid regex - should return False
    ("[", False),          # Invalid regex - should return False
    ("?", False),          # Invalid regex - should return False
    ("*", False),          # Invalid regex - should return False
    (1, False),            # Non-string - should return False
    (None, False),         # Non-string - should return False
    ([], False),           # Non-string - should return False
]

for test_input, expected in test_cases:
    result = is_re_compilable_fixed(test_input)
    status = "✓" if result == expected else "✗"
    print(f"{status} is_re_compilable_fixed({repr(test_input)}) = {result} (expected {expected})")