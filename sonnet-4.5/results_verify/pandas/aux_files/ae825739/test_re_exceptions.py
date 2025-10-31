import re

# Test what exceptions re.compile raises for various inputs
test_cases = [
    ("[", "Invalid regex - unterminated bracket"),
    (")", "Invalid regex - unbalanced parenthesis"),
    ("?", "Invalid regex - nothing to repeat"),
    ("\\", "Invalid regex - bad escape"),
    (123, "Non-string input"),
    (None, "None input"),
    ([], "List input"),
    (".*", "Valid regex"),
]

for pattern, description in test_cases:
    print(f"\nTesting: {description}")
    print(f"Pattern: {repr(pattern)}")
    try:
        result = re.compile(pattern)
        print(f"Success: compiled to {result}")
    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")
        print(f"Base classes: {[base.__name__ for base in type(e).__mro__]}")