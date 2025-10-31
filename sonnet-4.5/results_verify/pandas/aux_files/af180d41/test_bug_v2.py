#!/usr/bin/env python3
"""Test the reported bug in create_valid_python_identifier"""

import sys
import traceback
from hypothesis import given, strategies as st, settings

# Import the function
try:
    from pandas.core.computation.parsing import create_valid_python_identifier
    print("Successfully imported create_valid_python_identifier")
except ImportError as e:
    print(f"Failed to import: {e}")
    sys.exit(1)

# First, test the simple reproduction case as reported
print("\n--- Testing simple reproduction case from bug report ---")
print("Code: create_valid_python_identifier('\\x1f')")
try:
    result = create_valid_python_identifier('\x1f')
    print(f"Result: {result}")
    print(f"Is valid identifier: {result.isidentifier()}")
except SyntaxError as e:
    print(f"SyntaxError (as reported in bug): {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
    traceback.print_exc()

# Test other control characters mentioned
print("\n--- Testing other problematic characters from bug report ---")
test_chars = ['\x15', '\x17', '\x19', '\x1b', '\x1d', '\xa0', '😍']
for char in test_chars:
    try:
        result = create_valid_python_identifier(char)
        print(f"Character {repr(char)}: Success -> {result}")
    except SyntaxError as e:
        print(f"Character {repr(char)}: SyntaxError -> {e}")
    except Exception as e:
        print(f"Character {repr(char)}: Unexpected error -> {e}")

# Test normal cases to confirm function works for regular input
print("\n--- Testing normal cases (should work) ---")
normal_cases = ['hello', 'test_name', 'column1', 'data', 'my column', '!test', '€euro', 'a-b']
for name in normal_cases:
    try:
        result = create_valid_python_identifier(name)
        print(f"'{name}' -> '{result}' (valid: {result.isidentifier()})")
    except Exception as e:
        print(f"'{name}' -> Error: {e}")

# Now test with hypothesis to find more failure cases
print("\n--- Running hypothesis test to find more failures ---")

failures = []
success_count = 0

@given(st.text(min_size=1, max_size=1))
@settings(max_examples=100, deadline=None)
def test_single_char(char):
    global failures, success_count

    # Skip hashtag as it's documented to fail
    if '#' in char:
        return

    try:
        result = create_valid_python_identifier(char)
        if result.isidentifier():
            success_count += 1
        else:
            failures.append((char, f"Result '{result}' is not a valid identifier"))
    except SyntaxError as e:
        failures.append((char, str(e)))

try:
    test_single_char()
except Exception:
    pass  # Hypothesis will handle its own errors

print(f"\nHypothesis test results:")
print(f"  Successes: {success_count}")
print(f"  Failures: {len(failures)}")
if failures:
    print("\nFirst 20 failing characters:")
    for char, error in failures[:20]:
        print(f"  Character {repr(char)}: {error}")
    if len(failures) > 20:
        print(f"  ... and {len(failures) - 20} more failures")

# Let's also check what the function's docstring says about restrictions
print("\n--- Function documentation ---")
print(create_valid_python_identifier.__doc__)

# Let's trace through exactly what happens with '\x1f'
print("\n--- Detailed trace for '\\x1f' ---")
test_char = '\x1f'
print(f"Input character: {repr(test_char)}")
print(f"Character code: {ord(test_char)}")
print(f"Is original identifier?: {test_char.isidentifier()}")

# Looking at the code, it builds special_characters_replacements dict
# Then does: name = "".join([special_characters_replacements.get(char, char) for char in name])
# For '\x1f', it's not in the dict, so it passes through unchanged
# Then adds prefix: name = f"BACKTICK_QUOTED_STRING_{name}"
expected_result = f"BACKTICK_QUOTED_STRING_{test_char}"
print(f"Expected intermediate result: {repr(expected_result)}")
print(f"Is this a valid identifier?: {expected_result.isidentifier()}")

# The bug is clear: control characters like \x1f are not valid in Python identifiers
# but the function doesn't replace them