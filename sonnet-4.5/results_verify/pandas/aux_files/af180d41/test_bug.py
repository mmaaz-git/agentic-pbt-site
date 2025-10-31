#!/usr/bin/env python3
"""Test the reported bug in create_valid_python_identifier"""

import sys
import traceback
from hypothesis import given, strategies as st

# Import the function
try:
    from pandas.core.computation.parsing import create_valid_python_identifier
    print("Successfully imported create_valid_python_identifier")
except ImportError as e:
    print(f"Failed to import: {e}")
    sys.exit(1)

# First, test the simple reproduction case
print("\n--- Testing simple reproduction case ---")
try:
    result = create_valid_python_identifier('\x1f')
    print(f"Result for '\\x1f': {result}")
    print(f"Is valid identifier: {result.isidentifier()}")
except SyntaxError as e:
    print(f"SyntaxError as reported: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
    traceback.print_exc()

# Test other control characters mentioned
print("\n--- Testing other control characters ---")
test_chars = ['\x15', '\x17', '\x19', '\x1b', '\x1d', '\xa0', '😍']
for char in test_chars:
    try:
        result = create_valid_python_identifier(char)
        print(f"Character {repr(char)}: Success -> {result}")
    except SyntaxError as e:
        print(f"Character {repr(char)}: SyntaxError -> {e}")
    except Exception as e:
        print(f"Character {repr(char)}: Unexpected error -> {e}")

# Now run the hypothesis test
print("\n--- Running hypothesis test ---")
@given(st.text(min_size=1, max_size=1))
def test_create_valid_python_identifier_handles_all_chars(name):
    if '#' in name:
        return  # Skip hashtag as documented

    try:
        result = create_valid_python_identifier(name)
        assert result.isidentifier(), f"Result {result} is not a valid identifier"
    except SyntaxError:
        raise  # Re-raise to detect failures

# Run the hypothesis test manually to catch failures
from hypothesis import settings, Phase
from hypothesis.database import DirectoryBasedExampleDatabase

failures = []
test_count = 0

@given(st.text(min_size=1, max_size=1))
@settings(
    max_examples=200,
    phases=[Phase.generate, Phase.target],
    database=None,
    deadline=None,
    suppress_health_check=True
)
def run_test(name):
    global test_count, failures
    test_count += 1

    if '#' in name:
        return  # Skip hashtag as documented

    try:
        result = create_valid_python_identifier(name)
        if not result.isidentifier():
            failures.append((name, f"Result '{result}' is not valid identifier"))
    except SyntaxError as e:
        failures.append((name, str(e)))

# Run the test
print("Running hypothesis testing...")
try:
    run_test()
except Exception as e:
    print(f"Hypothesis test framework error: {e}")

print(f"\nTest summary: {test_count} tests run")
if failures:
    print(f"Found {len(failures)} failures:")
    for char, error in failures[:10]:  # Show first 10 failures
        print(f"  Character {repr(char)}: {error}")
    if len(failures) > 10:
        print(f"  ... and {len(failures) - 10} more")
else:
    print("No failures found in hypothesis testing")

# Test normal cases to confirm function works for regular input
print("\n--- Testing normal cases ---")
normal_cases = ['hello', 'test_name', 'column1', 'data', ' space ', '!test', '€euro']
for name in normal_cases:
    try:
        result = create_valid_python_identifier(name)
        print(f"'{name}' -> '{result}' (valid: {result.isidentifier()})")
    except Exception as e:
        print(f"'{name}' -> Error: {e}")