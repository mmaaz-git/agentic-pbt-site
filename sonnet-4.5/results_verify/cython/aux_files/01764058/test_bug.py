#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Utils import build_hex_version

test_cases = [
    ("", "empty string"),
    (".", "just a dot"),
    ("a", "alphabetic character"),
    ("1.0foo", "version with invalid suffix"),
    ("1..0", "double dot"),
    ("1.2.3", "valid version"),
    ("4.3a1", "valid version with alpha"),
    ("1.0b2", "valid version with beta"),
    ("2.0rc1", "valid version with rc"),
]

print("Testing build_hex_version function:")
print("=" * 50)

for test_input, description in test_cases:
    print(f"\nTest: {description}")
    print(f"Input: {repr(test_input)}")
    try:
        result = build_hex_version(test_input)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")

# Now test the hypothesis test
print("\n" + "=" * 50)
print("\nRunning Hypothesis test:")

from hypothesis import given, strategies as st

@given(st.text())
def test_build_hex_version_handles_all_strings(version_str):
    try:
        result = build_hex_version(version_str)
        assert result.startswith('0x')
        assert len(result) == 10
    except ValueError as e:
        assert "invalid literal for int()" not in str(e), \
            f"Should raise informative error, not '{e}'"

# Run a few specific examples from hypothesis
print("\nSpecific failing examples:")
for test in ["", ".", "a", "1.0foo"]:
    print(f"\nTesting: {repr(test)}")
    try:
        test_build_hex_version_handles_all_strings(test)
        print(f"  Hypothesis test passed")
    except AssertionError as e:
        print(f"  Hypothesis test failed: {e}")
    except Exception as e:
        print(f"  Unexpected error: {e}")