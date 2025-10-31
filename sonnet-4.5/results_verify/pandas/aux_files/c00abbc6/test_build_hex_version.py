#!/usr/bin/env python3
"""Test case to reproduce the build_hex_version bug."""

# First test: Hypothesis property-based test
from hypothesis import given, strategies as st, settings
from Cython.Utils import build_hex_version
import re

@settings(max_examples=500)
@given(st.from_regex(r'^[0-9]+\.[0-9]+(\.[0-9]+)?([ab]|rc)?[0-9]*$', fullmatch=True))
def test_build_hex_version_format(version_string):
    result = build_hex_version(version_string)
    assert re.match(r'^0x[0-9A-F]{8}$', result), f"Version {version_string} produced {result} with length {len(result)} (expected 10: '0x' + 8 hex digits)"

if __name__ == "__main__":
    print("Running Hypothesis test...")
    try:
        test_build_hex_version_format()
        print("Hypothesis test passed for all 500 examples!")
    except AssertionError as e:
        print(f"Hypothesis test failed: {e}")
    except Exception as e:
        print(f"Error running hypothesis test: {e}")

    print("\n" + "="*50 + "\n")

    # Second test: Manual reproduction with the specific failing case
    print("Running manual reproduction with '0.70000'...")
    version = '0.70000'
    result = build_hex_version(version)
    print(f"Input:  {version}")
    print(f"Output: {result}")
    print(f"Length: {len(result)} (expected 10: '0x' + 8 hex digits)")

    # Check if this is actually a bug
    if len(result) == 11:
        print("\n✗ BUG CONFIRMED: The output has 9 hex digits instead of 8")
        # Let's understand why
        import re
        segments = re.split(r'(\D+)', version)
        digits = []
        for segment in segments:
            if segment != '.' and segment:
                digits.append(int(segment))
        print(f"\nParsed version components: {digits}")
        print(f"Issue: Component value 70000 exceeds 255 (0xFF), the maximum for 2 hex digits")
        print(f"70000 in hex = 0x{70000:X}, which needs more than 2 hex digits")
    else:
        print("\n✓ No bug found - output has the correct length")