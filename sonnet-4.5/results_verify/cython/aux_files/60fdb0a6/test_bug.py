#!/usr/bin/env python3
"""Test script to reproduce the reported bug in _parse_pattern"""

import sys
import traceback

# First, let's test the basic reproduction
print("=" * 60)
print("Testing basic reproduction of the bug")
print("=" * 60)

try:
    from Cython.TestUtils import _parse_pattern
    print("Successfully imported _parse_pattern")
except ImportError as e:
    print(f"Failed to import: {e}")
    sys.exit(1)

# Test the exact failing case mentioned
print("\nTest 1: Calling _parse_pattern('/\\/')")
print("-" * 40)
try:
    result = _parse_pattern("/\\/")
    print(f"Result: {result}")
    print("No error - function returned successfully")
except ValueError as e:
    print(f"ValueError occurred: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"Unexpected error: {e}")
    traceback.print_exc()

# Test similar patterns to understand the behavior
test_patterns = [
    ("/test/pattern", "Normal pattern with start marker"),
    ("/test/:/end/pattern", "Pattern with start and end markers"),
    ("/\\/pattern", "Pattern with backslash as start marker"),
    ("/\\/", "Just backslash as start marker (the bug case)"),
    ("/\\//pattern", "Backslash followed by slash in start marker"),
    ("/a\\/b/pattern", "Start marker with escaped slash inside"),
    ("//pattern", "Empty start marker"),
    ("/", "Just a single slash"),
    ("//", "Two slashes"),
    ("pattern", "No slashes at all"),
]

print("\n" + "=" * 60)
print("Testing various pattern formats")
print("=" * 60)

for pattern, description in test_patterns:
    print(f"\nTest: {description}")
    print(f"Pattern: {repr(pattern)}")
    print("-" * 40)
    try:
        result = _parse_pattern(pattern)
        print(f"Result: start={repr(result[0])}, end={repr(result[1])}, pattern={repr(result[2])}")
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Now test the hypothesis test case
print("\n" + "=" * 60)
print("Testing the Hypothesis test case")
print("=" * 60)

try:
    from hypothesis import given, strategies as st

    @given(
        st.text(alphabet=st.characters(blacklist_characters='/'), min_size=0),
        st.text(min_size=1)
    )
    def test_parse_pattern_with_start_marker(start_marker, pattern):
        full_pattern = f"/{start_marker}/{pattern}"
        try:
            start, end, parsed = _parse_pattern(full_pattern)
            assert start == start_marker
            return True
        except ValueError:
            return False

    # Run the specific failing case
    print("\nTesting specific failing input: start_marker='\\\\', pattern='0'")
    full_pattern = f"/{'\\\\'}/{'0'}"
    print(f"Full pattern: {repr(full_pattern)}")
    try:
        result = test_parse_pattern_with_start_marker('\\', '0')
        print(f"Hypothesis test result: {result}")
    except Exception as e:
        print(f"Hypothesis test failed with: {e}")
        traceback.print_exc()

except ImportError:
    print("Hypothesis not installed, skipping property-based test")