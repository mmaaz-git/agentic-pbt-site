#!/usr/bin/env python3
"""Test script to reproduce the parse_list bug"""

import sys
import traceback
from hypothesis import given, strategies as st
from Cython.Build.Dependencies import parse_list

def test_basic():
    """Basic test for the parse_list function."""
    # Test the example cases from the bug report
    test_cases = [
        ("'", "Single quote"),
        ('"', "Double quote"),
        ("0'", "Digit and quote"),
        ("#", "Hash character"),
        ("[']", "Quote in brackets")
    ]

    print("Testing basic cases that should trigger the bug:")
    for test_input, description in test_cases:
        print(f"\nTesting: {description} - Input: {repr(test_input)}")
        try:
            result = parse_list(test_input)
            print(f"  Result: {result}")
        except KeyError as e:
            print(f"  KeyError: {e}")
        except Exception as e:
            print(f"  Exception ({type(e).__name__}): {e}")

# Run the hypothesis test
@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=0, max_size=100))
def test_parse_list_no_crash(s):
    """Property-based test: parse_list should never crash with KeyError."""
    try:
        result = parse_list(s)
        assert isinstance(result, list)
    except KeyError as e:
        # This is the bug we're looking for
        print(f"Found KeyError on input {repr(s)}: {e}")
        raise
    except Exception:
        # Other exceptions might be acceptable
        pass

if __name__ == "__main__":
    print("=" * 60)
    print("Testing parse_list function from Cython.Build.Dependencies")
    print("=" * 60)

    # Run basic tests
    test_basic()

    # Try to run hypothesis test
    print("\n" + "=" * 60)
    print("Running Hypothesis property-based test...")
    print("=" * 60)

    try:
        from hypothesis import find
        # Try to find a failing example
        failing_example = find(
            st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=0, max_size=10),
            lambda s: parse_list(s) or True  # Will raise if there's a KeyError
        )
        print(f"Found example that doesn't crash: {repr(failing_example)}")
    except Exception as e:
        print(f"Hypothesis found failure: {e}")
        traceback.print_exc()