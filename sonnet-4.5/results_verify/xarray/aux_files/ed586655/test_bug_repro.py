#!/usr/bin/env python3
"""Test reproduction of the cache mutation bug in xarray._load_static_files()"""

from hypothesis import given, strategies as st
from xarray.core.formatting_html import _load_static_files

# First, let's test the hypothesis test case
@given(st.text(min_size=1, max_size=100))
def test_cache_cannot_be_corrupted(corruption_text):
    _load_static_files.cache_clear()
    first = _load_static_files()
    original = first[0] if len(first) > 0 else None

    if original is not None:
        first[0] = corruption_text
        second = _load_static_files()

        assert second[0] == original, f"Cache was corrupted! Expected: {original[:50]}..., Got: {second[0][:50]}..."

# Test with manual reproduction
def test_manual_reproduction():
    print("=== Manual Reproduction Test ===")
    _load_static_files.cache_clear()

    original = _load_static_files()
    print(f"Original first element (first 50 chars): {original[0][:50]}")
    print(f"Type of returned value: {type(original)}")
    print(f"Number of elements: {len(original)}")

    # Modify the cached value
    original[0] = "CORRUPTED"

    # Get it again from cache
    second = _load_static_files()
    print(f"After mutation: {second[0]}")

    # This assertion should pass if bug exists
    assert second[0] == "CORRUPTED", "Bug not reproduced - cache was not corrupted"
    print("Bug confirmed: Cache was corrupted!")
    return True

if __name__ == "__main__":
    # Run manual test
    try:
        test_manual_reproduction()
    except Exception as e:
        print(f"Manual test error: {e}")

    # Run hypothesis test
    print("\n=== Running Hypothesis Test ===")
    try:
        test_cache_cannot_be_corrupted()
    except AssertionError as e:
        print(f"Hypothesis test failed as expected: {e}")
        print("This confirms the bug exists!")
    except Exception as e:
        print(f"Unexpected error: {e}")