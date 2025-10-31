#!/usr/bin/env python3
"""
Hypothesis-based property test for xarray._load_static_files cache mutation bug.
This tests the invariant that cached values should not be mutable.
"""

from hypothesis import given, strategies as st
from xarray.core.formatting_html import _load_static_files

@given(st.text(min_size=1, max_size=100))
def test_cache_cannot_be_corrupted(corruption_text):
    """Test that the cache cannot be corrupted by modifying returned values."""
    # Clear cache to ensure clean state
    _load_static_files.cache_clear()

    # Get the first result and store the original value
    first = _load_static_files()
    original = first[0]

    # Try to corrupt the cache by modifying the returned list
    first[0] = corruption_text

    # Get the result again - it should be unchanged (immutable)
    second = _load_static_files()

    # The cache should NOT have been corrupted
    assert second[0] == original, f"Cache was corrupted! Expected original value but got: {corruption_text}"

if __name__ == "__main__":
    # Run the test with Hypothesis
    import sys
    try:
        test_cache_cannot_be_corrupted()
        print("Test passed! (This shouldn't happen if the bug exists)")
    except AssertionError as e:
        print(f"Test failed as expected: {e}")
        sys.exit(1)