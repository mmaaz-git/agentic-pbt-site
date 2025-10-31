#!/usr/bin/env python3
"""Hypothesis property-based test for dask.utils.key_split"""

from hypothesis import given, strategies as st, settings
from dask.utils import key_split

@given(st.one_of(
    st.text(),
    st.binary(),
    st.tuples(st.text()),
    st.none(),
    st.lists(st.integers()),
    st.dictionaries(st.text(), st.integers()),
))
@settings(max_examples=100)
def test_key_split_never_raises(key):
    """Test that key_split never raises an exception and always returns a string."""
    try:
        result = key_split(key)
        assert isinstance(result, str), f"Expected string, got {type(result)}"
        print(f"✓ key_split({repr(key)[:50]}) = {result}")
    except Exception as e:
        print(f"✗ key_split({repr(key)[:50]}) raised {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    print("Running Hypothesis property-based test...")
    try:
        test_key_split_never_raises()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed with: {e}")