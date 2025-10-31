#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

from hypothesis import given, strategies as st, assume
import pytest
from scipy.datasets._utils import _clear_cache
from scipy.datasets._registry import method_files_map
import os

# Make sure test directory exists
os.makedirs("/tmp/test", exist_ok=True)

@given(datasets_input=st.one_of(
    st.integers(),
    st.text(),
    st.dictionaries(st.text(), st.integers())
))
def test_clear_cache_invalid_types(datasets_input):
    assume(not callable(datasets_input))

    # The test expects one of these exceptions
    with pytest.raises((AttributeError, AssertionError, TypeError)):
        _clear_cache(datasets_input, cache_dir="/tmp/test", method_map=method_files_map)

# Run the test
if __name__ == "__main__":
    print("Running hypothesis test...")
    print("=" * 50)
    try:
        test_clear_cache_invalid_types()
        print("Hypothesis test passed!")
    except Exception as e:
        print(f"Hypothesis test failed: {e}")
        import traceback
        traceback.print_exc()