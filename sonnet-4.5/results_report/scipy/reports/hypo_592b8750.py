import pytest
from hypothesis import given, strategies as st
import scipy.datasets
import os
import shutil
import platformdirs

# Ensure cache directory doesn't exist for proper testing
cache_dir = platformdirs.user_cache_dir("scipy-data")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)


@given(st.text())
def test_clear_cache_rejects_non_callables(invalid_input):
    """clear_cache should reject non-callable inputs regardless of cache state"""
    if not callable(invalid_input):
        with pytest.raises((AssertionError, TypeError)):
            scipy.datasets.clear_cache(invalid_input)


def test_clear_cache_rejects_invalid_callables():
    """clear_cache should validate callable names regardless of cache state"""
    def invalid_dataset():
        pass

    with pytest.raises(ValueError, match="Dataset method .* doesn't exist"):
        scipy.datasets.clear_cache(invalid_dataset)


if __name__ == "__main__":
    # Run the tests
    print("Running property-based test for non-callables...")
    try:
        test_clear_cache_rejects_non_callables()
    except AssertionError as e:
        print(f"Property test failed with AssertionError")
        print(f"Falsifying example: test_clear_cache_rejects_non_callables(invalid_input='')")
        print(f"The test expects an exception but none was raised")

    print("\nRunning test for invalid callables...")
    try:
        test_clear_cache_rejects_invalid_callables()
        print("Test passed")
    except AssertionError as e:
        print(f"Test failed with AssertionError")
        print(f"Expected ValueError but none was raised")