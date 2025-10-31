import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import numpy as np
from hypothesis import given, strategies as st
import pandas.core.strings.accessor as accessor


@given(st.text())
def test_cat_core_empty_list_returns_array(sep):
    result = accessor.cat_core([], sep)
    assert isinstance(result, np.ndarray), f"Expected np.ndarray, got {type(result).__name__}: {result}"


@given(st.text())
def test_cat_safe_empty_list_returns_array(sep):
    result = accessor.cat_safe([], sep)
    assert isinstance(result, np.ndarray), f"Expected np.ndarray, got {type(result).__name__}: {result}"


if __name__ == "__main__":
    # Run the tests
    print("Testing cat_core with empty list...")
    try:
        test_cat_core_empty_list_returns_array()
        print("cat_core test passed!")
    except Exception as e:
        print(f"cat_core test failed: {e}")

    print("\nTesting cat_safe with empty list...")
    try:
        test_cat_safe_empty_list_returns_array()
        print("cat_safe test passed!")
    except Exception as e:
        print(f"cat_safe test failed: {e}")