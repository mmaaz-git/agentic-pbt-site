#!/usr/bin/env python3
"""Property-based test that discovered the pandas.cut bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from hypothesis import given, strategies as st, settings, example

@given(
    x=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=5, max_size=50),
    bins=st.integers(min_value=2, max_value=10)
)
@example(x=[0.0, 0.0, 0.0, 0.0, 5e-324], bins=2)  # Known failing example
@settings(max_examples=300, deadline=None)
def test_cut_preserves_length(x, bins):
    """Test that pd.cut with duplicates='drop' preserves array length."""
    result = pd.cut(x, bins=bins, duplicates='drop')
    assert len(result) == len(x), f"Result length {len(result)} != input length {len(x)}"

if __name__ == "__main__":
    # Run the test
    print("Running property-based test for pandas.cut...")
    print("This test checks that pd.cut(x, bins, duplicates='drop') preserves input length.")
    print()

    try:
        test_cut_preserves_length()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed with assertion: {e}")
    except Exception as e:
        print(f"Test failed with exception: {type(e).__name__}: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()