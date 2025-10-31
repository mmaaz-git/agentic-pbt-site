import numpy as np
import sys
from hypothesis import given, strategies as st, settings
from pandas.core.ops import kleene_and

# Set recursion limit to catch the issue quickly
sys.setrecursionlimit(100)

@given(st.lists(st.booleans(), min_size=1, max_size=50),
       st.lists(st.booleans(), min_size=1, max_size=50))
@settings(max_examples=10)  # Reduced for quick testing
def test_kleene_and_without_na(left_vals, right_vals):
    min_len = min(len(left_vals), len(right_vals))
    left = np.array(left_vals[:min_len])
    right = np.array(right_vals[:min_len])
    try:
        result, mask = kleene_and(left, right, None, None)
        expected = left & right
        assert np.array_equal(result, expected)
        print(f"Test passed for inputs: left={left_vals[:min_len]}, right={right_vals[:min_len]}")
    except RecursionError as e:
        print(f"RecursionError for inputs: left={left_vals[:min_len]}, right={right_vals[:min_len]}")
        raise

# Run the test
print("Running property-based test...")
try:
    test_kleene_and_without_na()
except Exception as e:
    print(f"Test failed: {e}")