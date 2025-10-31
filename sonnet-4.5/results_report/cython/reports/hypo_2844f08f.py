import numpy as np
from hypothesis import given, strategies as st, settings, example
from pandas.core.ops import kleene_and

@given(st.lists(st.booleans(), min_size=1, max_size=50),
       st.lists(st.booleans(), min_size=1, max_size=50))
@example([True], [True])  # Add a specific example to ensure it fails
@settings(max_examples=10)  # Reduce examples to get cleaner output
def test_kleene_and_without_na(left_vals, right_vals):
    min_len = min(len(left_vals), len(right_vals))
    left = np.array(left_vals[:min_len])
    right = np.array(right_vals[:min_len])
    result, mask = kleene_and(left, right, None, None)
    expected = left & right
    assert np.array_equal(result, expected)

if __name__ == "__main__":
    # Run the test
    test_kleene_and_without_na()