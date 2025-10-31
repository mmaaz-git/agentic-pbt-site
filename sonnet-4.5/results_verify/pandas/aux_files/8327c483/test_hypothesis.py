import pandas as pd
from hypothesis import given, strategies as st, settings, assume

@given(
    arr=st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=20,
        max_size=100
    ),
    n_bins=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=200)
def test_qcut_approximately_equal_sized_bins(arr, n_bins):
    assume(len(set(arr)) >= n_bins)
    result = pd.qcut(arr, q=n_bins, duplicates='drop')
    value_counts = result.value_counts()
    if len(value_counts) > 1:
        max_bin_size = value_counts.max()
        min_bin_size = value_counts.min()
        assert max_bin_size - min_bin_size <= 2

# Test with the specific failing input
print("Testing with the specific failing input from the bug report...")
try:
    test_qcut_approximately_equal_sized_bins(
        arr=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5e-324],
        n_bins=2
    )
    print("Test passed!")
except ValueError as e:
    print(f"Test failed with ValueError: {e}")
except AssertionError as e:
    print(f"Test failed with AssertionError: {e}")
except Exception as e:
    print(f"Test failed with {type(e).__name__}: {e}")

print("\nRunning hypothesis tests...")
try:
    test_qcut_approximately_equal_sized_bins()
    print("All hypothesis tests passed!")
except Exception as e:
    print(f"Hypothesis tests failed: {type(e).__name__}: {e}")