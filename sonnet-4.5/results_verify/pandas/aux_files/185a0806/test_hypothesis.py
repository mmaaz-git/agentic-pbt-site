import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.api.extensions import take


@given(
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=100),
    num_fills=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=100)
def test_take_series_with_allow_fill(values, num_fills):
    series = pd.Series(values)
    indices = [0] * (len(values) // 2) + [-1] * num_fills
    result = take(series, indices, allow_fill=True)
    assert len(result) == len(indices)

# Test the specific failing input mentioned
print("Testing specific failing input: values=[0.0], num_fills=1")
try:
    test_take_series_with_allow_fill.hypothesis.inner_test(values=[0.0], num_fills=1)
    print("Test passed")
except Exception as e:
    print(f"Test failed with error: {e}")

# Run the full hypothesis test
print("\nRunning full hypothesis test...")
test_take_series_with_allow_fill()