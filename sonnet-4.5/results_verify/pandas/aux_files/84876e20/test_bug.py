import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.api.extensions import take


@given(
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000), min_size=2, max_size=20),
    fill_val=st.floats(allow_nan=False, allow_infinity=False, min_value=-10000, max_value=10000)
)
@settings(max_examples=300)
def test_index_allow_fill_with_value_should_use_fillvalue(values, fill_val):
    index = pd.Index(values, dtype='float64')
    arr = np.array(values)
    indices = [0, -1, 1]

    index_result = take(index, indices, allow_fill=True, fill_value=fill_val)
    array_result = take(arr, indices, allow_fill=True, fill_value=fill_val)

    assert index_result[1] == array_result[1], f"Index and array should return same fill_value"
    assert index_result[1] == fill_val, f"Expected fill_value {fill_val}, got {index_result[1]}"

# Test with the specific failing input
print("Testing specific failing input:")
values = [0.0, 0.0]
fill_val = 0.0
index = pd.Index(values, dtype='float64')
arr = np.array(values)
indices = [0, -1, 1]

index_result = take(index, indices, allow_fill=True, fill_value=fill_val)
array_result = take(arr, indices, allow_fill=True, fill_value=fill_val)

print(f"Index result: {list(index_result)}")
print(f"Array result: {list(array_result)}")
print(f"Index result[1]: {index_result[1]}")
print(f"Array result[1]: {array_result[1]}")
print(f"Fill value: {fill_val}")
print(f"Are they equal? {index_result[1] == array_result[1]}")
print(f"Is index_result[1] NaN? {pd.isna(index_result[1])}")

# Run hypothesis test
print("\nRunning hypothesis test:")
test_index_allow_fill_with_value_should_use_fillvalue()