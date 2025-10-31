import pandas as pd
from hypothesis import given, strategies as st
from pandas.core.arrays._utils import to_numpy_dtype_inference
from pandas._libs import lib


@given(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=100))
def test_to_numpy_dtype_inference_returns_dtype_for_string_arrays(values):
    arr = pd.array(values, dtype="string")

    dtype, na_value = to_numpy_dtype_inference(arr, None, lib.no_default, False)

    assert dtype is not None, f"dtype should not be None for string arrays, got {dtype} for values {values[:5]}"

# Run the test
if __name__ == "__main__":
    try:
        test_to_numpy_dtype_inference_returns_dtype_for_string_arrays()
        print("Hypothesis test passed!")
    except AssertionError as e:
        print(f"Hypothesis test failed: {e}")

    # Test specific failing example
    print("\nTesting specific example ['0']:")
    arr = pd.array(['0'], dtype="string")
    dtype, na_value = to_numpy_dtype_inference(arr, None, lib.no_default, False)
    print(f"dtype: {dtype}")
    print(f"dtype is None: {dtype is None}")