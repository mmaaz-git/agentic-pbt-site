import pandas as pd
from hypothesis import given, strategies as st, settings


@settings(max_examples=500)
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e3, max_value=1e3) | st.none(), min_size=1, max_size=50))
def test_floatingarray_na_propagation_consistency(values):
    arr = pd.array(values, dtype="Float64")

    pow_zero = arr ** 0
    mul_zero = arr * 0

    for i in range(len(arr)):
        if pd.isna(arr[i]):
            assert pd.isna(mul_zero[i]), f"NA * 0 correctly returns NA"
            assert pd.isna(pow_zero[i]), f"NA ** 0 should return NA but got {pow_zero[i]}"

# Run the test
if __name__ == "__main__":
    print("Running property-based test...")
    try:
        test_floatingarray_na_propagation_consistency()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed with assertion: {e}")
    except Exception as e:
        print(f"Test failed with exception: {e}")