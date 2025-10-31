import pandas.api.types as types
from hypothesis import given, strategies as st
import numpy as np


# Test from the bug report
@given(st.one_of(
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
))
def test_infer_dtype_accepts_scalars(val):
    result_scalar = types.infer_dtype(val, skipna=False)
    result_list = types.infer_dtype([val], skipna=False)
    assert result_scalar == result_list

# Run the test
if __name__ == "__main__":
    print("Testing with Hypothesis...")
    try:
        test_infer_dtype_accepts_scalars()
        print("Hypothesis test passed!")
    except Exception as e:
        print(f"Hypothesis test failed: {e}")

    # Test specific examples from the bug report
    print("\n--- Testing specific examples ---")

    test_values = [
        (0, "int"),
        (1.5, "float"),
        (True, "bool"),
        (1+2j, "complex"),
        (None, "None"),
        ("hello", "str"),
        (b"bytes", "bytes"),
        (np.int64(5), "np.int64"),
        (np.float64(5.5), "np.float64")
    ]

    for val, desc in test_values:
        print(f"\nTesting {desc}: {val}")
        try:
            result = types.infer_dtype(val, skipna=False)
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")

        # Also test with list version
        try:
            result_list = types.infer_dtype([val], skipna=False)
            print(f"  List result: {result_list}")
        except Exception as e:
            print(f"  List error: {type(e).__name__}: {e}")