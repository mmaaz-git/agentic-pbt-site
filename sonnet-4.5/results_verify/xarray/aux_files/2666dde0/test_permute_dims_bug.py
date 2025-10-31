"""Test file to reproduce the NamedArray.permute_dims() bug with duplicate dimensions"""

import numpy as np
import warnings
from hypothesis import given, strategies as st, settings
from xarray.namedarray.core import NamedArray


# Property-based test from bug report
@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=50)
def test_permute_dims_with_duplicate_names_transposes_data(rows, cols):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        arr = NamedArray(("x", "x"), np.arange(rows * cols).reshape(rows, cols))

    result = arr.permute_dims()

    np.testing.assert_array_equal(result.to_numpy(), arr.to_numpy().T,
                                   err_msg="permute_dims() should transpose data even with duplicate dimension names")


# Simple example from bug report
def reproduce_simple_example():
    print("=" * 60)
    print("REPRODUCING SIMPLE EXAMPLE FROM BUG REPORT")
    print("=" * 60)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        arr = NamedArray(("x", "x"), np.array([[1, 2], [3, 4]]))

    print("Original:")
    print(arr.to_numpy())

    result = arr.permute_dims()

    print("\nAfter permute_dims():")
    print(result.to_numpy())

    print("\nExpected (transposed):")
    print(arr.to_numpy().T)

    # Check if they're equal
    if np.array_equal(result.to_numpy(), arr.to_numpy()):
        print("\nERROR: The array was NOT transposed (returned unchanged)")
    else:
        print("\nSUCCESS: The array was properly transposed")


if __name__ == "__main__":
    # Run simple example first
    reproduce_simple_example()

    print("\n" + "=" * 60)
    print("RUNNING PROPERTY-BASED TEST")
    print("=" * 60)

    # Run property-based test
    try:
        test_permute_dims_with_duplicate_names_transposes_data()
        print("Property-based test PASSED")
    except AssertionError as e:
        print(f"Property-based test FAILED: {e}")