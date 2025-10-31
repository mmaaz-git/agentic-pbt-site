from hypothesis import given, settings, Verbosity
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst
import numpy as np

@given(npst.arrays(dtype=npst.floating_dtypes(), shape=npst.array_shapes()))
@settings(max_examples=100, verbosity=Verbosity.verbose)
def test_numpy_array_equiv_reflexivity(arr):
    """Test that array_equiv satisfies reflexivity: array_equiv(x, x) should always be True"""
    # An array should always be equivalent to itself
    result = np.array_equiv(arr, arr)
    assert result, f"array_equiv reflexivity violated: array_equiv({arr!r}, {arr!r}) returned False"

if __name__ == "__main__":
    # Run the test and let it find the failing case
    test_numpy_array_equiv_reflexivity()