from hypothesis import given
import hypothesis.extra.numpy as npst
import numpy as np

@given(npst.arrays(dtype=npst.floating_dtypes(), shape=npst.array_shapes()))
def test_array_equiv_reflexivity(arr):
    result = np.array_equiv(arr, arr)
    print(f"Testing array with shape {arr.shape} and dtype {arr.dtype}")
    print(f"Contains NaN: {np.any(np.isnan(arr))}")
    print(f"np.array_equiv(arr, arr) = {result}")
    assert result, f"Reflexivity violated: np.array_equiv(arr, arr) returned False for array: {arr}"

if __name__ == "__main__":
    # Run the test
    test_array_equiv_reflexivity()