from hypothesis import given, settings, strategies as st, assume
import hypothesis.extra.numpy as npst
import numpy as np
import scipy.ndimage

@given(
    npst.arrays(
        dtype=np.float64,
        shape=npst.array_shapes(min_dims=1, max_dims=2, min_side=5, max_side=15),
        elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
    ),
    st.integers(min_value=1, max_value=4)
)
@settings(max_examples=100)
def test_shift_wrap_invertible(arr, shift_amount):
    shifted = scipy.ndimage.shift(arr, shift_amount, order=0, mode='wrap')
    shifted_back = scipy.ndimage.shift(shifted, -shift_amount, order=0, mode='wrap')
    assert np.array_equal(arr, shifted_back), \
        f"Shift with mode='wrap' should be invertible. Array shape: {arr.shape}, shift: {shift_amount}"

if __name__ == "__main__":
    print("Running hypothesis test for scipy.ndimage.shift with mode='wrap'")
    print("=" * 60)

    failures = []

    try:
        test_shift_wrap_invertible()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")

        # Run with specific failing example
        print("\nRunning specific failing example from bug report:")
        arr = np.array([0., 1., 2., 3., 4.])
        shift_amount = 2
        print(f"Array: {arr}")
        print(f"Shift amount: {shift_amount}")

        shifted = scipy.ndimage.shift(arr, shift_amount, order=0, mode='wrap')
        print(f"After shift by {shift_amount}: {shifted}")

        shifted_back = scipy.ndimage.shift(shifted, -shift_amount, order=0, mode='wrap')
        print(f"After shift back by {-shift_amount}: {shifted_back}")
        print(f"Expected: {arr}")
        print(f"Invertible: {np.array_equal(arr, shifted_back)}")

    # Test with grid-wrap mode as comparison
    print("\n" + "=" * 60)
    print("Testing same property with mode='grid-wrap'")

    @given(
        npst.arrays(
            dtype=np.float64,
            shape=npst.array_shapes(min_dims=1, max_dims=2, min_side=5, max_side=15),
            elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
        ),
        st.integers(min_value=1, max_value=4)
    )
    @settings(max_examples=100)
    def test_shift_grid_wrap_invertible(arr, shift_amount):
        shifted = scipy.ndimage.shift(arr, shift_amount, order=0, mode='grid-wrap')
        shifted_back = scipy.ndimage.shift(shifted, -shift_amount, order=0, mode='grid-wrap')
        assert np.array_equal(arr, shifted_back), \
            f"Shift with mode='grid-wrap' should be invertible. Array shape: {arr.shape}, shift: {shift_amount}"

    try:
        test_shift_grid_wrap_invertible()
        print("All grid-wrap tests passed!")
    except AssertionError as e:
        print(f"Grid-wrap test failed: {e}")