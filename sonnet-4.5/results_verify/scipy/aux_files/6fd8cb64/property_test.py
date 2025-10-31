import numpy as np
from hypothesis import given, settings, strategies as st, assume
from hypothesis.extra import numpy as npst
import scipy.fftpack as fftpack


@given(
    st.data(),
    npst.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=100)
)
@settings(max_examples=200)
def test_hilbert_ihilbert_round_trip(data, shape):
    dtype = data.draw(st.sampled_from([np.float32, np.float64]))
    max_val = 1e6 if dtype == np.float32 else 1e10

    x = data.draw(npst.arrays(
        dtype=dtype,
        shape=shape,
        elements=st.floats(min_value=-max_val, max_value=max_val, allow_nan=False, allow_infinity=False)
    ))

    # Make sure sum is zero as required by documentation
    if np.sum(x) != 0:
        x = x - np.mean(x)

    # Apply the transformations
    y = fftpack.hilbert(x)
    result = fftpack.ihilbert(y)

    # Check if round-trip works
    atol, rtol = (1e-6, 1e-6) if dtype == np.float32 else (1e-10, 1e-10)

    # Only test for even-length arrays to confirm the bug
    if len(x) % 2 == 0:
        is_close = np.allclose(result, x, atol=atol, rtol=rtol)
        if not is_close:
            print(f"Failed for even-length array (n={len(x)})")
            print(f"  Input x: {x[:5]}..." if len(x) > 5 else f"  Input x: {x}")
            print(f"  Output: {result[:5]}..." if len(result) > 5 else f"  Output: {result}")
            print(f"  Max diff: {np.max(np.abs(result - x))}")
            return  # Expected failure for even-length arrays
    else:
        # For odd-length arrays, the round-trip should work
        assert np.allclose(result, x, atol=atol, rtol=rtol), \
            f"Round-trip failed for odd-length array (n={len(x)})"


# Run the test
if __name__ == "__main__":
    test_hilbert_ihilbert_round_trip()
    print("Property-based test completed.")

    # Explicitly test some edge cases
    print("\nExplicit test cases:")

    # Test even-length array with sum=0
    x_even = np.array([1.0, -1.0])
    y_even = fftpack.hilbert(x_even)
    result_even = fftpack.ihilbert(y_even)
    print(f"Even-length (n=2), sum={np.sum(x_even)}:")
    print(f"  Input: {x_even}, Output: {result_even}")
    print(f"  Round-trip works: {np.allclose(result_even, x_even)}")

    # Test another even-length
    x_even4 = np.array([1.0, -0.5, 0.5, -1.0])
    y_even4 = fftpack.hilbert(x_even4)
    result_even4 = fftpack.ihilbert(y_even4)
    print(f"Even-length (n=4), sum={np.sum(x_even4)}:")
    print(f"  Input: {x_even4}, Output: {result_even4}")
    print(f"  Round-trip works: {np.allclose(result_even4, x_even4)}")