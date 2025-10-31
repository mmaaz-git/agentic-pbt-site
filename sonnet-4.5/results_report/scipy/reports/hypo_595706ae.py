import numpy as np
from hypothesis import given, strategies as st, settings
from scipy import ndimage
from hypothesis.extra.numpy import arrays

@given(
    arr=arrays(dtype=np.float64, shape=st.tuples(
        st.integers(2, 10),
        st.integers(2, 10)
    ), elements=st.floats(
        min_value=-1e6, max_value=1e6,
        allow_nan=False, allow_infinity=False
    ))
)
@settings(max_examples=200)
def test_rotate_90_four_times_identity(arr):
    """
    Property: Rotating by 90 degrees 4 times should return the original array

    Evidence: Four 90-degree rotations equal a 360-degree rotation, which
    should be the identity transformation.
    """
    result = arr
    for _ in range(4):
        result = ndimage.rotate(result, 90, reshape=False)

    assert np.allclose(arr, result)

if __name__ == "__main__":
    # Run the test
    test_rotate_90_four_times_identity()