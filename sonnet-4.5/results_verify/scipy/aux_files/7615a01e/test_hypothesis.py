import numpy as np
from hypothesis import given, strategies as st, settings, example
from hypothesis.extra import numpy as npst
import scipy.ndimage as ndimage


@given(
    input_array=npst.arrays(
        dtype=np.float64,
        shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=5, max_side=10),
        elements=st.floats(
            min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    ),
    size=st.integers(min_value=3, max_value=5),
)
@settings(max_examples=50, deadline=None)
@example(input_array=np.array([[1., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0.]]), size=4)
def test_grey_dilation_is_maximum_filter(input_array, size):
    grey_dil_result = ndimage.grey_dilation(input_array, size=size)
    max_filter_result = ndimage.maximum_filter(input_array, size=size)

    assert np.allclose(
        grey_dil_result, max_filter_result, rtol=1e-10, atol=1e-10
    ), f"grey_dilation with flat structuring element should equal maximum_filter\nSize: {size}\nDifference:\n{grey_dil_result - max_filter_result}"


if __name__ == "__main__":
    test_grey_dilation_is_maximum_filter()
    print("All tests passed!")