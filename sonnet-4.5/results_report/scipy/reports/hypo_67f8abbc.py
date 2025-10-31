import numpy as np
from hypothesis import given, strategies as st, settings

@given(
    r=st.integers(min_value=0, max_value=255),
    g=st.integers(min_value=0, max_value=255),
    b=st.integers(min_value=0, max_value=255)
)
@settings(max_examples=1000)
def test_grayscale_formula_overflow(r, g, b):
    """
    Grayscale value should be bounded by min and max of RGB values.
    Fails for RGB=(1,1,1) which produces gray=0 instead of gray=1.
    """
    expected_gray_float = 0.21 * r + 0.71 * g + 0.07 * b

    rgb_array = np.array([[[r, g, b]]], dtype='uint8')

    gray = (0.21 * rgb_array[:, :, 0] +
            0.71 * rgb_array[:, :, 1] +
            0.07 * rgb_array[:, :, 2]).astype('uint8')

    actual_gray = gray[0, 0]

    min_val = min(r, g, b)
    max_val = max(r, g, b)

    assert min_val <= actual_gray <= max_val, \
        f"Gray {actual_gray} outside [{min_val}, {max_val}] " + \
        f"for RGB=({r}, {g}, {b}), expected_float={expected_gray_float:.2f}"

# Run the test
if __name__ == "__main__":
    test_grayscale_formula_overflow()