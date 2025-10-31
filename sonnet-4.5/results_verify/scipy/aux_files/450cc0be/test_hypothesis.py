from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis import strategies as st
import numpy as np
import scipy.ndimage as ndi

@given(
    binary_image=arrays(
        dtype=bool,
        shape=st.tuples(
            st.integers(min_value=5, max_value=20),
            st.integers(min_value=5, max_value=20)
        )
    )
)
@settings(max_examples=200)
def test_binary_closing_superset(binary_image):
    """
    Closing adds points: X ⊆ closing(X)
    Closing can only add points, never remove them.
    """
    closed = ndi.binary_closing(binary_image)
    assert np.all(binary_image <= closed), \
        "Input should be a subset of binary closing result"

# Run the test
if __name__ == "__main__":
    test_binary_closing_superset()