from hypothesis import given, strategies as st, settings
import numpy.ctypeslib
import numpy as np


@given(st.integers(min_value=-1000, max_value=-1))
@settings(max_examples=200)
def test_ndpointer_negative_ndim(ndim):
    """Test that ndpointer rejects negative ndim values."""
    try:
        ptr = numpy.ctypeslib.ndpointer(ndim=ndim)
        assert False, f"Should reject negative ndim {ndim}"
    except (TypeError, ValueError):
        pass

# Run the test
if __name__ == "__main__":
    test_ndpointer_negative_ndim()