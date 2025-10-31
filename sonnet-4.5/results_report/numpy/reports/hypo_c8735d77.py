from hypothesis import given, strategies as st, settings
import numpy.ctypeslib
import numpy as np


@given(st.tuples(st.integers(min_value=-10, max_value=-1), st.integers(min_value=1, max_value=10)))
@settings(max_examples=200)
def test_ndpointer_negative_shape(shape):
    try:
        ptr = numpy.ctypeslib.ndpointer(shape=shape)
        assert False, f"Should reject shape with negative dimensions {shape}"
    except (TypeError, ValueError):
        pass

if __name__ == "__main__":
    test_ndpointer_negative_shape()