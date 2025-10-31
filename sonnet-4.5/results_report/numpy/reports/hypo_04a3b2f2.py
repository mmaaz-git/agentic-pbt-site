from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
import numpy as np
from numpy import matrix


@given(arrays(np.float64, shape=st.tuples(st.integers(3, 20), st.integers(3, 20)),
              elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)))
@settings(max_examples=300)
def test_ptp_with_out_parameter(arr):
    m = matrix(arr)
    out = matrix(np.zeros((1, 1)))
    result = m.ptp(axis=None, out=out)
    assert result is out

if __name__ == "__main__":
    test_ptp_with_out_parameter()