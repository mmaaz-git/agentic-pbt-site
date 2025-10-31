import numpy as np
from hypothesis import given, strategies as st
from pandas.core.window.common import zsqrt


@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10))
def test_zsqrt_always_nonnegative(x):
    result = zsqrt(x)
    assert result >= 0

if __name__ == "__main__":
    # Run the test
    test_zsqrt_always_nonnegative()