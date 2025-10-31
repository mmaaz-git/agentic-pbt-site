from hypothesis import given, settings, strategies as st, assume
import numpy.random as nr


@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
       st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10))
@settings(max_examples=1000)
def test_uniform_bounds(low, high):
    assume(low < high)
    result = nr.uniform(low, high)
    assert low <= result < high, f"uniform({low}, {high}) = {result} not in range"

if __name__ == "__main__":
    test_uniform_bounds()