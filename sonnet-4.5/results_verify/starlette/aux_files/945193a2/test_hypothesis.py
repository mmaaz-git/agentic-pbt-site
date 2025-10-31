from hypothesis import given, strategies as st, assume, settings
import numpy as np
import numpy.lib.scimath as scimath

@given(
    st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=500)
def test_scimath_power_general(x, p):
    assume(abs(x) > 1e-100)
    assume(abs(p) > 1e-10 and abs(p) < 100)

    result = scimath.power(x, p)

    assert not np.isnan(result).any() if hasattr(result, '__iter__') else not np.isnan(result), \
        f"power({x}, {p}) should not be NaN"

if __name__ == "__main__":
    print("Running Hypothesis test...")
    try:
        test_scimath_power_general()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")