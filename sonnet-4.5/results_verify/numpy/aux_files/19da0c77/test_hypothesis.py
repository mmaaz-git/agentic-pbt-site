import numpy as np
import numpy.lib.scimath as scimath
from hypothesis import given, strategies as st, settings

@given(
    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    st.integers(min_value=-5, max_value=5)
)
@settings(max_examples=1000)
def test_power_definition(x, n):
    result = scimath.power(x, n)
    assert not np.isnan(result)

if __name__ == "__main__":
    # Test with the specific failing input
    x = -9.499558537778752e-188
    n = -2
    result = scimath.power(x, n)
    print(f"Testing x={x}, n={n}")
    print(f"scimath.power({x}, {n}) = {result}")
    print(f"Has NaN: {np.isnan(result)}")
    print(f"Result type: {type(result)}")

    # Also test the simpler case from the bug report
    x2 = -1e-200
    n2 = -2
    result2 = scimath.power(x2, n2)
    print(f"\nTesting x={x2}, n={n2}")
    print(f"scimath.power({x2}, {n2}) = {result2}")
    print(f"Has NaN: {np.isnan(result2)}")
    print(f"Result type: {type(result2)}")