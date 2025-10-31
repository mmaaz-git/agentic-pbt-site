import math
from hypothesis import given, strategies as st, settings
from scipy.special import boxcox1p, inv_boxcox1p


@settings(max_examples=1000)
@given(
    y=st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False),
    lmbda=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
)
def test_boxcox1p_round_trip_boxcox_first(y, lmbda):
    x = inv_boxcox1p(y, lmbda)
    if not math.isfinite(x) or x <= -1:
        return
    y_recovered = boxcox1p(x, lmbda)
    assert math.isfinite(y_recovered), f"boxcox1p returned non-finite value: {y_recovered}"
    assert math.isclose(y_recovered, y, rel_tol=1e-9, abs_tol=1e-9), \
        f"Round-trip failed: y={y}, lmbda={lmbda}, x={x}, y_recovered={y_recovered}"

if __name__ == "__main__":
    test_boxcox1p_round_trip_boxcox_first()
    print("Test completed")