from hypothesis import given, strategies as st, settings, example
import numpy as np
import scipy.interpolate as interp


@settings(max_examples=300)
@given(
    st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
             min_size=5, max_size=10, unique=True)
)
@example([0.0, 1.0, 2.0, 0.5, 5e-324])
def test_lagrange_interpolates_exactly(x_list):
    """
    Property: Lagrange polynomial should pass exactly through all data points
    """
    x = np.array(sorted(x_list))
    y = np.sin(x)

    poly = interp.lagrange(x, y)

    for xi, yi in zip(x, y):
        result = poly(xi)
        assert np.isclose(result, yi, rtol=1e-10, atol=1e-10), \
            f"Lagrange poly at {xi} = {result}, expected {yi}"

# Run the test
if __name__ == "__main__":
    test_lagrange_interpolates_exactly()