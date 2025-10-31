from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
import xarray as xr

@given(
    coord_data=arrays(
        dtype=np.float64,
        shape=st.integers(1, 10),
        elements=st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False),
    ),
    min_degree=st.integers(-5, -1),
    max_degree=st.integers(0, 3),
)
def test_polyval_preserves_all_coefficients(coord_data, min_degree, max_degree):
    degrees = list(range(min_degree, max_degree + 1))
    coeffs_data = np.random.uniform(-10, 10, len(degrees))

    coord = xr.DataArray(coord_data, dims=("x",))
    coeffs = xr.DataArray(coeffs_data, dims=("degree",), coords={"degree": degrees})

    result = xr.polyval(coord, coeffs)

    expected = sum(c * coord_data**d for c, d in zip(coeffs_data, degrees))

    assert np.allclose(result.values, expected, rtol=1e-10)

if __name__ == "__main__":
    # Run the test to find a failing example
    test_polyval_preserves_all_coefficients()