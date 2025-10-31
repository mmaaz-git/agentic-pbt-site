from hypothesis import given, settings, assume, strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
import xarray as xr

@given(
    data=arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(2, 10), st.integers(2, 10)),
        elements=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
    )
)
@settings(max_examples=200)
def test_corr_bounds(data):
    da_a = xr.DataArray(data[:, 0], dims=["x"])
    da_b = xr.DataArray(data[:, 1], dims=["x"])

    assume(da_a.std().values > 1e-10)
    assume(da_b.std().values > 1e-10)

    correlation = xr.corr(da_a, da_b)
    corr_val = correlation.values.item() if correlation.values.ndim == 0 else correlation.values

    assert -1.0 <= corr_val <= 1.0, f"Correlation {corr_val} is outside valid range [-1, 1]"

# Run the test
test_corr_bounds()