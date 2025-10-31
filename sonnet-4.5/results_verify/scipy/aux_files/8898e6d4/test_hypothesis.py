"""Hypothesis-based property test for xarray.corr"""
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays
import numpy as np
import xarray as xr

@given(
    data=arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(2, 10), st.integers(2, 10)),
        elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=500)
def test_corr_range(data):
    assume(np.std(data[:, 0]) > 1e-10)
    assume(np.std(data[:, 1]) > 1e-10)

    da_a = xr.DataArray(data[:, 0], dims=["x"])
    da_b = xr.DataArray(data[:, 1], dims=["x"])

    corr = xr.corr(da_a, da_b)

    # Check if correlation is within [-1, 1]
    if not (-1.0 <= corr.values <= 1.0):
        print(f"\nFound violation!")
        print(f"Data shape: {data.shape}")
        print(f"Data:\n{data}")
        print(f"Correlation: {corr.values.item()}")
        print(f"Exceeds bounds: {corr.values.item() > 1.0 or corr.values.item() < -1.0}")
        return False
    return True

# Run the test
print("Running hypothesis test...")
try:
    test_corr_range()
    print("Test completed. If no violations were printed, all tests passed.")
except Exception as e:
    print(f"Test failed with exception: {e}")