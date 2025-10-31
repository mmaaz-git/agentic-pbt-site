import numpy as np
import xarray as xr
from hypothesis import given, strategies as st
import warnings

@given(
    data=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=1,
        max_size=5
    ),
    ddof=st.integers(min_value=1, max_value=10)
)
def test_cov_ddof_handling(data, ddof):
    """xarray.cov should handle ddof >= valid_count gracefully"""
    if len(data) <= ddof:
        da_a = xr.DataArray(data, dims=["x"])
        da_b = xr.DataArray(data, dims=["x"])

        result = xr.cov(da_a, da_b, ddof=ddof)
        result_val = float(result.values)

        assert not np.isinf(result_val), f"xarray.cov returned inf for data={data}, ddof={ddof}"
        assert not (result_val < 0 and ddof >= len(data)), f"xarray.cov returned negative value for insufficient data"

# Run the test and capture failures
print("Running Hypothesis test...")
try:
    test_cov_ddof_handling()
    print("All tests passed!")
except AssertionError as e:
    print(f"Test failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

# Also test the specific failing cases mentioned
print("\nTesting specific failing inputs:")
test_cases = [
    ([1.0], 1),
    ([1.0, 2.0], 2),
    ([1.0, 2.0, 3.0], 5)
]

for data, ddof in test_cases:
    da = xr.DataArray(data, dims=["x"])
    result = xr.cov(da, da, ddof=ddof)
    result_val = float(result.values)
    print(f"data={data}, ddof={ddof} → xarray.cov returns {result_val}")