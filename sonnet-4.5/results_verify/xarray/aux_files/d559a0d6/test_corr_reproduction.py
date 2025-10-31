"""Test reproduction for xarray.corr() exceeding valid range [-1, 1]"""

import numpy as np
import xarray as xr
from hypothesis import given, settings, assume, strategies as st
from hypothesis.extra.numpy import arrays

# First, let's run the Hypothesis test
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

    assert -1.0 <= corr_val <= 1.0, f"Correlation {corr_val} exceeds valid range [-1, 1]"


# Now let's reproduce with the specific failing input
def test_specific_failing_case():
    print("\n=== Testing specific failing case ===")

    data = np.array([[1., 1.],
                      [0., 0.],
                      [0., 0.]])

    da_a = xr.DataArray(data[:, 0], dims=["x"])
    da_b = xr.DataArray(data[:, 1], dims=["x"])

    print(f"Data array A: {da_a.values}")
    print(f"Data array B: {da_b.values}")
    print(f"Std A: {da_a.std().values}")
    print(f"Std B: {da_b.std().values}")

    correlation = xr.corr(da_a, da_b)
    corr_val = correlation.values.item()

    print(f"Correlation value: {corr_val:.17f}")
    print(f"Exceeds 1.0: {corr_val > 1.0}")
    print(f"Difference from 1.0: {corr_val - 1.0:.2e}")

    # Compare with NumPy
    np_corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
    print(f"\nNumPy correlation: {np_corr:.17f}")
    print(f"NumPy exceeds 1.0: {np_corr > 1.0}")

    # Check if xarray violates the bounds
    if corr_val > 1.0 or corr_val < -1.0:
        print(f"\n❌ BUG CONFIRMED: xarray correlation {corr_val} is outside [-1, 1]")
        return False
    else:
        print(f"\n✅ No issue: correlation {corr_val} is within [-1, 1]")
        return True


if __name__ == "__main__":
    # Run the specific test case
    test_specific_failing_case()

    # Try to run hypothesis test
    print("\n=== Running Hypothesis test ===")
    try:
        test_corr_bounds()
        print("Hypothesis test completed without finding violations")
    except AssertionError as e:
        print(f"Hypothesis test found a violation: {e}")
    except Exception as e:
        print(f"Error running Hypothesis test: {e}")