import numpy as np
import xarray as xr

# Create test data with negative degree coefficient
coord = xr.DataArray([2.0], dims=("x",))
coeffs = xr.DataArray(
    [100.0, 1.0, 2.0],
    dims=("degree",),
    coords={"degree": [-1, 0, 1]}
)

# Evaluate the polynomial
result = xr.polyval(coord, coeffs)

# Calculate expected result manually
# For x = 2.0:
# Degree -1: 100 * (2.0)^(-1) = 100 * 0.5 = 50.0
# Degree 0:  1 * (2.0)^0 = 1 * 1 = 1.0
# Degree 1:  2 * (2.0)^1 = 2 * 2 = 4.0
# Expected total: 50.0 + 1.0 + 4.0 = 55.0

print(f"Input coordinate: {coord.values[0]}")
print(f"Coefficients: {coeffs.values} with degrees {list(coeffs.degree.values)}")
print(f"Result from xr.polyval: {result.values[0]}")
print(f"Expected result: 100*(2^-1) + 1*(2^0) + 2*(2^1) = 50.0 + 1.0 + 4.0 = 55.0")
print(f"Actual result: {result.values[0]}")
print(f"\nError: The coefficient for degree -1 (value 100.0) was silently dropped!")
print(f"The function only computed: 1*(2^0) + 2*(2^1) = 1.0 + 4.0 = 5.0")