import numpy as np
import xarray as xr

print("Testing xarray.polyval with negative degree coefficients")
print("=" * 60)

coord = xr.DataArray([2.0], dims=("x",))
coeffs = xr.DataArray(
    [100.0, 1.0, 2.0],
    dims=("degree",),
    coords={"degree": [-1, 0, 1]}
)

print(f"Coordinate: {coord.values}")
print(f"Coefficients: {coeffs.values}")
print(f"Degrees: {coeffs.coords['degree'].values}")
print()

result = xr.polyval(coord, coeffs)

print(f"Result: {result.values[0]}")
print(f"Expected: 100/2 + 1 + 2*2 = 55")
print(f"Actual: {result.values[0]}")
print()

# Let's manually compute what we expect
manual_result = 100.0 * (2.0 ** -1) + 1.0 * (2.0 ** 0) + 2.0 * (2.0 ** 1)
print(f"Manual computation: 100 * (2^-1) + 1 * (2^0) + 2 * (2^1)")
print(f"                  = 100 * 0.5 + 1 * 1 + 2 * 2")
print(f"                  = 50 + 1 + 4 = {manual_result}")