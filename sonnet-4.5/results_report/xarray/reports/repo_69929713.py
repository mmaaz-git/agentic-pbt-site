import numpy as np
import xarray as xr

# Create the specific failing input from the bug report
data = np.array([[1., 1.],
                  [0., 0.],
                  [0., 0.]])

# Create DataArrays from the columns
da_a = xr.DataArray(data[:, 0], dims=["x"])
da_b = xr.DataArray(data[:, 1], dims=["x"])

# Print input data for clarity
print("Input data:")
print(f"da_a values: {da_a.values}")
print(f"da_b values: {da_b.values}")
print()

# Compute correlation using xarray
correlation = xr.corr(da_a, da_b)
corr_val = correlation.values.item()

# Display results
print("xarray correlation results:")
print(f"Correlation value: {corr_val:.17f}")
print(f"Exceeds 1.0: {corr_val > 1.0}")
print(f"Amount over 1.0: {corr_val - 1.0:.2e}")
print()

# Compare with NumPy's corrcoef
np_corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
print("NumPy correlation results:")
print(f"Correlation value: {np_corr:.17f}")
print(f"Exceeds 1.0: {np_corr > 1.0}")