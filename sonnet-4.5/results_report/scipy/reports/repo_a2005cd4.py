import numpy as np
import xarray as xr

# Reproduce the exact bug case
data_a = np.array([[65., 65.,  0.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
data_b = np.array([[65., 65.,  0.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

da_a = xr.DataArray(data_a, dims=["x", "y"])
da_b = xr.DataArray(data_b, dims=["x", "y"])

# Check that the arrays have nonzero standard deviation
print(f"da_a standard deviation: {da_a.std().item()}")
print(f"da_b standard deviation: {da_b.std().item()}")

# Compute correlation
result = xr.corr(da_a, da_b)

# Display results
print(f"\nCorrelation result: {result.values}")
print(f"Result type: {type(result.values)}")
print(f"Result > 1.0: {result.values > 1.0}")
print(f"Result == 1.0: {result.values == 1.0}")
print(f"Difference from 1.0: {result.values - 1.0}")

# Check if it's in the valid range [-1, 1]
is_valid = -1.0 <= result.values <= 1.0
print(f"\nIs correlation in valid range [-1, 1]? {is_valid}")

# Compare with numpy's corrcoef
numpy_result = np.corrcoef(data_a.flatten(), data_b.flatten())[0, 1]
print(f"\nNumPy's corrcoef result: {numpy_result}")
print(f"NumPy result == 1.0: {numpy_result == 1.0}")
print(f"NumPy result > 1.0: {numpy_result > 1.0}")