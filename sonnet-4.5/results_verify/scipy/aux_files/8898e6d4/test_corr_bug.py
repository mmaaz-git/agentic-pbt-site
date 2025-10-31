"""Test case to reproduce the xarray.corr bounds issue"""
import numpy as np
import xarray as xr

# The failing input from the bug report
data = np.array([[1.9375, 1.    ],
                 [0.    , 0.    ],
                 [0.    , 0.    ],
                 [0.    , 0.    ],
                 [0.    , 0.    ]])

da_a = xr.DataArray(data[:, 0], dims=["x"])
da_b = xr.DataArray(data[:, 1], dims=["x"])

# Compute correlation with xarray
corr = xr.corr(da_a, da_b)

print(f"Input data:")
print(f"Column A: {data[:, 0]}")
print(f"Column B: {data[:, 1]}")
print()
print(f"xarray correlation: {corr.values.item()}")
print(f"xarray correlation exceeds 1.0: {corr.values.item() > 1.0}")
print(f"Exact value - 1.0 = {corr.values.item() - 1.0}")
print()

# Compare with numpy's corrcoef
numpy_corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
print(f"NumPy correlation: {numpy_corr}")
print(f"NumPy correlation exceeds 1.0: {numpy_corr > 1.0}")
print(f"NumPy within bounds [-1, 1]: {-1.0 <= numpy_corr <= 1.0}")
print()

# Manual calculation to verify
mean_a = np.mean(data[:, 0])
mean_b = np.mean(data[:, 1])
std_a = np.std(data[:, 0], ddof=0)
std_b = np.std(data[:, 1], ddof=0)
cov_ab = np.mean((data[:, 0] - mean_a) * (data[:, 1] - mean_b))
manual_corr = cov_ab / (std_a * std_b)

print(f"Manual calculation:")
print(f"Mean A: {mean_a}, Mean B: {mean_b}")
print(f"Std A: {std_a}, Std B: {std_b}")
print(f"Covariance: {cov_ab}")
print(f"Manual correlation: {manual_corr}")
print(f"Manual correlation exceeds 1.0: {manual_corr > 1.0}")