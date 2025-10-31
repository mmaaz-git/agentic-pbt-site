import numpy as np
from scipy.cluster.vq import whiten

# Test the specific failing case manually
n_obs = 7
n_features = 1

rng = np.random.default_rng(42)
obs = rng.standard_normal((n_obs, n_features))

zero_col = rng.integers(0, n_features)
constant_value = rng.uniform(-10, 10)
obs[:, zero_col] = constant_value

print(f"Setting column {zero_col} to constant value: {constant_value}")
print(f"Obs array shape: {obs.shape}")
print(f"All values in column {zero_col}: {obs[:, zero_col]}")
print(f"Are all values identical? {np.all(obs[:, zero_col] == constant_value)}")
print(f"Std of column: {np.std(obs, axis=0)[zero_col]}")
print(f"Std == 0? {np.std(obs, axis=0)[zero_col] == 0}")

whitened = whiten(obs)
print(f"\nAfter whitening:")
print(f"Whitened column values: {whitened[:, zero_col]}")
print(f"Expected value: {constant_value}")
print(f"Are they close? {np.allclose(whitened[:, zero_col], constant_value)}")
print(f"Actual difference: {whitened[0, zero_col] - constant_value}")