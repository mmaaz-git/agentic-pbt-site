import numpy as np
from scipy.cluster.vq import whiten

# Test with the specific failing input from the bug report
obs = np.array([[93206.82233024, 93206.82233024]] * 40)

print("=== Bug Reproduction Test ===")
print(f"Input shape: {obs.shape}")
print(f"Input: all values = {obs[0, 0]}")
print(f"All elements equal? {np.all(obs == obs[0, 0])}")
print()

# Check standard deviation
std_vals = np.std(obs, axis=0)
print(f"Std computed by numpy: {std_vals}")
print(f"Std == 0 (exact equality): {std_vals == 0}")
print(f"Std[0] exact value: {std_vals[0]:.20e}")
print()

# Apply whiten
result = whiten(obs)

print(f"Output after whiten: {result[0, 0]}")
print(f"Expected (from bug report): {obs[0, 0]} (unchanged)")
print(f"Actual change: {obs[0, 0]} → {result[0, 0]}")
print(f"Multiplication factor: {result[0, 0] / obs[0, 0]}")
print()

# Additional verification
print("=== Additional Checks ===")
print(f"Is output reasonable? {result[0, 0] < 1e10}")
print(f"Standard deviation of output: {np.std(result, axis=0)}")