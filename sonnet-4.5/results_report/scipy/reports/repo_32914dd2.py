import numpy as np
from scipy.spatial.distance import braycurtis

# Test case: all-zero vectors
u = np.array([0.0, 0.0, 0.0])
v = np.array([0.0, 0.0, 0.0])

print(f"Input vectors:")
print(f"u = {u}")
print(f"v = {v}")
print()

# Compute braycurtis distance
result = braycurtis(u, v)

print(f"braycurtis(u, v) = {result}")
print(f"Is result NaN? {np.isnan(result)}")
print()

# This should be 0 for identical vectors (identity property)
# But returns NaN due to division by zero
print("Expected: 0.0 (identity property: distance between identical vectors should be 0)")
print(f"Actual: {result}")