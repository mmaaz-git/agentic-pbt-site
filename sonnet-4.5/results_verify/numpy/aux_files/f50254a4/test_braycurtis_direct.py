import numpy as np
from scipy.spatial.distance import braycurtis

print("Testing direct reproduction case...")
u = np.array([0.0, 0.0, 0.0])
v = np.array([0.0, 0.0, 0.0])
result = braycurtis(u, v)

print(f"u = {u}")
print(f"v = {v}")
print(f"braycurtis(u, v) = {result}")

print(f"\nIs result NaN? {np.isnan(result)}")
assert np.isnan(result), f"Expected nan, got {result}"
print("Assertion passed: result is indeed NaN")

# Also test with non-zero vectors for comparison
print("\n--- Testing with non-zero identical vectors ---")
u2 = np.array([1.0, 2.0, 3.0])
v2 = np.array([1.0, 2.0, 3.0])
result2 = braycurtis(u2, v2)
print(f"u = {u2}")
print(f"v = {v2}")
print(f"braycurtis(u, v) = {result2}")
print(f"Is result 0? {np.isclose(result2, 0.0)}")

# Test with different vectors
print("\n--- Testing with different vectors ---")
u3 = np.array([1.0, 2.0, 3.0])
v3 = np.array([2.0, 3.0, 4.0])
result3 = braycurtis(u3, v3)
print(f"u = {u3}")
print(f"v = {v3}")
print(f"braycurtis(u, v) = {result3}")