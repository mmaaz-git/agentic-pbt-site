import numpy as np
from scipy.spatial.distance import braycurtis

u = np.array([0.0, 0.0, 0.0])
v = np.array([0.0, 0.0, 0.0])
result = braycurtis(u, v)

print(f"u = {u}")
print(f"v = {v}")
print(f"braycurtis(u, v) = {result}")
print(f"Is result NaN? {np.isnan(result)}")

# Test the assertion from the bug report
try:
    assert np.isnan(result)
    print("Assertion passed: result is indeed NaN")
except AssertionError:
    print(f"Assertion failed: result is not NaN, it is {result}")

# Also test with non-zero vectors for comparison
u2 = np.array([1.0, 2.0, 3.0])
v2 = np.array([1.0, 2.0, 3.0])
result2 = braycurtis(u2, v2)
print(f"\nFor comparison, with non-zero identical vectors:")
print(f"u2 = {u2}")
print(f"v2 = {v2}")
print(f"braycurtis(u2, v2) = {result2}")