from scipy.spatial.distance import braycurtis
import numpy as np

# Test case 1: All-zero arrays
u = np.array([0.0, 0.0, 0.0])
v = np.array([0.0, 0.0, 0.0])

print("Testing braycurtis with all-zero arrays:")
print(f"u = {u}")
print(f"v = {v}")

result = braycurtis(u, v)
print(f"braycurtis([0, 0, 0], [0, 0, 0]) = {result}")

# Verify it should be 0 for identical vectors
print(f"\nAssertion check: result == 0.0")
try:
    assert result == 0.0, f"Expected 0.0, got {result}"
    print("Assertion passed!")
except AssertionError as e:
    print(f"Assertion failed: {e}")

# Additional test: Identical non-zero vectors for comparison
print("\n--- For comparison: identical non-zero vectors ---")
u2 = np.array([1.0, 2.0, 3.0])
v2 = np.array([1.0, 2.0, 3.0])
result2 = braycurtis(u2, v2)
print(f"braycurtis([1, 2, 3], [1, 2, 3]) = {result2}")