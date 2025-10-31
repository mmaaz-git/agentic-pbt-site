from scipy.spatial.distance import braycurtis
import numpy as np

u = np.array([0.0, 0.0, 0.0])
v = np.array([0.0, 0.0, 0.0])

result = braycurtis(u, v)
print(f"braycurtis([0, 0, 0], [0, 0, 0]) = {result}")

try:
    assert result == 0.0, f"Expected 0.0, got {result}"
    print("Assertion passed: result is 0.0")
except AssertionError as e:
    print(f"Assertion failed: {e}")
    print(f"Result is NaN: {np.isnan(result)}")

# Test with other cases
u1 = np.array([1.0, 2.0, 3.0])
v1 = np.array([1.0, 2.0, 3.0])
result1 = braycurtis(u1, v1)
print(f"\nbraycurtis([1, 2, 3], [1, 2, 3]) = {result1}")

u2 = np.array([1.0, 0.0, 0.0])
v2 = np.array([0.0, 1.0, 0.0])
result2 = braycurtis(u2, v2)
print(f"braycurtis([1, 0, 0], [0, 1, 0]) = {result2}")