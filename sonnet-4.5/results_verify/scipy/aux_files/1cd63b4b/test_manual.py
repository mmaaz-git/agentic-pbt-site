from scipy.spatial.distance import dice
import numpy as np

u = np.array([False, False, False])
v = np.array([False, False, False])

result = dice(u, v)
print(f"dice([False, False, False], [False, False, False]) = {result}")

try:
    assert result == 0.0, f"Expected 0.0, got {result}"
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed: {e}")

# Test with identical vectors that have some True values
u2 = np.array([True, False, True])
v2 = np.array([True, False, True])
result2 = dice(u2, v2)
print(f"\ndice([True, False, True], [True, False, True]) = {result2}")
print(f"Expected 0.0 for identical vectors")

# Test with different vectors
u3 = np.array([True, False, False])
v3 = np.array([False, True, False])
result3 = dice(u3, v3)
print(f"\ndice([True, False, False], [False, True, False]) = {result3}")