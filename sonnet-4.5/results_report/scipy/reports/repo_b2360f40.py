import numpy as np
from scipy.spatial.distance import braycurtis

# Test case 1: All-zero vectors (the failing case)
u = np.array([0.0, 0.0, 0.0])
v = np.array([0.0, 0.0, 0.0])
result = braycurtis(u, v)
print(f"braycurtis([0, 0, 0], [0, 0, 0]) = {result}")
print(f"Is result nan? {np.isnan(result)}")
print()

# Test case 2: Identical non-zero vectors (should return 0)
u2 = np.array([1.0, 2.0, 3.0])
v2 = np.array([1.0, 2.0, 3.0])
result2 = braycurtis(u2, v2)
print(f"braycurtis([1, 2, 3], [1, 2, 3]) = {result2}")
print(f"Is result2 close to 0? {np.isclose(result2, 0.0)}")
print()

# Test case 3: Different vectors (should return a value between 0 and 1)
u3 = np.array([1.0, 0.0, 0.0])
v3 = np.array([0.0, 1.0, 0.0])
result3 = braycurtis(u3, v3)
print(f"braycurtis([1, 0, 0], [0, 1, 0]) = {result3}")