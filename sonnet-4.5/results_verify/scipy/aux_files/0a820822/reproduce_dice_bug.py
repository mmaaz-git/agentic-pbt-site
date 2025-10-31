import numpy as np
from scipy.spatial import distance

u = np.array([False])
v = np.array([False])

result = distance.dice(u, v)
print(f"dice([False], [False]) = {result}")

print(f"\nFor comparison:")
print(f"jaccard([False], [False]) = {distance.jaccard(u, v)}")

# Test symmetry
print(f"\nSymmetry test:")
print(f"dice(u, v) = {distance.dice(u, v)}")
print(f"dice(v, u) = {distance.dice(v, u)}")
print(f"Are they equal? {distance.dice(u, v) == distance.dice(v, u)}")

# Test with other all-false arrays
u2 = np.array([False, False, False])
v2 = np.array([False, False, False])
print(f"\ndice([False, False, False], [False, False, False]) = {distance.dice(u2, v2)}")