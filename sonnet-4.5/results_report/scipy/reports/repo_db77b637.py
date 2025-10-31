import numpy as np
from scipy.spatial import distance

# Test case: both arrays contain only False values
u = np.array([False])
v = np.array([False])

print("Testing dice distance with all-False arrays:")
print(f"u = {u}")
print(f"v = {v}")
print()

result = distance.dice(u, v)
print(f"dice([False], [False]) = {result}")
print()

# Test with larger all-False arrays
u_large = np.array([False, False, False])
v_large = np.array([False, False, False])
result_large = distance.dice(u_large, v_large)
print(f"dice([False, False, False], [False, False, False]) = {result_large}")
print()

# For comparison with other distance functions
print("Comparison with other distance metrics:")
print(f"jaccard([False], [False]) = {distance.jaccard(u, v)}")
print(f"hamming([False], [False]) = {distance.hamming(u, v)}")
print()

# Demonstrate the NaN equality issue
d_uv = distance.dice(u, v)
d_vu = distance.dice(v, u)
print("Symmetry check:")
print(f"dice(u, v) = {d_uv}")
print(f"dice(v, u) = {d_vu}")
print(f"dice(u, v) == dice(v, u): {d_uv == d_vu}")
print(f"Both are NaN: {np.isnan(d_uv) and np.isnan(d_vu)}")