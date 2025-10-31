import numpy as np
from scipy.spatial.distance import dice, jaccard

# Create two all-False boolean vectors
u = np.array([False, False, False, False, False])
v = np.array([False, False, False, False, False])

print("Testing dice function with all-False vectors:")
print(f"u = {u}")
print(f"v = {v}")
print()

# Test dice function
result = dice(u, v)
print(f"dice(all-False, all-False) = {result}")
print(f"Expected: 0.0 (since d(x,x) should be 0 for identical vectors)")
print()

# Compare with jaccard for reference
jaccard_result = jaccard(u, v)
print(f"For comparison, jaccard(all-False, all-False) = {jaccard_result}")
print("Note: jaccard was fixed in scipy 1.2.0 to return 0.0 for this case")