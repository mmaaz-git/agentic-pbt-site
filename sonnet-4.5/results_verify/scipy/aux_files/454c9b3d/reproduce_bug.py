import numpy as np
from scipy.spatial.distance import dice, jaccard

u = np.array([False, False, False, False, False])
v = np.array([False, False, False, False, False])

result = dice(u, v)
print(f"dice(all-False, all-False) = {result}")
print(f"Expected: 0.0")
print(f"For comparison, jaccard(all-False, all-False) = {jaccard(u, v)}")