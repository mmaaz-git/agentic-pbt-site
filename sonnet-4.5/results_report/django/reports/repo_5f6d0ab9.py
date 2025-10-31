from scipy.spatial.distance import correlation
import numpy as np

u = np.array([5.0, 5.0, 5.0])
v = np.array([5.0, 5.0, 5.0])

result = correlation(u, v)

print(f"correlation([5, 5, 5], [5, 5, 5]) = {result}")
assert result == 0.0, f"Expected 0.0 for identical arrays, got {result}"