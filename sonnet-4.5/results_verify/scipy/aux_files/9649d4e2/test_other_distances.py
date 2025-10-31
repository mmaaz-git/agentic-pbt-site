import numpy as np
from scipy.spatial.distance import braycurtis, jaccard, cosine, euclidean, cityblock
import warnings

# Test with all-zero vectors
u = np.array([0.0, 0.0, 0.0])
v = np.array([0.0, 0.0, 0.0])

print("Testing distance functions with all-zero vectors u = v = [0, 0, 0]:")
print("-" * 60)

# Test braycurtis
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = braycurtis(u, v)
    print(f"braycurtis(u, v) = {result}")

# Test jaccard (mentioned in the bug report as being fixed)
try:
    result = jaccard(u, v)
    print(f"jaccard(u, v) = {result}")
except Exception as e:
    print(f"jaccard(u, v) raised: {e}")

# Test cosine (also involves division)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = cosine(u, v)
    print(f"cosine(u, v) = {result}")

# Test euclidean (should work)
result = euclidean(u, v)
print(f"euclidean(u, v) = {result}")

# Test cityblock (manhattan, should work)
result = cityblock(u, v)
print(f"cityblock(u, v) = {result}")