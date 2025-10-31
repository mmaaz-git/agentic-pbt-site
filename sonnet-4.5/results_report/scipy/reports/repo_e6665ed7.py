import numpy as np
from scipy.spatial import distance

# Test case: all-False boolean arrays
x = np.array([False, False, False])
y = np.array([False, False, False])

# Compute dice dissimilarity
result = distance.dice(x, y)

print(f"dice({x}, {y}) = {result}")
print(f"Expected: 0.0 (identical arrays should have distance 0)")
print(f"Actual: {result}")
print()

# Comparison with other similar functions
print("Comparison with other boolean distance metrics:")
print(f"jaccard({x}, {y}) = {distance.jaccard(x, y)}")
print(f"hamming({x}, {y}) = {distance.hamming(x, y)}")
print(f"rogerstanimoto({x}, {y}) = {distance.rogerstanimoto(x, y)}")