import numpy as np
from scipy.spatial import distance

x = np.array([False, False, False])
y = np.array([False, False, False])

print("Testing similar distance functions with all-False arrays:")
print(f"x = {x}")
print(f"y = {y}")
print()

# Test dice
dice_result = distance.dice(x, y)
print(f"dice(x, y) = {dice_result}")

# Test jaccard
jaccard_result = distance.jaccard(x, y)
print(f"jaccard(x, y) = {jaccard_result}")

# Test hamming
hamming_result = distance.hamming(x, y)
print(f"hamming(x, y) = {hamming_result}")

# Test rogerstanimoto
rogerstanimoto_result = distance.rogerstanimoto(x, y)
print(f"rogerstanimoto(x, y) = {rogerstanimoto_result}")

print("\nNote: All these functions compute distance/dissimilarity between identical arrays.")