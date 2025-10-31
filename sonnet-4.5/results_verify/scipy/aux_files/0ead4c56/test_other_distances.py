#!/usr/bin/env python3
import numpy as np
from scipy.spatial import distance

# Test all-False vectors with various distance metrics
u = np.array([False, False])
v = np.array([False, False])

print("Testing various distance functions with all-False identical vectors:")
print("Input: u = [False, False], v = [False, False]")
print("-" * 50)

# Test Jaccard (similar to Dice)
try:
    result = distance.jaccard(u, v)
    print(f"Jaccard distance: {result}")
except Exception as e:
    print(f"Jaccard distance: Error - {e}")

# Test Rogers-Tanimoto
try:
    result = distance.rogerstanimoto(u, v)
    print(f"Rogers-Tanimoto distance: {result}")
except Exception as e:
    print(f"Rogers-Tanimoto distance: Error - {e}")

# Test Russell-Rao
try:
    result = distance.russellrao(u, v)
    print(f"Russell-Rao distance: {result}")
except Exception as e:
    print(f"Russell-Rao distance: Error - {e}")

# Test Sokal-Michener
try:
    result = distance.sokalmichener(u, v)
    print(f"Sokal-Michener distance: {result}")
except Exception as e:
    print(f"Sokal-Michener distance: Error - {e}")

# Test Sokal-Sneath
try:
    result = distance.sokalsneath(u, v)
    print(f"Sokal-Sneath distance: {result}")
except Exception as e:
    print(f"Sokal-Sneath distance: Error - {e}")

# Test Yule
try:
    result = distance.yule(u, v)
    print(f"Yule distance: {result}")
except Exception as e:
    print(f"Yule distance: Error - {e}")

# Test Hamming
try:
    result = distance.hamming(u, v)
    print(f"Hamming distance: {result}")
except Exception as e:
    print(f"Hamming distance: Error - {e}")

print("\n" + "=" * 50)
print("\nNow testing d(x,x) property for all-True vectors:")
u = np.array([True, True])
v = np.array([True, True])
print("Input: u = [True, True], v = [True, True]")
print("-" * 50)
print(f"Dice distance: {distance.dice(u, v)}")
print(f"Jaccard distance: {distance.jaccard(u, v)}")
print(f"Hamming distance: {distance.hamming(u, v)}")
print(f"Rogers-Tanimoto distance: {distance.rogerstanimoto(u, v)}")