from scipy.spatial import distance
import numpy as np

# Test all-False arrays with various dissimilarity metrics
u = np.array([False, False, False])
v = np.array([False, False, False])

print("Testing various dissimilarity metrics with all-False identical arrays:")
print("="*60)

# Test dice
try:
    result = distance.dice(u, v)
    print(f"dice: {result}")
except Exception as e:
    print(f"dice: ERROR - {e}")

# Test jaccard (similar to dice)
try:
    result = distance.jaccard(u, v)
    print(f"jaccard: {result}")
except Exception as e:
    print(f"jaccard: ERROR - {e}")

# Test hamming
try:
    result = distance.hamming(u, v)
    print(f"hamming: {result}")
except Exception as e:
    print(f"hamming: ERROR - {e}")

# Test rogerstanimoto
try:
    result = distance.rogerstanimoto(u, v)
    print(f"rogerstanimoto: {result}")
except Exception as e:
    print(f"rogerstanimoto: ERROR - {e}")

# Test sokalmichener
try:
    result = distance.sokalmichener(u, v)
    print(f"sokalmichener: {result}")
except Exception as e:
    print(f"sokalmichener: ERROR - {e}")

# Test sokalsneath
try:
    result = distance.sokalsneath(u, v)
    print(f"sokalsneath: {result}")
except Exception as e:
    print(f"sokalsneath: ERROR - {e}")

# Test yule
try:
    result = distance.yule(u, v)
    print(f"yule: {result}")
except Exception as e:
    print(f"yule: ERROR - {e}")

print("\n" + "="*60)
print("Testing dice with different cases:")
print("="*60)

# Test identical True vectors
u2 = np.array([True, True, True])
v2 = np.array([True, True, True])
result = distance.dice(u2, v2)
print(f"dice([True, True, True], [True, True, True]) = {result}")

# Test identical mixed vectors
u3 = np.array([True, False, True])
v3 = np.array([True, False, True])
result = distance.dice(u3, v3)
print(f"dice([True, False, True], [True, False, True]) = {result}")

# Test completely different vectors
u4 = np.array([True, True, True])
v4 = np.array([False, False, False])
result = distance.dice(u4, v4)
print(f"dice([True, True, True], [False, False, False]) = {result}")