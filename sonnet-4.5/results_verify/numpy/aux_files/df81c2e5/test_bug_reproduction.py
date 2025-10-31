import numpy as np
from numpy.matrixlib import matrix

# Test the specific reproduction example
arr_3d = np.zeros((1, 1, 2))
list_3d = [[[0.0, 0.0]]]

print("Testing 3D array input:")
try:
    m1 = matrix(arr_3d)
    print(f"3D array -> matrix: SUCCESS, shape = {m1.shape}")
except ValueError as e:
    print(f"3D array -> matrix: FAILED with error: {e}")

print("\nTesting 3D list input:")
try:
    m2 = matrix(list_3d)
    print(f"3D list -> matrix: SUCCESS, shape = {m2.shape}")
except ValueError as e:
    print(f"3D list -> matrix: FAILED with error: {e}")

print("\n" + "="*50)
print("Testing that arr_3d and list_3d represent the same data:")
print(f"arr_3d shape: {arr_3d.shape}")
print(f"arr_3d content: {arr_3d}")
print(f"list_3d structure: {list_3d}")
print(f"Converting list to array: {np.array(list_3d).shape}")
print(f"Are they equivalent? {np.array_equal(arr_3d, np.array(list_3d))}")