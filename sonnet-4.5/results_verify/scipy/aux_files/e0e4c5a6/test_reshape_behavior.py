import numpy as np
from scipy import ndimage

# Test to understand what reshape=False actually means according to the documentation

# Create a 2x3 array
arr = np.array([[1., 2., 3.],
                [4., 5., 6.]])

print("Original 2x3 array:")
print(arr)
print(f"Shape: {arr.shape}")

# Rotate 90 degrees with reshape=False
rotated_90_no_reshape = ndimage.rotate(arr, 90, reshape=False)
print("\nRotated 90° with reshape=False:")
print(rotated_90_no_reshape)
print(f"Shape: {rotated_90_no_reshape.shape} (same as original)")

# Rotate 90 degrees with reshape=True
rotated_90_reshape = ndimage.rotate(arr, 90, reshape=True)
print("\nRotated 90° with reshape=True:")
print(rotated_90_reshape)
print(f"Shape: {rotated_90_reshape.shape} (adapted to fit)")

# The key question: when rotating 90 degrees, a 2x3 array should become 3x2
# With reshape=False, it stays 2x3, which means data must be lost/cropped
print("\n" + "="*50)
print("Analysis:")
print("- A 90° rotation of a 2x3 array should naturally produce a 3x2 array")
print("- With reshape=False, output is forced to remain 2x3")
print("- This means the rotated content doesn't fit and gets cropped/interpolated")

# Test rotating back
print("\n" + "="*50)
print("Testing rotation back:")
rotated_back = ndimage.rotate(rotated_90_no_reshape, -90, reshape=False)
print("After rotating 90° then -90° with reshape=False:")
print(rotated_back)
print(f"Close to original? {np.allclose(arr, rotated_back)}")

# What about with mode='nearest' to avoid interpolation?
print("\n" + "="*50)
print("Testing with order=0 (nearest neighbor, no interpolation):")
arr_ones = np.ones((2, 3))
result = arr_ones.copy()
for i in range(4):
    result = ndimage.rotate(result, 90, reshape=False, order=0)
    print(f"After {i+1} rotations: {result}")
print(f"Close to original? {np.allclose(arr_ones, result)}")