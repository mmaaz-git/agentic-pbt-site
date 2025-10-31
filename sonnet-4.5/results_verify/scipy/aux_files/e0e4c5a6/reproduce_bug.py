import numpy as np
from scipy import ndimage

# Test the specific example from bug report
arr = np.array([[1., 1., 1.],
                [1., 1., 1.]])

print("Original array:")
print(arr)
print(f"Shape: {arr.shape}")

# Apply 4 rotations of 90 degrees with reshape=False
result = arr.copy()
for i in range(4):
    result = ndimage.rotate(result, 90, reshape=False)
    print(f"\nAfter rotation {i+1} (90° x {i+1} = {90*(i+1)}°):")
    print(result)

print("\n" + "="*50)
print("Final comparison:")
print("Original:", arr)
print("After 4x 90° rotations:", result)
print(f"Are they close? {np.allclose(arr, result)}")

# Also test with a single 360 degree rotation for comparison
print("\n" + "="*50)
print("Testing single 360° rotation:")
single_360 = ndimage.rotate(arr, 360, reshape=False)
print("After single 360° rotation:", single_360)
print(f"Is it close to original? {np.allclose(arr, single_360)}")

# Test with reshape=True for comparison
print("\n" + "="*50)
print("Testing 4x 90° rotations with reshape=True:")
result_reshape = arr.copy()
for i in range(4):
    result_reshape = ndimage.rotate(result_reshape, 90, reshape=True)
print("After 4x 90° rotations (reshape=True):", result_reshape)
print(f"Is it close to original? {np.allclose(arr, result_reshape)}")

# Test with square array
print("\n" + "="*50)
print("Testing 4x 90° rotations with square array:")
square_arr = np.ones((3, 3))
square_result = square_arr.copy()
for i in range(4):
    square_result = ndimage.rotate(square_result, 90, reshape=False)
print("Original square array:", square_arr)
print("After 4x 90° rotations:", square_result)
print(f"Is it close to original? {np.allclose(square_arr, square_result)}")