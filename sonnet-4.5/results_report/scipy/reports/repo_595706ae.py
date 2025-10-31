import numpy as np
from scipy import ndimage

# Create a simple non-square array
arr = np.array([[1., 1., 1.],
                [1., 1., 1.]])

print("Original array:")
print(arr)
print(f"Shape: {arr.shape}")
print()

# Apply 4 rotations of 90 degrees with reshape=False
result = arr.copy()
for i in range(4):
    result = ndimage.rotate(result, 90, reshape=False)
    print(f"After rotation {i+1} (90° × {i+1} = {90*(i+1)}°):")
    print(result)
    print(f"Shape: {result.shape}")
    print()

print("Expected: Should return to original array")
print("Actual result after 4×90° rotations:")
print(result)
print()

# Compare with a single 360° rotation
single_360 = ndimage.rotate(arr, 360, reshape=False)
print("Result of single 360° rotation:")
print(single_360)
print()

# Check if they are close
print("Are 4×90° rotations equal to original?", np.allclose(arr, result))
print("Is single 360° rotation equal to original?", np.allclose(arr, single_360))