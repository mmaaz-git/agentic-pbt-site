import numpy as np

# Test case demonstrating the grayscale conversion bug
r, g, b = 1, 1, 1

# Create a single pixel RGB image
rgb_array = np.array([[[r, g, b]]], dtype='uint8')

# Apply the same formula used in scipy.datasets.face(gray=True)
gray = (0.21 * rgb_array[:, :, 0] +
        0.71 * rgb_array[:, :, 1] +
        0.07 * rgb_array[:, :, 2]).astype('uint8')

actual_gray = gray[0, 0]

print(f"RGB = ({r}, {g}, {b})")
print(f"Expected grayscale (float) = 0.21 * {r} + 0.71 * {g} + 0.07 * {b} = {0.21 * r + 0.71 * g + 0.07 * b}")
print(f"Actual grayscale (uint8) = {actual_gray}")
print(f"Weight sum = 0.21 + 0.71 + 0.07 = {0.21 + 0.71 + 0.07}")
print()

# Show the issue across multiple gray values
print("Demonstration of systematic underestimation:")
print("RGB Value -> Gray Value (Expected Float vs Actual uint8)")
print("-" * 50)
for val in [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 255]:
    r, g, b = val, val, val
    rgb_array = np.array([[[r, g, b]]], dtype='uint8')
    gray = (0.21 * rgb_array[:, :, 0] +
            0.71 * rgb_array[:, :, 1] +
            0.07 * rgb_array[:, :, 2]).astype('uint8')
    actual_gray = gray[0, 0]
    expected_float = 0.21 * r + 0.71 * g + 0.07 * b
    print(f"RGB=({val:3d}, {val:3d}, {val:3d}) -> gray={actual_gray:3d}, expected_float={expected_float:6.2f}, diff={val - actual_gray}")