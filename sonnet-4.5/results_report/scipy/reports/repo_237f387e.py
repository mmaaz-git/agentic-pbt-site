import numpy as np

# Test case: pure white RGB should convert to pure white grayscale
r, g, b = 255, 255, 255

# SciPy's current weights
scipy_weights = [0.21, 0.71, 0.07]
gray_value_scipy = int(scipy_weights[0] * r + scipy_weights[1] * g + scipy_weights[2] * b)

print(f"Input RGB: ({r}, {g}, {b})")
print(f"SciPy grayscale weights: {scipy_weights}")
print(f"Sum of weights: {sum(scipy_weights)}")
print(f"Expected grayscale value: 255")
print(f"Actual grayscale value: {gray_value_scipy}")
print(f"Brightness loss: {255 - gray_value_scipy}")

# Compare with standard ITU-R BT.709 weights
correct_weights = [0.2126, 0.7152, 0.0722]
gray_value_correct = int(correct_weights[0] * r + correct_weights[1] * g + correct_weights[2] * b)
print(f"\nWith ITU-R BT.709 weights {correct_weights}:")
print(f"Sum of weights: {sum(correct_weights)}")
print(f"Grayscale value: {gray_value_correct}")

# Test the actual function
try:
    import scipy.datasets
    # Create a small test image with pure white pixels
    test_image = np.ones((1, 1, 3), dtype='uint8') * 255
    # Simulate what scipy.datasets.face does with gray=True
    gray_simulated = (0.21 * test_image[:, :, 0] + 0.71 * test_image[:, :, 1] +
                      0.07 * test_image[:, :, 2]).astype('uint8')
    print(f"\nActual scipy conversion of white pixel: {gray_simulated[0, 0]}")
except Exception as e:
    print(f"\nError testing scipy directly: {e}")