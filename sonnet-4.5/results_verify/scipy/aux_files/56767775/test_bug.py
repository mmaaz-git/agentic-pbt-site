#!/usr/bin/env python3
"""Test the reported bug about grayscale conversion weights"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

# Test 1: Property-based test from bug report
def test_grayscale_weights_sum_to_one():
    weights = [0.21, 0.71, 0.07]
    weight_sum = sum(weights)

    assert weight_sum == 1.0, (
        f"Grayscale conversion weights {weights} sum to {weight_sum}, not 1.0. "
        f"This causes grayscale images to be {abs(1.0 - weight_sum) * 100:.1f}% "
        f"{'darker' if weight_sum < 1.0 else 'brighter'} than expected."
    )

# Test 2: Reproduce the bug
print("=== Reproducing the bug ===")
weights = [0.21, 0.71, 0.07]
print(f"Sum of grayscale weights: {sum(weights)}")
print(f"Expected: 1.0")
print(f"Difference: {1.0 - sum(weights)}")

# Test 3: Check standard weights
print("\n=== Standard RGB to Grayscale Weights ===")
print("ITU-R BT.601: (0.299, 0.587, 0.114) -> Sum:", sum([0.299, 0.587, 0.114]))
print("ITU-R BT.709: (0.2126, 0.7152, 0.0722) -> Sum:", sum([0.2126, 0.7152, 0.0722]))

# Test 4: Check if scipy weights match any standard
print("\n=== Checking scipy weights against standards ===")
scipy_weights = [0.21, 0.71, 0.07]
bt709_weights = [0.2126, 0.7152, 0.0722]

print(f"Scipy weights: {scipy_weights}")
print(f"BT.709 weights: {bt709_weights}")
print(f"Difference from BT.709: [{scipy_weights[0] - bt709_weights[0]:.4f}, "
      f"{scipy_weights[1] - bt709_weights[1]:.4f}, "
      f"{scipy_weights[2] - bt709_weights[2]:.4f}]")

# Test 5: Run the property test
print("\n=== Running property test ===")
try:
    test_grayscale_weights_sum_to_one()
    print("Property test PASSED")
except AssertionError as e:
    print(f"Property test FAILED: {e}")

# Test 6: Test actual scipy function
print("\n=== Testing actual scipy.datasets.face() ===")
try:
    import scipy.datasets
    import numpy as np

    # Get color and grayscale faces
    face_color = scipy.datasets.face(gray=False)
    face_gray = scipy.datasets.face(gray=True)

    print(f"Color face shape: {face_color.shape}")
    print(f"Grayscale face shape: {face_gray.shape}")
    print(f"Color face dtype: {face_color.dtype}")
    print(f"Grayscale face dtype: {face_gray.dtype}")

    # Manual conversion with scipy weights
    manual_gray = (0.21 * face_color[:, :, 0] +
                   0.71 * face_color[:, :, 1] +
                   0.07 * face_color[:, :, 2]).astype('uint8')

    # Check if manual matches scipy's output
    print(f"Manual conversion matches scipy: {np.array_equal(manual_gray, face_gray)}")

    # Compare with proper BT.709 weights
    proper_gray = (0.2126 * face_color[:, :, 0] +
                   0.7152 * face_color[:, :, 1] +
                   0.0722 * face_color[:, :, 2]).astype('uint8')

    # Calculate average difference
    diff = proper_gray.astype('int') - face_gray.astype('int')
    print(f"Average pixel difference (proper - scipy): {np.mean(diff):.4f}")
    print(f"Max pixel difference: {np.max(np.abs(diff))}")
    print(f"Percentage of pixels different: {np.sum(diff != 0) / diff.size * 100:.2f}%")

except ImportError as e:
    print(f"Could not import scipy: {e}")