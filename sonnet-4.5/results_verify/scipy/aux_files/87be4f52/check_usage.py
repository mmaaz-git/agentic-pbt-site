#!/usr/bin/env python3
"""Check how Rotation.mean is typically used."""

from scipy.spatial.transform import Rotation
import numpy as np

print("="*60)
print("TESTING ROTATION.MEAN WITH DIFFERENT INPUTS")
print("="*60)

# Test 1: Multiple rotations as a single Rotation object
print("\nTest 1: Creating multiple rotations at once")
try:
    r_multi = Rotation.from_euler('zyx', [[0, 0, 0],
                                           [1, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 1]], degrees=True)
    print(f"Created Rotation object with shape: {r_multi.as_quat().shape}")
    print(f"Number of rotations: {len(r_multi)}")

    print("Calling r_multi.mean()...")
    mean_rot = r_multi.mean()
    print(f"Success! Mean: {mean_rot}")
    print(f"Mean as quaternion: {mean_rot.as_quat()}")

except Exception as e:
    print(f"Failed: {e}")

# Test 2: Single rotation created from single input
print("\n" + "-"*40)
print("\nTest 2: Single rotation from single input")
try:
    r_single = Rotation.from_euler('zyx', [0, 0, 0], degrees=True)
    print(f"Created single Rotation: {r_single}")
    print(f"Shape: {r_single.as_quat().shape}")
    print(f"Length: {len(r_single) if hasattr(r_single, '__len__') else 'N/A'}")

    print("Calling r_single.mean()...")
    mean_rot = r_single.mean()
    print(f"Success! Mean: {mean_rot}")

except Exception as e:
    print(f"Failed: {e}")

# Test 3: Check if mean is a class method or instance method
print("\n" + "-"*40)
print("\nTest 3: Method type analysis")
print(f"Is mean a classmethod? {type(Rotation.__dict__.get('mean', None))}")

# Check if there's a way to call mean on a list of Rotation objects
print("\n" + "-"*40)
print("\nTest 4: Documentation example behavior")
try:
    # This is from the documentation
    r = Rotation.from_euler('zyx', [[0, 0, 0],
                                     [1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]], degrees=True)
    result = r.mean()
    print(f"Doc example works: {result.as_euler('zyx', degrees=True)}")
except Exception as e:
    print(f"Doc example failed: {e}")

# Test 5: Check type of Rotation.mean
print("\n" + "-"*40)
print("\nTest 5: Checking if mean is a classmethod that takes a list")
import inspect
print(f"Rotation.mean type: {type(Rotation.mean)}")
print(f"Is it a classmethod? {'classmethod' in str(type(Rotation.mean))}")