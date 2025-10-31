#!/usr/bin/env python3
"""Check if Rotation.mean can be called as a class method."""

from scipy.spatial.transform import Rotation
import numpy as np

print("="*60)
print("TESTING ROTATION.MEAN AS CLASS METHOD")
print("="*60)

# Try calling mean as a class method
print("\nTest: Calling Rotation.mean() with list of Rotation objects")

r1 = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
r2 = Rotation.from_quat([0.707, 0.0, 0.0, 0.707])

print(f"r1: {r1}")
print(f"r2: {r2}")

# Check if this is how it's supposed to be called
print("\nAttempting various ways to call mean:")

print("\n1. r1.mean() - instance method on single rotation:")
try:
    result = r1.mean()
    print(f"Success: {result}")
except Exception as e:
    print(f"Failed: {e}")

print("\n2. Rotation.mean([r1, r2]) - class method with list:")
print("This is what the bug report claims should work...")
try:
    result = Rotation.mean([r1, r2])
    print(f"Success: {result}")
except Exception as e:
    print(f"Failed: {e}")

print("\n3. Rotation.mean(r1) - class method with single rotation:")
try:
    result = Rotation.mean(r1)
    print(f"Success: {result}")
except Exception as e:
    print(f"Failed: {e}")

# The correct way based on the documentation
print("\n4. Creating multiple rotations and calling mean():")
try:
    r_multi = Rotation.from_quat([[0.0, 0.0, 0.0, 1.0],
                                   [0.707, 0.0, 0.0, 0.707]])
    print(f"Created multi-rotation object: {r_multi}")
    print(f"Shape: {r_multi.as_quat().shape}")
    result = r_multi.mean()
    print(f"Success: {result}")
except Exception as e:
    print(f"Failed: {e}")

print("\n" + "="*60)
print("CONCLUSION:")
print("Based on the documentation, mean() is an instance method")
print("that should be called on a Rotation object containing")
print("multiple rotations, NOT a class method accepting a list.")