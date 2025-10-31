from scipy.spatial.transform import Rotation
import numpy as np

# Check what constitutes a valid Rotation instance
r_single = Rotation.random()
r_array = Rotation.random(5)
r_identity = Rotation.identity()
r_concat = Rotation.concatenate([Rotation.random()])

print(f"Single rotation shape: {r_single.as_quat().shape}")
print(f"Array rotation shape: {r_array.as_quat().shape}")
print(f"Identity rotation shape: {r_identity.as_quat().shape}")
print(f"Concatenated single rotation shape: {r_concat.as_quat().shape}")

print(f"\nSingle rotation is single: {r_single.single}")
print(f"Array rotation is single: {r_array.single}")
print(f"Identity rotation is single: {r_identity.single}")
print(f"Concatenated single is single: {r_concat.single}")

# Check if the documentation says they should be interchangeable
print("\n--- Testing interchangeability in other methods ---")

# Test multiplication (which works with mixed types)
try:
    result = r_single * r_identity
    print(f"Single * identity works: {result.single}")
except Exception as e:
    print(f"Single * identity failed: {e}")

try:
    result = r_array[0:1] * r_identity
    print(f"Array[0:1] * identity works: {result.single}")
except Exception as e:
    print(f"Array[0:1] * identity failed: {e}")

# Test inv() method
try:
    inv_single = r_single.inv()
    print(f"Single.inv() works: {inv_single.single}")
except Exception as e:
    print(f"Single.inv() failed: {e}")