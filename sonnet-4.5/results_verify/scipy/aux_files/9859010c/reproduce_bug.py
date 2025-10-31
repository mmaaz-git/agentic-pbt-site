from scipy.spatial.transform import Rotation
import numpy as np

np.random.seed(1)
r = Rotation.random()
identity = Rotation.identity()

print("Single rotation with single identity:")
try:
    reduced = r.reduce(left=identity)
    print(f"Success: {reduced}")
except Exception as e:
    print(f"FAILED: {e}")

print("\nSingle rotation with array identity:")
identity_array = Rotation.concatenate([Rotation.identity()])
try:
    reduced = r.reduce(left=identity_array)
    print(f"Success: {reduced}")
except Exception as e:
    print(f"FAILED: {e}")

print("\nArray rotation with single identity:")
r_array = Rotation.concatenate([r])
try:
    reduced = r_array.reduce(left=identity)
    print(f"Success: {reduced}")
except Exception as e:
    print(f"FAILED: {e}")

print("\nArray rotation with array identity:")
try:
    reduced = r_array.reduce(left=identity_array)
    print(f"Success: {reduced}")
except Exception as e:
    print(f"FAILED: {e}")