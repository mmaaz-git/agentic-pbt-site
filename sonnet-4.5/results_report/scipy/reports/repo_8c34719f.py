import numpy as np
from scipy.spatial.transform import Rotation

# Create a rotation with magnitude 1.0 (rotation by 1 radian around x-axis)
r = Rotation.from_rotvec([1.0, 0.0, 0.0])
print(f"Original rotation magnitude: {r.magnitude()}")

# Create a single-element group containing only this rotation
group = Rotation.concatenate([r])
print(f"Group size: {len(group)}")

# Reduce the rotation by the group containing itself
reduced = r.reduce(group)
print(f"Reduced rotation magnitude: {reduced.magnitude()}")

# Expected behavior: when a rotation that is a member of a group is reduced
# by that group, the result should be the identity rotation (magnitude 0.0)
print(f"\nExpected magnitude: 0.0 (identity rotation)")
print(f"Actual magnitude: {reduced.magnitude()}")

# Verify this is indeed incorrect
if np.isclose(reduced.magnitude(), 0.0, atol=1e-10):
    print("✓ Test passed: reduced to identity")
else:
    print("✗ Test failed: did not reduce to identity")
    print(f"  Magnitude is {reduced.magnitude() / r.magnitude()}x the original")