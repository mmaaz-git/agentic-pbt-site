import numpy as np
from scipy.spatial.transform import Rotation

print("Testing various rotation creation methods:\n")

# Test 1: from_rotvec
r1 = Rotation.from_rotvec([1.0, 0.0, 0.0])
group1 = Rotation.concatenate([r1])
reduced1 = r1.reduce(group1)
print(f"from_rotvec: original={r1.magnitude():.4f}, reduced={reduced1.magnitude():.4f}")

# Test 2: from_quat
r2 = Rotation.from_quat([0, 0, 0.7071068, 0.7071068])  # 90 degree rotation
group2 = Rotation.concatenate([r2])
reduced2 = r2.reduce(group2)
print(f"from_quat: original={r2.magnitude():.4f}, reduced={reduced2.magnitude():.4f}")

# Test 3: from_euler
r3 = Rotation.from_euler('xyz', [45, 0, 0], degrees=True)
group3 = Rotation.concatenate([r3])
reduced3 = r3.reduce(group3)
print(f"from_euler: original={r3.magnitude():.4f}, reduced={reduced3.magnitude():.4f}")

# Test 4: identity rotation
r4 = Rotation.identity()
group4 = Rotation.concatenate([r4])
reduced4 = r4.reduce(group4)
print(f"identity: original={r4.magnitude():.4f}, reduced={reduced4.magnitude():.4f}")

print("\nTesting with predefined groups:")

# Test 5: Using create_group
g = Rotation.create_group('I')  # Icosahedral group
print(f"Icosahedral group size: {len(g)}")
# Reduce last element by the whole group
reduced5 = g[-1].reduce(g)
print(f"Last element of 'I' group reduced by group: magnitude={reduced5.magnitude():.4f}")

# Test 6: Multiple element group via concatenate
r6a = Rotation.from_rotvec([1.0, 0.0, 0.0])
r6b = Rotation.from_rotvec([0.0, 1.0, 0.0])
group6 = Rotation.concatenate([r6a, r6b])
reduced6 = r6a.reduce(group6)
print(f"\nTwo-element group via concatenate:")
print(f"First rotation reduced by group: magnitude={reduced6.magnitude():.4f}")