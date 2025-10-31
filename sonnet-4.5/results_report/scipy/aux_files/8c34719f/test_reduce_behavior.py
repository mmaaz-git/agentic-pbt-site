import numpy as np
from scipy.spatial.transform import Rotation

print("Understanding what reduce() should do:")
print("=" * 50)

# The reduce() method should find l and r from the groups such that
# q = l * p * r has the smallest magnitude

r = Rotation.from_rotvec([1.0, 0.0, 0.0])
print(f"Original rotation r: magnitude = {r.magnitude():.4f}")

# When we have a single-element group containing r itself:
# We're looking for q = group[i] * r * identity (since right is None)
# where group[i] minimizes the magnitude of q

group = Rotation.concatenate([r])
print(f"\nGroup contains single element: {group}")

# The optimal choice should be group[0] (which is r)
# So q = r * r * identity = r^2
# But we expect that when r is in the group, the reduced rotation should be identity

# Let's manually compute what's happening:
q_manual = group[0] * r  # Since right=None, it's just left * p
print(f"\nManual computation (group[0] * r): magnitude = {q_manual.magnitude():.4f}")

# Compare with reduce
reduced = r.reduce(group)
print(f"Using reduce(): magnitude = {reduced.magnitude():.4f}")

# The issue is clear: when reducing by a group containing the rotation itself,
# we expect identity (0 magnitude), but we get double the magnitude instead

print("\n" + "=" * 50)
print("Expected behavior (from scipy tests):")
print("When a rotation is a member of the group, reducing by the group")
print("should yield the identity rotation (magnitude 0)")

# Let's verify with the inverse
r_inv = r.inv()
group_inv = Rotation.concatenate([r_inv])
identity_check = r_inv * r
print(f"\nInverse * Original = Identity? magnitude = {identity_check.magnitude():.4f}")

# What we'd expect:
# If group contains r, and we're reducing r by the group,
# The best choice is group[0] = r, giving us r^(-1) * r = identity
# But instead, reduce() seems to be computing r * r = r^2