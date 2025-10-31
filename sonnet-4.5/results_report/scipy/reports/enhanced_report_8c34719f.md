# Bug Report: scipy.spatial.transform.Rotation.reduce Incorrect Behavior for Single-Element Groups

**Target**: `scipy.spatial.transform.Rotation.reduce`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `reduce()` method incorrectly doubles the rotation magnitude when reducing a rotation by a single-element group containing itself (created via `Rotation.concatenate()`), instead of returning the identity rotation with magnitude 0.0 as expected.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from scipy.spatial.transform import Rotation
import numpy as np

@st.composite
def rotation_strategy(draw):
    rotvec = draw(st.lists(
        st.floats(min_value=-2*np.pi, max_value=2*np.pi,
                  allow_nan=False, allow_infinity=False),
        min_size=3, max_size=3
    ))
    return Rotation.from_rotvec(rotvec)

@given(rotation_strategy())
@settings(max_examples=100)
def test_reduce_self_is_identity(r):
    """Reducing a rotation by a group containing itself should yield identity"""
    group = Rotation.concatenate([r])
    reduced = r.reduce(group)
    assert np.isclose(reduced.magnitude(), 0.0, atol=1e-10), \
        f"Expected magnitude 0.0, got {reduced.magnitude()}"

if __name__ == "__main__":
    test_reduce_self_is_identity()
```

<details>

<summary>
**Failing input**: `Rotation.from_matrix(array([[ 0.54030231, -0.84147098,  0.        ], [ 0.84147098,  0.54030231,  0.        ], [ 0.        ,  0.        ,  1.        ]]))`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 24, in <module>
    test_reduce_self_is_identity()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 15, in test_reduce_self_is_identity
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 20, in test_reduce_self_is_identity
    assert np.isclose(reduced.magnitude(), 0.0, atol=1e-10), \
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected magnitude 0.0, got 2.0
Falsifying example: test_reduce_self_is_identity(
    r=Rotation.from_matrix(array([[ 0.54030231, -0.84147098,  0.        ],
                                [ 0.84147098,  0.54030231,  0.        ],
                                [ 0.        ,  0.        ,  1.        ]])),
)
```
</details>

## Reproducing the Bug

```python
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
```

<details>

<summary>
Output shows magnitude doubled instead of reduced to identity
</summary>
```
Original rotation magnitude: 1.0
Group size: 1
Reduced rotation magnitude: 2.0

Expected magnitude: 0.0 (identity rotation)
Actual magnitude: 2.0
✗ Test failed: did not reduce to identity
  Magnitude is 2.0x the original
```
</details>

## Why This Is A Bug

According to the scipy documentation, the `reduce()` method performs a transformation `q = l * p * r` where `l` and `r` are chosen from the left and right groups to minimize the magnitude of `q`. When a rotation `p` is reduced by a group containing itself, the optimal choice should be to select the group element that produces the identity rotation (magnitude 0.0).

The scipy test suite explicitly validates this behavior in `test_rotation_groups.py::test_single_reduction`, which shows that when a rotation that is a member of a group is reduced by that group, the result should be the identity rotation. However, when creating a single-element group via `Rotation.concatenate([r])` instead of using predefined groups like `create_group()`, the reduce method incorrectly computes `r * r` (doubling the magnitude) instead of finding the combination that yields identity.

The bug specifically affects:
- Single-element groups created via `Rotation.concatenate()`
- Non-identity rotations (the identity rotation correctly reduces to itself)
- All rotation creation methods (from_rotvec, from_quat, from_euler, etc.)

Multi-element groups and predefined symmetry groups (created via `create_group()`) work correctly, indicating this is an edge case in the handling of single-element concatenated groups.

## Relevant Context

The `reduce()` method is used in crystallography and robotics to find equivalent orientations considering symmetries. This bug could lead to incorrect calculations when working with trivial symmetry groups (single-element groups) in applications like:
- Crystallographic orientation analysis with trivial point groups
- Robotics path planning with single-configuration symmetries
- Unit testing of rotation group operations

Testing revealed:
- Identity rotation correctly reduces to identity (magnitude 0.0 → 0.0)
- All non-identity rotations double their magnitude when reduced by single-element groups
- Predefined groups (e.g., icosahedral 'I' with 60 elements) work correctly
- Multi-element concatenated groups produce different (non-doubled) results

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.reduce.html

## Proposed Fix

The bug likely resides in the Cython implementation in `scipy/spatial/transform/_rotation.pyx`. A high-level fix would involve ensuring that single-element groups are handled consistently regardless of how they're created. The algorithm should recognize when the rotation being reduced is equal to (or the inverse of) an element in the group and return the identity rotation.

Workaround for users until the bug is fixed:

```python
def safe_reduce(rotation, group):
    """Workaround for single-element group reduce bug"""
    if len(group) == 1:
        # Check if rotation matches the single group element
        # Using matrix comparison for numerical stability
        if np.allclose(rotation.as_matrix(), group[0].as_matrix()):
            return Rotation.identity()
    return rotation.reduce(group)
```