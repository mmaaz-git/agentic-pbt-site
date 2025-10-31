# Bug Report: scipy.spatial.transform.Rotation.reduce Single-Element Group Bug

**Target**: `scipy.spatial.transform.Rotation.reduce`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a single `Rotation` object is reduced by a group containing only itself (created via `Rotation.concatenate()`), the method incorrectly returns a rotation with magnitude equal to double the original rotation's magnitude, instead of the identity rotation (magnitude 0.0).

## Property-Based Test

```python
from hypothesis import given, strategies as st
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
def test_reduce_self_is_identity(r):
    """Reducing a rotation by a group containing itself should yield identity"""
    group = Rotation.concatenate([r])
    reduced = r.reduce(group)
    assert np.isclose(reduced.magnitude(), 0.0, atol=1e-10)
```

**Failing input**: Any single rotation created via `from_rotvec`, `from_quat`, etc., when reduced by a single-element group created via `concatenate`.

Example: `r = Rotation.from_rotvec([1.0, 0.0, 0.0])`

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial.transform import Rotation

r = Rotation.from_rotvec([1.0, 0.0, 0.0])
print(f"Original magnitude: {r.magnitude()}")

group = Rotation.concatenate([r])
reduced = r.reduce(group)
print(f"Reduced magnitude: {reduced.magnitude()}")

print(f"Expected: 0.0 (identity)")
print(f"Actual: {reduced.magnitude()}")
```

**Output:**
```
Original magnitude: 1.0
Reduced magnitude: 2.0
Expected: 0.0 (identity)
Actual: 2.0
```

## Why This Is A Bug

According to the scipy documentation and test suite (see `test_rotation_groups.py::test_single_reduction`), when a rotation that is a member of a group is reduced by that group, the result should be the identity rotation (magnitude 0.0).

The test `test_single_reduction` demonstrates this expected behavior:
```python
g = Rotation.create_group(name)
f = g[-1].reduce(g)
assert_array_almost_equal(f.magnitude(), 0)
```

However, when a single-element group is created via `Rotation.concatenate([r])` instead of using a pre-defined group like `create_group()`, the reduce method incorrectly doubles the rotation magnitude instead of returning identity.

**Key observations:**
1. Reducing by multi-element groups (e.g., tetrahedral group) works correctly
2. Reducing the identity rotation by `concatenate([identity])` works correctly
3. Reducing non-identity single rotations by `concatenate([rotation])` fails with magnitude = 2 × original

This suggests an edge case bug in how `reduce` handles single-element groups created via `concatenate`, specifically for non-identity rotations.

## Fix

The bug likely resides in the Cython implementation of `Rotation.reduce` in `_rotation.pyx`. The method appears to have different code paths for:
- Single rotations vs rotation sequences
- Groups created via `create_group` vs `concatenate`

A proper fix would require:
1. Examining the `reduce` implementation in `scipy/spatial/transform/_rotation.pyx`
2. Ensuring consistent handling of single-element groups regardless of how they're created
3. Verifying that the quaternion matching logic correctly accounts for the double cover property

Since the source is in Cython and complex, a high-level fix approach would be:
- Normalize the handling of single-element groups in `reduce`
- Ensure the "find closest rotation in group" logic works identically for all group sizes
- Add proper handling for the edge case where the group contains only the rotation being reduced

Without access to modify the Cython source, users can work around this bug by:
```python
if len(group) == 1:
    if np.allclose(r.as_matrix(), group[0].as_matrix()):
        reduced = Rotation.identity()
    else:
        reduced = r.reduce(group)
else:
    reduced = r.reduce(group)
```