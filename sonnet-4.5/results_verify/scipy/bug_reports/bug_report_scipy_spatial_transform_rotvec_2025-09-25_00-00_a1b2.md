# Bug Report: scipy.spatial.transform.Rotation rotvec Round-Trip Failure

**Target**: `scipy.spatial.transform.Rotation`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The round-trip property for rotation vectors fails when the rotation magnitude is exactly π radians. Converting a rotation to a rotation vector with `as_rotvec()`, then back with `from_rotvec()`, and again to a rotation vector produces a negated result instead of the original rotation vector.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st
from scipy.spatial.transform import Rotation


@st.composite
def quaternions(draw):
    q = draw(st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
                      min_size=4, max_size=4))
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return [0, 0, 0, 1]
    return (np.array(q) / norm).tolist()


@st.composite
def rotation_vectors(draw):
    axis = draw(st.lists(st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False),
                         min_size=3, max_size=3))
    angle = draw(st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False))
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        return [0, 0, 0]
    return (np.array(axis) / axis_norm * angle).tolist()


@st.composite
def rotations(draw):
    choice = draw(st.integers(min_value=0, max_value=1))
    if choice == 0:
        q = draw(quaternions())
        return Rotation.from_quat(q)
    else:
        rv = draw(rotation_vectors())
        return Rotation.from_rotvec(rv)


@given(rotations())
@settings(max_examples=500)
def test_rotvec_round_trip(r):
    rv1 = r.as_rotvec()
    r2 = Rotation.from_rotvec(rv1)
    rv2 = r2.as_rotvec()
    assert np.allclose(rv1, rv2, atol=1e-10)
```

**Failing input**: `Rotation.from_matrix(array([[-1., 0., 0.], [0., 0.6, 0.8], [0., 0.8, -0.6]]))`

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial.transform import Rotation

matrix = np.array([[-1.,  0.,  0.],
                   [ 0.,  0.6, 0.8],
                   [ 0.,  0.8, -0.6]])

r = Rotation.from_matrix(matrix)

rv1 = r.as_rotvec()
print(f"rv1 = {rv1}")

r2 = Rotation.from_rotvec(rv1)
rv2 = r2.as_rotvec()
print(f"rv2 = {rv2}")

print(f"|rv1| = {np.linalg.norm(rv1)}")
print(f"rv1 == rv2: {np.allclose(rv1, rv2)}")
print(f"rv1 == -rv2: {np.allclose(rv1, -rv2)}")
```

Output:
```
rv1 = [0.         2.80992589 1.40496295]
rv2 = [-0.         -2.80992589 -1.40496295]
|rv1| = 3.141592653589793
rv1 == rv2: False
rv1 == -rv2: True
```

## Why This Is A Bug

The round-trip property `from_rotvec(as_rotvec(r)).as_rotvec() == r.as_rotvec()` is violated for rotations with magnitude π. This violates user expectations for serialization/deserialization. When a user saves a rotation vector and later loads it, they expect to get the same representation back, even though the underlying rotation is equivalent.

While it's true that rotation vectors with magnitudes π and -π around opposite axes represent the same rotation mathematically, the library should maintain consistency in representation. The documentation does not mention this behavior, and the examples in the docstring show round-trip conversions working correctly.

This is particularly problematic for:
1. Serialization/deserialization workflows where users expect stable representations
2. Numerical algorithms that rely on the stability of rotation vector representations
3. Testing and debugging where inconsistent representations cause confusion

## Fix

The issue occurs at the boundary case where the rotation magnitude is exactly π. The conversion from rotation vector to quaternion and back introduces a sign flip. One possible fix is to ensure that `as_rotvec()` always returns a rotation vector in a canonical form (e.g., with a positive first non-zero component) to avoid the sign ambiguity.

Alternatively, the library could document this behavior explicitly and provide guidance on how to handle it.