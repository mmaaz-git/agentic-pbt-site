# Bug Report: scipy.spatial.transform.Rotation Euler Angle Round-Trip Fails Near Gimbal Lock

**Target**: `scipy.spatial.transform.Rotation.as_euler()` / `scipy.spatial.transform.Rotation.from_euler()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The round-trip conversion `Rotation.from_euler(seq, angles).as_euler(seq)` does not preserve the rotation when the middle Euler angle is very close to (but not exactly at) gimbal lock. This violates the expected property that `from_euler(seq, r.as_euler(seq))` should equal `r`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from scipy.spatial.transform import Rotation
import numpy as np

@given(st.floats(min_value=-np.pi, max_value=np.pi),
       st.floats(min_value=0, max_value=2*np.pi),
       st.floats(min_value=0, max_value=2*np.pi))
@settings(max_examples=500)
def test_euler_round_trip_ZYZ(alpha, beta, gamma):
    seq = 'ZYZ'
    angles = np.array([alpha, beta, gamma])

    r = Rotation.from_euler(seq, angles)
    angles2 = r.as_euler(seq)
    r2 = Rotation.from_euler(seq, angles2)

    assert r.approx_equal(r2, atol=1e-10)
```

**Failing input**: `alpha=0.0, beta=5.960464477539063e-08, gamma=1.0` (sequence='ZYZ')

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial.transform import Rotation

alpha = 0.0
beta = 5.960464477539063e-08
gamma = 1.0

angles_original = np.array([alpha, beta, gamma])
r_original = Rotation.from_euler('ZYZ', angles_original)

angles_recovered = r_original.as_euler('ZYZ')
r_recovered = Rotation.from_euler('ZYZ', angles_recovered)

angle_diff = (r_original.inv() * r_recovered).magnitude()
print(f"Angle between rotations: {angle_diff:.2e} radians")

test_vector = np.array([100.0, 200.0, 300.0])
v1 = r_original.apply(test_vector)
v2 = r_recovered.apply(test_vector)
print(f"Vector difference: {np.max(np.abs(v1 - v2)):.2e}")
```

Output:
```
Angle between rotations: 5.72e-08 radians
Vector difference: 1.50e-05
```

## Why This Is A Bug

The property `Rotation.from_euler(seq, r.as_euler(seq)) ≈ r` should hold for all rotations `r` and all valid Euler angle sequences `seq`. This is a fundamental round-trip property.

When `beta ≈ 0` (near gimbal lock for ZYZ), `as_euler()` detects gimbal lock and sets `gamma=0`, redistributing the angle to `alpha=(alpha+gamma)`. However, when `beta` is very small but not exactly zero:

```
R_z(0.0) × R_y(5.96e-08) × R_z(1.0) ≠ R_z(1.0) × R_y(5.96e-08) × R_z(0.0)
```

These are **different rotations** because the small Y-axis rotation does not commute with the Z-axis rotations.

The warning message "Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles" is misleading. While it's true that angles cannot be uniquely determined **at exact gimbal lock**, when beta is merely **close to** (but not at) gimbal lock, the angles **are** uniquely determined, and changing them changes the rotation.

## Fix

The gimbal lock detection threshold should be more conservative. Currently, the code detects gimbal lock too aggressively and modifies angles even when the rotation is not actually at gimbal lock.

**Option 1**: Only apply gimbal lock handling when `|beta|` or `|beta - π|` is below a much stricter threshold (e.g., 1e-12 instead of current ~1e-6).

**Option 2**: When near (but not at) gimbal lock, do not redistribute the angles. Accept that the conversion may be numerically less stable rather than returning incorrect angles.

**Option 3**: When gimbal lock is detected and angles are redistributed, adjust `beta` to exactly 0 (or π) so that the angle redistribution is mathematically valid.

The fix should be applied in the Cython implementation of `as_euler()` in `scipy/spatial/transform/_rotation.pyx`.