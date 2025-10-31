# Bug Report: scipy.spatial.transform.Rotation.mean Segmentation Fault on Invalid Usage

**Target**: `scipy.spatial.transform.Rotation.mean`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Calling `Rotation.mean()` as a class method with a list argument causes a segmentation fault instead of raising a proper Python exception.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
from scipy.spatial.transform import Rotation
import numpy as np
import hypothesis.extra.numpy as hnp

@st.composite
def quaternions(draw):
    q = draw(hnp.arrays(np.float64, (4,), elements=st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False)))
    assume(np.linalg.norm(q) > 1e-10)
    return q / np.linalg.norm(q)

def rotation_equal(r1, r2, atol=1e-10):
    """Check if two rotations are equal within tolerance"""
    q1 = r1.as_quat()
    q2 = r2.as_quat()
    # Account for quaternion double cover (q and -q represent same rotation)
    return np.allclose(q1, q2, atol=atol) or np.allclose(q1, -q2, atol=atol)

@given(quaternions())
@settings(max_examples=200)
def test_rotation_mean_single(q):
    """Property: mean of single rotation should be itself"""
    r = Rotation.from_quat(q)
    r_mean = Rotation.mean([r])

    assert rotation_equal(r, r_mean, atol=1e-10), "Mean of single rotation is not itself"

if __name__ == "__main__":
    test_rotation_mean_single()
```

<details>

<summary>
**Failing input**: `q = np.array([1.0, 0.0, 0.0, 0.0])` (any quaternion triggers the crash)
</summary>
```
Process terminated with exit code: 139
```
</details>

## Reproducing the Bug

```python
from scipy.spatial.transform import Rotation
import numpy as np

q = np.array([0.0, 0.0, 0.0, 1.0])
r = Rotation.from_quat(q)

print("Attempting to call Rotation.mean([r]) with a single rotation in a list...")
r_mean = Rotation.mean([r])
print("Mean computed:", r_mean.as_quat())
```

<details>

<summary>
Segmentation fault (core dumped)
</summary>
```
Process terminated with exit code: 139
```
</details>

## Why This Is A Bug

This is a bug because Python code should never cause segmentation faults, even when the API is used incorrectly. The issue stems from a fundamental misunderstanding of the API: `Rotation.mean()` is an instance method, not a class method.

According to the scipy documentation:
- **Correct usage**: `rotation_instance.mean()` where `rotation_instance` contains multiple rotations
- **Incorrect usage**: `Rotation.mean([rotation_object])` - treating it as a class method

The documentation signature clearly shows this is an instance method: `Rotation.mean(self, weights=None)` where `self` is the Rotation instance containing the rotations to average.

While the usage `Rotation.mean([r])` is incorrect, the Cython/C extension code should handle this gracefully by raising a TypeError like "descriptor 'mean' requires a 'Rotation' object but received a 'list'", not by crashing the Python interpreter.

## Relevant Context

The `mean()` method works correctly when used as intended:
```python
# Correct usage - multiple rotations in one Rotation object
r = Rotation.from_quat([[0, 0, 0, 1], [1, 0, 0, 0]])
mean_r = r.mean()  # Works correctly
```

When attempting to use `mean()` on a single rotation instance:
```python
r = Rotation.from_quat([0, 0, 0, 1])  # Single rotation
mean_r = r.mean()  # Raises TypeError: Single rotation has no len()
```

The segfault only occurs when incorrectly calling `Rotation.mean()` as a class method with a list argument, which bypasses normal Python method binding and likely causes the Cython code to receive unexpected input types.

Documentation reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.mean.html

## Proposed Fix

The fix should be in the Cython extension to add proper type checking when the method is called improperly. Since this is compiled Cython code, a high-level fix approach would be:

1. Add input validation at the beginning of the `mean` method in `_rotation.pyx`
2. Check if `self` is actually a Rotation instance
3. Raise a descriptive TypeError if called incorrectly

The actual implementation would require modifying the scipy Cython source code to ensure the `mean` method properly validates its implicit `self` parameter before attempting any operations that could cause memory access violations.