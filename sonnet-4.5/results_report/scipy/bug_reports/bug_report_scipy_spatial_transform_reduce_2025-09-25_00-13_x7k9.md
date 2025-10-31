# Bug Report: scipy.spatial.transform Rotation.reduce Fails With Single Rotation Arguments

**Target**: `scipy.spatial.transform.Rotation.reduce`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`Rotation.reduce()` fails with a cryptic ValueError when called on a single rotation with a single rotation as the left or right argument, even though it accepts single rotations as arguments in all other combinations.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from scipy.spatial.transform import Rotation
import numpy as np

@given(st.integers(1, 100))
def test_reduce_identity_composition(seed):
    """Test that reducing a rotation composed with identity gives the original rotation"""
    np.random.seed(seed)
    r = Rotation.random()

    identity = Rotation.identity()
    composed = r * identity

    reduced = composed.reduce(left=identity)

    assert r.approx_equal(reduced, atol=1e-14)
```

**Failing input**: Any seed value (e.g., `seed=1`)

## Reproducing the Bug

```python
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
```

Output:
```
Single rotation with single identity:
FAILED: Expected `quat` to have shape (4,) or (N, 4), got ().

Single rotation with array identity:
Success: Rotation.from_matrix(...)

Array rotation with single identity:
Success: Rotation.from_matrix(...)
```

## Why This Is A Bug

The `reduce()` method has inconsistent behavior:

| Target Type | Argument Type | Result |
|------------|---------------|--------|
| Single | Single | ✗ FAILS |
| Single | Array | ✓ Works |
| Array | Single | ✓ Works |
| Array | Array | ✓ Works |

Since the method accepts single rotations as arguments in 3 out of 4 cases, it should also accept them when both the target and argument are single rotations. This inconsistency is confusing for users and breaks the principle of least surprise.

The error message "Expected `quat` to have shape (4,) or (N, 4), got ()" is also unhelpful, as it doesn't explain that the user should wrap their single rotation in an array.

## Fix

The fix should normalize single rotation arguments to arrays at the beginning of the `reduce()` method:

```diff
def reduce(self, left=None, right=None, return_indices=False):
    # ... existing validation code ...

+   # Normalize single rotations to arrays for consistent handling
+   if left is not None and not hasattr(left, '__len__'):
+       left = Rotation.concatenate([left])
+   if right is not None and not hasattr(right, '__len__'):
+       right = Rotation.concatenate([right])

    # ... existing reduction algorithm ...
```

This ensures that single rotations are handled consistently as 1-element arrays, matching the behavior when the target is an array.