# Bug Report: scipy.spatial.transform.Rotation.reduce Fails With Single Rotation Arguments

**Target**: `scipy.spatial.transform.Rotation.reduce`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `Rotation.reduce()` method crashes with a ValueError when both the target rotation and the left/right argument are single rotations, despite the documentation stating it accepts "Rotation instance" without restrictions.

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

# Run the test
if __name__ == "__main__":
    test_reduce_identity_composition()
```

<details>

<summary>
**Failing input**: `seed=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 20, in <module>
    test_reduce_identity_composition()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 6, in test_reduce_identity_composition
    def test_reduce_identity_composition(seed):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 14, in test_reduce_identity_composition
    reduced = composed.reduce(left=identity)
  File "scipy/spatial/transform/_rotation.pyx", line 3065, in scipy.spatial.transform._rotation.Rotation.reduce
  File "scipy/spatial/transform/_rotation.pyx", line 850, in scipy.spatial.transform._rotation.Rotation.__init__
ValueError: Expected `quat` to have shape (4,) or (N, 4), got ().
Falsifying example: test_reduce_identity_composition(
    seed=1,  # or any other generated value
)
```
</details>

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

print("\nArray rotation with array identity:")
try:
    reduced = r_array.reduce(left=identity_array)
    print(f"Success: {reduced}")
except Exception as e:
    print(f"FAILED: {e}")
```

<details>

<summary>
ValueError when single rotation is used with single rotation argument
</summary>
```
Single rotation with single identity:
FAILED: Expected `quat` to have shape (4,) or (N, 4), got ().

Single rotation with array identity:
Success: Rotation.from_matrix(array([[ 0.70595742, -0.70241983, -0.09072214],
                            [-0.19221012, -0.31329391,  0.93000118],
                            [-0.68167397, -0.63910352, -0.35618436]]))

Array rotation with single identity:
Success: Rotation.from_matrix(array([[[ 0.70595742, -0.70241983, -0.09072214],
                             [-0.19221012, -0.31329391,  0.93000118],
                             [-0.68167397, -0.63910352, -0.35618436]]]))

Array rotation with array identity:
Success: Rotation.from_matrix(array([[[ 0.70595742, -0.70241983, -0.09072214],
                             [-0.19221012, -0.31329391,  0.93000118],
                             [-0.68167397, -0.63910352, -0.35618436]]]))
```
</details>

## Why This Is A Bug

This violates expected behavior because the `reduce()` method documentation explicitly states that both `left` and `right` parameters accept "Rotation instance" without distinguishing between single and array rotations. The inconsistent behavior matrix shows:

| Target Type | Argument Type | Result |
|------------|---------------|--------|
| Single | Single | ✗ **FAILS** |
| Single | Array | ✓ Works |
| Array | Single | ✓ Works |
| Array | Array | ✓ Works |

The method works correctly in 3 out of 4 combinations, failing only when both the target and argument are single rotations. This contradicts:

1. **The documented interface**: Parameters are described as "Rotation instance" which includes both single rotations (shape `(4,)`) and rotation arrays (shape `(N, 4)`)
2. **Principle of least surprise**: Other Rotation methods like multiplication handle single-single combinations without issue
3. **API consistency**: The same operation succeeds when either rotation is wrapped in an array

The cryptic error message "Expected `quat` to have shape (4,) or (N, 4), got ()" provides no guidance that users need to wrap single rotations in arrays, making this a confusing developer experience.

## Relevant Context

The scipy.spatial.transform.Rotation class represents rotations in 3D space and can be either:
- A single rotation with quaternion shape `(4,)`
- An array of rotations with quaternion shape `(N, 4)`

The `reduce()` method is used to find equivalent rotations with minimal rotation angle by applying symmetry transformations from rotation groups. This is particularly useful in crystallography and robotics applications where symmetries can be exploited to find optimal rotations.

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.reduce.html

The issue appears to stem from internal handling where single rotations aren't properly normalized to arrays before processing, causing the quaternion initialization to receive an empty shape `()` instead of the expected `(4,)` or `(N, 4)`.

## Proposed Fix

```diff
def reduce(self, left=None, right=None, return_indices=False):
    # ... existing validation code ...

+   # Normalize single rotations to arrays for consistent handling
+   if left is not None and left.single:
+       left = Rotation.concatenate([left])
+   if right is not None and right.single:
+       right = Rotation.concatenate([right])

    # ... existing reduction algorithm ...
```