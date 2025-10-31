# Bug Report: scipy.spatial.transform.Rotation.mean() Segmentation Fault

**Target**: `scipy.spatial.transform.Rotation.mean`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Calling `Rotation.mean()` with a Python list containing a single Rotation object causes a segmentation fault, crashing the Python interpreter.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from scipy.spatial.transform import Rotation
import numpy as np

@st.composite
def quaternions(draw):
    q = draw(hnp.arrays(np.float64, (4,), elements=st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False)))
    assume(np.linalg.norm(q) > 1e-10)
    return q / np.linalg.norm(q)

@given(quaternions())
@settings(max_examples=200)
def test_rotation_mean_single(q):
    """Property: mean of single rotation should be itself"""
    r = Rotation.from_quat(q)
    r_mean = Rotation.mean([r])

    assert rotation_equal(r, r_mean, atol=1e-10), "Mean of single rotation is not itself"
```

**Failing input**: Any quaternion, e.g., `q = [0.0, 0.0, 0.0, 1.0]`

## Reproducing the Bug

```python
from scipy.spatial.transform import Rotation
import numpy as np

q = np.array([0.0, 0.0, 0.0, 1.0])
r = Rotation.from_quat(q)

r_mean = Rotation.mean([r])
```

**Expected behavior**: The mean of a single rotation should be that rotation itself.

**Actual behavior**: Segmentation fault - Python interpreter crashes immediately.

**Output**:
```
Fatal Python error: Segmentation fault
```

## Why This Is A Bug

1. **Crashes on valid input**: A list containing a single Rotation object is valid input according to the API. The function should either:
   - Handle it gracefully and return the rotation itself
   - Raise a clear ValueError stating minimum requirements

2. **Memory safety violation**: Segmentation faults indicate undefined behavior in the C/Cython extension code, likely dereferencing a null pointer or accessing invalid memory.

3. **Inconsistent with expectations**: Taking the mean of a single element should return that element, as is standard in statistics and NumPy.

## Additional Context

Testing reveals:
- `Rotation.mean([r1, r2])` with 2 or more rotations works correctly
- `Rotation.mean(r)` with a single Rotation object (not in a list) may also fail, but with a different error

The bug is in the compiled extension module `scipy.spatial.transform._rotation`, likely in the C/Cython implementation of the `mean` method.

## Fix

The fix should be implemented in the `_rotation.pyx` file (or corresponding C extension). The code should:

1. Add a check for the length of the input before processing
2. If length is 1, return a copy of the single rotation
3. If length is 0, raise a clear ValueError

The fix would look something like:

```python
# Pseudocode for the fix in _rotation.pyx
def mean(cls, rotations, weights=None):
    # Convert input to array of rotations
    rotations_array = _make_array_from_input(rotations)

    # Add this check:
    if len(rotations_array) == 0:
        raise ValueError("At least one rotation is required to compute the mean")

    if len(rotations_array) == 1:
        # Return a copy of the single rotation
        return rotations_array[0]

    # Continue with existing mean computation for 2+ rotations
    ...
```

Since this is in a compiled Cython extension, the actual fix requires modifying the Cython source and ensuring proper bounds checking before array access.