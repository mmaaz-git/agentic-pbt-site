# Bug Report: scipy.spatial.distance.is_valid_dm TypeError in Error Message Formatting

**Target**: `scipy.spatial.distance.is_valid_dm`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_valid_dm` function crashes with `TypeError` instead of raising the intended `ValueError` when validating a matrix with non-zero diagonal values, tolerance > 0, and `name=None`.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.spatial.distance import is_valid_dm


@given(st.integers(min_value=2, max_value=20))
@settings(max_examples=100)
def test_is_valid_dm_error_handling_without_name(n):
    mat = np.eye(n) * 5.0 + np.ones((n, n))

    try:
        is_valid_dm(mat, tol=0.1, throw=True, name=None)
        assert False, "Should raise ValueError for non-zero diagonal"
    except ValueError:
        pass


if __name__ == "__main__":
    test_is_valid_dm_error_handling_without_name()
```

<details>

<summary>
**Failing input**: `n=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 19, in <module>
    test_is_valid_dm_error_handling_without_name()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 7, in test_is_valid_dm_error_handling_without_name
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 12, in test_is_valid_dm_error_handling_without_name
    is_valid_dm(mat, tol=0.1, throw=True, name=None)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py", line 2610, in is_valid_dm
    raise ValueError(('Distance matrix \'{}\' diagonal must be close '
                     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                      'to zero within tolerance {:5.5f}.').format(*tol))
                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
TypeError: str.format() argument after * must be an iterable, not float
Falsifying example: test_is_valid_dm_error_handling_without_name(
    n=2,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial.distance import is_valid_dm

# Create a 2x2 matrix with non-zero diagonal
mat = np.array([[5.0, 1.0], [1.0, 5.0]])

# This should raise ValueError but raises TypeError instead
# when name=None and tol > 0
is_valid_dm(mat, tol=0.1, throw=True, name=None)
```

<details>

<summary>
TypeError: str.format() argument after * must be an iterable, not float
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/40/repo.py", line 9, in <module>
    is_valid_dm(mat, tol=0.1, throw=True, name=None)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py", line 2610, in is_valid_dm
    raise ValueError(('Distance matrix \'{}\' diagonal must be close '
                     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                      'to zero within tolerance {:5.5f}.').format(*tol))
                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
TypeError: str.format() argument after * must be an iterable, not float
```
</details>

## Why This Is A Bug

The function is documented to validate distance matrices and raise an exception when `throw=True` and validation fails. According to the code structure, all validation errors should raise `ValueError` with descriptive error messages. However, at lines 2610-2611 of `distance.py`, when `name=None` and the diagonal exceeds tolerance, the code contains a string formatting error:

```python
raise ValueError(('Distance matrix \'{}\' diagonal must be close '
                  'to zero within tolerance {:5.5f}.').format(*tol))
```

The `*tol` attempts to unpack `tol` as an iterable, but `tol` is a float value. This causes Python to raise `TypeError` instead of the intended `ValueError`. The function works correctly when a name is provided (line 2607-2608) but fails in this specific code path.

This violates the function's contract because:
1. The documentation states that "An exception is thrown if the distance matrix passed is not valid" - implying a validation exception, not a string formatting error
2. All other validation failures in the function correctly raise `ValueError`
3. The function behaves inconsistently based on whether `name` is provided or not

## Relevant Context

The `is_valid_dm` function performs three main validation checks on distance matrices:
1. Must be 2-dimensional (square matrix)
2. Must be symmetric (within tolerance)
3. Must have zeros on the diagonal (within tolerance)

The bug occurs only in the specific case where:
- `name=None` (or empty string)
- `tol > 0` (tolerance is specified)
- The diagonal values exceed the tolerance

The function has 12 different error paths, and this is the only one with incorrect string formatting. Comparing with the parallel code path when `name` is provided (lines 2607-2608), we can see the correct formatting approach.

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.is_valid_dm.html
Source code: scipy/spatial/distance.py, lines 2569-2618

## Proposed Fix

```diff
--- a/scipy/spatial/distance.py
+++ b/scipy/spatial/distance.py
@@ -2607,8 +2607,8 @@ def is_valid_dm(D, tol=0.0, throw=False, name="D", warning=False):
                     raise ValueError(f'Distance matrix \'{name}\' diagonal must be '
                                      f'close to zero within tolerance {tol:5.5f}.')
                 else:
-                    raise ValueError(('Distance matrix \'{}\' diagonal must be close '
-                                      'to zero within tolerance {:5.5f}.').format(*tol))
+                    raise ValueError(f'Distance matrix diagonal must be close '
+                                     f'to zero within tolerance {tol:5.5f}.')
     except Exception as e:
         if throw:
```