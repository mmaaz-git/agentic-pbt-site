# Bug Report: scipy.spatial.distance.is_valid_dm TypeError in Error Message Formatting

**Target**: `scipy.spatial.distance.is_valid_dm`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `is_valid_dm` is called with `name=None` or empty string and encounters a non-zero diagonal with `tol > 0`, it raises `TypeError` instead of the intended `ValueError` due to incorrect string formatting.

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
```

**Failing input**: `n=2`

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial.distance import is_valid_dm

mat = np.array([[5.0, 1.0], [1.0, 5.0]])
is_valid_dm(mat, tol=0.1, throw=True, name=None)
```

Output:
```
TypeError: str.format() argument after * must be an iterable, not float
```

## Why This Is A Bug

The function is supposed to raise `ValueError` with a descriptive error message when given an invalid distance matrix. Instead, when `name=None` and the diagonal check fails with `tol > 0`, it crashes with `TypeError` due to malformed string formatting at line 2610-2611:

```python
raise ValueError(('Distance matrix \'{}\' diagonal must be close '
                  'to zero within tolerance {:5.5f}.').format(*tol))
```

The `*tol` unpacking operator expects an iterable but `tol` is a float, causing the TypeError.

## Fix

```diff
--- a/scipy/spatial/distance.py
+++ b/scipy/spatial/distance.py
@@ -2607,7 +2607,7 @@ def is_valid_dm(D, tol=0.0, throw=False, name="D", warning=False):
                     raise ValueError(f'Distance matrix \'{name}\' diagonal must be '
                                      f'close to zero within tolerance {tol:5.5f}.')
                 else:
-                    raise ValueError(('Distance matrix \'{}\' diagonal must be close '
-                                      'to zero within tolerance {:5.5f}.').format(*tol))
+                    raise ValueError(('Distance matrix diagonal must be close '
+                                      'to zero within tolerance {:5.5f}.').format(tol))
     except Exception as e:
         if throw:
```