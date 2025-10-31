# Bug Report: Cython.Shadow.index_type Step Validation Failure

**Target**: `Cython.Shadow.index_type`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `index_type` function fails to properly validate that step is provided only once in multi-dimensional array type specifications, allowing invalid specifications like `double[:, :]` (step in both dimensions) to pass validation.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from Cython.Shadow import index_type
import pytest


def test_index_type_tuple_with_multiple_steps_fails():
    """Step should only be provided once, not in multiple dimensions"""
    slices = (slice(None, None, 1), slice(None, None, 1))
    with pytest.raises(Exception):
        index_type(int, slices)
```

**Failing input**: `slices = (slice(None, None, 1), slice(None, None, 1))`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Shadow import index_type

slices_2d = (slice(None, None, 1), slice(None, None, 1))
result = index_type(int, slices_2d)

slices_3d = (slice(None, None, 1), slice(None, None, None), slice(None, None, 1))
result = index_type(int, slices_3d)
```

Both calls succeed when they should raise `InvalidTypeSpecification` with the message "Step may only be provided once, and only in the first or last dimension."

## Why This Is A Bug

The error message in the code states: "Step may only be provided once, and only in the first or last dimension." However, the validation logic at line 52 uses a truthiness check on `step_idx`:

```python
if s.step and (step_idx or idx not in (0, len(item) - 1)):
    raise InvalidTypeSpecification(...)
```

When `step_idx = 0` (step provided in first dimension, index 0), the expression `step_idx or ...` evaluates to `False or ...` because 0 is falsy in Python. This allows a second step to be provided in the last dimension without raising an exception.

For a 2D array with steps in both dimensions:
- Iteration 0: step_idx=None, check passes (no raise), then step_idx=0
- Iteration 1: `step_idx or idx not in (0,1)` = `0 or False` = `False` (no raise!)

The intended check should use `step_idx is not None` to properly detect when a step has already been provided.

## Fix

```diff
--- a/Shadow.py
+++ b/Shadow.py
@@ -49,7 +49,7 @@ def index_type(base_type, item):
         step_idx = None
         for idx, s in enumerate(item):
             verify_slice(s)
-            if s.step and (step_idx or idx not in (0, len(item) - 1)):
+            if s.step and (step_idx is not None or idx not in (0, len(item) - 1)):
                 raise InvalidTypeSpecification(
                     "Step may only be provided once, and only in the "
                     "first or last dimension.")
```