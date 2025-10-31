# Bug Report: pandas.core.array_algos.putmask_without_repeat Allows Repetition with Length-1 Arrays

**Target**: `pandas.core.array_algos.putmask.putmask_without_repeat`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `putmask_without_repeat` function violates its documented contract by allowing numpy's `putmask` to repeat values when `new` has length 1, despite the documentation stating "We require an exact match."

## Property-Based Test

```python
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume
from pandas.core.array_algos.putmask import putmask_without_repeat


@given(
    values_size=st.integers(min_value=10, max_value=50),
    new_size=st.integers(min_value=1, max_value=9)
)
@settings(max_examples=300)
def test_putmask_without_repeat_length_mismatch_error(values_size, new_size):
    assume(new_size != values_size)

    values = np.arange(values_size)
    mask = np.ones(values_size, dtype=bool)
    new = np.arange(new_size)

    with pytest.raises(ValueError, match="cannot assign mismatch"):
        putmask_without_repeat(values, mask, new)
```

**Failing input**: `values_size=10, new_size=1`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.array_algos.putmask import putmask_without_repeat

values = np.arange(10)
mask = np.ones(10, dtype=bool)
new = np.array([999])

putmask_without_repeat(values, mask, new)
print(values)
```

**Output**: `[999 999 999 999 999 999 999 999 999 999]`

The function did not raise `ValueError` despite the length mismatch (1 vs 10 masked positions), and instead repeated the value 999, which is exactly the behavior the function was designed to prevent.

## Why This Is A Bug

The function's docstring explicitly states:
> `np.putmask will truncate or repeat if new is a listlike with len(new) != len(values). We require an exact match.`

However, the implementation contains a special case at line 93:
```python
elif mask.shape[-1] == shape[-1] or shape[-1] == 1:
    np.putmask(values, mask, new)
```

When `shape[-1] == 1`, the function calls `np.putmask`, which will repeat the single value across all masked positions. This violates the "exact match" requirement stated in the documentation. The function should either:
1. Raise a `ValueError` when `len(new) != nlocs` (number of masked positions), or
2. Update its documentation to clarify that length-1 arrays are allowed and will be broadcast

## Fix

Remove the `shape[-1] == 1` special case from the condition, so that length-1 arrays are also subject to the exact match requirement:

```diff
--- a/pandas/core/array_algos/putmask.py
+++ b/pandas/core/array_algos/putmask.py
@@ -90,7 +90,7 @@ def putmask_without_repeat(
             np.place(values, mask, new)
             # i.e. values[mask] = new
-        elif mask.shape[-1] == shape[-1] or shape[-1] == 1:
+        elif mask.shape[-1] == shape[-1]:
             np.putmask(values, mask, new)
         else:
             raise ValueError("cannot assign mismatch length to masked array")
```

This ensures that when `new` is a 1-element array but there are more than 1 masked positions, a `ValueError` is raised as documented.