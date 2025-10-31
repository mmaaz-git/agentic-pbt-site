# Bug Report: xarray.core.utils.is_uniform_spaced Crashes on Single-Element Arrays

**Target**: `xarray.core.utils.is_uniform_spaced`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_uniform_spaced` function crashes with a `ValueError` when given a single-element array or an empty array, due to calling `.min()` and `.max()` on an empty differences array.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from hypothesis import given, strategies as st, settings
from xarray.core.utils import is_uniform_spaced


@given(st.floats(allow_nan=False, allow_infinity=False))
@settings(max_examples=500)
def test_single_element_uniform(x):
    result = is_uniform_spaced([x])
    assert result, f"Single element array should be uniformly spaced: [{x}]"
```

**Failing input**: Any single value, e.g., `x=0.0`

## Reproducing the Bug

```python
from xarray.core.utils import is_uniform_spaced

result = is_uniform_spaced([5.0])
```

Output:
```
Traceback (most recent call last):
  File "test.py", line 3, in <module>
    result = is_uniform_spaced([5.0])
  File "xarray/core/utils.py", line 737, in is_uniform_spaced
    return bool(np.isclose(diffs.min(), diffs.max(), **kwargs))
ValueError: zero-size array to reduction operation minimum which has no identity
```

## Why This Is A Bug

The function's docstring and examples don't restrict the input to arrays with 2+ elements. Single-element arrays and empty arrays are valid inputs for many array-processing functions, and should be handled gracefully.

Mathematically, a single-element array is trivially uniformly spaced (there are no gaps between elements). The current implementation fails because:
1. `np.diff([x])` returns an empty array `[]`
2. Calling `.min()` on an empty array raises a `ValueError`

The function should handle edge cases:
- Empty array: Could return True (vacuously true) or False (depending on convention)
- Single element: Should return True (trivially uniformly spaced)
- Two elements: Should always return True (only one gap, so it's uniform)

## Fix

Add explicit handling for arrays with fewer than 2 elements:

```diff
def is_uniform_spaced(arr, **kwargs) -> bool:
    """Return True if values of an array are uniformly spaced and sorted.

    >>> is_uniform_spaced(range(5))
    True
    >>> is_uniform_spaced([-4, 0, 100])
    False

    kwargs are additional arguments to ``np.isclose``
    """
    arr = np.array(arr, dtype=float)
+   if len(arr) < 2:
+       return True
    diffs = np.diff(arr)
    return bool(np.isclose(diffs.min(), diffs.max(), **kwargs))
```

This fix:
- Returns `True` for empty and single-element arrays (vacuously/trivially uniformly spaced)
- Maintains existing behavior for arrays with 2+ elements
- Prevents the crash from calling `.min()/.max()` on empty arrays