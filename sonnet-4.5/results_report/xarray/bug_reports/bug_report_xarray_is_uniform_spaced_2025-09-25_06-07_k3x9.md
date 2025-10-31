# Bug Report: xarray.core.utils.is_uniform_spaced ValueError on Small Arrays

**Target**: `xarray.core.utils.is_uniform_spaced`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_uniform_spaced` function crashes with ValueError when passed arrays with fewer than 2 elements, despite not documenting this restriction and the question being mathematically meaningful (vacuously true).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from xarray.core.utils import is_uniform_spaced
import numpy as np

@given(n=st.integers(min_value=0, max_value=100))
@settings(max_examples=300)
def test_linspace_always_uniform(n):
    arr = np.linspace(0, 10, n)
    result = is_uniform_spaced(arr)
    assert result == True, f"linspace with {n} points should be uniformly spaced"

@given(size=st.integers(min_value=0, max_value=2))
@settings(max_examples=100)
def test_small_arrays_dont_crash(size):
    arr = list(range(size))
    result = is_uniform_spaced(arr)
    assert isinstance(result, bool)
```

**Failing inputs**:
- Empty array: `[]`
- Single element array: `[1]`
- Any array with length < 2

## Reproducing the Bug

```python
from xarray.core.utils import is_uniform_spaced

result = is_uniform_spaced([])
```

**Output:**
```
ValueError: zero-size array to reduction operation minimum which has no identity
```

**Another failing case:**
```python
from xarray.core.utils import is_uniform_spaced

result = is_uniform_spaced([5])
```

**Output:**
```
ValueError: zero-size array to reduction operation minimum which has no identity
```

## Why This Is A Bug

1. **Undocumented restriction**: The function's docstring does not mention any minimum array size requirement:
   ```python
   def is_uniform_spaced(arr, **kwargs) -> bool:
       """Return True if values of an array are uniformly spaced and sorted.
       ...
       """
   ```

2. **Mathematically meaningful**: Asking whether an empty array or single-element array is "uniformly spaced" has a clear answer (vacuously true - there are no pairs of adjacent elements that have different spacing).

3. **Reasonable use case**: This function may be called on arrays of varying sizes in dynamic contexts. Users shouldn't need to add special-case checks for small arrays before calling what appears to be a general-purpose utility function.

4. **Inconsistent with similar functions**: Many numpy and pandas functions handle edge cases like empty arrays gracefully rather than crashing.

5. **`np.linspace` compatibility**: `np.linspace(0, 10, 0)` and `np.linspace(0, 10, 1)` are valid calls that produce arrays of length 0 and 1, but passing these results to `is_uniform_spaced` crashes.

## Fix

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

**Rationale for the fix:**
- Arrays with 0 or 1 elements are vacuously uniformly spaced (no adjacent pairs exist to violate uniformity)
- This matches the behavior for length-2 arrays (always uniformly spaced, since there's only one interval)
- The fix is minimal and preserves all existing behavior for arrays with ≥ 2 elements