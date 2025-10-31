# Bug Report: numpy.strings.index - Inconsistent ValueError Behavior

**Target**: `numpy.strings.index` and `numpy.strings.rindex`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`numpy.strings.index` and `numpy.strings.rindex` raise `ValueError` when ANY element in the input array doesn't contain the substring, making them unusable for vectorized operations with mixed results. This is inconsistent with `find`/`rfind` which return -1 for elements where the substring is not found.

## Property-Based Test

```python
import numpy as np
import numpy.strings as ns
from hypothesis import given, settings, strategies as st

string_arrays_with_substring = st.lists(st.text(), min_size=1, max_size=10).flatmap(
    lambda strings: st.tuples(st.just(np.array(strings)), st.sampled_from(strings))
)

@given(string_arrays_with_substring)
@settings(max_examples=500)
def test_find_index_consistency(arr_and_sub):
    arr, sub = arr_and_sub
    find_result = ns.find(arr, sub)

    for i in range(len(arr)):
        if find_result[i] >= 0:
            try:
                index_result = ns.index(arr, sub)
                assert find_result[i] == index_result[i]
            except ValueError:
                assert False, f"find returned {find_result[i]} but index raised ValueError"
```

**Failing input**: `arr = np.array(['0', ''])`, `sub = '0'`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as ns

arr = np.array(['0', ''])
sub = '0'

find_result = ns.find(arr, sub)
print(f"find(arr, '0'): {find_result}")

try:
    index_result = ns.index(arr, sub)
    print(f"index(arr, '0'): {index_result}")
except ValueError as e:
    print(f"index(arr, '0'): raises ValueError: {e}")
    print("BUG: find succeeds and returns [0, -1], but index raises ValueError")
```

**Output:**
```
find(arr, '0'): [ 0 -1]
index(arr, '0'): raises ValueError: substring not found
BUG: find succeeds and returns [0, -1], but index raises ValueError
```

The same bug exists for `rindex`:

```python
rfind_result = ns.rfind(arr, sub)
print(f"rfind(arr, '0'): {rfind_result}")

try:
    rindex_result = ns.rindex(arr, sub)
    print(f"rindex(arr, '0'): {rindex_result}")
except ValueError as e:
    print(f"rindex(arr, '0'): raises ValueError: {e}")
```

## Why This Is A Bug

The documentation states that `index` is "Like `find`, but raises `ValueError` when the substring is not found." However, in a vectorized context, this behavior is problematic:

1. **Inconsistency**: `find` returns -1 for elements where the substring is not found, allowing vectorized processing. `index` raises `ValueError` if ANY element doesn't contain the substring, even when other elements do.

2. **Breaks vectorization**: Users cannot use `index` for element-wise operations on arrays with mixed results. The function only works when ALL elements contain the substring.

3. **Contract violation**: The function returns "Output array of ints" according to its docstring, but raises an exception instead when processing multiple elements with mixed results.

**Expected behavior**: Either:
- Option A: Return an array like `find`, and let users handle -1 values (or raise for individual element access)
- Option B: Return an array with valid indices for found substrings and raise ValueError only when accessing elements that failed
- Option C: Document clearly that the function is only for arrays where all elements contain the substring

**Current behavior**: Raises ValueError for the entire array operation if any single element doesn't contain the substring, making the function practically unusable for vectorized operations.

## Fix

The underlying `_index_ufunc` and `_rindex_ufunc` functions need to be modified to handle the "not found" case element-wise rather than raising a ValueError for the entire array. The fix would likely be in the C/C++ implementation of these ufuncs in NumPy's core.

Alternatively, the Python wrappers could catch the ValueError and convert it to return -1 for elements where the substring is not found, matching the behavior of `find`/`rfind`:

```diff
@set_module("numpy.strings")
def index(a, sub, start=0, end=None):
    """
    Like `find`, but raises :exc:`ValueError` when the substring is not found.

    ...
    """
    end = end if end is not None else MAX
-   return _index_ufunc(a, sub, start, end)
+   result = _find_ufunc(a, sub, start, end)
+   if np.any(result < 0):
+       raise ValueError("substring not found")
+   return result
```

However, this would change the behavior to raise ValueError only when checking the result, not during the operation itself. A better fix would be at the ufunc level to allow element-wise error handling.