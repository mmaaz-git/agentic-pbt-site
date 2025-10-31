# Bug Report: numpy.char.split/rsplit/splitlines Return Non-Iterable 0-d Arrays

**Target**: `numpy.char.split`, `numpy.char.rsplit`, `numpy.char.splitlines`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When `numpy.char.split()`, `rsplit()`, or `splitlines()` are called on scalar strings, they return 0-dimensional arrays that cannot be iterated, violating the documented behavior and user expectations. This makes these functions unusable in common iteration patterns.

## Property-Based Test

```python
import numpy.char as char
import numpy as np
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=1, max_size=20), st.text(min_size=1, max_size=5))
@settings(max_examples=1000)
def test_split_always_returns_iterable_array(s, sep):
    result = char.split(s, sep)
    assert isinstance(result, np.ndarray)

    try:
        list(result)
    except TypeError as e:
        raise AssertionError(
            f"split('{s}', '{sep}') returned non-iterable array "
            f"with shape {result.shape}: {e}"
        )
```

**Failing input**: Any scalar string, e.g., `s='test'`, `sep=','`

## Reproducing the Bug

```python
import numpy.char as char

result = char.split('a,b,c', ',')
print(f"Result: {result}")
print(f"Shape: {result.shape}")

try:
    for item in result:
        print(f"  {item}")
    print("SUCCESS: Can iterate")
except TypeError as e:
    print(f"ERROR: {e}")
```

Output:
```
Result: ['a', 'b', 'c']
Shape: ()
ERROR: iteration over a 0-d array
```

## Why This Is A Bug

1. **Violates documentation**: The docstring states "Returns: out : ndarray - Array of list objects". A 0-d array cannot be iterated to access these list objects.

2. **Inconsistent behavior**: When called on 1-d arrays, these functions return iterable 1-d arrays. When called on scalars, they return non-iterable 0-d arrays.

3. **Breaks user expectations**: Users expect split() to return an iterable collection of substrings, matching Python's `str.split()` behavior.

4. **Affects practical usage**: Common iteration patterns fail:
   ```python
   for word in char.split(text, ' '):
       process(word)
   ```

5. **Affects all three functions**: `split`, `rsplit`, and `splitlines` all have this bug.

## Fix

The functions should return 1-d arrays when called on scalar strings, consistent with their behavior on array inputs.

```diff
--- a/numpy/_core/defchararray.py
+++ b/numpy/_core/defchararray.py
@@ -split_function
     result = apply_split_function(a, sep, maxsplit)
+    if result.ndim == 0:
+        result = np.array([result.item()])
     return result
```