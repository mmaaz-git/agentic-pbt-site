# Bug Report: numpy.strings.mod Empty Tuple Handling

**Target**: `numpy.strings.mod`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.strings.mod` incorrectly handles empty tuple formatting arguments, returning an empty array instead of the original strings unchanged. When the `values` parameter is an empty tuple `()`, the function should return the format strings unchanged (matching Python's `%` operator), but instead returns an empty array.

## Property-Based Test

```python
import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st

@given(st.lists(st.text(alphabet='abc', min_size=0, max_size=10), min_size=1, max_size=5))
def test_mod_no_format_unchanged(strings):
    arr = np.array(strings)

    for i, s in enumerate(strings):
        if '%' not in s:
            result = nps.mod(arr[i:i+1], ())
            assert result[0] == s
```

**Failing input**: `strings=['']`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

arr = np.array([''])
result = nps.mod(arr, ())
print(f"Input: {arr}")
print(f"Output: {result}")
print(f"Expected: {arr}")
print(f"Python: '' % () = {repr('' % ())}")

arr = np.array(['hello'])
result = nps.mod(arr, ())
print(f"\nInput: {arr}")
print(f"Output: {result}")
print(f"Expected: {arr}")
print(f"Python: 'hello' % () = {repr('hello' % ())}")
```

Output:
```
Input: ['']
Output: []
Expected: ['']
Python: '' % () = ''

Input: ['hello']
Output: []
Expected: ['hello']
Python: 'hello' % () = 'hello'
```

## Why This Is A Bug

1. **Violates documented behavior**: The docstring states that `mod` implements "pre-Python 2.6 string formatting (interpolation)", which should match Python's `%` operator behavior
2. **Data loss**: The function returns an empty array, losing all input data
3. **Inconsistent with Python**: Python's `str % ()` returns the string unchanged, but numpy returns an empty array
4. **Shape mismatch for multi-element arrays**: For arrays with more than one element, the function crashes with a ValueError about shape mismatch

## Fix

The issue appears to be in how the function handles empty tuples during broadcasting. The fix would likely involve:

1. Special-casing the empty tuple scenario to return the input unchanged
2. Or ensuring that the broadcasting logic correctly handles zero-length value arrays

```diff
--- a/numpy/_core/strings.py
+++ b/numpy/_core/strings.py
@@ -265,6 +265,9 @@ def mod(a, values):
     """
     a = np.asanyarray(a)
+    # Handle empty tuple case - no formatting needed
+    if isinstance(values, tuple) and len(values) == 0:
+        return a.copy()
     return _to_bytes_or_str_array(
         _vec_string(a, np.object_, '__mod__', (values,)), a)
```

Note: This is a suggested fix. The actual implementation may need to handle more edge cases depending on how the broadcasting logic works internally.