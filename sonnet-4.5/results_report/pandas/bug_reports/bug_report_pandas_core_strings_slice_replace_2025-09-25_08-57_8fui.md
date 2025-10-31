# Bug Report: pandas Series.str.slice_replace Data Loss

**Target**: `pandas.core.strings.object_array.ObjectStringArrayMixin._str_slice_replace`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Series.str.slice_replace()` method silently loses data when `start > stop`, causing characters between `stop` and `start` to be deleted from the result.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd

@given(
    st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=20),
    st.integers(min_value=-10, max_value=10),
    st.integers(min_value=-10, max_value=10)
)
@settings(max_examples=1000)
def test_slice_replace_consistency(strings, start, stop):
    s = pd.Series(strings)
    replaced = s.str.slice_replace(start, stop, 'X')

    for orig_str, repl in zip(strings, replaced):
        if start is None:
            actual_start = 0
        elif start < 0:
            actual_start = max(0, len(orig_str) + start)
        else:
            actual_start = start

        if stop is None:
            actual_stop = len(orig_str)
        elif stop < 0:
            actual_stop = max(0, len(orig_str) + stop)
        else:
            actual_stop = stop

        expected_repl = orig_str[:actual_start] + 'X' + orig_str[actual_stop:]
        assert repl == expected_repl
```

**Failing input**: `strings=['abc'], start=2, stop=1`

## Reproducing the Bug

```python
import pandas as pd

s = pd.Series(['abc'])
result = s.str.slice_replace(start=2, stop=1, repl='X').iloc[0]
expected = 'abc'[:2] + 'X' + 'abc'[1:]

print(f"Result:   {result!r}")
print(f"Expected: {expected!r}")
```

Output:
```
Result:   'abXc'
Expected: 'abXbc'
```

The character 'b' at index 1 is incorrectly removed.

## Why This Is A Bug

When `start > stop`, the slice `s[start:stop]` is an empty string (valid Python slicing behavior). Replacing an empty slice should insert the replacement while preserving all original characters: `s[:start] + repl + s[stop:]`.

This bug commonly occurs with negative indices. For example:
```python
s = pd.Series(['hello'])
s.str.slice_replace(-1, -3, 'X')
```
Results in `'hellXo'` (losing "ll") instead of `'hellXllo'`.

The current implementation incorrectly uses `start` as the stop position when the slice is empty, causing data loss.

## Fix

The bug is in `ObjectStringArrayMixin._str_slice_replace()`:

```diff
--- a/pandas/core/strings/object_array.py
+++ b/pandas/core/strings/object_array.py
@@ -XXX,XX +XXX,XX @@ class ObjectStringArrayMixin:
     def _str_slice_replace(self, start=None, stop=None, repl=None):
         if repl is None:
             repl = ""

         def f(x):
-            if x[start:stop] == "":
-                local_stop = start
-            else:
-                local_stop = stop
             y = ""
             if start is not None:
                 y += x[:start]
             y += repl
             if stop is not None:
-                y += x[local_stop:]
+                y += x[stop:]
             return y

         return self._str_map(f)
```

The fix removes the special case handling for empty slices. Python's slicing already handles `start > stop` correctly (returns empty string), so the replacement logic should simply use: `x[:start] + repl + x[stop:]`