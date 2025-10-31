# Bug Report: pandas Series.str.slice_replace data loss when start > stop

**Target**: `pandas.core.strings.object_array.ObjectStringArrayMixin._str_slice_replace`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Series.str.slice_replace()` method silently loses data when `start > stop`, incorrectly treating the empty slice case.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings

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

**Failing input**: `strings=['0'], start=1, stop=0`

## Reproducing the Bug

```python
import pandas as pd

s = pd.Series(['abc'])

result = s.str.slice_replace(start=2, stop=1, repl='X').iloc[0]
expected = 'abc'[:2] + 'X' + 'abc'[1:]

print(f"Result:   {result!r}")
print(f"Expected: {expected!r}")
print(f"Match: {result == expected}")
```

Output:
```
Result:   'abXc'
Expected: 'abXbc'
Match: False
```

## Why This Is A Bug

When `start > stop`, the slice `s[start:stop]` is an empty string (valid Python slicing). Replacing an empty slice should insert the replacement string at that position while preserving all original characters: `s[:start] + repl + s[stop:]`.

However, the current implementation loses characters between `stop` and `start` when the slice is empty. The character 'b' at position 1 is lost in the example above.

## Fix

```diff
--- a/pandas/core/strings/object_array.py
+++ b/pandas/core/strings/object_array.py
@@ -XXX,11 +XXX,7 @@ class ObjectStringArrayMixin:
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
```