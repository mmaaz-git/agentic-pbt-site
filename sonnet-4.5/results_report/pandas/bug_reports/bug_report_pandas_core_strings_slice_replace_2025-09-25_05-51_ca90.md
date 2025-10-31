# Bug Report: pandas.core.strings.accessor.StringMethods.slice_replace - Incorrect behavior when start >= stop

**Target**: `pandas.core.strings.accessor.StringMethods.slice_replace`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`slice_replace()` deviates from documented behavior and Python slicing semantics when `start >= stop`. Instead of following the documented pattern `s[:start] + repl + s[stop:]`, it uses `s[:start] + repl + s[start:]`, effectively ignoring the `stop` parameter.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings


@given(st.lists(st.text(min_size=1), min_size=1, max_size=100), st.integers(min_value=0, max_value=10), st.integers(min_value=0, max_value=10), st.text())
@settings(max_examples=1000)
def test_slice_replace_consistency(strings, start, stop, repl):
    s = pd.Series(strings)
    replaced = s.str.slice_replace(start, stop, repl)

    for i in range(len(s)):
        if pd.notna(s.iloc[i]) and pd.notna(replaced.iloc[i]):
            expected = s.iloc[i][:start] + repl + s.iloc[i][stop:]
            assert replaced.iloc[i] == expected
```

**Failing input**: `strings=['hello'], start=1, stop=0, repl='X'`

## Reproducing the Bug

```python
import pandas as pd

s = pd.Series(['hello'])
result = s.str.slice_replace(start=1, stop=0, repl='X')

print(f"Result: {result.iloc[0]}")
print(f"Expected: {s.iloc[0][:1] + 'X' + s.iloc[0][0:]}")

assert result.iloc[0] == 'hXhello', f"Bug: got '{result.iloc[0]}' instead of 'hXhello'"
```

Output:
```
Result: hXello
Expected: hXhello
AssertionError: Bug: got 'hXello' instead of 'hXhello'
```

## Why This Is A Bug

The documentation states: "the slice from `start` to `stop` is replaced with `repl`", which implies the operation should be `s[:start] + repl + s[stop:]` following standard Python slicing semantics.

However, the implementation in `pandas/core/strings/object_array.py` contains special-case logic:

```python
if x[start:stop] == "":
    local_stop = start
else:
    local_stop = stop
```

When `start > stop` (or `start == stop`), the slice `x[start:stop]` is empty, triggering this special case. This changes the behavior to `s[:start] + repl + s[start:]`, which is an insertion at `start` rather than the documented slice replacement.

This violates:
1. The API documentation
2. Python's standard slicing semantics
3. User expectations based on the method name

## Fix

```diff
--- a/pandas/core/strings/object_array.py
+++ b/pandas/core/strings/object_array.py
@@ -XXX,11 +XXX,6 @@ class ObjectStringArrayMixin(BaseStringArrayMethods):
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