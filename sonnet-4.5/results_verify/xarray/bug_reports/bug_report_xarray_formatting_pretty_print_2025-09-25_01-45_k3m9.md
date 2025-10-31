# Bug Report: xarray.core.formatting.pretty_print Length Contract Violation

**Target**: `xarray.core.formatting.pretty_print`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `pretty_print` function violates its documented contract to return a string of exactly `numchars` length when the input string needs truncation for small `numchars` values.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import xarray.core.formatting as fmt

@given(st.integers(min_value=1, max_value=100))
@settings(max_examples=1000)
def test_pretty_print_length(numchars):
    obj = "test"
    result = fmt.pretty_print(obj, numchars)
    assert len(result) == numchars
```

**Failing input**: `numchars=1`

## Reproducing the Bug

```python
import xarray.core.formatting as fmt

obj = "test"
numchars = 1

result = fmt.pretty_print(obj, numchars)

print(f"Input: obj='{obj}', numchars={numchars}")
print(f"Output: '{result}'")
print(f"Output length: {len(result)}")

assert len(result) == numchars
```

Output:
```
Input: obj='test', numchars=1
Output: 'te...'
Output length: 5
AssertionError: Expected length 1, got 5
```

## Why This Is A Bug

The function `pretty_print` has a clear contract in its docstring: "format the returned string so that it is numchars long". However, when `numchars` is small (1, 2), the function returns strings longer than `numchars`.

The root cause is in `maybe_truncate`, which unconditionally appends "..." (3 characters) when truncating, even if `maxlen` is less than 3. This causes the truncated string to exceed `maxlen`.

## Fix

```diff
--- a/xarray/core/formatting.py
+++ b/xarray/core/formatting.py
@@ -49,7 +49,10 @@ def pretty_print(x, numchars: int):

 def maybe_truncate(obj, maxlen=500):
     s = str(obj)
     if len(s) > maxlen:
-        s = s[: (maxlen - 3)] + "..."
+        if maxlen < 3:
+            s = s[:maxlen]
+        else:
+            s = s[: (maxlen - 3)] + "..."
     return s
```