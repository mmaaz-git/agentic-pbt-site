# Bug Report: xarray.core.formatting maybe_truncate and pretty_print violate length constraints

**Target**: `xarray.core.formatting.maybe_truncate` and `xarray.core.formatting.pretty_print`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`maybe_truncate()` can return strings longer than `maxlen` when `maxlen < 3`, and `pretty_print()` can return strings with length different from `numchars`, violating their documented behavior.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.core.formatting import maybe_truncate, pretty_print

@given(st.text(), st.integers(min_value=1, max_value=1000))
def test_maybe_truncate_at_most_maxlen(text, maxlen):
    result = maybe_truncate(text, maxlen)
    assert len(result) <= maxlen

@given(st.text(), st.integers(min_value=1, max_value=1000))
def test_pretty_print_produces_exact_length(text, numchars):
    result = pretty_print(text, numchars)
    assert len(result) == numchars
```

**Failing inputs**:
- `maybe_truncate('00', maxlen=1)` returns `'...'` (length 3, expected <= 1)
- `pretty_print('00', numchars=1)` returns `'...'` (length 3, expected exactly 1)

## Reproducing the Bug

```python
from xarray.core.formatting import maybe_truncate, pretty_print

result = maybe_truncate('00', maxlen=1)
print(f"{result!r} has length {len(result)}, expected <= 1")

result = maybe_truncate('hello world', maxlen=2)
print(f"{result!r} has length {len(result)}, expected <= 2")

result = pretty_print('00', numchars=1)
print(f"{result!r} has length {len(result)}, expected exactly 1")

result = pretty_print('hello', numchars=2)
print(f"{result!r} has length {len(result)}, expected exactly 2")
```

## Why This Is A Bug

1. **`maybe_truncate()` violates its implicit contract**: When a string is truncated, the result should be at most `maxlen` characters. However, when `len(s) > maxlen` and `maxlen < 3`, the function returns `s[:(maxlen - 3)] + "..."` which has length at least 3, exceeding `maxlen`.

2. **`pretty_print()` violates its documented behavior**: The docstring states "format the returned string so that it is numchars long", but it can return strings of different length when `numchars < 3` because it relies on `maybe_truncate()` which has the bug above.

## Fix

```diff
--- a/xarray/core/formatting.py
+++ b/xarray/core/formatting.py
@@ -49,7 +49,11 @@ def pretty_print(x, numchars: int):

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

This ensures that `maybe_truncate()` never returns a string longer than `maxlen`, which in turn fixes `pretty_print()` to always return a string of exactly `numchars` length.