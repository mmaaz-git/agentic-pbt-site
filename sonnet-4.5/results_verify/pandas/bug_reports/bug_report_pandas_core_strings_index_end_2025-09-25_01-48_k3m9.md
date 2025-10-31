# Bug Report: pandas.core.strings._str_index and _str_rindex Incorrect end Parameter Handling

**Target**: `pandas.core.strings.object_array.ObjectStringArrayMixin._str_index` and `_str_rindex`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_str_index` and `_str_rindex` methods in `object_array.py` have identical code in both branches of their conditional, causing incorrect behavior when `end=0` or `end=False` is passed. The methods should omit the `end` parameter when `end is None`, but currently pass it unconditionally in the else branch.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings, assume


@given(st.text(min_size=10), st.text(min_size=1))
@settings(max_examples=500, deadline=None)
def test_str_index_end_zero_behavior(text, sub):
    assume(len(text) >= 10)
    assume(sub in text)

    s = pd.Series([text])

    python_result = None
    python_error = None
    try:
        python_result = text.index(sub, 0, 0)
    except ValueError as e:
        python_error = str(e)

    pandas_result = None
    pandas_error = None
    try:
        pandas_result = s.str.index(sub, start=0, end=0).iloc[0]
    except ValueError as e:
        pandas_error = str(e)

    assert (python_error is None) == (pandas_error is None), \
        f"Behavior mismatch: Python {'raised' if python_error else 'returned'} vs Pandas {'raised' if pandas_error else 'returned'}"

    if python_result is not None and pandas_result is not None:
        assert python_result == pandas_result
```

**Failing input**: Any string where the substring exists, e.g., `text="hello world"`, `sub="world"`, `end=0`

## Reproducing the Bug

```python
import pandas as pd

s = pd.Series(['hello world'])

print("Python behavior with end=0:")
try:
    result = 'hello world'.index('world', 0, 0)
    print(f"  Found at index: {result}")
except ValueError as e:
    print(f"  ValueError (expected): {e}")

print("\nPandas behavior with end=0:")
try:
    result = s.str.index('world', start=0, end=0).iloc[0]
    print(f"  Found at index: {result}")
    print("  BUG: Should raise ValueError when searching in empty range [0:0]")
except ValueError as e:
    print(f"  ValueError: {e}")
```

## Why This Is A Bug

Lines 316-321 in `object_array.py`:
```python
def _str_index(self, sub, start: int = 0, end=None):
    if end:                                      # line 317
        f = lambda x: x.index(sub, start, end)   # line 318
    else:                                        # line 319
        f = lambda x: x.index(sub, start, end)   # line 320 - IDENTICAL TO LINE 318!
    return self._str_map(f, dtype="int64")
```

Both branches (lines 318 and 320) are identical, making the conditional pointless. This causes incorrect behavior when `end` is a falsy value other than `None`:
- `end=0`: Should search in empty range `[0:0]` but instead searches in `[start:None]`
- `end=False`: Treated as 0 by Python, same issue

The correct pattern is shown in `_str_find_` (lines 296-299), which properly checks `if end is None:` and omits the parameter in that case.

Same bug exists in `_str_rindex` (lines 323-328).

## Fix

```diff
--- a/pandas/core/strings/object_array.py
+++ b/pandas/core/strings/object_array.py
@@ -314,10 +314,10 @@ class ObjectStringArrayMixin(BaseStringArrayMethods):
         return self._str_map(f)

     def _str_index(self, sub, start: int = 0, end=None):
-        if end:
+        if end is not None:
             f = lambda x: x.index(sub, start, end)
         else:
-            f = lambda x: x.index(sub, start, end)
+            f = lambda x: x.index(sub, start)
         return self._str_map(f, dtype="int64")

     def _str_rindex(self, sub, start: int = 0, end=None):
-        if end:
+        if end is not None:
             f = lambda x: x.rindex(sub, start, end)
         else:
-            f = lambda x: x.rindex(sub, start, end)
+            f = lambda x: x.rindex(sub, start)
         return self._str_map(f, dtype="int64")
```