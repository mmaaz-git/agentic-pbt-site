# Bug Report: pandas.core.strings slice_replace with None boundaries

**Target**: `pandas.core.strings.object_array.ObjectStringArrayMixin._str_slice_replace`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_str_slice_replace` method incorrectly handles `None` values for `start` and `stop` parameters, returning an empty string instead of preserving the original string content outside the slice boundaries.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings


@given(
    strings=st.lists(st.text(), min_size=1),
    start=st.integers(min_value=-5, max_value=5) | st.none(),
    stop=st.integers(min_value=-5, max_value=10) | st.none(),
    repl=st.text()
)
@settings(max_examples=500)
def test_slice_replace_property(strings, start, stop, repl):
    s = pd.Series(strings)
    result = s.str.slice_replace(start, stop, repl)

    for i, string in enumerate(strings):
        if pd.isna(string):
            assert pd.isna(result.iloc[i])
            continue

        expected = string[:start] + repl + string[stop:]
        actual = result.iloc[i]

        assert actual == expected
```

**Failing input**: `strings=['0'], start=None, stop=None, repl=''`

## Reproducing the Bug

```python
import pandas as pd

s = pd.Series(['0'])
result = s.str.slice_replace(start=None, stop=None, repl='')
print(f"Result: {result.iloc[0]!r}")
print(f"Expected: {'0'[:None] + '' + '0'[None:]!r}")

s = pd.Series(['hello'])
result = s.str.slice_replace(start=None, stop=None, repl='')
print(f"Result: {result.iloc[0]!r}")
print(f"Expected: {'hello'[:None] + '' + 'hello'[None:]!r}")
```

## Why This Is A Bug

In Python, slicing with `None` boundaries means "use the default boundary". So `x[:None]` is the entire string up to the end, and `x[None:]` is the entire string from the beginning. Therefore, `x[:None] + repl + x[None:]` should equal `x + repl + x` when both start and stop are None.

However, the current implementation checks `if start is not None` and `if stop is not None` to decide whether to include the portions of the string outside the slice. This causes the function to skip adding the string content when the parameters are explicitly `None`, even though `None` is a valid and meaningful value for slice boundaries.

The bug is in lines 357-361 of `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/strings/object_array.py`:

```python
y = ""
if start is not None:
    y += x[:start]
y += repl
if stop is not None:
    y += x[local_stop:]
```

This should unconditionally perform the slicing operations since Python's slice notation handles `None` correctly.

## Fix

```diff
--- a/pandas/core/strings/object_array.py
+++ b/pandas/core/strings/object_array.py
@@ -354,11 +354,7 @@ class ObjectStringArrayMixin(BaseStringArrayMethods):
                 local_stop = start
             else:
                 local_stop = stop
-            y = ""
-            if start is not None:
-                y += x[:start]
-            y += repl
-            if stop is not None:
-                y += x[local_stop:]
+            y = x[:start] + repl + x[local_stop:]
             return y
```