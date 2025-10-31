# Bug Report: pandas.core.strings slice_replace with start > stop

**Target**: `pandas.core.strings.object_array.ObjectStringArrayMixin._str_slice_replace`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `slice_replace` method produces incorrect results when `start > stop`, incorrectly handling the empty slice case and omitting parts of the original string that should be preserved.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings

@given(st.lists(st.text(min_size=1), min_size=1), st.integers(min_value=-10, max_value=10), st.integers(min_value=-10, max_value=10), st.text(max_size=5))
@settings(max_examples=1000)
def test_slice_replace_consistency_with_python(strings, start, stop, repl):
    s = pd.Series(strings)
    replaced = s.str.slice_replace(start, stop, repl)
    for orig, repl_result in zip(s, replaced):
        if pd.notna(orig):
            expected = orig[:start] + repl + orig[stop:]
            assert repl_result == expected
```

**Failing input**: `strings=['0'], start=1, stop=0, repl=''`

## Reproducing the Bug

```python
import pandas as pd

s = pd.Series(['0'])
result = s.str.slice_replace(start=1, stop=0, repl='')
print(result.iloc[0])

expected = '0'[:1] + '' + '0'[0:]
print(expected)

assert result.iloc[0] == expected
```

**Expected output**: `'00'` (string with '0' before position 1, empty replacement, and '0' from position 0 onward)

**Actual output**: `'0'`

Additional examples demonstrating the bug:
- `'hello'.slice_replace(3, 1, 'X')` → pandas: `'helXlo'`, expected: `'helXello'`
- `'abc'.slice_replace(2, 1, '')` → pandas: `'abc'`, expected: `'abbc'`
- `'test'.slice_replace(4, 2, 'XX')` → pandas: `'testXX'`, expected: `'testXXst'`

## Why This Is A Bug

When `start > stop`, Python's slicing semantics treat `x[start:stop]` as an empty slice. The slice_replace operation should insert the replacement at position `start` while preserving `x[:start]` and `x[stop:]`.

The current implementation incorrectly sets `local_stop = start` when detecting an empty slice, which causes it to use `x[start:]` instead of `x[stop:]`, thereby omitting the substring `x[stop:start]`.

## Fix

```diff
--- a/pandas/core/strings/object_array.py
+++ b/pandas/core/strings/object_array.py
@@ -xxx,10 +xxx,6 @@ def _str_slice_replace(self, start=None, stop=None, repl=None):
         repl = ""

     def f(x):
-        if x[start:stop] == "":
-            local_stop = start
-        else:
-            local_stop = stop
         y = ""
         if start is not None:
             y += x[:start]
         y += repl
         if stop is not None:
-            y += x[local_stop:]
+            y += x[stop:]
         return y
```