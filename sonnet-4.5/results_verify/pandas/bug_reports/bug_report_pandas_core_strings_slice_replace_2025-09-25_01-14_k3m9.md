# Bug Report: pandas.core.strings slice_replace empty slice handling

**Target**: `pandas.core.strings.object_array.ObjectStringArrayMixin._str_slice_replace`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `slice_replace` is called with `start > stop` (creating an empty slice), it incorrectly treats this as an insertion at position `start`, ignoring the `stop` parameter, instead of following the literal slice replacement semantics.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=1, max_size=10),
       st.integers(min_value=0, max_value=10),
       st.integers(min_value=0, max_value=10),
       st.text(max_size=5))
@settings(max_examples=1000)
def test_slice_replace_literal_interpretation(string, start, stop, repl):
    s = pd.Series([string], dtype=object)
    result = s.str.slice_replace(start, stop, repl)[0]
    expected = string[:start] + repl + string[stop:]
    assert result == expected
```

**Failing input**: `string='0', start=1, stop=0, repl=''`

## Reproducing the Bug

```python
import pandas as pd

s = pd.Series(['0'])
result = s.str.slice_replace(start=1, stop=0, repl='')
print(result[0])

s = pd.Series(['abcde'])
result = s.str.slice_replace(start=3, stop=1, repl='X')
print(result[0])
```

**Output:**
```
0
abcXde
```

**Expected:**
```
00
abcXbcde
```

## Why This Is A Bug

The function is documented as "Replace a positional slice of a string with another value". The natural interpretation is that `slice_replace(start, stop, repl)` should produce `string[:start] + repl + string[stop:]` for all values of start and stop, including when the slice `[start:stop]` is empty.

However, when `start > stop`, the implementation has special case logic that sets `local_stop = start` when the slice is empty (line 353 in object_array.py), effectively computing `string[:start] + repl + string[start:]` instead. This is inconsistent with:

1. The literal interpretation of the function signature
2. The behavior is undocumented
3. The Arrow string implementation doesn't have this special case

## Fix

```diff
--- a/pandas/core/strings/object_array.py
+++ b/pandas/core/strings/object_array.py
@@ -349,16 +349,10 @@ class ObjectStringArrayMixin(BaseStringArrayMethods):
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