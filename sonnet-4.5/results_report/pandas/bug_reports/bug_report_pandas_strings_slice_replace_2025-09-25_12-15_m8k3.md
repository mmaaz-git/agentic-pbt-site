# Bug Report: pandas.core.strings slice_replace with start > stop

**Target**: `pandas.core.strings.object_array.ObjectStringArrayMixin._str_slice_replace`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_str_slice_replace` method incorrectly handles cases where `start > stop`, producing different results than Python's native string slicing behavior.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import pandas as pd
from hypothesis import given, strategies as st, settings


@given(st.text(max_size=50), st.integers(min_value=-10, max_value=10), st.integers(min_value=-10, max_value=10), st.text(max_size=20))
@settings(max_examples=1000)
def test_slice_replace_matches_python(text, start, stop, repl):
    s = pd.Series([text])
    pandas_result = s.str.slice_replace(start, stop, repl)[0]
    python_slice = text[:start] + repl + text[stop:]
    assert pandas_result == python_slice
```

**Failing input**: `text='0', start=1, stop=0, repl=''`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import pandas as pd

text = '0'
start = 1
stop = 0
repl = ''

s = pd.Series([text])
pandas_result = s.str.slice_replace(start, stop, repl)[0]
python_result = text[:start] + repl + text[stop:]

print(f"Text: '{text}'")
print(f"start={start}, stop={stop}, repl='{repl}'")
print(f"Python: '{python_result}'")
print(f"Pandas: '{pandas_result}'")
assert pandas_result == python_result
```

## Why This Is A Bug

The `_str_slice_replace` method in `object_array.py` lines 347-364 contains logic that adjusts the `stop` parameter when the slice is empty:

```python
def f(x):
    if x[start:stop] == "":
        local_stop = start
    else:
        local_stop = stop
    y = ""
    if start is not None:
        y += x[:start]
    y += repl
    if stop is not None:
        y += x[local_stop:]
    return y
```

When `start > stop`, Python slicing `x[start:stop]` produces an empty slice, triggering the condition. The code then incorrectly sets `local_stop = start` instead of using the original `stop` value. This causes the final concatenation `y += x[local_stop:]` to use the wrong index.

For the failing case `text='0', start=1, stop=0, repl=''`:
- Expected: `'0'[:1] + '' + '0'[0:]` = `'0' + '' + '0'` = `'00'`
- Actual: Uses `local_stop=1`, resulting in `'0'[:1] + '' + '0'[1:]` = `'0' + '' + ''` = `'0'`

## Fix

```diff
--- a/pandas/core/strings/object_array.py
+++ b/pandas/core/strings/object_array.py
@@ -349,11 +349,6 @@ class ObjectStringArrayMixin(BaseStringArrayMethods):
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