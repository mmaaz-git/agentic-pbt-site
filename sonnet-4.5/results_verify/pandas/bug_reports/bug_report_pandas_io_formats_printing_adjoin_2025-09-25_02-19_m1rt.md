# Bug Report: pandas.io.formats.printing.adjoin - Crashes on Empty Lists

**Target**: `pandas.io.formats.printing.adjoin`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `adjoin` function crashes with a `ValueError` when given empty lists as input, even though the function does not document any requirement that lists must be non-empty.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.formats.printing import adjoin


@given(st.integers(min_value=0, max_value=10), st.lists(st.lists(st.text()), min_size=1))
def test_adjoin_handles_empty_lists(space, lists):
    result = adjoin(space, *lists)
    assert isinstance(result, str)
```

**Failing input**: `adjoin(0, [])`

## Reproducing the Bug

```python
from pandas.io.formats.printing import adjoin

adjoin(0, [])
```

Expected: Function should handle empty lists gracefully (either return empty string or handle the edge case).

Actual: Raises `ValueError: max() iterable argument is empty`

## Why This Is A Bug

The function's docstring does not specify that lists must be non-empty. Empty lists are valid Python lists and should be handled gracefully. The crash occurs because the function calls `max()` on empty sequences without checking if the lists are empty first.

The bug occurs at two locations in the code:
1. Line 51: `lengths = [max(map(strlen, x)) + space for x in lists[:-1]]` - crashes when any list except the last is empty
2. Line 53: `lengths.append(max(map(len, lists[-1])))` - crashes when the last list is empty

## Fix

```diff
--- a/pandas/io/formats/printing.py
+++ b/pandas/io/formats/printing.py
@@ -48,9 +48,9 @@ def adjoin(space: int, *lists: list[str], **kwargs) -> str:
     justfunc = kwargs.pop("justfunc", _adj_justify)

     newLists = []
-    lengths = [max(map(strlen, x)) + space for x in lists[:-1]]
+    lengths = [max(map(strlen, x), default=0) + space for x in lists[:-1]]
     # not the last one
-    lengths.append(max(map(len, lists[-1])))
+    lengths.append(max(map(len, lists[-1]), default=0))
     maxLen = max(map(len, lists))
     for i, lst in enumerate(lists):
         nl = justfunc(lst, lengths[i], mode="left")
```
