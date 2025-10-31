# Bug Report: pandas.io.formats.printing.adjoin - Inconsistent Use of strlen Parameter

**Target**: `pandas.io.formats.printing.adjoin`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `adjoin` function does not consistently apply the `strlen` parameter to all input lists. It uses `strlen` for all lists except the last one, where it uses the built-in `len()` function instead. This breaks the documented purpose of the `strlen` parameter for unicode handling.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.formats.printing import adjoin


@given(st.lists(st.text(min_size=1), min_size=1), st.lists(st.text(min_size=1), min_size=1))
def test_adjoin_uses_strlen_consistently(list1, list2):
    call_count = {"count": 0}

    def counting_strlen(s):
        call_count["count"] += 1
        return len(s)

    adjoin(1, list1, list2, strlen=counting_strlen)

    total_strings = len(list1) + len(list2)
    assert call_count["count"] >= total_strings
```

**Failing input**: Any input where the function is called with a custom `strlen` parameter and multiple lists.

## Reproducing the Bug

```python
from pandas.io.formats.printing import adjoin


def custom_strlen(s):
    return len(s) + 10


result = adjoin(1, ["a"], ["b"], strlen=custom_strlen)
print(repr(result))
```

Expected: Both lists should use `custom_strlen` for width calculation, resulting in consistent spacing based on the custom length function.

Actual: Only the first list uses `custom_strlen`; the last list uses `len()`, resulting in inconsistent column width calculations.

## Why This Is A Bug

The function's docstring explicitly states: "strlen : callable - function used to calculate the length of each str. Needed for unicode handling."

However, the implementation only applies `strlen` to lists[:-1] (all lists except the last):
- Line 51: `lengths = [max(map(strlen, x)) + space for x in lists[:-1]]` - uses custom `strlen`
- Line 53: `lengths.append(max(map(len, lists[-1])))` - uses built-in `len()`

This inconsistency would cause incorrect formatting when using `adjoin` with unicode text that has display widths different from string lengths, which is the exact use case the `strlen` parameter is designed to handle.

## Fix

```diff
--- a/pandas/io/formats/printing.py
+++ b/pandas/io/formats/printing.py
@@ -50,7 +50,7 @@ def adjoin(space: int, *lists: list[str], **kwargs) -> str:
     newLists = []
     lengths = [max(map(strlen, x)) + space for x in lists[:-1]]
     # not the last one
-    lengths.append(max(map(len, lists[-1])))
+    lengths.append(max(map(strlen, lists[-1])))
     maxLen = max(map(len, lists))
     for i, lst in enumerate(lists):
         nl = justfunc(lst, lengths[i], mode="left")
```
