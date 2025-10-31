# Bug Report: get_level_lengths IndexError on Variable-Length Levels

**Target**: `pandas.io.formats.excel.get_level_lengths`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`get_level_lengths` crashes with an `IndexError` when processing levels of different lengths, despite the function signature and documentation not requiring uniform lengths.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.formats.excel import get_level_lengths

@given(st.lists(st.lists(st.text(max_size=5), min_size=1, max_size=10), min_size=1, max_size=5))
def test_get_level_lengths_sum_invariant(levels):
    result = get_level_lengths(levels)

    assert len(result) == len(levels)

    for i, level_lengths in enumerate(result):
        total_length = sum(level_lengths.values())
        assert total_length == len(levels[i])
```

**Failing input**: `levels=[[''], ['', '']]`

## Reproducing the Bug

```python
from pandas.io.formats.excel import get_level_lengths

levels = [[''], ['', '']]
result = get_level_lengths(levels)
```

Output:
```
IndexError: list index out of range
```

## Why This Is A Bug

The function's signature accepts `levels: Any` and the docstring describes it as a "list of lists" without requiring uniform lengths. However, the implementation initializes a `control` array to the length of the first level and assumes all subsequent levels have the same length. When a level is longer than `levels[0]`, accessing `control[i]` raises an `IndexError`.

## Fix

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -2017,11 +2017,13 @@ def get_level_lengths(
     if len(levels) == 0:
         return []

-    control = [True] * len(levels[0])
+    max_len = max(len(level) for level in levels)
+    control = [True] * max_len

     result = []
     for level in levels:
         last_index = 0
+        level_len = len(level)

         lengths = {}
         for i, key in enumerate(level):
```