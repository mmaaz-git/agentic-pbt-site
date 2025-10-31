# Bug Report: Cython.Tempita.looper odd/even Properties Are Swapped

**Target**: `Cython.Tempita._looper.loop_pos`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `odd` and `even` properties of the `loop_pos` class return the opposite boolean values from what their names suggest. Position 0 (the first item) is reported as odd when it should be even.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Tempita._looper import looper


@given(st.lists(st.integers(), min_size=1))
def test_looper_odd_even_properties(seq):
    result = list(looper(seq))
    for loop_obj, item in result:
        pos = loop_obj.index
        if pos % 2 == 0:
            assert loop_obj.even == True
            assert loop_obj.odd == False
        else:
            assert loop_obj.odd == True
            assert loop_obj.even == False
```

**Failing input**: `seq=[0]` (or any non-empty list)

## Reproducing the Bug

```python
from Cython.Tempita._looper import looper

for loop_obj, item in looper([1, 2, 3, 4]):
    pos = loop_obj.index
    print(f"Position {pos}: odd={loop_obj.odd}, even={loop_obj.even}")
```

Output:
```
Position 0: odd=True, even=0
Position 1: odd=False, even=1
Position 2: odd=True, even=0
Position 3: odd=False, even=1
```

Position 0 should be even (True) but reports odd=True and even=0 (falsy).

## Why This Is A Bug

In standard zero-based indexing, position 0 is considered even (0 % 2 == 0). The implementation has the `odd` and `even` properties backwards. Additionally, the `even` property returns an integer (0 or 1) instead of a boolean, which is inconsistent with the `odd` property that returns a boolean.

## Fix

```diff
--- a/Cython/Tempita/_looper.py
+++ b/Cython/Tempita/_looper.py
@@ -96,11 +96,11 @@ class loop_pos:
     previous = property(previous)

     def odd(self):
-        return not self.pos % 2
+        return self.pos % 2 == 1
     odd = property(odd)

     def even(self):
-        return self.pos % 2
+        return self.pos % 2 == 0
     even = property(even)

     def first(self):
```