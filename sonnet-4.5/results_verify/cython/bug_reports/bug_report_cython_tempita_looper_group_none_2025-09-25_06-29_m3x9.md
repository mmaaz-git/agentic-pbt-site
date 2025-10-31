# Bug Report: Cython.Tempita looper first_group/last_group None Handling

**Target**: `Cython.Tempita._looper.loop_pos._compare_group`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `first_group()` and `last_group()` methods crash with AttributeError when used with non-None getters at sequence boundaries, because they attempt to call getattr/index operations on None values (previous/next items).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Tempita._looper import looper


@given(st.lists(st.integers(), min_size=1, max_size=10))
@settings(max_examples=100)
def test_looper_last_group_with_callable_getter(values):
    class Item:
        def __init__(self, val):
            self.val = val

    items = [Item(v) for v in values]

    for loop, item in looper(items):
        if loop.last:
            try:
                result = loop.last_group('.val')
                assert result == True
            except AttributeError:
                assert False, "last_group should not crash on last item"
```

**Failing input**: Any sequence with at least 1 item when using attribute/callable/index getter with `last_group`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita._looper import looper


class Item:
    def __init__(self, value):
        self.value = value


items = [Item(1), Item(2), Item(3)]

for loop, item in looper(items):
    if loop.last:
        result = loop.last_group('.value')
        print(f"Result: {result}")
```

Output:
```
AttributeError: 'NoneType' object has no attribute 'value'
```

## Why This Is A Bug

Line 138 in `Cython/Tempita/_looper.py` calls `_compare_group(self.item, self.__next__, getter)` where `self.__next__` is None for the last item (line 89 returns None when IndexError).

The `_compare_group` method (lines 140-154) doesn't handle None values:
- Line 148: `getattr(other, getter)()` - crashes if other is None
- Line 150: `getattr(other, getter)` - crashes if other is None
- Line 154: `other[getter]` - crashes if other is None

The same issue affects `first_group()` at line 127 when pos == 0 and previous is None.

The docstrings for first_group and last_group (lines 118-138) don't mention this limitation, suggesting the methods should work with all getter types at boundaries.

## Fix

```diff
--- a/Cython/Tempita/_looper.py
+++ b/Cython/Tempita/_looper.py
@@ -139,6 +139,8 @@ class loop_pos:

     def _compare_group(self, item, other, getter):
+        if other is None:
+            return True
         if getter is None:
             return item != other
         elif (isinstance(getter, str)
```