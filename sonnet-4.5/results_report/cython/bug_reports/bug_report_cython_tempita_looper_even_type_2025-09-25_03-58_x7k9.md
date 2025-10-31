# Bug Report: Cython.Tempita looper.even Type Inconsistency

**Target**: `Cython.Tempita._looper.loop_pos.even`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `looper.even` property returns int (0 or 1) instead of bool, creating type inconsistency with its counterpart `odd` property which correctly returns bool.

## Property-Based Test

```python
@given(st.lists(st.integers(), min_size=2, max_size=20))
def test_looper_odd_even_type_consistency(seq):
    results = list(looper(seq))
    for loop, item in results:
        assert isinstance(loop.odd, bool), f"odd should return bool, got {type(loop.odd)}"
        assert isinstance(loop.even, bool), f"even should return bool, got {type(loop.even)}"
```

**Failing input**: Any sequence with 2+ items, e.g., `[0, 0]`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita._looper import looper

seq = [10, 20]
results = list(looper(seq))

first_loop, _ = results[0]
print(f"First item - odd: {first_loop.odd!r} (type: {type(first_loop.odd).__name__})")
print(f"First item - even: {first_loop.even!r} (type: {type(first_loop.even).__name__})")
```

## Why This Is A Bug

In `_looper.py`:
- Line 98-100: `odd` property returns `not self.pos % 2` which produces a bool
- Line 102-104: `even` property returns `self.pos % 2` which produces an int (0 or 1)

This creates type inconsistency between paired properties. While functionally correct due to Python's truthy/falsy semantics, it violates the principle of least surprise and could cause issues with:
- Type checkers expecting bool
- Code using identity checks (`is True`, `is False`)
- Template logic expecting consistent boolean types

Similar properties like `first` and `last` (lines 106-112) correctly return bool values.

## Fix

```diff
--- a/Cython/Tempita/_looper.py
+++ b/Cython/Tempita/_looper.py
@@ -100,7 +100,7 @@ class loop_pos:
     odd = property(odd)

     def even(self):
-        return self.pos % 2
+        return bool(self.pos % 2)
     even = property(even)

     def first(self):
```