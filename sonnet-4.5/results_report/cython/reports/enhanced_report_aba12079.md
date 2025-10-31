# Bug Report: Cython.Tempita._looper.loop_pos.even Returns int Instead of bool

**Target**: `Cython.Tempita._looper.loop_pos.even`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `even` property in Cython.Tempita._looper returns int (0 or 1) instead of bool, creating a type inconsistency with its counterpart `odd` property which correctly returns bool.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from Cython.Tempita._looper import looper

@given(st.lists(st.integers(), min_size=2, max_size=20))
def test_looper_odd_even_type_consistency(seq):
    results = list(looper(seq))
    for loop, item in results:
        assert isinstance(loop.odd, bool), f"odd should return bool, got {type(loop.odd)}"
        assert isinstance(loop.even, bool), f"even should return bool, got {type(loop.even)}"

# Run the test
if __name__ == "__main__":
    test_looper_odd_even_type_consistency()
```

<details>

<summary>
**Failing input**: `[0, 0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 16, in <module>
    test_looper_odd_even_type_consistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 8, in test_looper_odd_even_type_consistency
    def test_looper_odd_even_type_consistency(seq):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 12, in test_looper_odd_even_type_consistency
    assert isinstance(loop.even, bool), f"even should return bool, got {type(loop.even)}"
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^
AssertionError: even should return bool, got <class 'int'>
Falsifying example: test_looper_odd_even_type_consistency(
    seq=[0, 0],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita._looper import looper

seq = [10, 20]
results = list(looper(seq))

print("Testing type inconsistency between odd and even properties:")
print("=" * 60)

for i, (loop, item) in enumerate(results):
    print(f"\nItem at index {i} (value: {item}):")
    print(f"  loop.odd: {loop.odd!r} (type: {type(loop.odd).__name__})")
    print(f"  loop.even: {loop.even!r} (type: {type(loop.even).__name__})")

    # Also check other similar properties for comparison
    print(f"  loop.first: {loop.first!r} (type: {type(loop.first).__name__})")
    print(f"  loop.last: {loop.last!r} (type: {type(loop.last).__name__})")

print("\n" + "=" * 60)
print("BUG: The 'even' property returns int (0 or 1) instead of bool")
print("     while 'odd' property correctly returns bool (True or False)")
```

<details>

<summary>
Type inconsistency demonstrated: even returns int, odd returns bool
</summary>
```
Testing type inconsistency between odd and even properties:
============================================================

Item at index 0 (value: 10):
  loop.odd: True (type: bool)
  loop.even: 0 (type: int)
  loop.first: True (type: bool)
  loop.last: False (type: bool)

Item at index 1 (value: 20):
  loop.odd: False (type: bool)
  loop.even: 1 (type: int)
  loop.first: False (type: bool)
  loop.last: True (type: bool)

============================================================
BUG: The 'even' property returns int (0 or 1) instead of bool
     while 'odd' property correctly returns bool (True or False)
```
</details>

## Why This Is A Bug

This violates expected behavior for several reasons:

1. **Type Inconsistency Between Paired Properties**: The `odd` and `even` properties are semantic opposites that should have consistent return types. In the source code (_looper.py):
   - Line 98-100: `odd` returns `not self.pos % 2` which produces a bool
   - Line 102-104: `even` returns `self.pos % 2` which produces an int (0 or 1)

2. **Inconsistent with Similar Properties**: Other boolean-semantic properties like `first` (line 107) and `last` (line 111) correctly return bool values using comparisons.

3. **Violates Principle of Least Surprise**: Properties named `odd` and `even` naturally suggest boolean return values. Users would reasonably expect both to return True/False.

4. **Potential Issues with Type-Sensitive Code**:
   - Type checkers (mypy, pyright) expecting bool will flag this as an error
   - Code using identity checks (`is True`, `is False`) will behave unexpectedly
   - JSON serialization might produce different outputs (0/1 vs true/false)

While this doesn't break functionality due to Python's truthy/falsy semantics (0 is falsy, 1 is truthy), it creates an unnecessary inconsistency in the API that could lead to subtle bugs in type-aware contexts.

## Relevant Context

The Cython.Tempita._looper module provides a helper for iterating over sequences with context about position. It's particularly useful in template engines where you need to know if you're on an odd/even row for styling, or if you're at the first/last item.

The looper was likely derived from the original Tempita templating project. The bug appears to be a simple oversight where the developer forgot to wrap the modulo result in a bool() call for the `even` property, while correctly doing the boolean conversion for `odd` by using the `not` operator.

Documentation: The module has inline documentation showing usage examples but doesn't explicitly specify return types for properties. However, the semantic meaning of property names and consistency with other properties establishes clear expectations.

Source code location: `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Tempita/_looper.py`

## Proposed Fix

```diff
--- a/Cython/Tempita/_looper.py
+++ b/Cython/Tempita/_looper.py
@@ -101,7 +101,7 @@ class loop_pos:

     def even(self):
-        return self.pos % 2
+        return bool(self.pos % 2)
     even = property(even)

     def first(self):
```