# Bug Report: xarray.core.dtypes.AlwaysGreaterThan violates total ordering invariants

**Target**: `xarray.core.dtypes.AlwaysGreaterThan`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `AlwaysGreaterThan` class is decorated with `@functools.total_ordering` but violates the fundamental invariant that `a > a` must be `False` for any object `a`. This class returns `True` for all comparisons including self-comparison, breaking the irreflexivity property of strict ordering.

## Property-Based Test

```python
from hypothesis import given
from hypothesis import strategies as st
from xarray.core.dtypes import AlwaysGreaterThan

@given(st.integers())
def test_alwaysgt_comparison_properties(x):
    agt = AlwaysGreaterThan()

    if not isinstance(x, AlwaysGreaterThan):
        assert agt > x

    agt2 = AlwaysGreaterThan()
    assert agt == agt2
    assert not (agt > agt2)

if __name__ == "__main__":
    test_alwaysgt_comparison_properties()
```

<details>

<summary>
**Failing input**: `x=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 17, in <module>
    test_alwaysgt_comparison_properties()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 6, in test_alwaysgt_comparison_properties
    def test_alwaysgt_comparison_properties(x):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 14, in test_alwaysgt_comparison_properties
    assert not (agt > agt2)
           ^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_alwaysgt_comparison_properties(
    x=0,
)
```
</details>

## Reproducing the Bug

```python
from xarray.core.dtypes import AlwaysGreaterThan

# Create two instances of AlwaysGreaterThan
agt1 = AlwaysGreaterThan()
agt2 = AlwaysGreaterThan()

# Test equality - should be True
print(f"agt1 == agt2: {agt1 == agt2}")

# Test greater than - should be False since they're equal, but returns True (BUG!)
print(f"agt1 > agt2: {agt1 > agt2}")

# This violates the irreflexivity property
print(f"agt1 > agt1: {agt1 > agt1}")

# Test assertions
assert agt1 == agt2, "Expected two AlwaysGreaterThan instances to be equal"
assert agt1 > agt2, "This assertion passes but shouldn't - violates ordering properties!"
```

<details>

<summary>
Output shows violation of ordering properties
</summary>
```
agt1 == agt2: True
agt1 > agt2: True
agt1 > agt1: True
```
</details>

## Why This Is A Bug

The `@functools.total_ordering` decorator expects classes to implement comparison operators that satisfy the mathematical properties of a total ordering. The current implementation violates these fundamental properties:

1. **Irreflexivity of >**: For any object `a`, `a > a` must be `False`. The current implementation returns `True` for `agt > agt`, violating this property.

2. **Antisymmetry**: If `a >= b` and `b >= a`, then `a == b`. With the current implementation, `agt1 > agt2` and `agt2 > agt1` are both `True` even though `agt1 == agt2` is also `True`, violating the consistency between equality and ordering.

3. **Transitivity implications**: When `a == b`, the expectation is that `a > b` is `False`. Having both `agt1 == agt2` and `agt1 > agt2` be `True` violates this expectation.

This is problematic because:
- It breaks the contract expected by `@functools.total_ordering`, which may lead to incorrect behavior in derived comparison methods
- It violates universal expectations for comparison operators that Python developers rely on
- It could cause unexpected behavior in sorting algorithms, ordered containers, or any code that relies on comparison transitivity
- The same bug affects the `AlwaysLessThan` class with the `__lt__` method

## Relevant Context

The `AlwaysGreaterThan` and `AlwaysLessThan` classes are used as sentinel values in xarray, serving as object-type equivalents to `np.inf` and `-np.inf`. They're instantiated as singleton-like objects `INF` and `NINF` at the module level (lines 36-37 of `/home/npc/miniconda/lib/python3.13/site-packages/xarray/core/dtypes.py`).

These sentinel values are used in functions like `get_pos_infinity()` and `get_neg_infinity()` when dealing with object dtypes that don't have a native infinity representation. While the singleton pattern means that in practice the same instance is usually compared, the bug still manifests if new instances are created.

The documentation at line 35 states these are "Equivalence to np.inf (-np.inf) for object-type", suggesting they should behave similarly to numeric infinities, where `np.inf == np.inf` is `True` but `np.inf > np.inf` is `False`.

## Proposed Fix

```diff
--- a/xarray/core/dtypes.py
+++ b/xarray/core/dtypes.py
@@ -17,7 +17,8 @@ from xarray.core import utils
 @functools.total_ordering
 class AlwaysGreaterThan:
     def __gt__(self, other):
-        return True
+        if isinstance(other, type(self)):
+            return False
+        return True

     def __eq__(self, other):
         return isinstance(other, type(self))
@@ -26,7 +27,8 @@ class AlwaysGreaterThan:
 @functools.total_ordering
 class AlwaysLessThan:
     def __lt__(self, other):
-        return True
+        if isinstance(other, type(self)):
+            return False
+        return True

     def __eq__(self, other):
         return isinstance(other, type(self))
```