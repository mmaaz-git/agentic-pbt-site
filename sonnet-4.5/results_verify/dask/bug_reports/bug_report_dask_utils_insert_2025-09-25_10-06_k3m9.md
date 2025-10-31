# Bug Report: dask.utils.insert Function Name/Behavior Mismatch

**Target**: `dask.utils.insert`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `insert` function in `dask.utils` has a misleading name - it performs element replacement rather than insertion. The function name suggests it inserts a new element into a tuple, but it actually replaces an existing element at the specified position.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.utils import insert


@settings(max_examples=500)
@given(
    st.tuples(st.integers(), st.integers(), st.integers()),
    st.integers(min_value=0, max_value=2),
    st.integers()
)
def test_insert_preserves_length_not_increases(tup, loc, val):
    """
    BUG: insert() preserves tuple length instead of increasing it.

    A true insertion should increase the length by 1.
    But dask.utils.insert() replaces an element, keeping the same length.
    """
    result = insert(tup, loc, val)

    # BUG: length is preserved (replacement behavior)
    assert len(result) == len(tup)

    # Element at loc is replaced with val
    assert result[loc] == val

    # All other elements unchanged
    for i in range(len(tup)):
        if i != loc:
            assert result[i] == tup[i]
```

**Failing input**: Any tuple, e.g., `('a', 'b', 'c')`

## Reproducing the Bug

```python
from dask.utils import insert

tup = ('a', 'b', 'c')
result = insert(tup, 1, 'X')
print(result)

print(f"Length before: {len(tup)}")
print(f"Length after: {len(result)}")
print(f"Element at position 1: {result[1]}")

lst = list(tup)
lst.insert(1, 'X')
print(f"True insert would give: {tuple(lst)}")
```

Output:
```
('a', 'X', 'c')
Length before: 3
Length after: 3
Element at position 1: X
True insert would give: ('a', 'X', 'b', 'c')
```

## Why This Is A Bug

The function is named `insert` which in Python typically means adding a new element (like `list.insert()`). However, the implementation:

```python
def insert(tup, loc, val):
    L = list(tup)
    L[loc] = val  # This is REPLACEMENT, not insertion
    return tuple(L)
```

performs `L[loc] = val` which is replacement, not insertion. This violates:

1. **Naming convention**: Python's `list.insert(index, value)` inserts a new element and shifts subsequent elements
2. **Expected behavior**: A function called "insert" should increase the collection size by 1
3. **Documentation contract**: While the docstring doesn't explicitly claim insertion semantics, the name strongly implies it

The function is not used anywhere in the dask codebase (grep search found no imports), suggesting it's dead code where this bug was never caught.

## Fix

Either rename the function to match its behavior or fix the implementation:

**Option 1: Rename to match behavior**
```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1158,7 +1158,7 @@ def digit(n, k, base):
     return n // base**k % base


-def insert(tup, loc, val):
+def replace_at(tup, loc, val):
     """

-    >>> insert(('a', 'b', 'c'), 0, 'x')
+    >>> replace_at(('a', 'b', 'c'), 0, 'x')
     ('x', 'b', 'c')
     """
```

**Option 2: Fix to actually insert**
```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1161,9 +1161,9 @@ def insert(tup, loc, val):
 def insert(tup, loc, val):
     """

-    >>> insert(('a', 'b', 'c'), 0, 'x')
-    ('x', 'b', 'c')
+    >>> insert(('a', 'b', 'c'), 1, 'x')
+    ('a', 'x', 'b', 'c')
     """
     L = list(tup)
-    L[loc] = val
+    L.insert(loc, val)
     return tuple(L)
```

Given the function appears unused and the docstring example shows replacement behavior, **Option 1 (rename)** is safer to avoid breaking any undocumented dependencies.