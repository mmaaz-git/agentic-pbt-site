# Bug Report: DeepChainMap.__delitem__ Violates Deletion Invariant

**Target**: `pandas.core.computation.scope.DeepChainMap.__delitem__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `DeepChainMap.__delitem__` method only deletes a key from the first map where it's found, but if the same key exists in multiple maps, the key remains accessible from subsequent maps. This violates the fundamental deletion invariant: after `del container[key]`, `key` should not be accessible via `key in container`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from pandas.core.computation.scope import DeepChainMap

@given(
    st.dictionaries(st.text(min_size=1, max_size=5), st.integers(), min_size=1, max_size=5),
    st.dictionaries(st.text(min_size=1, max_size=5), st.integers(), min_size=1, max_size=5)
)
def test_deepchainmap_deletion_invariant(map1, map2):
    common_keys = set(map1.keys()) & set(map2.keys())
    assume(len(common_keys) > 0)

    dcm = DeepChainMap(map1.copy(), map2.copy())
    key = list(common_keys)[0]

    del dcm[key]

    assert key not in dcm, f"After deletion, {key} should not be accessible"
```

**Failing input**: Any two dictionaries with at least one common key, e.g., `map1={'x': 1}`, `map2={'x': 2}`

## Reproducing the Bug

```python
from pandas.core.computation.scope import DeepChainMap

map1 = {'x': 100, 'y': 200}
map2 = {'x': 999, 'z': 300}

dcm = DeepChainMap(map1, map2)

print(f"Before deletion: 'x' in dcm = {('x' in dcm)}")
print(f"  dcm['x'] = {dcm['x']}")

del dcm['x']

print(f"\nAfter deletion: 'x' in dcm = {('x' in dcm)}")
if 'x' in dcm:
    print(f"  dcm['x'] = {dcm['x']}")
    print("  BUG: Key is still accessible after deletion!")
```

Output:
```
Before deletion: 'x' in dcm = True
  dcm['x'] = 100

After deletion: 'x' in dcm = True
  dcm['x'] = 999
  BUG: Key is still accessible after deletion!
```

## Why This Is A Bug

The fundamental invariant of the `__delitem__` method is:
```python
del container[key]
=> key not in container
```

The current implementation violates this invariant when a key exists in multiple maps within the `DeepChainMap`. After deletion, the key is still accessible from subsequent maps, which is unexpected and inconsistent with standard Python container semantics.

This is particularly problematic because:
1. `DeepChainMap` is used to manage variable scopes in pandas eval expressions
2. Users expect that deleting a variable makes it inaccessible
3. The behavior is inconsistent with `__setitem__`, which properly updates the key in the map where it exists

## Fix

The fix depends on the intended semantics. Two options:

**Option 1: Delete from all maps** (preserves deletion invariant)
```diff
--- a/pandas/core/computation/scope.py
+++ b/pandas/core/computation/scope.py
@@ -40,10 +40,13 @@ class DeepChainMap(ChainMap[_KT, _VT]):
     def __delitem__(self, key: _KT) -> None:
         """
         Raises
         ------
         KeyError
             If `key` doesn't exist.
         """
+        found = False
         for mapping in self.maps:
             if key in mapping:
                 del mapping[key]
-                return
+                found = True
-        raise KeyError(key)
+        if not found:
+            raise KeyError(key)
```

**Option 2: Document current behavior** (if intentional)
If the current behavior is intentional (delete only from first occurrence), the docstring should explicitly document this edge case to avoid user confusion.