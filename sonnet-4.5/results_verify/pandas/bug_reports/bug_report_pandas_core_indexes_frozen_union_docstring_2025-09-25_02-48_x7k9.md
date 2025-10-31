# Bug Report: FrozenList.union Incorrect Docstring

**Target**: `pandas.core.indexes.frozen.FrozenList.union`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `FrozenList.union` method's docstring incorrectly states it returns "The collection difference between self and other" when it actually returns the concatenation/union.

## Reproducing the Bug

```python
import pandas as pd
from pandas.core.indexes.frozen import FrozenList

fl1 = FrozenList([1, 2, 3])
fl2 = FrozenList([4, 5, 6])

result = fl1.union(fl2)

print(f"fl1: {fl1}")
print(f"fl2: {fl2}")
print(f"fl1.union(fl2): {result}")
print()
print("Expected (based on docstring): difference = [1, 2, 3]")
print("Actual: concatenation = [1, 2, 3, 4, 5, 6]")
```

**Output:**
```
fl1: [1, 2, 3]
fl2: [4, 5, 6]
fl1.union(fl2): [1, 2, 3, 4, 5, 6]

Expected (based on docstring): difference = [1, 2, 3]
Actual: concatenation = [1, 2, 3, 4, 5, 6]
```

## Why This Is A Bug

1. **Incorrect docstring**: The Returns section at `frozen.py:44-46` states:
   > `Returns: FrozenList - The collection difference between self and other.`

2. **Actual behavior**: The method concatenates lists (union), not computes difference.

3. **Misleading to users**: Someone reading the docstring would expect the method to compute set difference, but it actually concatenates.

4. **Copy-paste error**: This appears to be a copy-paste error from the `difference` method docstring.

## Fix

Correct the docstring to accurately describe the concatenation behavior:

```diff
@@ -34,7 +34,7 @@ class FrozenList(PandasObject, list):
     def union(self, other) -> FrozenList:
         """
         Returns a FrozenList with other concatenated to the end of self.

         Parameters
         ----------
         other : array-like
             The array-like whose elements we are concatenating.

         Returns
         -------
         FrozenList
-            The collection difference between self and other.
+            A new FrozenList containing all elements from self followed by all elements from other.
         """
```