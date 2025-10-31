# Bug Report: pandas.arrays factorize() IndexError on all-NA arrays

**Target**: `pandas.arrays` masked array types: `BooleanArray`, `IntegerArray`, `FloatingArray`, `ArrowStringArray`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `factorize()` is called on masked array types where all values are NA (missing), it returns codes containing -1 (the NA sentinel) and an empty uniques array. This breaks the standard reconstruction pattern `uniques[codes]`, causing an IndexError. Bug affects BooleanArray, IntegerArray, FloatingArray, and ArrowStringArray.

## Property-Based Test

```python
import numpy as np
import pandas.arrays as pa
from hypothesis import given, strategies as st, settings


@settings(max_examples=200)
@given(
    values=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100),
    mask_indices=st.lists(st.integers(min_value=0, max_value=99), max_size=10)
)
def test_integerarray_factorize_roundtrip(values, mask_indices):
    mask = np.zeros(len(values), dtype=bool)
    for idx in mask_indices:
        if idx < len(values):
            mask[idx] = True

    arr = pa.IntegerArray(np.array(values, dtype='int64'), mask=mask)
    codes, uniques = arr.factorize()

    reconstructed = uniques[codes]
```

**Failing input**: `values=[0], mask_indices=[0]` (produces an all-NA array)

## Reproducing the Bug

```python
import numpy as np
import pandas.arrays as pa

arr = pa.IntegerArray(np.array([0], dtype='int64'), mask=np.array([True]))
codes, uniques = arr.factorize()
reconstructed = uniques[codes]
```

Output:
```
IndexError: index -1 is out of bounds for axis 0 with size 0
```

The same bug occurs with BooleanArray, FloatingArray, and ArrowStringArray. Interestingly, ArrowExtensionArray handles this correctly by including `<NA>` in the uniques array.

## Why This Is A Bug

The `factorize()` method is documented as encoding the array as an enumerated type, returning codes and unique values. The standard usage pattern for factorization is reconstruction via `uniques[codes]`. This pattern is broken when:

1. All values in the array are NA
2. `use_na_sentinel=True` (the default)
3. This produces `codes=[-1, -1, ...]` and `uniques=[]`
4. Attempting `uniques[codes]` raises IndexError

The same operation with `use_na_sentinel=False` works correctly, producing codes `[0, 0, ...]` and `uniques=[<NA>]`, allowing successful reconstruction.

## Fix

The fix should ensure that when using the NA sentinel, the uniques array can still be indexed with -1 (Python's negative indexing). Options include:

1. Include NA in uniques array even when using sentinel (breaking change to return format)
2. Document that `uniques[codes]` pattern doesn't work with all-NA arrays when using sentinels
3. Return codes with different sentinel value that won't cause indexing issues
4. Handle the all-NA case specially to maintain the roundtrip property

The cleanest fix is likely option 1, making uniques always include NA as its last element when sentinel is used and any NAs exist:

```diff
--- a/pandas/core/arrays/masked.py
+++ b/pandas/core/arrays/masked.py
@@ -factorize method location
-        codes, uniques = super().factorize(use_na_sentinel=use_na_sentinel)
-        return codes, uniques
+        codes, uniques_arr = super().factorize(use_na_sentinel=use_na_sentinel)
+        # Ensure uniques can be indexed with -1 when NA sentinel is used
+        if use_na_sentinel and len(uniques_arr) == 0 and (codes == -1).any():
+            # Append NA to uniques so uniques[-1] works
+            na_value = type(self)._from_sequence([pd.NA])
+            uniques_arr = type(self)._concat_same_type([uniques_arr, na_value])
+        return codes, uniques_arr
```

Note: The exact implementation would need to handle the NA value construction properly for each array type.