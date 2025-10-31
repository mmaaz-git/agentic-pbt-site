# Bug Report: pandas.arrays factorize() IndexError on all-NA masked arrays

**Target**: `pandas.arrays.IntegerArray`, `pandas.arrays.BooleanArray`, `pandas.arrays.FloatingArray`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `factorize()` method crashes with IndexError when called on masked array types containing only NA values using the default `use_na_sentinel=True`, breaking both direct indexing and the documented reconstruction pattern.

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

# Run the test
test_integerarray_factorize_roundtrip()
```

<details>

<summary>
**Failing input**: `values=[0], mask_indices=[0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 23, in <module>
    test_integerarray_factorize_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 7, in test_integerarray_factorize_roundtrip
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 20, in test_integerarray_factorize_roundtrip
    reconstructed = uniques[codes]
                    ~~~~~~~^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/masked.py", line 188, in __getitem__
    newmask = self._mask[item]
              ~~~~~~~~~~^^^^^^
IndexError: index -1 is out of bounds for axis 0 with size 0
Falsifying example: test_integerarray_factorize_roundtrip(
    values=[0],
    mask_indices=[0],
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import pandas.arrays as pa

# Create an IntegerArray with all values masked (NA)
arr = pa.IntegerArray(np.array([0], dtype='int64'), mask=np.array([True]))
print(f"Original array: {arr}")

# Factorize the array
codes, uniques = arr.factorize()
print(f"Codes: {codes}")
print(f"Uniques: {uniques}")

# Attempt reconstruction using indexing (fails)
print("\nAttempting reconstruction with uniques[codes]:")
reconstructed = uniques[codes]
print(f"Reconstructed: {reconstructed}")
```

<details>

<summary>
IndexError when attempting to reconstruct all-NA array
</summary>
```
Original array: <IntegerArray>
[<NA>]
Length: 1, dtype: Int64
Codes: [-1]
Uniques: <IntegerArray>
[]
Length: 0, dtype: Int64

Attempting reconstruction with uniques[codes]:
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/repo.py", line 15, in <module>
    reconstructed = uniques[codes]
                    ~~~~~~~^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/masked.py", line 188, in __getitem__
    newmask = self._mask[item]
              ~~~~~~~~~~^^^^^^
IndexError: index -1 is out of bounds for axis 0 with size 0
```
</details>

## Why This Is A Bug

The pandas documentation for `factorize()` explicitly states that "uniques.take(codes) will have the same values as values" in the return value documentation. However, this promise is violated when:

1. **All values in the array are NA (masked)**
2. **Using `use_na_sentinel=True` (the default)**

When these conditions are met:
- `factorize()` returns `codes=[-1, -1, ...]` (using -1 as NA sentinel) and `uniques=[]` (empty array)
- The documented reconstruction method `uniques.take(codes)` fails with: "cannot do a non-empty take from an empty axes"
- Direct indexing `uniques[codes]` fails with IndexError trying to access index -1 on empty array
- Even for mixed NA/non-NA arrays, both methods produce incorrect results without special parameters

The bug affects `IntegerArray`, `BooleanArray`, and `FloatingArray` consistently. The same operation works correctly with `use_na_sentinel=False`, producing `codes=[0, 0, ...]` and `uniques=[<NA>]`.

## Relevant Context

Testing revealed three important findings:

1. **Documentation mismatch**: The pandas documentation promises `uniques.take(codes)` works, but it fails for all-NA arrays without additional parameters (`allow_fill=True, fill_value=pd.NA`)

2. **Mixed NA handling**: For arrays with both NA and non-NA values, the default reconstruction methods produce incorrect results where NA values are replaced with the last unique value

3. **Workaround exists**: `uniques.take(codes, allow_fill=True, fill_value=pd.NA)` works correctly for both all-NA and mixed cases

The arrays are documented as "experimental" and their APIs may change. However, the factorize method is a core pandas operation that should handle edge cases gracefully.

Documentation reference: https://pandas.pydata.org/docs/reference/api/pandas.factorize.html

## Proposed Fix

The cleanest fix is to ensure `factorize()` returns output that can be reconstructed using the documented method. When all values are NA and `use_na_sentinel=True`, the method should either:

1. Include NA in the uniques array (even though documentation says it won't)
2. Raise a more informative error with guidance on handling all-NA arrays
3. Document the need for special parameters in reconstruction

Here's a potential fix in the masked array base class:

```diff
--- a/pandas/core/arrays/masked.py
+++ b/pandas/core/arrays/masked.py
@@ -1234,6 +1234,15 @@ class BaseMaskedArray(NDArrayBackedExtensionArray):
         codes, uniques = factorize_array(
             arr, use_na_sentinel=use_na_sentinel, mask=mask
         )
+
+        # Handle all-NA case when using sentinel
+        if use_na_sentinel and len(uniques) == 0 and len(codes) > 0:
+            # All values were NA, return a single NA in uniques
+            # This ensures uniques.take(codes, allow_fill=True, fill_value=pd.NA) works
+            # and provides a more informative error for the default case
+            from pandas import NA
+            uniques = type(self)._from_sequence([NA], dtype=self.dtype)
+            codes[:] = 0  # Point to the NA value

         # the hashtables don't handle all different types of bits
         uniques = uniques.astype(self.dtype.numpy_dtype, copy=False)
```

Alternatively, update documentation to clearly state that for arrays with NA values, reconstruction requires:
`uniques.take(codes, allow_fill=True, fill_value=pd.NA)`