# Bug Report: numpy.lib.format.dtype_to_descr Loses Shape Information for Sub-Array Dtypes

**Target**: `numpy.lib.format.dtype_to_descr` and `numpy.lib.format.descr_to_dtype`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `dtype_to_descr` function fails to preserve shape information for sub-array dtypes, violating its documented contract that the result "can be passed to `numpy.dtype()` in order to replicate the input dtype". This makes the round-trip `descr_to_dtype(dtype_to_descr(dtype))` fail for any dtype with a shape attribute.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st

@given(
    base_type=st.sampled_from(['i4', 'f8', 'c16']),
    shape_size=st.integers(min_value=1, max_value=5)
)
def test_dtype_descr_round_trip_with_shapes(base_type, shape_size):
    shape = tuple(range(1, shape_size + 1))
    dtype = np.dtype((base_type, shape))

    descr = np.lib.format.dtype_to_descr(dtype)
    restored = np.lib.format.descr_to_dtype(descr)

    assert restored == dtype, \
        f"Round-trip failed: {dtype} -> {descr} -> {restored}"

# Run the test
if __name__ == "__main__":
    test_dtype_descr_round_trip_with_shapes()
```

<details>

<summary>
**Failing input**: `dtype=np.dtype(('i4', (1,)))`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 20, in <module>
    test_dtype_descr_round_trip_with_shapes()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 5, in test_dtype_descr_round_trip_with_shapes
    base_type=st.sampled_from(['i4', 'f8', 'c16']),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 15, in test_dtype_descr_round_trip_with_shapes
    assert restored == dtype, \
           ^^^^^^^^^^^^^^^^^
AssertionError: Round-trip failed: ('<i4', (1,)) -> |V4 -> |V4
Falsifying example: test_dtype_descr_round_trip_with_shapes(
    base_type='i4',  # or any other generated value
    shape_size=1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.lib.format as fmt

dtype_with_shape = np.dtype(('i4', (1,)))
print(f"Original dtype: {dtype_with_shape}")
print(f"Original dtype.shape: {dtype_with_shape.shape}")
print(f"Original dtype.base: {dtype_with_shape.base}")
print(f"Original dtype.str: {dtype_with_shape.str}")

descr = fmt.dtype_to_descr(dtype_with_shape)
print(f"\nAfter dtype_to_descr: {descr}")
print(f"Type of descr: {type(descr)}")

restored = fmt.descr_to_dtype(descr)
print(f"\nRestored dtype: {restored}")
print(f"Restored dtype.shape: {restored.shape}")
print(f"Restored dtype.base: {restored.base if hasattr(restored, 'base') else 'N/A'}")

print(f"\nAre they equal? {restored == dtype_with_shape}")

# This will fail with AssertionError
assert restored == dtype_with_shape, f"Round-trip failed: {dtype_with_shape} -> {descr} -> {restored}"
```

<details>

<summary>
AssertionError: Round-trip failed
</summary>
```
Original dtype: ('<i4', (1,))
Original dtype.shape: (1,)
Original dtype.base: int32
Original dtype.str: |V4

After dtype_to_descr: |V4
Type of descr: <class 'str'>

Restored dtype: |V4
Restored dtype.shape: ()
Restored dtype.base: |V4

Are they equal? False
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/32/repo.py", line 22, in <module>
    assert restored == dtype_with_shape, f"Round-trip failed: {dtype_with_shape} -> {descr} -> {restored}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Round-trip failed: ('<i4', (1,)) -> |V4 -> |V4
```
</details>

## Why This Is A Bug

The NumPy documentation explicitly states that `dtype_to_descr` and `descr_to_dtype` should be inverse operations:

1. The docstring for `dtype_to_descr` (line 269-271 in `/home/npc/miniconda/lib/python3.13/site-packages/numpy/lib/_format_impl.py`) states:
   > "An object that can be passed to `numpy.dtype()` in order to replicate the input dtype."

2. The docstring for `descr_to_dtype` (line 315) states:
   > "This is essentially the reverse of `~lib.format.dtype_to_descr`."

However, for sub-array dtypes (dtypes with a shape attribute), the function returns only `dtype.str` (line 307), which is a string like `"|V4"` that represents the total byte size but completely loses the shape information. When `descr_to_dtype` processes this string, it creates a void dtype with no shape, failing to recover the original structure.

The bug occurs because:
- A sub-array dtype like `np.dtype(('i4', (1,)))` has a base type (`'i4'`) and shape `(1,)`
- The dtype's string representation `"|V4"` only encodes the total size (4 bytes)
- This loses the critical information that it's 1 element of 4-byte integers, not just 4 bytes of void data
- The `descr_to_dtype` function already supports tuple inputs `(base_dtype, shape)` (lines 335-338), but `dtype_to_descr` never generates this format for sub-array dtypes

## Relevant Context

Sub-array dtypes are a documented NumPy feature used to create arrays where each element is itself an array of a fixed shape. They are created using the syntax `np.dtype((base_type, shape))`.

Interestingly, NPY file operations work around this issue by flattening array dimensions before saving (see `header_data_from_array_1_0` at line 382), so typical users saving/loading NPY files don't encounter this bug. However, anyone directly using these functions for dtype serialization will face this issue.

The `descr_to_dtype` function already has code to handle tuple descriptors for sub-array dtypes (lines 335-338):
```python
elif isinstance(descr, tuple):
    # subtype, will always have a shape descr[1]
    dt = descr_to_dtype(descr[0])
    return numpy.dtype((dt, descr[1]))
```

This shows the functions were designed to handle sub-array dtypes, but the implementation in `dtype_to_descr` is incomplete.

## Proposed Fix

The fix is to check if the dtype has a shape and return a tuple format that preserves this information:

```diff
--- a/numpy/lib/_format_impl.py
+++ b/numpy/lib/_format_impl.py
@@ -303,8 +303,11 @@ def dtype_to_descr(dtype):
                       "allow_pickle=True to be set.",
                       UserWarning, stacklevel=2)
         return "|O"
-    else:
+    elif dtype.shape == ():
+        # Simple dtype without shape
         return dtype.str
+    else:
+        # Sub-array dtype: preserve shape information
+        return (dtype.base.str, dtype.shape)


 @set_module("numpy.lib.format")
```

This ensures that sub-array dtypes can be properly round-tripped through the descriptor format, fulfilling the documented API contract.