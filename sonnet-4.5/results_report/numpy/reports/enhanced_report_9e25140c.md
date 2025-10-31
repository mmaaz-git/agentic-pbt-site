# Bug Report: numpy.rec.fromarrays Silent Data Corruption with Dtype Mismatch

**Target**: `numpy.rec.fromarrays`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `numpy.rec.fromarrays` function silently corrupts data through integer overflow when converting from uint64 to int64 dtypes, transforming positive values into negative ones without warning or error.

## Property-Based Test

```python
import numpy as np
import numpy.rec
from hypothesis import given, strategies as st

@given(st.lists(st.integers(), min_size=1, max_size=10))
def test_fromarrays_dtype_preserves_data(data):
    arr1 = np.array(data)
    arr2 = np.array(data)
    dtype = np.dtype([('a', 'i8'), ('b', 'i8')])
    rec = numpy.rec.fromarrays([arr1, arr2], dtype=dtype)
    assert np.array_equal(rec.a, arr1)

if __name__ == "__main__":
    test_fromarrays_dtype_preserves_data()
```

<details>

<summary>
**Failing input**: `data=[9_223_372_036_854_775_808]`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 14, in <module>
  |     test_fromarrays_dtype_preserves_data()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 6, in test_fromarrays_dtype_preserves_data
  |     def test_fromarrays_dtype_preserves_data(data):
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 10, in test_fromarrays_dtype_preserves_data
    |     rec = numpy.rec.fromarrays([arr1, arr2], dtype=dtype)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/records.py", line 659, in fromarrays
    |     _array[name] = obj
    |     ~~~~~~^^^^^^
    | OverflowError: Python int too large to convert to C long
    | Falsifying example: test_fromarrays_dtype_preserves_data(
    |     data=[-9_223_372_036_854_775_809],
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 11, in test_fromarrays_dtype_preserves_data
    |     assert np.array_equal(rec.a, arr1)
    |            ~~~~~~~~~~~~~~^^^^^^^^^^^^^
    | AssertionError
    | Falsifying example: test_fromarrays_dtype_preserves_data(
    |     data=[9_223_372_036_854_775_808],
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.rec

data = [9_223_372_036_854_775_808]
arr1 = np.array(data)
arr2 = np.array(data)

dtype = np.dtype([('a', 'i8'), ('b', 'i8')])
rec = numpy.rec.fromarrays([arr1, arr2], dtype=dtype)

print(f"Original: {arr1[0]}")
print(f"After fromarrays: {rec.a[0]}")
print(f"Data corrupted: {arr1[0] != rec.a[0]}")
```

<details>

<summary>
Data silently corrupted from positive to negative value
</summary>
```
Original: 9223372036854775808
After fromarrays: -9223372036854775808
Data corrupted: True
```
</details>

## Why This Is A Bug

This violates expected behavior in several critical ways:

1. **Silent Data Corruption**: The value 9,223,372,036,854,775,808 (2^63) is silently converted from a positive number to -9,223,372,036,854,775,808, completely changing its meaning without any warning or error. This is particularly dangerous in scientific computing where data integrity is paramount.

2. **Inconsistent with NumPy's Safety Mechanisms**: While NumPy's `astype()` method provides a `casting` parameter that defaults to 'unsafe' for backwards compatibility, it at least offers users the option to request safe casting. The `fromarrays` function provides no such control mechanism, forcing unsafe casting with no alternative.

3. **Documentation Gap**: The `fromarrays` documentation states that the dtype parameter is a "valid dtype for all arrays" but fails to warn users about potential overflow/underflow when the input data exceeds the target dtype's range. This omission is significant given the severity of silent data corruption.

4. **Violates Principle of Least Surprise**: Users reasonably expect that either their data will be preserved correctly, or they will receive an error indicating incompatible types. Silent corruption is the worst possible outcome as it can propagate undetected through subsequent calculations.

## Relevant Context

- **Integer Type Boundaries**:
  - int64 max: 9,223,372,036,854,775,807 (2^63 - 1)
  - int64 min: -9,223,372,036,854,775,808 (-2^63)
  - uint64 max: 18,446,744,073,709,551,615 (2^64 - 1)

- **NumPy Behavior**: When creating an array from the value 9,223,372,036,854,775,808, NumPy automatically selects uint64 as the dtype since the value exceeds int64's maximum. The issue occurs at line 659 in `/numpy/_core/records.py` where `_array[name] = obj` performs an implicit unsafe cast.

- **Related NumPy Functions**: The `astype()` method offers casting control with options like 'safe' (only value-preserving conversions), 'same_kind', and 'unsafe' (default). The `fromarrays` function should provide similar control to prevent data corruption.

- **Documentation Reference**: https://numpy.org/doc/stable/reference/generated/numpy.rec.fromarrays.html

## Proposed Fix

```diff
--- a/numpy/_core/records.py
+++ b/numpy/_core/records.py
@@ -656,7 +656,11 @@ def fromarrays(arrayList, dtype=None, shape=None, formats=None,
         if testshape != shape:
             raise ValueError(f'array-shape mismatch in array {k} ("{name}")')

-        _array[name] = obj
+        # Check if the assignment would cause data loss
+        if not np.can_cast(obj.dtype, descr[k].base, casting='safe'):
+            raise ValueError(f'Cannot safely cast array {k} ("{name}") '
+                           f'from {obj.dtype} to {descr[k].base} without data loss')
+        _array[name] = obj.astype(descr[k].base, casting='safe')

     return _array
```