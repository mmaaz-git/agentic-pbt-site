# Bug Report: pandas.api.extensions.take Fails with Series/Index and allow_fill=True

**Target**: `pandas.api.extensions.take` / `pandas.core.array_algos.take.take_nd`
**Severity**: High
**Bug Type**: Crash / Logic
**Date**: 2025-09-25

## Summary

The `pandas.api.extensions.take` function crashes when passed a Series with `allow_fill=True` and produces incorrect results when passed an Index, despite documentation explicitly stating these are supported input types since version 2.1.0.

## Property-Based Test

```python
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.api.extensions import take


@given(
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=100),
    num_fills=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=100)
def test_take_series_with_allow_fill(values, num_fills):
    series = pd.Series(values)
    indices = [0] * (len(values) // 2) + [-1] * num_fills
    result = take(series, indices, allow_fill=True)
    assert len(result) == len(indices)


if __name__ == "__main__":
    # Run the test to find the failing case
    test_take_series_with_allow_fill()
```

<details>

<summary>
**Failing input**: `values=[0.0], num_fills=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 21, in <module>
    test_take_series_with_allow_fill()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 8, in test_take_series_with_allow_fill
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 15, in test_take_series_with_allow_fill
    result = take(series, indices, allow_fill=True)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/algorithms.py", line 1234, in take
    result = take_nd(
        arr, indices, axis=axis, allow_fill=True, fill_value=fill_value
    )
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/array_algos/take.py", line 110, in take_nd
    return arr.take(
           ~~~~~~~~^
        indexer, fill_value=fill_value, allow_fill=allow_fill, axis=axis
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/generic.py", line 4122, in take
    nv.validate_take((), kwargs)
    ~~~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/compat/numpy/function.py", line 86, in __call__
    validate_kwargs(fname, kwargs, self.defaults)
    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/util/_validators.py", line 164, in validate_kwargs
    _check_for_invalid_keys(fname, kwargs, compat_args)
    ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/util/_validators.py", line 138, in _check_for_invalid_keys
    raise TypeError(f"{fname}() got an unexpected keyword argument '{bad_arg}'")
TypeError: take() got an unexpected keyword argument 'allow_fill'
Falsifying example: test_take_series_with_allow_fill(
    # The test always failed when commented parts were varied together.
    values=[0.0],  # or any other generated value
    num_fills=1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import numpy as np
from pandas.api.extensions import take

# Test case demonstrating the bug

# First show that it works with numpy arrays
print("="*60)
print("Testing with numpy array (expected behavior):")
print("="*60)
arr = np.array([10.0, 20.0, 30.0])
print(f"Input array: {arr}")
print(f"Indices: [0, -1] with allow_fill=True")
result = take(arr, [0, -1], allow_fill=True)
print(f"Result: {result}")
print(f"Expected: [10. nan] - CORRECT!\n")

# Test with Series - this crashes
print("="*60)
print("Testing with pandas Series:")
print("="*60)
series = pd.Series([10.0, 20.0, 30.0])
print(f"Input Series: {series.values}")
print(f"Indices: [0, -1] with allow_fill=True")
try:
    result = take(series, [0, -1], allow_fill=True)
    print(f"Result: {result}")
except TypeError as e:
    print(f"ERROR - TypeError: {e}")
    print("Expected: Should work like numpy array and return Series([10.0, NaN])\n")

# Test with Index - this has inconsistent behavior
print("="*60)
print("Testing with pandas Index (integer):")
print("="*60)

# Integer Index with fill_value=None
print("\n1. Integer Index with fill_value=None:")
index = pd.Index([10, 20, 30])
print(f"Input Index: {index.values}")
print(f"Indices: [0, -1] with allow_fill=True, fill_value=None")
result = take(index, [0, -1], allow_fill=True, fill_value=None)
print(f"Result: {result.values}")
print(f"Expected: Should have NaN in second position")
print(f"Actual: Got {result[1]} (the last element) instead of NaN - BUG!")

# Integer Index with fill_value=-999
print("\n2. Integer Index with fill_value=-999:")
try:
    result = take(index, [0, -1], allow_fill=True, fill_value=-999)
    print(f"Result: {result.values}")
except ValueError as e:
    print(f"ERROR - ValueError: {e}")
    print("Expected: Should work and fill with -999")

# Float Index behavior
print("\n3. Float Index with fill_value=None:")
float_index = pd.Index([10.0, 20.0, 30.0])
print(f"Input Index: {float_index.values}")
print(f"Indices: [0, -1] with allow_fill=True, fill_value=None")
result = take(float_index, [0, -1], allow_fill=True, fill_value=None)
print(f"Result: {result.values}")
print(f"Expected: Should have NaN in second position")
print(f"Actual: Got {result[1]} (the last element) instead of NaN - BUG!")

print("\n4. Float Index with fill_value=np.nan:")
result = take(float_index, [0, -1], allow_fill=True, fill_value=np.nan)
print(f"Result: {result.values}")
print(f"This works correctly with NaN in second position!")

print("\n" + "="*60)
print("SUMMARY OF BUGS:")
print("="*60)
print("1. Series with allow_fill=True crashes with TypeError")
print("2. Index with fill_value=None silently ignores allow_fill parameter")
print("3. Integer Index cannot accept fill values (raises ValueError)")
print("4. Behavior is inconsistent across different input types")
```

<details>

<summary>
Multiple failure modes with Series and Index inputs
</summary>
```
============================================================
Testing with numpy array (expected behavior):
============================================================
Input array: [10. 20. 30.]
Indices: [0, -1] with allow_fill=True
Result: [10. nan]
Expected: [10. nan] - CORRECT!

============================================================
Testing with pandas Series:
============================================================
Input Series: [10. 20. 30.]
Indices: [0, -1] with allow_fill=True
ERROR - TypeError: take() got an unexpected keyword argument 'fill_value'
Expected: Should work like numpy array and return Series([10.0, NaN])

============================================================
Testing with pandas Index (integer):
============================================================

1. Integer Index with fill_value=None:
Input Index: [10 20 30]
Indices: [0, -1] with allow_fill=True, fill_value=None
Result: [10 30]
Expected: Should have NaN in second position
Actual: Got 30 (the last element) instead of NaN - BUG!

2. Integer Index with fill_value=-999:
ERROR - ValueError: Unable to fill values because Index cannot contain NA
Expected: Should work and fill with -999

3. Float Index with fill_value=None:
Input Index: [10. 20. 30.]
Indices: [0, -1] with allow_fill=True, fill_value=None
Result: [10. 30.]
Expected: Should have NaN in second position
Actual: Got 30.0 (the last element) instead of NaN - BUG!

4. Float Index with fill_value=np.nan:
Result: [10. nan]
This works correctly with NaN in second position!

============================================================
SUMMARY OF BUGS:
============================================================
1. Series with allow_fill=True crashes with TypeError
2. Index with fill_value=None silently ignores allow_fill parameter
3. Integer Index cannot accept fill values (raises ValueError)
4. Behavior is inconsistent across different input types
```
</details>

## Why This Is A Bug

This violates the documented behavior of `pandas.api.extensions.take` in multiple ways:

1. **Documentation explicitly states Series and Index are supported**: The function documentation (pandas/core/algorithms.py:1147-1149) clearly states:
   > ".. deprecated:: 2.1.0
   >     Passing an argument other than a numpy.ndarray, ExtensionArray,
   >     Index, or Series is deprecated."

   This means Series and Index are officially supported input types, not deprecated ones.

2. **Series.take() has incompatible signature**: When `take_nd` receives a Series with a numpy dtype (like float64), it incorrectly delegates to `Series.take()` which only accepts `(indices, axis, **kwargs)` and doesn't support `allow_fill` or `fill_value` parameters. This causes a TypeError crash.

3. **Index.take() silently produces wrong results**: When `fill_value=None`, Index.take() ignores the `allow_fill=True` parameter and treats -1 as a regular negative index (returning the last element) instead of a sentinel value for filling with NaN.

4. **Inconsistent behavior across types**: The same operation produces different results:
   - numpy array: Works correctly, fills with NaN
   - Series: Crashes with TypeError
   - Index with None: Silently returns wrong data
   - Index with non-None value: Raises ValueError

The root cause is in `take_nd` (pandas/core/array_algos/take.py:104-114) which incorrectly assumes that all non-numpy arrays supporting `.take()` method will accept the same parameters, when in reality Series.take() has a different signature than ExtensionArray.take().

## Relevant Context

The bug occurs in the delegation logic at pandas/core/array_algos/take.py:104-114:

```python
if not isinstance(arr, np.ndarray):
    # i.e. ExtensionArray,
    # includes for EA to catch DatetimeArray, TimedeltaArray
    if not is_1d_only_ea_dtype(arr.dtype):
        # i.e. DatetimeArray, TimedeltaArray
        arr = cast("NDArrayBackedExtensionArray", arr)
        return arr.take(
            indexer, fill_value=fill_value, allow_fill=allow_fill, axis=axis
        )

    return arr.take(indexer, fill_value=fill_value, allow_fill=allow_fill)
```

When `arr` is a Series:
- `isinstance(arr, np.ndarray)` returns False
- `is_1d_only_ea_dtype(arr.dtype)` returns False for numpy dtypes
- The code calls `arr.take()` with parameters that Series.take() doesn't accept

For Index objects, Index.take() does accept these parameters but has buggy implementation when `fill_value=None`.

Documentation links:
- pandas.api.extensions.take: https://pandas.pydata.org/docs/reference/api/pandas.api.extensions.take.html
- Series.take: https://pandas.pydata.org/docs/reference/api/pandas.Series.take.html
- Index.take: https://pandas.pydata.org/docs/reference/api/pandas.Index.take.html

## Proposed Fix

Extract the underlying array from Series/Index/DataFrame objects before processing:

```diff
--- a/pandas/core/array_algos/take.py
+++ b/pandas/core/array_algos/take.py
@@ -101,6 +101,14 @@ def take_nd(
             # so for that case cast upfront
             arr = arr.astype(dtype)

+    # Extract underlying array from Series/Index/DataFrame
+    # These have incompatible .take() methods
+    from pandas.core.generic import ABCDataFrame, ABCIndex, ABCSeries
+    if isinstance(arr, (ABCSeries, ABCIndex, ABCDataFrame)):
+        # Use ._values to get the underlying array
+        # This preserves ExtensionArray types when present
+        arr = arr._values
+
     if not isinstance(arr, np.ndarray):
         # i.e. ExtensionArray,
         # includes for EA to catch DatetimeArray, TimedeltaArray
```