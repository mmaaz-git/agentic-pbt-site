# Bug Report: pandas.core.array_algos.masked_accumulations Unexpectedly Mutates Input Arrays

**Target**: `pandas.core.array_algos.masked_accumulations.cumsum` (and related: `cumprod`, `cummin`, `cummax`)
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The masked accumulation functions in pandas unexpectedly modify the input array when the mask contains `True` values, violating the contract that functions returning new arrays should not mutate their inputs.

## Property-Based Test

```python
import numpy as np
from pandas.core.array_algos.masked_accumulations import cumsum
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as npst


@given(
    values=npst.arrays(dtype=np.int64, shape=st.integers(5, 20),
                      elements=st.integers(-100, 100))
)
def test_cumsum_should_not_mutate_input(values):
    """cumsum should not modify the input array."""
    original = values.copy()
    mask = np.zeros(len(values), dtype=bool)
    mask[len(values) // 2] = True  # Set at least one mask value to True

    result_values, result_mask = cumsum(values, mask, skipna=True)

    # This assertion fails!
    np.testing.assert_array_equal(
        values, original,
        err_msg="cumsum modified the input array!"
    )

    print(f"✓ Test passed with values: {original[:5]}... (length {len(original)})")


if __name__ == "__main__":
    # Run the test
    test_cumsum_should_not_mutate_input()
```

<details>

<summary>
**Failing input**: `values=array([1, 1, 1, 1, 1])`
</summary>
```
✓ Test passed with values: [0 0 0 0 0]... (length 5)
✓ Test passed with values: [0 0 0 0 0]... (length 16)
✓ Test passed with values: [0 0 0 0 0]... (length 7)
✓ Test passed with values: [0 0 0 0 0]... (length 14)
✓ Test passed with values: [0 0 0 0 0]... (length 5)
✓ Test passed with values: [0 0 0 0 0]... (length 19)
✓ Test passed with values: [0 0 0 0 0]... (length 15)
✓ Test passed with values: [0 0 0 0 0]... (length 14)
✓ Test passed with values: [0 0 0 0 0]... (length 11)
✓ Test passed with values: [0 0 0 0 0]... (length 20)
✓ Test passed with values: [  0 -88   0   0   0]... (length 20)
✓ Test passed with values: [  0 -88   0   0   0]... (length 20)
✓ Test passed with values: [ 0 20  0  0  0]... (length 20)
✓ Test passed with values: [0 0 0 0 0]... (length 16)
✓ Test passed with values: [0 0 0 0 0]... (length 16)
✓ Test passed with values: [0 0 0 0 0]... (length 17)
✓ Test passed with values: [0 0 0 0 0]... (length 17)
✓ Test passed with values: [0 0 0 0 0]... (length 17)
✓ Test passed with values: [0 0 0 0 0]... (length 17)
✓ Test passed with values: [-88   0   0   0   0]... (length 20)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 30, in <module>
    test_cumsum_should_not_mutate_input()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 8, in test_cumsum_should_not_mutate_input
    values=npst.arrays(dtype=np.int64, shape=st.integers(5, 20),
              ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 20, in test_cumsum_should_not_mutate_input
    np.testing.assert_array_equal(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        values, original,
        ^^^^^^^^^^^^^^^^^
        err_msg="cumsum modified the input array!"
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 1051, in assert_array_equal
    assert_array_compare(operator.__eq__, actual, desired, err_msg=err_msg,
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         verbose=verbose, header='Arrays are not equal',
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         strict=strict)
                         ^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 916, in assert_array_compare
    raise AssertionError(msg)
AssertionError:
Arrays are not equal
cumsum modified the input array!
Mismatched elements: 1 / 5 (20%)
Max absolute difference among violations: 1
Max relative difference among violations: 1.
 ACTUAL: array([1, 1, 0, 1, 1])
 DESIRED: array([1, 1, 1, 1, 1])
Falsifying example: test_cumsum_should_not_mutate_input(
    values=array([1, 1, 1, 1, 1]),
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1009
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1010
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1093
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1615
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py:862
        (and 4 more with settings.verbosity >= verbose)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.array_algos.masked_accumulations import cumsum

# Create test data
values = np.array([1, 2, 3, 4, 5])
mask = np.array([False, True, False, True, False])

print("Before cumsum:")
print(f"  values: {values}")
print(f"  mask:   {mask}")

# Call cumsum function
result_values, result_mask = cumsum(values, mask, skipna=True)

print("\nAfter cumsum:")
print(f"  values (input):  {values}")
print(f"  result_values:   {result_values}")
print(f"  result_mask:     {result_mask}")

# Check if input was modified
if not np.array_equal(values, np.array([1, 2, 3, 4, 5])):
    print("\n❌ BUG CONFIRMED: Input array was modified!")
    print(f"   Original values: [1, 2, 3, 4, 5]")
    print(f"   Modified values: {values}")
else:
    print("\n✓ Input array was not modified")
```

<details>

<summary>
Output shows input array mutation at masked positions
</summary>
```
Before cumsum:
  values: [1 2 3 4 5]
  mask:   [False  True False  True False]

After cumsum:
  values (input):  [1 0 3 0 5]
  result_values:   [1 1 4 4 9]
  result_mask:     [False  True False  True False]

❌ BUG CONFIRMED: Input array was modified!
   Original values: [1, 2, 3, 4, 5]
   Modified values: [1 0 3 0 5]
```
</details>

## Why This Is A Bug

This behavior violates expected function contracts and causes data corruption:

1. **Unexpected mutation**: The function modifies its input array at positions where `mask=True`, setting them to fill values (0 for cumsum, 1 for cumprod, etc.). This happens on line 68 of `masked_accumulations.py`: `values[mask] = fill_value`.

2. **Contradictory behavior**: The function both modifies the input AND returns a new array (from `np.cumsum` on line 73), which is highly unusual. Standard NumPy/pandas functions either modify in-place OR return a new array, never both.

3. **Documentation mismatch**: While the docstring mentions "We will modify values in place", this is misleading because:
   - The function returns a new array from the numpy accumulation function
   - The caller (`BaseMaskedArray._accumulate`) expects to receive new arrays to pass to `_simple_new`
   - No public API documentation warns users about this side effect

4. **Real impact on pandas users**: When using pandas nullable dtypes (Int64, Float64, etc.), calling `.cumsum()` on a Series will corrupt the underlying data array. The corruption happens silently and persists, potentially affecting all subsequent operations on that Series.

## Relevant Context

The bug occurs in the internal implementation used by pandas' masked arrays (nullable integer and float dtypes). The problematic code path:

1. User calls `Series.cumsum()` on a Series with nullable dtype (e.g., `pd.Int64Dtype()`)
2. This calls `BaseMaskedArray._accumulate()` in `/pandas/core/arrays/masked.py:1574`
3. Line 1577 takes a reference (not a copy) of the internal data: `data = self._data`
4. Line 1581 calls the accumulation function which modifies this data in place
5. The Series' internal data is now corrupted with fill values at masked positions

The bug affects all four accumulation functions: `cumsum`, `cumprod`, `cummin`, and `cummax`.

Documentation: https://pandas.pydata.org/docs/reference/api/pandas.Series.cumsum.html
Source code: https://github.com/pandas-dev/pandas/blob/main/pandas/core/array_algos/masked_accumulations.py

## Proposed Fix

```diff
--- a/pandas/core/array_algos/masked_accumulations.py
+++ b/pandas/core/array_algos/masked_accumulations.py
@@ -65,6 +65,7 @@ def _cum_func(
             f"No accumulation for {func} implemented on BaseMaskedArray"
         )

+    values = values.copy()
     values[mask] = fill_value

     if not skipna:
```