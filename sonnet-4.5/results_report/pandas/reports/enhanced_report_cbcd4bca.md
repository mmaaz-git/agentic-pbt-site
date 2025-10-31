# Bug Report: pandas.core.arrays.arrow ListAccessor.__getitem__ Crashes on Variable-Length Lists

**Target**: `pandas.core.arrays.arrow.accessors.ListAccessor.__getitem__`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The ListAccessor.__getitem__ method crashes with an ArrowInvalid exception when accessing an index that is out of bounds for any list in a Series containing lists of different lengths, instead of returning NA for lists that don't have that index.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
import pyarrow as pa


@settings(max_examples=500)
@given(
    st.lists(
        st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=10),
        min_size=1,
        max_size=20
    ),
    st.integers(min_value=0, max_value=9)
)
def test_list_accessor_getitem_returns_correct_element(lists_of_ints, index):
    s = pd.Series(lists_of_ints, dtype=pd.ArrowDtype(pa.list_(pa.int64())))
    result = s.list[index]

    expected = [lst[index] if index < len(lst) else None for lst in lists_of_ints]

    for i, (res, exp) in enumerate(zip(result, expected)):
        if exp is None:
            assert pd.isna(res)
        else:
            assert res == exp
```

<details>

<summary>
**Failing input**: `lists_of_ints=[[0]], index=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 29, in <module>
    test_list_accessor_getitem_returns_correct_element()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 7, in test_list_accessor_getitem_returns_correct_element
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 17, in test_list_accessor_getitem_returns_correct_element
    result = s.list[index]
             ~~~~~~^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/accessors.py", line 155, in __getitem__
    element = pc.list_element(self._pa_array, key)
  File "/home/npc/.local/lib/python3.13/site-packages/pyarrow/compute.py", line 252, in wrapper
    return func.call(args, None, memory_pool)
           ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pyarrow/_compute.pyx", line 407, in pyarrow._compute.Function.call
  File "pyarrow/error.pxi", line 155, in pyarrow.lib.pyarrow_internal_check_status
  File "pyarrow/error.pxi", line 92, in pyarrow.lib.check_status
pyarrow.lib.ArrowInvalid: Index 1 is out of bounds: should be in [0, 1)
Falsifying example: test_list_accessor_getitem_returns_correct_element(
    lists_of_ints=[[0]],
    index=1,
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import pyarrow as pa

# Create a Series with lists of different lengths
s = pd.Series(
    [[1, 2, 3], [4]],
    dtype=pd.ArrowDtype(pa.list_(pa.int64()))
)

# Try accessing index 0 (this should work)
print("Accessing index 0:")
print(s.list[0])
print()

# Try accessing index 1 (this should fail for the second list)
print("Accessing index 1:")
try:
    result = s.list[1]
    print(result)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
ArrowInvalid exception raised when accessing out-of-bounds index
</summary>
```
Accessing index 0:
0    1
1    4
dtype: int64[pyarrow]

Accessing index 1:
Error: ArrowInvalid: Index 1 is out of bounds: should be in [0, 1)
```
</details>

## Why This Is A Bug

1. **Violates pandas' established patterns**: Throughout pandas, out-of-bounds access returns NA/null rather than raising exceptions. The Series.str accessor, for example, returns NaN when accessing an index beyond string length. Users expect `s.list[1]` on `[[1,2,3], [4]]` to return `[2, NA]`, not crash.

2. **Contradicts the docstring**: The __getitem__ docstring states "Index or slice of indices to access from each list" (emphasis on "from each list"), implying independent processing of each list. There's no mention that the index must be valid for ALL lists simultaneously.

3. **Confusing error message**: The PyArrow error "Index 1 is out of bounds: should be in [0, 1)" doesn't clarify that the issue is with ONE list being shorter than the requested index, making debugging difficult for users.

4. **Inconsistent with regular Python lists in Series**: Using `apply()` on a Series of regular Python lists handles this gracefully by returning None for out-of-bounds indices.

5. **Common use case**: Variable-length lists are extremely common in real-world data (e.g., tokenized text, transaction histories, sensor readings). Requiring all lists to have the same minimum length is an unrealistic constraint.

## Relevant Context

The issue stems from line 155 in `/pandas/core/arrays/arrow/accessors.py` where the method directly calls `pc.list_element(self._pa_array, key)` without checking if the index is valid for all lists. PyArrow's `list_element` function requires the index to be valid for all lists in the array.

The ListAccessor was recently added to pandas (PR #55777) and this edge case appears to have been overlooked. Other pandas accessors like Series.str already handle this pattern correctly, establishing a clear precedent for the expected behavior.

Relevant documentation:
- pandas ListAccessor: https://pandas.pydata.org/docs/reference/api/pandas.Series.list.__getitem__.html
- PyArrow list_element: https://arrow.apache.org/docs/python/generated/pyarrow.compute.list_element.html

## Proposed Fix

```diff
--- a/pandas/core/arrays/arrow/accessors.py
+++ b/pandas/core/arrays/arrow/accessors.py
@@ -148,9 +148,35 @@ class ListAccessor(ArrowAccessor):
         from pandas import Series

         if isinstance(key, int):
-            # TODO: Support negative key but pyarrow does not allow
-            # element index to be an array.
-            element = pc.list_element(self._pa_array, key)
+            if key < 0:
+                raise NotImplementedError("Negative indexing is not yet supported")
+
+            # Check if all lists have the requested index
+            lengths = pc.list_value_length(self._pa_array)
+            min_length = pc.min(lengths).as_py()
+
+            if min_length <= key:
+                # Some lists don't have this index - handle element-wise
+                import pyarrow as pa
+
+                # Build a new array with nulls for out-of-bounds indices
+                result_chunks = []
+                for chunk in self._pa_array.chunks:
+                    chunk_lengths = pc.list_value_length(chunk)
+                    mask = pc.greater(chunk_lengths, key)
+
+                    # Get elements where mask is true, null otherwise
+                    valid_elements = pc.list_element(
+                        pc.filter(chunk, mask),
+                        key
+                    )
+
+                    # Construct result array with nulls for invalid indices
+                    result = pc.if_else(mask, pc.list_element(chunk, key), None)
+                    result_chunks.append(result)
+
+                element = pa.chunked_array(result_chunks)
+            else:
+                element = pc.list_element(self._pa_array, key)
             return Series(element, dtype=ArrowDtype(element.type))
         elif isinstance(key, slice):
```