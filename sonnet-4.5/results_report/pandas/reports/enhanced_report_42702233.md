# Bug Report: pandas ListAccessor Empty Slice Crash

**Target**: `pandas.core.arrays.arrow.accessors.ListAccessor.__getitem__`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `ListAccessor.__getitem__` method crashes with `ArrowInvalid` error when attempting to perform an empty slice (where start equals stop, e.g., `[0:0]`), violating Python's standard slice semantics where such operations should return empty sequences.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray
from hypothesis import given, strategies as st, settings


@given(
    lists=st.lists(
        st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=10),
        min_size=1, max_size=20
    ),
    start=st.integers(min_value=-15, max_value=15),
    stop=st.integers(min_value=-15, max_value=15) | st.none(),
    step=st.integers(min_value=1, max_value=3) | st.none()
)
@settings(max_examples=500)
def test_list_accessor_slice_consistency(lists, start, stop, step):
    pa_array = pa.array(lists, type=pa.list_(pa.int64()))
    arr = ArrowExtensionArray(pa_array)
    s = pd.Series(arr)

    try:
        sliced = s.list[start:stop:step]

        for i in range(len(s)):
            original_list = lists[i]
            sliced_value = sliced.iloc[i]
            expected_slice = original_list[start:stop:step]

            assert len(sliced_value) == len(expected_slice)
    except NotImplementedError:
        pass

# Run the test
test_list_accessor_slice_consistency()
```

<details>

<summary>
**Failing input**: `lists=[[0]], start=0, stop=0, step=None`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 38, in <module>
    test_list_accessor_slice_consistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 11, in test_list_accessor_slice_consistency
    lists=st.lists(
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 26, in test_list_accessor_slice_consistency
    sliced = s.list[start:stop:step]
             ~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/accessors.py", line 173, in __getitem__
    sliced = pc.list_slice(self._pa_array, start, stop, step)
  File "/home/npc/.local/lib/python3.13/site-packages/pyarrow/compute.py", line 269, in wrapper
    return func.call(args, options, memory_pool)
           ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pyarrow/_compute.pyx", line 407, in pyarrow._compute.Function.call
  File "pyarrow/error.pxi", line 155, in pyarrow.lib.pyarrow_internal_check_status
  File "pyarrow/error.pxi", line 92, in pyarrow.lib.check_status
pyarrow.lib.ArrowInvalid: `start`(0) should be greater than 0 and smaller than `stop`(0)
Falsifying example: test_list_accessor_slice_consistency(
    # The test sometimes passed when commented parts were varied together.
    lists=[[0]],  # or any other generated value
    start=0,  # or any other generated value
    stop=0,
    step=None,  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/33/hypo.py:34
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
import pyarrow as pa

# Create a Series with a list containing integers
lists = [[0, 1, 2, 3]]
pa_array = pa.array(lists, type=pa.list_(pa.int64()))
s = pd.Series(pa_array, dtype=pd.ArrowDtype(pa.list_(pa.int64())))

print("Original Series:")
print(s)
print()

print("Attempting to perform empty slice s.list[0:0]...")
try:
    result = s.list[0:0]
    print("Result:")
    print(result)
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
```

<details>

<summary>
Error: ArrowInvalid when attempting empty slice
</summary>
```
Original Series:
0    [0 1 2 3]
dtype: list<item: int64>[pyarrow]

Attempting to perform empty slice s.list[0:0]...
Error occurred: ArrowInvalid: `start`(0) should be greater than 0 and smaller than `stop`(0)
```
</details>

## Why This Is A Bug

This behavior violates Python's standard slice semantics and creates an inconsistency in pandas' API:

1. **Python slice semantics violation**: In standard Python, `lst[n:n]` is a valid operation that returns an empty list. For example, `[1,2,3][1:1]` returns `[]`. The ListAccessor should follow these same semantics to maintain consistency with Python's behavior.

2. **Inconsistency within pandas**: Regular pandas Series slicing correctly handles empty slices - `pd.Series([1,2,3])[1:1]` returns an empty Series. However, `s.list[1:1]` crashes with an ArrowInvalid error when `s` is a Series with ArrowDtype list.

3. **Undocumented limitation**: The `ListAccessor.__getitem__` docstring (lines 118-147 in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/arrow/accessors.py`) states it can "Index or slice lists in the Series" but does not document any restriction on slice values where start equals stop.

4. **Common edge case**: Empty slices naturally occur in data processing when indices are computed dynamically. For instance, when processing variable-length data or implementing sliding windows, it's common to encounter situations where start and stop indices are equal.

5. **PyArrow limitation not handled**: The error originates from PyArrow's `pc.list_slice` function which doesn't support empty slices. However, pandas, as a higher-level interface, should handle this edge case gracefully before delegating to PyArrow.

## Relevant Context

The crash occurs at line 173 in `pandas/core/arrays/arrow/accessors.py`:
```python
sliced = pc.list_slice(self._pa_array, start, stop, step)
```

PyArrow's `list_slice` function validates that `start` must be less than `stop`, which fails for empty slices. This is a known limitation of PyArrow's implementation, but pandas can and should handle this case before calling the underlying PyArrow function.

For reference, Python's built-in list slicing documentation states that for slice notation `s[i:j]`, "The slice of s from i to j is defined as the sequence of items with index k such that i <= k < j. If i or j is greater than len(s), use len(s). If i is omitted or None, use 0. If j is omitted or None, use len(s). If i is greater than or equal to j, the slice is empty."

## Proposed Fix

```diff
--- a/pandas/core/arrays/arrow/accessors.py
+++ b/pandas/core/arrays/arrow/accessors.py
@@ -170,6 +170,14 @@ class ListAccessor(ArrowAccessor):
                 start = 0
             if step is None:
                 step = 1
+
+            # Handle empty slice case where start equals stop
+            # PyArrow's list_slice doesn't support this, but Python semantics require it
+            if stop is not None and start == stop:
+                # Return a Series of empty lists with the same type as the original
+                empty_lists = pa.array([[]] * len(self._pa_array),
+                                      type=self._pa_array.type)
+                return Series(empty_lists, dtype=ArrowDtype(empty_lists.type))
+
             sliced = pc.list_slice(self._pa_array, start, stop, step)
             return Series(sliced, dtype=ArrowDtype(sliced.type))
         else:
```