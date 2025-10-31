# Bug Report: pandas ListAccessor Empty Slice Crash

**Target**: `pandas.core.arrays.arrow.accessors.ListAccessor.__getitem__`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `ListAccessor.__getitem__` method crashes with `ArrowInvalid` when slicing with equal start and stop values (e.g., `[n:n]`), which should return an empty list according to Python slice semantics.

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
```

**Failing input**: `lists=[[0]], start=0, stop=0, step=None`

## Reproducing the Bug

```python
import pandas as pd
import pyarrow as pa

lists = [[0, 1, 2, 3]]
pa_array = pa.array(lists, type=pa.list_(pa.int64()))
s = pd.Series(pa_array, dtype=pd.ArrowDtype(pa.list_(pa.int64())))

result = s.list[0:0]
```

**Error**:
```
pyarrow.lib.ArrowInvalid: `start`(0) should be greater than 0 and smaller than `stop`(0)
```

**Expected behavior**: Should return a Series containing an empty list, matching Python's slice behavior where `lst[n:n]` returns `[]`.

## Why This Is A Bug

1. **Violates Python semantics**: In Python, `lst[n:n]` is a valid slice that returns an empty list. Pandas list accessor should follow the same semantics for consistency.

2. **Undocumented limitation**: The `ListAccessor.__getitem__` docstring (lines 118-147 in accessors.py) does not document that slices with equal start/stop values are invalid.

3. **User impact**: Users naturally expect Python slice semantics to work, and this crashes on a common edge case (empty slices).

4. **Inconsistency**: Regular pandas Series slicing handles `s[n:n]` correctly (returns empty Series), but `s.list[n:n]` crashes.

## Fix

The bug occurs at line 173 in `pandas/core/arrays/arrow/accessors.py`:

```python
sliced = pc.list_slice(self._pa_array, start, stop, step)
```

PyArrow's `list_slice` doesn't handle the `start == stop` edge case. Pandas should add validation before calling it:

```diff
--- a/pandas/core/arrays/arrow/accessors.py
+++ b/pandas/core/arrays/arrow/accessors.py
@@ -170,6 +170,11 @@ class ListAccessor(ArrowAccessor):
                 start = 0
             if step is None:
                 step = 1
+            # Handle empty slice case (start == stop) which PyArrow doesn't support
+            if stop is not None and start == stop:
+                # Return empty lists for all elements
+                empty_type = self._pa_array.type.value_type
+                return Series([[] for _ in range(len(self._pa_array))], dtype=ArrowDtype(pa.list_(empty_type)))
             sliced = pc.list_slice(self._pa_array, start, stop, step)
             return Series(sliced, dtype=ArrowDtype(sliced.type))
         else:
```