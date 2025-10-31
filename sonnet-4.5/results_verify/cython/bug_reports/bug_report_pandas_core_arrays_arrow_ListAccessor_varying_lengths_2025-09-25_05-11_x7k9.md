# Bug Report: pandas.core.arrays.arrow ListAccessor Indexing Crash

**Target**: `pandas.core.arrays.arrow.ListAccessor.__getitem__`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

ListAccessor crashes when indexing lists of varying lengths, even though the docstring example shows this is expected to work. Accessing `series.list[i]` fails with an ArrowInvalid error when some lists don't have an element at index `i`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import pandas as pd
import pyarrow as pa


@st.composite
def list_arrays(draw, min_size=0, max_size=20):
    num_lists = draw(st.integers(min_value=min_size, max_value=max_size))
    list_data = []
    for _ in range(num_lists):
        list_size = draw(st.integers(min_value=0, max_value=10))
        inner_list = draw(st.lists(
            st.integers(min_value=-100, max_value=100),
            min_size=list_size,
            max_size=list_size
        ))
        list_data.append(inner_list)

    pa_array = pa.array(list_data, type=pa.list_(pa.int64()))
    arr = pd.arrays.ArrowExtensionArray(pa_array)
    return pd.Series(arr), list_data


@given(list_arrays(min_size=1, max_size=30), st.data())
def test_list_positive_indexing(series_and_data, data):
    series, list_data = series_and_data
    max_list_len = max(len(lst) for lst in list_data) if list_data else 0
    assume(max_list_len > 0)

    idx = data.draw(st.integers(min_value=0, max_value=max_list_len - 1))

    # This should not crash - pandas typically handles missing data gracefully
    result = series.list[idx]
    assert len(result) == len(series)
```

**Failing input**: `series = Series([[], [10, 20]]), idx = 0`

## Reproducing the Bug

```python
import pandas as pd
import pyarrow as pa

list_data = [[], [10, 20]]
pa_array = pa.array(list_data, type=pa.list_(pa.int64()))
arr = pd.arrays.ArrowExtensionArray(pa_array)
series = pd.Series(arr)

result = series.list[0]
```

Output:
```
ArrowInvalid: Index 0 is out of bounds: should be in [0, 0)
```

## Why This Is A Bug

1. **API Contract Violation**: The docstring example in `ListAccessor.__getitem__` shows a Series with varying-length lists: `[[1, 2, 3], [3]]`. This example only works because the chosen index (0) is valid for all lists. There's no documentation stating that all lists must have the same length or that the index must be valid for all lists.

2. **Inconsistent with pandas behavior**: Pandas typically handles missing/invalid data gracefully by returning None/NA, not by crashing. For example:
   - `Series([1, 2, 3])[10]` raises IndexError with a clear message
   - `DataFrame.loc` with missing keys returns None/NA
   - Arrow-backed nullable arrays handle None values gracefully

3. **Poor error message**: The error "Index 0 is out of bounds: should be in [0, 0)" comes from PyArrow and is confusing for pandas users. It doesn't explain that the issue is with varying list lengths.

4. **Limits usability**: Real-world data often has lists of varying lengths. Requiring all lists to have the same length or manually checking lengths before indexing is burdensome.

## Fix

The fix should handle lists of varying lengths gracefully. Two possible approaches:

**Approach 1: Return None/NA for out-of-bounds indices** (preferred)
```python
def __getitem__(self, key: int | slice) -> Series:
    from pandas import Series

    if isinstance(key, int):
        # Handle negative indices
        if key < 0:
            list_lengths = pc.list_value_length(self._pa_array)
            key = pc.add(key, list_lengths)

        # Use PyArrow's safe element access that returns null for out-of-bounds
        # Or implement bounds checking
        list_lengths = pc.list_value_length(self._pa_array)
        is_valid = pc.greater_equal(list_lengths, key + 1)

        # Create result with null for invalid indices
        element = pc.if_else(
            is_valid,
            pc.list_element(self._pa_array, key),
            pa.scalar(None, type=self._pa_array.type.value_type)
        )
        return Series(element, dtype=ArrowDtype(element.type))
```

**Approach 2: Document the limitation**
If the fix is not feasible due to PyArrow limitations, at minimum:
- Add a clear warning in the docstring
- Raise a more informative error message (e.g., "Cannot index list at position 0: some lists are too short. All lists must have length > 0.")

Given PyArrow's constraints, Approach 1 may require element-wise operations. A hybrid solution might be to:
1. Check if all lists have sufficient length
2. If yes, use fast PyArrow `list_element`
3. If no, either use element-wise operations or raise a clear error