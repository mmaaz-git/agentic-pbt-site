# Bug Report: pandas.core.sparse.SparseArray fill_value Setter Mutates Data

**Target**: `pandas.core.arrays.sparse.SparseArray.fill_value` (setter)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Setting the `fill_value` property of a SparseArray can unexpectedly mutate the logical data represented by the array when the original data contains values equal to the original fill_value.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.arrays import SparseArray

@given(
    data=st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50)
)
def test_fill_value_setter_preserves_data(data):
    sparse = SparseArray(data, fill_value=0)
    original_dense = sparse.to_dense()

    sparse.fill_value = 999

    assert np.array_equal(sparse.to_dense(), original_dense)
```

**Failing input**: `data=[0]`

## Reproducing the Bug

```python
import numpy as np
from pandas.arrays import SparseArray

data = [0, 0, 0]
sparse = SparseArray(data, fill_value=0)
print(f"Original data: {sparse.to_dense()}")

sparse.fill_value = 999
print(f"After setting fill_value=999: {sparse.to_dense()}")
```

**Output:**
```
Original data: [0 0 0]
After setting fill_value=999: [999 999 999]
```

**Expected:** Data should remain `[0 0 0]`
**Actual:** Data becomes `[999 999 999]`

## Why This Is A Bug

The `fill_value` setter is a property that users expect to be a metadata change, not a data-mutating operation. When all original data values equal the fill_value, the SparseArray stores them implicitly (sp_values is empty). Changing fill_value then changes what these implicit values represent when converting back to dense format.

This violates the principle of least surprise - a property setter should not mutate the logical data. The current implementation at `array.py:659-661` only updates the dtype:

```python
@fill_value.setter
def fill_value(self, value) -> None:
    self._dtype = SparseDtype(self.dtype.subtype, value)
```

When `to_dense()` is called (line 593), it creates an array filled with the **current** fill_value, not the original:
```python
out = np.full(self.shape, fill_value, dtype=dtype)
out[self.sp_index.indices] = self.sp_values
```

## Fix

The fill_value setter should materialize implicit values before changing the fill_value. Here's a high-level approach:

1. When setting a new fill_value, check if any positions currently hold the implicit fill_value
2. If so, materialize those positions into sp_values and sp_index
3. Then update the fill_value

Alternatively, document this as a known limitation and recommend users create a new SparseArray instead of mutating fill_value:

```python
# Instead of:
sparse.fill_value = new_value

# Use:
sparse = SparseArray(sparse.to_dense(), fill_value=new_value)
```

Or disable the setter entirely and make fill_value read-only after construction.