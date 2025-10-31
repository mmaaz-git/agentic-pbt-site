# Bug Report: pandas.Index.insert Large Integer ValueError

**Target**: `pandas.core.indexes.base.Index.insert`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Inserting a very large integer (beyond int64 range) into an int64 Index causes a cryptic `ValueError: Invalid integer data type 'O'` instead of either succeeding by upcasting to object dtype or failing with a clear TypeError.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st, assume
import pandas as pd


@st.composite
def index_strategy(draw):
    elements = draw(st.lists(
        st.one_of(
            st.integers(min_value=-1000, max_value=1000),
            st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
            st.text(min_size=0, max_size=10)
        ),
        min_size=0,
        max_size=50
    ))
    try:
        return pd.Index(elements)
    except:
        assume(False)


@given(index_strategy(), st.integers(min_value=0, max_value=10), st.integers())
@settings(max_examples=500)
def test_insert_increases_length(idx, pos, item):
    assume(pos <= len(idx))
    inserted = idx.insert(pos, item)
    assert len(inserted) == len(idx) + 1
```

**Failing input**: `idx=Index([0], dtype='int64'), pos=0, item=18446744073709551616`

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

idx = pd.Index([0], dtype='int64')
item = 18_446_744_073_709_551_616

result = idx.insert(0, item)
```

**Output:**
```
ValueError: Invalid integer data type 'O'.
```

**Full traceback shows the error occurs in:**
- `pandas/core/dtypes/cast.py:1344` in `find_result_type`
- When calling `np.iinfo(right_dtype)` where `right_dtype` is object dtype

## Why This Is A Bug

1. **Incorrect assumption**: The code in `find_result_type` (cast.py:1344) assumes that `np.min_scalar_type(right)` always returns a numeric dtype, but for very large integers it returns object dtype.

2. **Improper error handling**: When the value doesn't fit in int64, pandas attempts to find a compatible dtype but crashes with a cryptic internal error instead of:
   - Successfully converting to object dtype (which can hold any Python object), OR
   - Raising a clear TypeError explaining the incompatibility

3. **Inconsistent behavior**: Inserting the same large integer into an object-dtype Index works fine, but int64 Index crashes instead of auto-converting.

## Fix

The bug is in `pandas/core/dtypes/cast.py` around line 1344. The code needs to check if `right_dtype` is an object dtype before calling `np.iinfo()`:

```diff
--- a/pandas/core/dtypes/cast.py
+++ b/pandas/core/dtypes/cast.py
@@ -1336,11 +1336,12 @@
             # We should check if the corresponding int dtype (e.g. int64 for uint64)
             # can hold the number
             right_dtype = np.min_scalar_type(right)
             if right == 0:
                 # Special case 0
                 right = left_dtype
             elif (
+                right_dtype.kind != 'O'
+                and not np.issubdtype(left_dtype, np.unsignedinteger)
-                not np.issubdtype(left_dtype, np.unsignedinteger)
                 and 0 < right <= np.iinfo(right_dtype).max
             ):
                 # If left dtype isn't unsigned, check if it fits in the signed dtype
                 right = np.dtype(f"i{right_dtype.itemsize}")
             else:
                 right = right_dtype
```

This ensures that `np.iinfo()` is only called on actual integer dtypes, not object dtype.