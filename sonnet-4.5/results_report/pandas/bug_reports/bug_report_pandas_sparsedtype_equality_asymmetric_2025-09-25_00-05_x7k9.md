# Bug Report: SparseDtype Equality Not Symmetric

**Target**: `pandas.core.dtypes.dtypes.SparseDtype.__eq__`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

SparseDtype's equality operator is not symmetric: `dtype1 == dtype2` can return a different result than `dtype2 == dtype1`, violating a fundamental property of equality.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.core.dtypes.dtypes import SparseDtype

@st.composite
def valid_sparse_dtypes(draw):
    base_dtype = draw(st.sampled_from([np.float32, np.float64]))
    use_default = draw(st.booleans())
    if use_default:
        return SparseDtype(base_dtype)
    fill_value = draw(st.one_of(
        st.floats(allow_nan=True, allow_infinity=True),
        st.floats(allow_nan=False, allow_infinity=False)
    ))
    return SparseDtype(base_dtype, fill_value)

@given(valid_sparse_dtypes(), valid_sparse_dtypes())
def test_equality_symmetric(dtype1, dtype2):
    """Property: If dtype1 == dtype2, then dtype2 == dtype1"""
    if dtype1 == dtype2:
        assert dtype2 == dtype1, \
            f"Equality not symmetric: {dtype1} == {dtype2} but {dtype2} != {dtype1}"
```

**Failing input**: `dtype1=Sparse[float32, nan]`, `dtype2=Sparse[float32, 0.0]`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.dtypes.dtypes import SparseDtype

dtype1 = SparseDtype(np.float32, np.nan)
dtype2 = SparseDtype(np.float32, 0.0)

print(f"dtype1 == dtype2: {dtype1 == dtype2}")
print(f"dtype2 == dtype1: {dtype2 == dtype1}")
```

**Output:**
```
dtype1 == dtype2: True
dtype2 == dtype1: False
```

## Why This Is A Bug

Equality must be symmetric - if `a == b` is True, then `b == a` must also be True. This is a fundamental requirement for equality operators in Python and mathematics.

The bug occurs when comparing SparseDtype objects where one has a NaN fill_value and the other has a non-NaN fill_value. The asymmetry arises from different code paths in `__eq__`:

1. When `self._is_na_fill_value` is True (NaN fill value):
   - Uses type checking: `isinstance(other.fill_value, type(self.fill_value))`
   - For `Sparse[float32, nan] == Sparse[float32, 0.0]`: checks `isinstance(0.0, type(nan))` → True

2. When `self._is_na_fill_value` is False (non-NaN fill value):
   - Uses value equality: `self.fill_value == other.fill_value`
   - For `Sparse[float32, 0.0] == Sparse[float32, nan]`: checks `0.0 == nan` → False

This asymmetry violates:
- Python's equality contract
- Hash consistency (equal objects must have equal hashes)
- User expectations
- Use in sets, dictionaries, and other containers that rely on consistent equality

## Fix

The issue is in the `__eq__` method at lines 1707-1717. The type-based check creates asymmetry. The fix should ensure that when comparing a NaN fill value with a non-NaN fill value, both directions return False.

```diff
diff --git a/pandas/core/dtypes/dtypes.py b/pandas/core/dtypes/dtypes.py
index 1234567..abcdefg 100644
--- a/pandas/core/dtypes/dtypes.py
+++ b/pandas/core/dtypes/dtypes.py
@@ -1707,10 +1707,11 @@ class SparseDtype(ExtensionDtype):
             if self._is_na_fill_value:
                 # this case is complicated by two things:
                 # SparseDtype(float, float(nan)) == SparseDtype(float, np.nan)
                 # SparseDtype(float, np.nan)     != SparseDtype(float, pd.NaT)
                 # i.e. we want to treat any floating-point NaN as equal, but
                 # not a floating-point NaN and a datetime NaT.
+                # Also ensure that NaN != non-NaN for symmetry
                 fill_value = (
                     other._is_na_fill_value
-                    and isinstance(self.fill_value, type(other.fill_value))
-                    or isinstance(other.fill_value, type(self.fill_value))
+                    and (isinstance(self.fill_value, type(other.fill_value))
+                         or isinstance(other.fill_value, type(self.fill_value)))
                 )
             else:
                 with warnings.catch_warnings():
```

The fix changes the boolean logic to ensure that `fill_value` is True only when BOTH dtypes have NaN fill values AND their types are compatible. This makes the comparison symmetric.