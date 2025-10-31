# Bug Report: is_integer() and is_integer_dtype() Inconsistency

**Target**: `pandas.api.types.is_integer` and `pandas.api.types.is_integer_dtype`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`is_integer(x)` returns `True` for Python integers that don't fit in int64, but `is_integer_dtype(np.array([x]))` returns `False` for the same value. This creates an inconsistency between the scalar type checker and the dtype checker.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import pandas.api.types as pd_types


@given(st.one_of(st.integers(), st.booleans()))
def test_scalar_and_dtype_consistency_int(value):
    is_int_scalar = pd_types.is_integer(value)

    arr = np.array([value])
    is_int_dtype = pd_types.is_integer_dtype(arr)

    if is_int_scalar:
        assert is_int_dtype, f"is_integer({value})={is_int_scalar} but is_integer_dtype(np.array([{value}]))={is_int_dtype}"
```

**Failing input**: `value=-9_223_372_036_854_775_809` (one less than minimum int64)

## Reproducing the Bug

```python
import numpy as np
import pandas.api.types as pd_types

value = -9_223_372_036_854_775_809

print(f"is_integer({value}) = {pd_types.is_integer(value)}")

arr = np.array([value])
print(f"Array dtype: {arr.dtype}")
print(f"is_integer_dtype(array) = {pd_types.is_integer_dtype(arr)}")
```

Output:
```
is_integer(-9223372036854775809) = True
Array dtype: object
is_integer_dtype(array) = False
```

## Why This Is A Bug

The functions `is_integer()` and `is_integer_dtype()` are inconsistent:
- `is_integer(x)` returns `True` for any Python `int`, including values outside int64 range
- When such a value is placed in a NumPy array, it becomes object dtype
- `is_integer_dtype()` returns `False` for object dtype, even if it contains only integers

This violates user expectations: if a value is identified as an integer, an array containing only that value should be considered to have integer dtype, or at minimum the behavior should be clearly documented.

The root cause is that Python integers are arbitrary precision, while NumPy int64 has a fixed range `[-9223372036854775808, 9223372036854775807]`. Values outside this range are stored as object dtype.

## Fix

The fix depends on the intended semantics:

**Option 1**: Make `is_integer()` only return True for integers that fit in int64:

```diff
--- a/pandas/core/dtypes/common.py
+++ b/pandas/core/dtypes/common.py
@@ -xxx,x +xxx,x @@ def is_integer(obj):
     """
-    return isinstance(obj, (int, np.integer)) and not isinstance(obj, (bool, np.bool_))
+    if isinstance(obj, bool) or isinstance(obj, np.bool_):
+        return False
+    if isinstance(obj, np.integer):
+        return True
+    if isinstance(obj, int):
+        # Check if it fits in int64 range
+        return np.iinfo(np.int64).min <= obj <= np.iinfo(np.int64).max
+    return False
```

**Option 2**: Make `is_integer_dtype()` return True for object dtype if it contains integers (more complex, may have performance implications).

**Option 3**: Document this edge case clearly in both functions' docstrings.

I recommend Option 1 or Option 3, as Option 2 would require inspecting array contents which breaks the dtype-level abstraction.