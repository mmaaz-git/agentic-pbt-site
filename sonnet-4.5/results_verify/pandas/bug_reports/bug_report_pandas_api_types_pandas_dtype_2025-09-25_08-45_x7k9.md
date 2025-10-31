# Bug Report: pandas_dtype None Handling Inconsistency

**Target**: `pandas.api.types.pandas_dtype`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `pandas_dtype` function silently accepts `None` and returns `float64`, while the related internal function `_get_dtype` raises a `TypeError` for `None`. This inconsistency violates the expected behavior documented in `_get_dtype` and creates surprising behavior where `None` is treated as a valid dtype.

## Property-Based Test

```python
from pandas.api.types import pandas_dtype
from pandas.core.dtypes.common import _get_dtype
from hypothesis import given, strategies as st

@given(st.none())
def test_pandas_dtype_none_should_raise_typeerror(value):
    try:
        result = pandas_dtype(value)
        assert False, f"pandas_dtype(None) should raise TypeError, but returned {result}"
    except TypeError:
        pass
```

**Failing input**: `value=None`

## Reproducing the Bug

```python
from pandas.api.types import pandas_dtype
from pandas.core.dtypes.common import _get_dtype

result = pandas_dtype(None)
print(f"pandas_dtype(None) = {result}")

try:
    result = _get_dtype(None)
except TypeError as e:
    print(f"_get_dtype(None) raised TypeError: {e}")
```

**Output:**
```
pandas_dtype(None) = float64
_get_dtype(None) raised TypeError: Cannot deduce dtype from null object
```

## Why This Is A Bug

1. **Inconsistent behavior**: The internal helper function `_get_dtype` explicitly checks for `None` and raises a `TypeError` with message "Cannot deduce dtype from null object", but the public API `pandas_dtype` does not.

2. **Documentation mismatch**: The `pandas_dtype` docstring states "Raises TypeError if not a dtype", which implies `None` should raise an error.

3. **Surprising behavior**: Silently converting `None` to `float64` is unexpected and could lead to hard-to-debug issues. This happens because `pandas_dtype` relies on numpy's `np.dtype(None)` which returns `float64`.

4. **API inconsistency**: Related dtype checking functions like `is_dtype_equal` and other functions in the module treat `None` specially, often returning `False` or raising errors.

## Fix

```diff
--- a/pandas/core/dtypes/common.py
+++ b/pandas/core/dtypes/common.py
@@ -1606,6 +1606,10 @@ def validate_all_hashable(*args, error_name: str | None = None) -> None:
 def pandas_dtype(dtype) -> DtypeObj:
     """
     Convert input into a pandas only dtype object or a numpy dtype object.

     Parameters
     ----------
     dtype : object to be converted

     Returns
     -------
     np.dtype or a pandas dtype

     Raises
     ------
     TypeError if not a dtype

     Examples
     --------
     >>> pd.api.types.pandas_dtype(int)
     dtype('int64')
     """
+    # Explicitly check for None to match _get_dtype behavior
+    if dtype is None:
+        raise TypeError("Cannot deduce dtype from null object")
+
     # short-circuit
     if isinstance(dtype, np.ndarray):
         return dtype.dtype
```