# Bug Report: pandas_dtype Raises ValueError Instead of Documented TypeError

**Target**: `pandas.api.types.pandas_dtype`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `pandas_dtype` function's docstring states it "Raises TypeError if not a dtype", but for certain invalid string inputs (e.g., strings starting with digits followed by special characters like '0:', '0;', '0/'), it raises ValueError instead.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.api.types as pt

@given(st.text(min_size=1, max_size=10))
@settings(max_examples=1000)
def test_pandas_dtype_raises_typeerror_on_invalid_input(s):
    try:
        pt.pandas_dtype(s)
    except TypeError:
        pass
    except ValueError as e:
        raise AssertionError(f"pandas_dtype raised ValueError instead of documented TypeError for input {s!r}: {e}")
```

**Failing input**: `'0:'`

## Reproducing the Bug

```python
import pandas.api.types as pt

try:
    result = pt.pandas_dtype('0:')
    print(f"Unexpected success: {result}")
except TypeError as e:
    print(f"Got TypeError (as documented): {e}")
except ValueError as e:
    print(f"Got ValueError (BUG - should be TypeError): {e}")
```

Output:
```
Got ValueError (BUG - should be TypeError): format number 1 of "0:" is not recognized
```

Compare with a similar invalid input that correctly raises TypeError:
```python
try:
    result = pt.pandas_dtype('invalid')
except TypeError as e:
    print(f"Got TypeError (correct): {e}")
```

Output:
```
Got TypeError (correct): data type 'invalid' not understood
```

## Why This Is A Bug

The function's docstring explicitly states:
```
Raises
------
TypeError if not a dtype
```

However, for inputs like `'0:'`, `'1:'`, `'0;'`, `'0/'`, and similar patterns (digit followed by special character), the function raises ValueError instead. This violates the documented API contract and makes exception handling unpredictable for users.

Other invalid string inputs like `'invalid'` and `'foo:bar'` correctly raise TypeError, so the behavior is inconsistent.

## Fix

The issue stems from the function calling `np.dtype(dtype)` which can raise either TypeError or ValueError depending on the input. The fix should catch both exceptions and normalize them to TypeError as documented.

```diff
--- a/pandas/core/dtypes/common.py
+++ b/pandas/core/dtypes/common.py
@@ -1660,7 +1660,11 @@ def pandas_dtype(dtype) -> DtypeObj:
     elif isinstance(dtype, (np.dtype, ExtensionDtype)):
         return dtype

-    npdtype = np.dtype(dtype)
+    try:
+        npdtype = np.dtype(dtype)
+    except ValueError as e:
+        raise TypeError(str(e)) from e
+
     result = registry.find(npdtype)
     if result is not None:
         return result
```
