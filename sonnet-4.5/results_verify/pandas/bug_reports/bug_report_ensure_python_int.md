# Bug Report: pandas.core.dtypes.common.ensure_python_int Type Signature Mismatch

**Target**: `pandas.core.dtypes.common.ensure_python_int`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ensure_python_int` function has a type signature `value: int | np.integer` but the implementation accepts float values, creating a mismatch between the declared API and actual behavior.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.core.dtypes.common import ensure_python_int

@given(st.floats(allow_nan=False, allow_infinity=False).filter(lambda x: x == int(x)))
def test_ensure_python_int_type_signature_contract(value):
    """
    The type signature says: value: int | np.integer
    But the function accepts floats that equal their integer conversion.
    This violates the type contract.
    """
    try:
        result = ensure_python_int(value)
        assert False, f"Function with signature 'int | np.integer' accepted float {value}"
    except TypeError:
        pass
```

**Failing input**: `42.0` (type: `float`)

## Reproducing the Bug

```python
import numpy as np
from pandas.core.dtypes.common import ensure_python_int

float_value = 42.0
result = ensure_python_int(float_value)
print(f"Result: {result}")
print(f"Input type: {type(float_value)}")
```

**Output:**
```
Result: 42
Input type: <class 'float'>
```

The function accepts a `float` despite the type signature declaring `value: int | np.integer`.

## Why This Is A Bug

1. **Type signature violation**: The function signature at line 93 declares `value: int | np.integer`, explicitly excluding `float`

2. **Documentation inconsistency**: The docstring (lines 94-108) states parameter type is "int or numpy.integer", but the implementation (line 109) checks `is_integer(value) or is_float(value)`

3. **Contract violation**: Type checkers (mypy, pyright) will reject valid calls like `ensure_python_int(42.0)` even though the runtime accepts them

4. **Caller confusion**: In `pandas/core/indexes/range.py`, the error message says "RangeIndex(...) must be called with integers" but then calls `ensure_python_int()` which silently accepts floats, creating inconsistent behavior

The docstring hints at conversion with "if the value isn't an int **or can't be converted to one**", suggesting float acceptance is intentional. The bug is that the type signature is too restrictive.

## Fix

Update the type signature to match the actual behavior:

```diff
-def ensure_python_int(value: int | np.integer) -> int:
+def ensure_python_int(value: int | np.integer | float) -> int:
     """
     Ensure that a value is a python int.

     Parameters
     ----------
-    value: int or numpy.integer
+    value: int, numpy.integer, or float
+        If float, must equal its integer conversion (i.e., no fractional part).

     Returns
     -------
     int

     Raises
     ------
-    TypeError: if the value isn't an int or can't be converted to one.
+    TypeError: if the value isn't an integer-like value (int, np.integer, or
+        whole-number float) or can't be converted to int.
     """
     if not (is_integer(value) or is_float(value)):
```

Alternatively, if the intention is to only accept integers, change line 109:

```diff
-    if not (is_integer(value) or is_float(value)):
+    if not is_integer(value):
```

And update the docstring to remove "or can't be converted to one" since that implies accepting other types.