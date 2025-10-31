# Bug Report: pandas.io.parsers validate_integer Min Value Validation

**Target**: `pandas.io.parsers.readers.validate_integer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `validate_integer` function fails to validate that float values satisfy the minimum value requirement. When a float can be losslessly converted to an integer, the function skips the minimum value check, allowing negative or otherwise invalid values to pass validation.

## Property-Based Test

```python
import pytest
from hypothesis import given, settings, strategies as st
from pandas.io.parsers.readers import validate_integer


@settings(max_examples=500)
@given(
    val=st.floats(allow_nan=False, allow_infinity=False),
    min_val=st.integers(min_value=0, max_value=1000)
)
def test_validate_integer_respects_min_val_for_floats(val, min_val):
    if val != int(val):
        return

    if int(val) >= min_val:
        result = validate_integer("test", val, min_val)
        assert result >= min_val
    else:
        with pytest.raises(ValueError, match="must be an integer"):
            validate_integer("test", val, min_val)
```

**Failing input**: `val=-1.0, min_val=0`

## Reproducing the Bug

```python
from pandas.io.parsers.readers import validate_integer

result = validate_integer("chunksize", -1.0, 1)
print(f"Result: {result}")
```

Expected: ValueError("'chunksize' must be an integer >=1")
Actual: Returns -1 without raising an error

This affects real usage when users pass invalid parameters:
```python
import pandas as pd
import io

csv_data = "a,b,c\n1,2,3\n4,5,6"
reader = pd.read_csv(io.StringIO(csv_data), chunksize=-1.0)
```

## Why This Is A Bug

The function's docstring explicitly states: "Minimum allowed value (val < min_val will result in a ValueError)". However, when `val` is a float that can be losslessly converted to int, the minimum value check is skipped entirely.

This violates the documented contract and allows invalid parameter values (like negative chunksizes) to be silently accepted, which could cause unexpected behavior or errors later in the parsing process.

## Fix

```diff
--- a/pandas/io/parsers/readers.py
+++ b/pandas/io/parsers/readers.py
@@ -547,9 +547,11 @@ def validate_integer(

     msg = f"'{name:s}' must be an integer >={min_val:d}"
     if is_float(val):
         if int(val) != val:
             raise ValueError(msg)
         val = int(val)
+        if val < min_val:
+            raise ValueError(msg)
     elif not (is_integer(val) and val >= min_val):
         raise ValueError(msg)

     return int(val)
```
