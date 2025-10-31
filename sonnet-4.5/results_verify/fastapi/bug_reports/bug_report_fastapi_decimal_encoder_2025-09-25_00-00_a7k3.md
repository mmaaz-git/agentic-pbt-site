# Bug Report: decimal_encoder Crashes on Special Decimal Values

**Target**: `fastapi.encoders.decimal_encoder`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `decimal_encoder` function crashes with a TypeError when given special Decimal values like `Infinity`, `-Infinity`, or `NaN`. This violates the function's implicit contract of handling all Decimal values.

## Property-Based Test

```python
from decimal import Decimal
from hypothesis import given, strategies as st
from fastapi.encoders import decimal_encoder


@st.composite
def decimal_with_special_values(draw):
    """Generate Decimal values including special values like Infinity and NaN."""
    choice = draw(st.integers(0, 9))
    if choice == 0:
        return Decimal('Infinity')
    elif choice == 1:
        return Decimal('-Infinity')
    elif choice == 2:
        return Decimal('NaN')
    else:
        return draw(st.decimals(allow_nan=False, allow_infinity=False))


@given(decimal_with_special_values())
def test_decimal_encoder_handles_all_decimal_values(dec_value):
    result = decimal_encoder(dec_value)
    assert isinstance(result, (int, float))
```

**Failing input**: `Decimal('Infinity')`

## Reproducing the Bug

```python
from decimal import Decimal
from fastapi.encoders import decimal_encoder

decimal_encoder(Decimal('Infinity'))
```

**Output:**
```
TypeError: '>=' not supported between instances of 'str' and 'int'
```

## Why This Is A Bug

The `decimal_encoder` function is used by `jsonable_encoder` to convert Decimal objects to JSON-serializable values. While special values like Infinity and NaN are edge cases, they are valid Decimal values that users might encounter. The function should either:

1. Handle these values gracefully by converting them to appropriate Python float equivalents (`float('inf')`, `float('-inf')`, `float('nan')`), or
2. Raise a more informative error message

The current behavior crashes with a confusing TypeError because `Decimal.as_tuple().exponent` returns the string `'F'` for Infinity (and `'n'` for NaN), which can't be compared with the integer `0`.

**Root cause:** Line 52 in `/home/npc/miniconda/lib/python3.13/site-packages/fastapi/encoders.py`:
```python
if dec_value.as_tuple().exponent >= 0:  # type: ignore[operator]
```

When `dec_value` is Infinity, `dec_value.as_tuple().exponent` is the string `'F'`, causing the comparison to fail.

## Fix

```diff
--- a/fastapi/encoders.py
+++ b/fastapi/encoders.py
@@ -49,6 +49,12 @@ def decimal_encoder(dec_value: Decimal) -> Union[int, float]:
     >>> decimal_encoder(Decimal("1"))
     1
     """
+    # Handle special values (Infinity, -Infinity, NaN)
+    if dec_value.is_infinite():
+        return float('inf') if dec_value > 0 else float('-inf')
+    if dec_value.is_nan():
+        return float('nan')
+
     if dec_value.as_tuple().exponent >= 0:  # type: ignore[operator]
         return int(dec_value)
     else:
```