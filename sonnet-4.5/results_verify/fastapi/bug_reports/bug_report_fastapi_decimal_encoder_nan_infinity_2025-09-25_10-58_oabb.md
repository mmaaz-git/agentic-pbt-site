# Bug Report: fastapi.encoders.decimal_encoder TypeError with NaN and Infinity

**Target**: `fastapi.encoders.decimal_encoder`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `decimal_encoder` function crashes with a `TypeError` when given Decimal values of NaN, Infinity, or -Infinity because it attempts to compare the exponent (which is a string for these special values) with an integer.

## Property-Based Test

```python
from decimal import Decimal
from hypothesis import given, strategies as st
from fastapi.encoders import decimal_encoder


@given(st.decimals(allow_nan=True, allow_infinity=True))
def test_decimal_encoder_handles_special_values(dec):
    result = decimal_encoder(dec)
```

**Failing input**: `Decimal('NaN')`, `Decimal('Infinity')`, `Decimal('-Infinity')`

## Reproducing the Bug

```python
from decimal import Decimal
from fastapi.encoders import decimal_encoder

decimal_encoder(Decimal('NaN'))
```

This raises:
```
TypeError: '>=' not supported between instances of 'str' and 'int'
```

The same error occurs with `Decimal('Infinity')` and `Decimal('-Infinity')`.

## Why This Is A Bug

For special Decimal values (NaN, Infinity, -Infinity), the `as_tuple().exponent` field is a string ('n' for NaN, 'F' for Infinity) rather than an integer. The function attempts to compare this string with 0 using `>=`, which causes a TypeError.

Normal Decimal values work fine:
- `Decimal('1')` has `exponent=0` (int)
- `Decimal('1.0')` has `exponent=-1` (int)

But special values have:
- `Decimal('NaN')` has `exponent='n'` (str)
- `Decimal('Infinity')` has `exponent='F'` (str)

## Fix

```diff
--- a/fastapi/encoders.py
+++ b/fastapi/encoders.py
@@ -49,6 +49,10 @@ def decimal_encoder(dec_value: Decimal) -> Union[int, float]:
     >>> decimal_encoder(Decimal("1"))
     1
     """
+    # Handle special values (NaN, Infinity, -Infinity)
+    if not dec_value.is_finite():
+        return float(dec_value)
+
     if dec_value.as_tuple().exponent >= 0:  # type: ignore[operator]
         return int(dec_value)
     else:
```