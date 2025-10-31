# Bug Report: decimal_encoder Precision Loss in Round-Trip

**Target**: `fastapi.encoders.decimal_encoder`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `decimal_encoder` function loses precision when encoding `Decimal` values with fractional parts, violating its documented purpose of avoiding "failed round-tripping between encode and parse."

## Property-Based Test

```python
from decimal import Decimal
from hypothesis import given, strategies as st
from fastapi.encoders import decimal_encoder

@given(st.decimals(allow_nan=False, allow_infinity=False))
def test_decimal_encoder_round_trip(dec_value):
    encoded = decimal_encoder(dec_value)
    decoded = Decimal(str(encoded))
    assert decoded == dec_value, f"Round-trip failed: {dec_value} -> {encoded} -> {decoded}"
```

**Failing input**: `Decimal('9202420.974752872')`

## Reproducing the Bug

```python
from decimal import Decimal
from fastapi.encoders import decimal_encoder

failing_value = Decimal('9202420.974752872')

print(f"Original decimal: {failing_value}")
encoded = decimal_encoder(failing_value)
print(f"Encoded value: {encoded}")

decoded = Decimal(str(encoded))
print(f"Decoded value: {decoded}")

print(f"Round-trip preserved? {decoded == failing_value}")
print(f"Loss of precision: {failing_value - decoded}")
```

Output:
```
Original decimal: 9202420.974752872
Encoded value: 9202420.974752871
Decoded value: 9202420.974752871
Round-trip preserved? False
Loss of precision: 1E-9
```

## Why This Is A Bug

The docstring for `decimal_encoder` states:

> Encodes a Decimal as int of there's no exponent, otherwise float
>
> This is useful when we use ConstrainedDecimal to represent Numeric(x,0)
> where a integer (but not int typed) is used. **Encoding this as a float
> results in failed round-tripping between encode and parse.**

The function is registered in `ENCODERS_BY_TYPE` for **all** `Decimal` instances, not just integer decimals. When a `Decimal` with fractional precision is encoded, it's converted to a `float`, which loses precision due to float's binary representation limitations. This directly contradicts the function's stated purpose of avoiding round-trip failures.

Users relying on `Decimal` for precision (financial calculations, scientific measurements) will experience silent data corruption when their values are encoded and decoded through FastAPI's JSON serialization.

## Fix

The function should preserve precision for all `Decimal` values by converting them to strings instead of floats:

```diff
 def decimal_encoder(dec_value: Decimal) -> Union[int, float]:
     """
     Encodes a Decimal as int of there's no exponent, otherwise float

     This is useful when we use ConstrainedDecimal to represent Numeric(x,0)
     where a integer (but not int typed) is used. Encoding this as a float
     results in failed round-tripping between encode and parse.
     Our Id type is a prime example of this.

     >>> decimal_encoder(Decimal("1.0"))
-    1.0
+    "1.0"

     >>> decimal_encoder(Decimal("1"))
     1
     """
     if dec_value.as_tuple().exponent >= 0:  # type: ignore[operator]
         return int(dec_value)
     else:
-        return float(dec_value)
+        return str(dec_value)
```

Alternatively, if backward compatibility with float output is required, the function could be updated to only handle integer decimals as documented, and a new encoder could handle fractional decimals:

```diff
 def decimal_encoder(dec_value: Decimal) -> Union[int, float, str]:
     """
     Encodes a Decimal as int of there's no exponent, otherwise string

     This is useful when we use ConstrainedDecimal to represent Numeric(x,0)
     where a integer (but not int typed) is used. Encoding this as a float
     results in failed round-tripping between encode and parse.
     Our Id type is a prime example of this.

     >>> decimal_encoder(Decimal("1.0"))
-    1.0
+    "1.0"

     >>> decimal_encoder(Decimal("1"))
     1
     """
     if dec_value.as_tuple().exponent >= 0:  # type: ignore[operator]
         return int(dec_value)
     else:
-        return float(dec_value)
+        return str(dec_value)
```