# Bug Report: pydantic.deprecated.json.decimal_encoder Integer Detection

**Target**: `pydantic.deprecated.json.decimal_encoder`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `decimal_encoder` function incorrectly detects integer-valued Decimals, causing values like `Decimal('1.0')` and `Decimal('42.00')` to be encoded as floats instead of ints, contradicting the function's documented purpose of preserving integer values.

## Property-Based Test

```python
from decimal import Decimal
from hypothesis import given, strategies as st, example
from pydantic.deprecated.json import decimal_encoder


@given(st.decimals(allow_nan=False, allow_infinity=False))
@example(Decimal('1.0'))
@example(Decimal('42.00'))
def test_decimal_encoder_integer_values_encode_as_int(dec):
    encoded = decimal_encoder(dec)
    is_integer_value = dec == dec.to_integral_value()

    if is_integer_value:
        assert isinstance(encoded, int), (
            f"Integer-valued Decimal {dec} (exponent={dec.as_tuple().exponent}) "
            f"should encode as int, got {type(encoded).__name__}"
        )
```

**Failing inputs**:
- `Decimal('1.0')` - exponent=-1, encodes as float instead of int
- `Decimal('42.00')` - exponent=-2, encodes as float instead of int
- `Decimal('100.000')` - exponent=-3, encodes as float instead of int

## Reproducing the Bug

```python
from decimal import Decimal
from pydantic.deprecated.json import decimal_encoder

dec = Decimal('1.0')
encoded = decimal_encoder(dec)

print(f"Input: {dec}")
print(f"Is integer value: {dec == dec.to_integral_value()}")
print(f"Exponent: {dec.as_tuple().exponent}")
print(f"Encoded as: {type(encoded).__name__} = {encoded}")
print(f"Expected: int (to preserve integer value)")
print(f"Actual: {type(encoded).__name__}")
```

Output:
```
Input: 1.0
Is integer value: True
Exponent: -1
Encoded as: float = 1.0
Expected: int (to preserve integer value)
Actual: float
```

## Why This Is A Bug

The function's docstring (lines 34-39) states:

> "This is useful when we use ConstrainedDecimal to represent Numeric(x,0) where a integer (but not int typed) is used. Encoding this as a float results in failed round-tripping between encode and parse."

Numeric(x,0) represents integers with 0 decimal places. Values like `Decimal('1.0')` and `Decimal('42.00')` are integer values and should be encoded as `int` to avoid round-trip issues, but the current implementation only checks `exponent >= 0`, which fails for these cases.

The issue is:
- `Decimal('1')` has exponent=0 → encoded as int ✓
- `Decimal('1.0')` has exponent=-1 → encoded as float ✗
- Both represent the integer value 1 and should encode as int

## Fix

```diff
--- a/pydantic/deprecated/json.py
+++ b/pydantic/deprecated/json.py
@@ -45,7 +45,7 @@ def decimal_encoder(dec_value: Decimal) -> Union[int, float]:
         1
     """
     exponent = dec_value.as_tuple().exponent
-    if isinstance(exponent, int) and exponent >= 0:
+    if dec_value == dec_value.to_integral_value():
         return int(dec_value)
     else:
         return float(dec_value)
```

This change correctly identifies integer-valued Decimals regardless of their exponent representation.