# Bug Report: pydantic.deprecated.json.decimal_encoder - Precision Loss for Fractional Decimals

**Target**: `pydantic.deprecated.json.decimal_encoder`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `decimal_encoder` function silently loses precision when encoding `Decimal` values with fractional components. While the function correctly preserves integer-valued decimals by encoding them as `int`, it encodes fractional decimals as `float`, which introduces floating-point representation errors that violate the round-trip property implied by the function's docstring.

## Property-Based Test

```python
from decimal import Decimal
from hypothesis import given, strategies as st
from pydantic.deprecated.json import decimal_encoder

@given(st.decimals(allow_nan=False, allow_infinity=False, places=10))
def test_decimal_encoder_round_trip(dec_value):
    encoded = decimal_encoder(dec_value)
    reconstructed = Decimal(encoded)

    assert dec_value == reconstructed, (
        f"Round-trip failed: {dec_value} -> {encoded} -> {reconstructed}"
    )
```

**Failing input**: `Decimal('0.1')`

## Reproducing the Bug

```python
from decimal import Decimal
from pydantic.deprecated.json import decimal_encoder

original = Decimal("0.1")
encoded = decimal_encoder(original)
reconstructed = Decimal(encoded)

print(f"Original:      {original}")
print(f"Encoded:       {encoded} ({type(encoded).__name__})")
print(f"Reconstructed: {reconstructed}")
print(f"Match: {original == reconstructed}")
```

Output:
```
Original:      0.1
Encoded:       0.1 (float)
Reconstructed: 0.1000000000000000055511151231257827021181583404541015625
Match: False
```

Additional failing examples:
- `Decimal('0.01')` → `0.01` → `Decimal('0.01000000000000000020816681711721685132943093776702880859375')`
- `Decimal('3.14159')` → `3.14159` → `Decimal('3.14158999999999988261834005243144929409027099609375')`
- `Decimal('0.333333333333')` → `0.333333333333` → `Decimal('0.333333333333000025877623784253955818712711334228515625')`

## Why This Is A Bug

The function's docstring states:

> "Encoding this as a float results in failed round-tripping between encode and parse."

This implies the function is designed to prevent round-trip failures. However, the function only prevents round-trip failures for integer-valued decimals (those with `exponent >= 0`). For fractional decimals, it still encodes as `float`, which causes the exact precision-loss problem the docstring warns about.

This is a contract violation because:
1. The docstring warns about round-trip failures but doesn't clarify this applies only to fractional decimals
2. The function silently loses precision without any warning or validation
3. Users relying on `Decimal` for exact precision (common in financial applications) will experience silent data corruption

The function is used in `ENCODERS_BY_TYPE` (line 61 of json.py), meaning any user encoding a `Decimal` through `pydantic_encoder` will encounter this precision loss.

## Fix

The function has two legitimate use cases with conflicting requirements:
1. Integer-valued decimals should encode as `int` (current behavior, works correctly)
2. Fractional decimals should preserve precision (current behavior fails)

**Option 1: Document the limitation** (minimal fix)

Update the docstring to clearly state that precision is only preserved for integer-valued decimals:

```diff
 def decimal_encoder(dec_value: Decimal) -> Union[int, float]:
-    """Encodes a Decimal as int of there's no exponent, otherwise float.
+    """Encodes a Decimal as int if exponent >= 0, otherwise float.

-    This is useful when we use ConstrainedDecimal to represent Numeric(x,0)
-    where a integer (but not int typed) is used. Encoding this as a float
-    results in failed round-tripping between encode and parse.
-    Our Id type is a prime example of this.
+    This prevents round-trip failures for integer-valued Decimals (those
+    representing values like Numeric(x,0) in databases). For fractional
+    Decimals, precision may be lost due to float representation.
+
+    Warning: Fractional Decimals (e.g., Decimal("0.1")) will lose precision
+    when encoded as float. Use string encoding if exact precision is required.

     >>> decimal_encoder(Decimal("1.0"))
     1.0

     >>> decimal_encoder(Decimal("1"))
     1
+
+    >>> # Note: Precision loss for fractional decimals
+    >>> decimal_encoder(Decimal("0.1"))  # Returns 0.1 but loses precision
+    0.1
     """
```

**Option 2: Encode fractional decimals as strings** (breaking change, better precision)

```diff
-def decimal_encoder(dec_value: Decimal) -> Union[int, float]:
+def decimal_encoder(dec_value: Decimal) -> Union[int, float, str]:
     """Encodes a Decimal as int, float, or str depending on the value.

-    This is useful when we use ConstrainedDecimal to represent Numeric(x,0)
-    where a integer (but not int typed) is used. Encoding this as a float
-    results in failed round-tripping between encode and parse.
+    - Integer-valued Decimals (exponent >= 0): encoded as int
+    - Fractional Decimals with exact float representation: encoded as float
+    - Other fractional Decimals: encoded as string to preserve precision

     >>> decimal_encoder(Decimal("1.0"))
     1.0

     >>> decimal_encoder(Decimal("1"))
     1
+
+    >>> decimal_encoder(Decimal("0.1"))
+    '0.1'
     """
     exponent = dec_value.as_tuple().exponent
     if isinstance(exponent, int) and exponent >= 0:
         return int(dec_value)
     else:
-        return float(dec_value)
+        # Check if float representation is exact
+        float_val = float(dec_value)
+        if Decimal(float_val) == dec_value:
+            return float_val
+        else:
+            return str(dec_value)
```

Given that this is deprecated code (`pydantic.deprecated`), **Option 1** (documentation fix) is recommended to avoid breaking existing users.