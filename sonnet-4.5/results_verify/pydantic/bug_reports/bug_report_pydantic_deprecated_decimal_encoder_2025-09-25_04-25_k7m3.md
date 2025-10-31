# Bug Report: pydantic.deprecated.json.decimal_encoder Precision Loss

**Target**: `pydantic.deprecated.json.decimal_encoder`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `decimal_encoder` function causes precision loss when encoding Decimal values with fractional parts, despite its docstring claiming it helps avoid "failed round-tripping between encode and parse."

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from decimal import Decimal
import json
from pydantic.deprecated.json import decimal_encoder


@given(st.decimals(allow_nan=False, allow_infinity=False))
@settings(max_examples=1000)
def test_decimal_encoder_round_trip(dec):
    encoded = decimal_encoder(dec)
    assert isinstance(encoded, (int, float))

    json_str = json.dumps(encoded)
    decoded = json.loads(json_str)

    assert Decimal(str(decoded)) == dec
```

**Failing input**: `Decimal('22367635711314.143')`

## Reproducing the Bug

```python
import json
from decimal import Decimal
from pydantic.deprecated.json import decimal_encoder

dec = Decimal('22367635711314.143')
encoded = decimal_encoder(dec)
json_str = json.dumps(encoded)
decoded = json.loads(json_str)
restored = Decimal(str(decoded))

print(f"Original: {dec}")
print(f"After round-trip: {restored}")
print(f"Equal? {restored == dec}")
```

Output:
```
Original: 22367635711314.143
After round-trip: 22367635711314.145
Equal? False
```

## Why This Is A Bug

The function's docstring explicitly states: "Encodes a Decimal as int of there's no exponent, otherwise float. This is useful when we use ConstrainedDecimal to represent Numeric(x,0) where a integer (but not int typed) is used. **Encoding this as a float results in failed round-tripping between encode and parse.**"

However, the current implementation only returns `int` when `exponent >= 0`, and returns `float` for all other cases. This means any Decimal with a fractional part (negative exponent) gets converted to float, causing the exact precision loss the function claims to avoid.

## Fix

The function cannot avoid precision loss for decimals with fractional parts when converting to JSON primitives (int/float). The real issue is that the docstring overpromises. The function should either:

1. Return a string representation for decimals with fractional parts to preserve precision
2. Update the docstring to clarify it only preserves precision for integer-valued decimals

Here's a fix that returns strings for fractional decimals:

```diff
--- a/pydantic/deprecated/json.py
+++ b/pydantic/deprecated/json.py
@@ -46,7 +46,7 @@ def decimal_encoder(dec_value: Decimal) -> Union[int, float]:
     exponent = dec_value.as_tuple().exponent
     if isinstance(exponent, int) and exponent >= 0:
         return int(dec_value)
     else:
-        return float(dec_value)
+        return str(dec_value)
```

Alternatively, update the docstring to clarify limitations:

```diff
--- a/pydantic/deprecated/json.py
+++ b/pydantic/deprecated/json.py
@@ -33,9 +33,9 @@ def isoformat(o: Union[datetime.date, datetime.time]) -> str:
 def decimal_encoder(dec_value: Decimal) -> Union[int, float]:
-    """Encodes a Decimal as int of there's no exponent, otherwise float.
+    """Encodes a Decimal as int if there's no exponent, otherwise float.

-    This is useful when we use ConstrainedDecimal to represent Numeric(x,0)
+    This preserves precision for integer-valued decimals (Numeric(x,0))
     where a integer (but not int typed) is used. Encoding this as a float
-    results in failed round-tripping between encode and parse.
-    Our Id type is a prime example of this.
+    results in failed round-tripping for fractional decimals due to
+    floating-point precision limits.
```