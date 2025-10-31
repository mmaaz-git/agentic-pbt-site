# Bug Report: pydantic.deprecated.json.decimal_encoder Precision Loss

**Target**: `pydantic.deprecated.json.decimal_encoder`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `decimal_encoder` function loses precision when encoding `Decimal` values with fractional parts, violating its documented purpose of avoiding "failed round-tripping between encode and parse."

## Property-Based Test

```python
from decimal import Decimal
from hypothesis import given, strategies as st
from pydantic.deprecated.json import decimal_encoder
import warnings


@given(st.decimals(allow_nan=False, allow_infinity=False))
def test_decimal_encoder_round_trip(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        encoded = decimal_encoder(x)
        assert isinstance(encoded, (int, float))

        decoded = Decimal(str(encoded))

        assert decoded == x, f"Round-trip failed: {x} -> {encoded} -> {decoded}"
```

**Failing input**: `Decimal('252579977670696.67')`

## Reproducing the Bug

```python
from decimal import Decimal
from pydantic.deprecated.json import decimal_encoder
import warnings

warnings.filterwarnings("ignore")

x = Decimal('252579977670696.67')
encoded = decimal_encoder(x)
decoded = Decimal(str(encoded))

print(f"Original: {x}")
print(f"Encoded:  {encoded}")
print(f"Decoded:  {decoded}")
print(f"Round-trip equal? {decoded == x}")
```

Output:
```
Original: 252579977670696.67
Encoded:  252579977670696.66
Decoded:  252579977670696.66
Round-trip equal? False
```

## Why This Is A Bug

The function's docstring explicitly states:

> "Encoding this as a float results in failed round-tripping between encode and parse. Our Id type is a prime example of this."

The function was specifically designed to prevent round-trip failures, yet it fails to do so for `Decimal` values with fractional parts that exceed float precision (typically ~15-17 significant digits). When a `Decimal` with a fractional component is encoded as a `float`, precision is lost due to float's limited precision.

## Fix

The root cause is that the function converts `Decimal` values with fractional parts to `float`, which has limited precision. For true round-trip preservation, the encoder should use `str` representation instead of `float` for Decimals with fractional parts:

```diff
def decimal_encoder(dec_value: Decimal) -> Union[int, float, str]:
    """Encodes a Decimal as int of there's no exponent, otherwise string.

    This is useful when we use ConstrainedDecimal to represent Numeric(x,0)
    where a integer (but not int typed) is used. Encoding this as a float
    results in failed round-tripping between encode and parse.
    Our Id type is a prime example of this.

    >>> decimal_encoder(Decimal("1.0"))
-    1.0
+    '1.0'

    >>> decimal_encoder(Decimal("1"))
    1
    """
    exponent = dec_value.as_tuple().exponent
    if isinstance(exponent, int) and exponent >= 0:
        return int(dec_value)
    else:
-        return float(dec_value)
+        return str(dec_value)
```