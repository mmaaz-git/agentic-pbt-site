# Bug Report: pydantic.deprecated.json.decimal_encoder Precision Loss

**Target**: `pydantic.deprecated.json.decimal_encoder`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `decimal_encoder` function loses precision when encoding `Decimal` values with fractional parts by converting them to floats, directly violating its documented purpose of avoiding "failed round-tripping between encode and parse."

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

if __name__ == "__main__":
    test_decimal_encoder_round_trip()
```

<details>

<summary>
**Failing input**: `Decimal('9.8902070606274132E-16')`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 19, in <module>
    test_decimal_encoder_round_trip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 8, in test_decimal_encoder_round_trip
    def test_decimal_encoder_round_trip(x):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 16, in test_decimal_encoder_round_trip
    assert decoded == x, f"Round-trip failed: {x} -> {encoded} -> {decoded}"
           ^^^^^^^^^^^^
AssertionError: Round-trip failed: 9.8902070606274132E-16 -> 9.890207060627413e-16 -> 9.890207060627413E-16
Falsifying example: test_decimal_encoder_round_trip(
    x=fraction_to_decimal(Fraction(1, 1_011_101_177_022_842)),
)
```
</details>

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

<details>

<summary>
Precision loss during float conversion
</summary>
```
Original: 252579977670696.67
Encoded:  252579977670696.66
Decoded:  252579977670696.66
Round-trip equal? False
```
</details>

## Why This Is A Bug

The function's docstring at lines 34-39 explicitly states: "Encoding this as a float results in failed round-tripping between encode and parse. Our Id type is a prime example of this." This statement acknowledges that float encoding causes round-trip failures - the exact problem the function was supposedly created to prevent.

Despite this documentation, the implementation at line 51 converts any Decimal with a negative exponent (fractional part) to a float: `return float(dec_value)`. This conversion loses precision for Decimal values that exceed float's ~15-17 significant digit precision limit (IEEE 754 double precision).

The contradiction is stark: the docstring warns about float encoding causing round-trip failures, yet the function deliberately performs float encoding for fractional Decimals. This violates the fundamental purpose of the Decimal type - maintaining exact decimal precision.

## Relevant Context

The function resides in `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/deprecated/json.py`, indicating it's part of Pydantic's deprecated API. While deprecated, the function is still accessible and may be in use by existing codebases.

The function is registered in `ENCODERS_BY_TYPE` (line 61) and used by the deprecated `pydantic_encoder` function. The docstring examples show:
- `Decimal("1.0")` returns `1.0` (float)
- `Decimal("1")` returns `1` (int)

The function checks if the decimal's exponent is non-negative (line 48) to determine whether to return an integer. For negative exponents (fractional values), it converts to float, causing precision loss.

Industry standard practice for JSON serialization of Decimals is to use string representation to preserve precision, as implemented by Django REST Framework and other major serialization libraries.

## Proposed Fix

```diff
def decimal_encoder(dec_value: Decimal) -> Union[int, float]:
    """Encodes a Decimal as int of there's no exponent, otherwise float.

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