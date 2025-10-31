# Bug Report: DecimalValidator Rejects Zero with Decimal Places

**Target**: `django.core.validators.DecimalValidator`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`DecimalValidator` incorrectly rejects zero values when represented with decimal places (e.g., `Decimal("0.0")`) if `decimal_places=0` is configured, even though zero is mathematically valid regardless of its representation.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from decimal import Decimal
from django.core.validators import DecimalValidator
from django.core.exceptions import ValidationError

@given(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=0, max_value=20),
)
def test_decimal_validator_accepts_zero(max_digits, decimal_places):
    assume(decimal_places <= max_digits)

    validator = DecimalValidator(max_digits, decimal_places)

    decimal_value = Decimal("0.0")
    validator(decimal_value)
```

**Failing input**: `max_digits=1, decimal_places=0`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.conf import settings
settings.configure(USE_I18N=False)

from decimal import Decimal
from django.core.validators import DecimalValidator

validator = DecimalValidator(max_digits=1, decimal_places=0)

print("Decimal('0'):", end=" ")
try:
    validator(Decimal("0"))
    print("PASSED ✓")
except:
    print("FAILED ✗")

print("Decimal('0.0'):", end=" ")
try:
    validator(Decimal("0.0"))
    print("PASSED ✓")
except:
    print("FAILED ✗ - BUG")
```

Output:
```
Decimal('0'): PASSED ✓
Decimal('0.0'): FAILED ✗ - BUG
```

## Why This Is A Bug

Zero is mathematically a single value regardless of how it's represented. `Decimal("0")`, `Decimal("0.0")`, and `Decimal("0.00")` all represent the same value (zero) and should all be valid for any reasonable `DecimalValidator` configuration.

The bug occurs because the validator counts the decimal places based on the exponent in the Decimal's internal representation rather than the actual significant digits. For `Decimal("0.0")`, the internal representation is `DecimalTuple(sign=0, digits=(0,), exponent=-1)`, which leads the validator to calculate `decimals=1`, causing rejection when `decimal_places=0`.

## Fix

The fix should special-case zero values. When the digit tuple is `(0,)`, the number of decimal places should be treated as 0 regardless of the exponent:

```diff
--- a/django/core/validators.py
+++ b/django/core/validators.py
@@ -541,6 +541,10 @@ class DecimalValidator:
     def __call__(self, value):
         digit_tuple, exponent = value.as_tuple()[1:]
         if exponent in {"F", "n", "N"}:
             raise ValidationError(
                 self.messages["invalid"], code="invalid", params={"value": value}
             )
+        # Special case: zero should always be valid regardless of representation
+        if digit_tuple == (0,):
+            digits = decimals = 0
+            whole_digits = 0
+        elif exponent >= 0:
-        if exponent >= 0:
             digits = len(digit_tuple)
             if digit_tuple != (0,):
                 # A positive exponent adds that many trailing zeros.
                 digits += exponent
             decimals = 0
         else:
             # If the absolute value of the negative exponent is larger than the
             # number of digits, then it's the same as the number of digits,
             # because it'll consume all of the digits in digit_tuple and then
             # add abs(exponent) - len(digit_tuple) leading zeros after the
             # decimal point.
             if abs(exponent) > len(digit_tuple):
                 digits = decimals = abs(exponent)
             else:
                 digits = len(digit_tuple)
                 decimals = abs(exponent)
-        whole_digits = digits - decimals
+            whole_digits = digits - decimals

         if self.max_digits is not None and digits > self.max_digits:
             raise ValidationError(
                 self.messages["max_digits"],
                 code="max_digits",
                 params={"max": self.max_digits, "value": value},
             )
```