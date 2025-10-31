# Bug Report: django_core_DecimalValidator DecimalValidator Incorrectly Rejects Zero with Decimal Representation

**Target**: `django.core.validators.DecimalValidator`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`DecimalValidator` incorrectly rejects zero values when represented with decimal places (e.g., `Decimal("0.0")`) when `decimal_places=0` is configured, even though all representations of zero are mathematically equivalent.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.conf import settings
settings.configure(USE_I18N=False)

from hypothesis import given, strategies as st, assume
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

if __name__ == "__main__":
    test_decimal_validator_accepts_zero()
```

<details>

<summary>
**Failing input**: `max_digits=1, decimal_places=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 25, in <module>
    test_decimal_validator_accepts_zero()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 13, in test_decimal_validator_accepts_zero
    st.integers(min_value=1, max_value=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 22, in test_decimal_validator_accepts_zero
    validator(decimal_value)
    ~~~~~~~~~^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/core/validators.py", line 570, in __call__
    raise ValidationError(
    ...<3 lines>...
    )
django.core.exceptions.ValidationError: ['Ensure that there are no more than 0 decimal places.']
Falsifying example: test_decimal_validator_accepts_zero(
    max_digits=1,
    decimal_places=0,
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.conf import settings
settings.configure(USE_I18N=False)

from decimal import Decimal
from django.core.validators import DecimalValidator
from django.core.exceptions import ValidationError

validator = DecimalValidator(max_digits=1, decimal_places=0)

print("Testing DecimalValidator(max_digits=1, decimal_places=0)")
print("="*60)

# Test Decimal('0')
print("\nDecimal('0'):", end=" ")
try:
    validator(Decimal("0"))
    print("PASSED ✓")
except ValidationError as e:
    print(f"FAILED ✗ - Error: {e}")

# Test Decimal('0.0')
print("Decimal('0.0'):", end=" ")
try:
    validator(Decimal("0.0"))
    print("PASSED ✓")
except ValidationError as e:
    print(f"FAILED ✗ - Error: {e}")

# Test Decimal('0.00')
print("Decimal('0.00'):", end=" ")
try:
    validator(Decimal("0.00"))
    print("PASSED ✓")
except ValidationError as e:
    print(f"FAILED ✗ - Error: {e}")

print("\n" + "="*60)
print("Mathematical equivalence check:")
print(f"Decimal('0') == Decimal('0.0'): {Decimal('0') == Decimal('0.0')}")
print(f"Decimal('0') == Decimal('0.00'): {Decimal('0') == Decimal('0.00')}")

print("\n" + "="*60)
print("Internal representation analysis:")
print(f"Decimal('0') internal: {Decimal('0').as_tuple()}")
print(f"Decimal('0.0') internal: {Decimal('0.0').as_tuple()}")
print(f"Decimal('0.00') internal: {Decimal('0.00').as_tuple()}")
```

<details>

<summary>
DecimalValidator rejects Decimal("0.0") and Decimal("0.00") despite mathematical equivalence to Decimal("0")
</summary>
```
Testing DecimalValidator(max_digits=1, decimal_places=0)
============================================================

Decimal('0'): PASSED ✓
Decimal('0.0'): FAILED ✗ - Error: ['Ensure that there are no more than 0 decimal places.']
Decimal('0.00'): FAILED ✗ - Error: ['Ensure that there are no more than 1 digit in total.']

============================================================
Mathematical equivalence check:
Decimal('0') == Decimal('0.0'): True
Decimal('0') == Decimal('0.00'): True

============================================================
Internal representation analysis:
Decimal('0') internal: DecimalTuple(sign=0, digits=(0,), exponent=0)
Decimal('0.0') internal: DecimalTuple(sign=0, digits=(0,), exponent=-1)
Decimal('0.00') internal: DecimalTuple(sign=0, digits=(0,), exponent=-2)
```
</details>

## Why This Is A Bug

This violates expected behavior because:

1. **Mathematical Inconsistency**: All three representations (`Decimal("0")`, `Decimal("0.0")`, `Decimal("0.00")`) are mathematically identical - they all equal zero. Python's Decimal class confirms this: `Decimal("0") == Decimal("0.0")` returns `True`.

2. **Zero Has No Significant Decimal Places**: Mathematically, zero has no significant decimal places regardless of representation. The trailing zeros after the decimal point in "0.0" or "0.00" are purely formatting concerns, not mathematical precision.

3. **Violates Principle of Least Surprise**: Users expect validators to validate based on mathematical value, not internal representation details. A user storing zero should not need to worry about whether it's formatted as "0" or "0.0".

4. **Django Documentation Gap**: The Django documentation for DecimalValidator does not specify that different representations of zero would be treated differently. Users would reasonably expect consistent handling of mathematically equivalent values.

5. **Real-World Impact**: This bug affects data import/export scenarios where zero might come from different sources with different formatting (databases, APIs, user input, CSV files). Form inputs might generate "0.0" while model defaults might use "0", causing unexpected validation failures.

## Relevant Context

The bug occurs in the DecimalValidator.__call__ method (django/core/validators.py lines 538-584). The validator counts decimal places based on the Decimal's internal exponent representation rather than considering the actual mathematical significance:

- For `Decimal("0.0")` with internal representation `DecimalTuple(sign=0, digits=(0,), exponent=-1)`, the code at line 560 calculates `decimals = abs(exponent) = 1`
- This causes validation failure at line 569-574 when `decimal_places=0`

Django's DecimalValidator documentation: https://docs.djangoproject.com/en/stable/ref/validators/#decimalvalidator

The validator is commonly used with DecimalField in models and forms throughout Django applications for validating monetary values, measurements, and other decimal data where zero is a common and valid value.

## Proposed Fix

```diff
--- a/django/core/validators.py
+++ b/django/core/validators.py
@@ -538,6 +538,11 @@ class DecimalValidator:
     def __call__(self, value):
         digit_tuple, exponent = value.as_tuple()[1:]
         if exponent in {"F", "n", "N"}:
             raise ValidationError(
                 self.messages["invalid"], code="invalid", params={"value": value}
             )
+        # Special case: zero should always be valid regardless of representation
+        if digit_tuple == (0,):
+            digits = 1 if self.max_digits is not None else 0
+            decimals = 0
+            whole_digits = digits
-        if exponent >= 0:
+        elif exponent >= 0:
             digits = len(digit_tuple)
             if digit_tuple != (0,):
                 # A positive exponent adds that many trailing zeros.
                 digits += exponent
             decimals = 0
+            whole_digits = digits - decimals
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
```