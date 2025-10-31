# Bug Report: DecimalField.to_python Precision Handling Inconsistency

**Target**: `django.db.models.fields.DecimalField.to_python`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`DecimalField.to_python()` inconsistently applies precision limits based on input type: float inputs are precision-limited using `Context(prec=max_digits)`, but Decimal inputs bypass this limitation, leading to different Decimal objects for the same numeric value.

## Property-Based Test

```python
from decimal import Decimal
from hypothesis import given, strategies as st, assume, settings
from django.db.models.fields import DecimalField
from django.core import exceptions

@given(
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
    st.integers(min_value=1, max_value=20),
    st.integers(min_value=0, max_value=10)
)
@settings(max_examples=200)
def test_decimal_field_float_vs_decimal_consistency(float_val, max_digits, decimal_places):
    assume(decimal_places < max_digits)

    field = DecimalField(max_digits=max_digits, decimal_places=decimal_places)
    decimal_val = Decimal(str(float_val))

    try:
        result_float = field.to_python(float_val)
        result_decimal = field.to_python(decimal_val)

        assert result_float == result_decimal, (
            f"Inconsistent results for same value: "
            f"float({float_val}) -> {result_float}, "
            f"Decimal('{decimal_val}') -> {result_decimal}"
        )
    except exceptions.ValidationError:
        pass
```

**Failing input**: `float_val=123.456789, max_digits=5, decimal_places=2`

## Reproducing the Bug

```python
from decimal import Decimal, Context
from django.db.models.fields import DecimalField

field = DecimalField(max_digits=5, decimal_places=2)

float_input = 123.456789
decimal_input = Decimal('123.456789')

result_from_float = field.to_python(float_input)
result_from_decimal = field.to_python(decimal_input)

print(f"Float input result: {result_from_float}")
print(f"Decimal input result: {result_from_decimal}")
print(f"Results equal: {result_from_float == result_from_decimal}")

ctx = Context(prec=5)
expected = ctx.create_decimal_from_float(float_input)
print(f"Expected (with context): {expected}")
```

## Why This Is A Bug

The inconsistency violates the principle that `to_python()` should normalize inputs to a canonical Python representation regardless of input type.

**Code analysis** (django/db/models/fields/__init__.py lines 1814-1817):

```python
if isinstance(value, float):
    decimal_value = self.context.create_decimal_from_float(value)
else:
    decimal_value = decimal.Decimal(value)
```

- **Float path**: Uses `self.context` with `prec=max_digits`, applying precision limiting
- **Decimal path**: Creates a copy via `Decimal(value)` without applying context, preserving all precision

This means:
1. Same numeric value produces different Decimal objects based on input type
2. Decimal inputs can bypass `max_digits` constraint that floats respect
3. Code calling `to_python()` twice gets different behavior on first vs second call if the first input was high-precision

**Impact**: This affects `get_db_prep_value()` which calls `to_python()` twice (lines 1833-1835), potentially producing unexpected results when the input is a high-precision Decimal.

## Fix

Apply the context consistently to both float and Decimal inputs:

```diff
 def to_python(self, value):
     if value is None:
         return value
     try:
         if isinstance(value, float):
             decimal_value = self.context.create_decimal_from_float(value)
-        else:
+        elif isinstance(value, Decimal):
+            decimal_value = self.context.create_decimal(value)
+        else:
             decimal_value = decimal.Decimal(value)
+            decimal_value = self.context.create_decimal(decimal_value)
     except (decimal.InvalidOperation, TypeError, ValueError):
         raise exceptions.ValidationError(
             self.error_messages["invalid"],
             code="invalid",
             params={"value": value},
         )
     if not decimal_value.is_finite():
         raise exceptions.ValidationError(
             self.error_messages["invalid"],
             code="invalid",
             params={"value": value},
         )
     return decimal_value
```

Alternatively, apply context normalization once at the end:

```diff
 def to_python(self, value):
     if value is None:
         return value
     try:
         if isinstance(value, float):
             decimal_value = self.context.create_decimal_from_float(value)
         else:
             decimal_value = decimal.Decimal(value)
+            decimal_value = self.context.create_decimal(decimal_value)
     except (decimal.InvalidOperation, TypeError, ValueError):
         raise exceptions.ValidationError(
             self.error_messages["invalid"],
             code="invalid",
             params={"value": value},
         )
     if not decimal_value.is_finite():
         raise exceptions.ValidationError(
             self.error_messages["invalid"],
             code="invalid",
             params={"value": value},
         )
     return decimal_value
```