# Bug Report: Django DecimalField Precision Handling Inconsistency

**Target**: `django.db.models.fields.DecimalField.to_python`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

DecimalField.to_python() applies precision limits differently based on input type: float inputs are precision-limited using Context(prec=max_digits), but Decimal inputs preserve their original precision, causing the same numeric value to produce different Decimal objects.

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

if __name__ == "__main__":
    test_decimal_field_float_vs_decimal_consistency()
```

<details>

<summary>
**Failing input**: `float_val=11.0, max_digits=1, decimal_places=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 31, in <module>
    test_decimal_field_float_vs_decimal_consistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 7, in test_decimal_field_float_vs_decimal_consistency
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 22, in test_decimal_field_float_vs_decimal_consistency
    assert result_float == result_decimal, (
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Inconsistent results for same value: float(11.0) -> 1E+1, Decimal('11.0') -> 11.0
Falsifying example: test_decimal_field_float_vs_decimal_consistency(
    float_val=11.0,
    max_digits=1,
    decimal_places=0,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/46/hypo.py:23
        /home/npc/pbt/agentic-pbt/worker_/46/hypo.py:27
```
</details>

## Reproducing the Bug

```python
from decimal import Decimal, Context
from django.db.models.fields import DecimalField

# Create a DecimalField with max_digits=5 and decimal_places=2
field = DecimalField(max_digits=5, decimal_places=2)

# Test the same numeric value in different forms
float_input = 123.456789
decimal_input = Decimal('123.456789')

print("Testing DecimalField.to_python() with same numeric value in different forms:")
print(f"float_input: {float_input}")
print(f"decimal_input: {decimal_input}")
print()

# Call to_python on both inputs
result_from_float = field.to_python(float_input)
result_from_decimal = field.to_python(decimal_input)

print(f"Result from float input: {result_from_float}")
print(f"Result from Decimal input: {result_from_decimal}")
print(f"Results are equal: {result_from_float == result_from_decimal}")
print()

# Show what the context would do
ctx = Context(prec=5)
expected_from_float = ctx.create_decimal_from_float(float_input)
print(f"Expected result with Context(prec=5) for float: {expected_from_float}")
print()

# Show the precision difference
print("Precision analysis:")
print(f"Float result precision: {len(str(result_from_float).replace('.', '').replace('-', ''))}")
print(f"Decimal result precision: {len(str(result_from_decimal).replace('.', '').replace('-', ''))}")
```

<details>

<summary>
DecimalField produces different results for float vs Decimal inputs with same numeric value
</summary>
```
Testing DecimalField.to_python() with same numeric value in different forms:
float_input: 123.456789
decimal_input: 123.456789

Result from float input: 123.46
Result from Decimal input: 123.456789
Results are equal: False

Expected result with Context(prec=5) for float: 123.46

Precision analysis:
Float result precision: 5
Decimal result precision: 9
```
</details>

## Why This Is A Bug

This bug violates the fundamental contract of the `to_python()` method, which according to Django documentation should "convert the value into the correct Python object" and serve as a normalization point for all inputs. The method is expected to produce consistent, canonical representations regardless of the input type.

The inconsistency stems from lines 1814-1817 in `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/db/models/fields/__init__.py`:

1. **For float inputs (line 1815)**: The method uses `self.context.create_decimal_from_float(value)` where `self.context` is defined as `decimal.Context(prec=self.max_digits)` (line 1797). This applies precision limiting based on max_digits.

2. **For Decimal and other inputs (line 1817)**: The method uses `decimal.Decimal(value)` which creates a new Decimal without applying any context, preserving the original precision.

This causes several problems:

- **Normalization failure**: The same numeric value (e.g., 123.456789) produces different Decimal objects depending on whether it arrives as a float or Decimal
- **Constraint bypass**: Decimal inputs can exceed the max_digits constraint that float inputs must respect
- **Double processing issues**: The `get_db_prep_value()` method (lines 1833-1835) calls `to_python()` twice in sequence, which could produce unexpected results when processing high-precision Decimals
- **Validation inconsistency**: The field enforces precision limits differently based on arbitrary input type distinctions

The bug is particularly problematic for financial applications where decimal precision handling must be consistent and predictable.

## Relevant Context

The DecimalField class defines a context property at line 1796-1797:
```python
@cached_property
def context(self):
    return decimal.Context(prec=self.max_digits)
```

This context is used for float inputs but not for Decimal inputs, creating the inconsistency. The Django documentation for model fields states that `to_python()` should handle normalization, but this implementation fails to normalize consistently.

Django's DecimalField is commonly used for financial data where precision and consistency are critical. The bug can lead to subtle data integrity issues where the same value stored and retrieved might have different precision depending on how it was initially provided.

Documentation reference: https://docs.djangoproject.com/en/stable/howto/custom-model-fields/#converting-values-to-python-objects

## Proposed Fix

```diff
--- a/django/db/models/fields/__init__.py
+++ b/django/db/models/fields/__init__.py
@@ -1813,8 +1813,10 @@ class DecimalField(Field):
         try:
             if isinstance(value, float):
                 decimal_value = self.context.create_decimal_from_float(value)
+            elif isinstance(value, decimal.Decimal):
+                decimal_value = self.context.create_decimal(value)
             else:
-                decimal_value = decimal.Decimal(value)
+                decimal_value = self.context.create_decimal(decimal.Decimal(value))
         except (decimal.InvalidOperation, TypeError, ValueError):
             raise exceptions.ValidationError(
                 self.error_messages["invalid"],
```