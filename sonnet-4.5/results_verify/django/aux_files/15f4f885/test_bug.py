#!/usr/bin/env python3
import decimal
from django.db.models.fields import DecimalField

# Test the reported bug
field = DecimalField(max_digits=1, decimal_places=0)

result_from_float = field.to_python(11.0)
result_from_string = field.to_python("11.0")

print("Result from float (11.0):", result_from_float)
print("Result from string ('11.0'):", result_from_string)
print("Are they equal?", result_from_float == result_from_string)
print()
print("Float result type:", type(result_from_float))
print("String result type:", type(result_from_string))
print()

# Let's understand what's happening with the context
print("Field context precision:", field.context.prec)
print()

# Let's see what create_decimal_from_float does
test_context = decimal.Context(prec=1)
print("Creating decimal from float with prec=1:")
print("  Result:", test_context.create_decimal_from_float(11.0))
print()

# Compare with regular decimal creation
print("Creating decimal from string:")
print("  Result:", decimal.Decimal("11.0"))
print()

# Test with some other values
print("Testing with other values:")
for value in [1.0, 9.9, 10.0, 11.0, 99.0, 100.0]:
    field = DecimalField(max_digits=1, decimal_places=0)
    try:
        float_result = field.to_python(value)
        string_result = field.to_python(str(value))
        match = float_result == string_result
        print(f"  {value}: float={float_result}, string={string_result}, match={match}")
    except Exception as e:
        print(f"  {value}: Error - {e}")