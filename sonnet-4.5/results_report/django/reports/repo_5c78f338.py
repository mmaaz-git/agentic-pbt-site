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