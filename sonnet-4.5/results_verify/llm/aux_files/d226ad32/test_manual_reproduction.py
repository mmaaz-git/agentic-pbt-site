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