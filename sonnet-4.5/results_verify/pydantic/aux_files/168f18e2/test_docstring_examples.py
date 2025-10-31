from decimal import Decimal
from pydantic.deprecated.json import decimal_encoder

# Test the examples from the docstring
result1 = decimal_encoder(Decimal("1.0"))
print(f'decimal_encoder(Decimal("1.0")) = {result1}, type = {type(result1)}')
print(f'Expected: 1.0')

result2 = decimal_encoder(Decimal("1"))
print(f'decimal_encoder(Decimal("1")) = {result2}, type = {type(result2)}')
print(f'Expected: 1')

# Let's also test some edge cases
result3 = decimal_encoder(Decimal("100"))
print(f'decimal_encoder(Decimal("100")) = {result3}, type = {type(result3)}')

result4 = decimal_encoder(Decimal("10E2"))  # Scientific notation for 1000
print(f'decimal_encoder(Decimal("10E2")) = {result4}, type = {type(result4)}')

result5 = decimal_encoder(Decimal("10E-2"))  # 0.1
print(f'decimal_encoder(Decimal("10E-2")) = {result5}, type = {type(result5)}')