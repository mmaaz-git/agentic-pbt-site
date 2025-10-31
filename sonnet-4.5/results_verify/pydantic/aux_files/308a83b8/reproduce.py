from decimal import Decimal
from pydantic.deprecated.json import decimal_encoder

print("Testing decimal_encoder behavior:")
print("-" * 50)

test_cases = [
    Decimal('1'),
    Decimal('1.0'),
    Decimal('42.00'),
    Decimal('100.000'),
    Decimal('0'),
    Decimal('0.0'),
    Decimal('123'),
    Decimal('123.0'),
    Decimal('1.5'),
    Decimal('0.5'),
]

for dec in test_cases:
    encoded = decimal_encoder(dec)
    is_integer_value = dec == dec.to_integral_value()
    exponent = dec.as_tuple().exponent

    print(f"Input: {dec}")
    print(f"  Is integer value: {is_integer_value}")
    print(f"  Exponent: {exponent}")
    print(f"  Encoded as: {type(encoded).__name__} = {encoded}")
    print(f"  Expected type if integer value: {'int' if is_integer_value else 'float'}")
    print(f"  Matches expectation: {(isinstance(encoded, int) and is_integer_value) or (isinstance(encoded, float) and not is_integer_value)}")
    print()

print("\nSpecific example from bug report:")
print("-" * 50)
dec = Decimal('1.0')
encoded = decimal_encoder(dec)

print(f"Input: {dec}")
print(f"Is integer value: {dec == dec.to_integral_value()}")
print(f"Exponent: {dec.as_tuple().exponent}")
print(f"Encoded as: {type(encoded).__name__} = {encoded}")
print(f"Expected: int (to preserve integer value)")
print(f"Actual: {type(encoded).__name__}")

print("\nAnalyzing the current implementation:")
print("-" * 50)
print("Current logic: exponent >= 0 → int, otherwise → float")
print("\nExamples:")
print("  Decimal('1') has exponent=0 → encoded as int ✓")
print("  Decimal('10') has exponent=1 → encoded as int ✓")
print("  Decimal('1.0') has exponent=-1 → encoded as float")
print("  Decimal('1.00') has exponent=-2 → encoded as float")
print("\nBut Decimal('1.0') and Decimal('1') both represent the integer 1!")

print("\nChecking docstring examples:")
print("-" * 50)
print('From docstring: decimal_encoder(Decimal("1.0")) → 1.0')
actual = decimal_encoder(Decimal("1.0"))
print(f'Actual result: {actual} (type: {type(actual).__name__})')
print(f'Matches docstring: {actual == 1.0 and isinstance(actual, float)}')

print('\nFrom docstring: decimal_encoder(Decimal("1")) → 1')
actual = decimal_encoder(Decimal("1"))
print(f'Actual result: {actual} (type: {type(actual).__name__})')
print(f'Matches docstring: {actual == 1 and isinstance(actual, int)}')