from pydantic.v1.types import BYTE_SIZES

# Test the calculation that happens in validate
test_cases = [
    ("0.5", "b"),
    ("1.7", "kb"),
    ("2.5", "mb"),
    ("0.1", "b"),
    ("0.9999", "b"),
]

for scalar_str, unit in test_cases:
    scalar = float(scalar_str)
    unit_mult = BYTE_SIZES[unit.lower()]
    result = float(scalar) * unit_mult
    int_result = int(result)

    print(f"Input: {scalar_str}{unit}")
    print(f"  scalar={scalar}, unit_mult={unit_mult}")
    print(f"  float(scalar) * unit_mult = {result}")
    print(f"  int(float(scalar) * unit_mult) = {int_result}")
    print(f"  Data loss: {result - int_result}")
    print()