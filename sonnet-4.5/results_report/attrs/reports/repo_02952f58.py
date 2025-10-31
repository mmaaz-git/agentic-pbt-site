import attr
import attrs

@attrs.define
class DoubleConverter:
    x: int = attr.field(converter=lambda v: v * 2)

# Create an instance with x=5
# The converter will make x = 5 * 2 = 10
obj = DoubleConverter(x=5)
print(f"Original: obj.x = {obj.x}")

# Evolve without changing anything
# This should preserve x=10, but it applies the converter again
evolved = attr.evolve(obj)
print(f"Evolved: evolved.x = {evolved.x}")
print(f"Expected: 10, Got: {evolved.x}")

# Verify the bug happens every time
print("\nDemonstrating the issue with different values:")
for value in [1, 3, 7, 10]:
    obj = DoubleConverter(x=value)
    evolved = attr.evolve(obj)
    print(f"  Value={value}: original.x={obj.x}, evolved.x={evolved.x}, should be equal but aren't")