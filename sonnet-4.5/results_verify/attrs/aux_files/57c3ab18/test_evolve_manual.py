import attr
import attrs

@attrs.define
class DoubleConverter:
    x: int = attr.field(converter=lambda v: v * 2)

obj = DoubleConverter(x=5)
print(f"Original: obj.x = {obj.x}")

evolved = attr.evolve(obj)
print(f"Evolved: evolved.x = {evolved.x}")
print(f"Expected: 10, Got: {evolved.x}")

# Also test with explicit value
evolved_explicit = attr.evolve(obj, x=7)
print(f"\nWith explicit change:")
print(f"Original: obj.x = {obj.x}")
print(f"Evolved with x=7: evolved_explicit.x = {evolved_explicit.x}")
print(f"Expected: 14, Got: {evolved_explicit.x}")

# Let's also verify the behavior is the issue
print(f"\nVerification: evolved.x == original.x? {evolved.x == obj.x}")