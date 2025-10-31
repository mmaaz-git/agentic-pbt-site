import copy
import attrs

def double_it(x):
    print(f"Converter called with value: {x}")
    return x * 2

@attrs.define
class Container:
    x: int = attrs.field(converter=double_it)
    y: str = "default"

print("Testing copy.replace() behavior on Python 3.13+")
print("=" * 60)

obj1 = Container(5, "hello")
print(f"obj1.x = {obj1.x}")
print(f"obj1.y = {obj1.y}")

print("\n1. Using copy.replace (Python 3.13+)...")
try:
    obj2 = copy.replace(obj1, y="world")
    print(f"copy.replace result:")
    print(f"obj2.x = {obj2.x}")
    print(f"obj2.y = {obj2.y}")
    print(f"Converter was {'NOT' if obj2.x == 10 else ''} called for unchanged field")
except Exception as e:
    print(f"Error: {e}")
    print(f"copy.replace might not support attrs classes directly")

print("\n2. Using attrs.evolve...")
obj3 = attrs.evolve(obj1, y="world")
print(f"attrs.evolve result:")
print(f"obj3.x = {obj3.x}")
print(f"obj3.y = {obj3.y}")
print(f"Converter was {'NOT' if obj3.x == 10 else ''} called for unchanged field")

print("\n" + "=" * 60)
print("Comparison:")
print(f"Original obj1.x:    {obj1.x}")
if 'obj2' in locals():
    print(f"copy.replace obj2.x: {obj2.x}")
print(f"attrs.evolve obj3.x: {obj3.x}")