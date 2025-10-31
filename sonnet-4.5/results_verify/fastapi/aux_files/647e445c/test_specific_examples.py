import attr

# Test with integer
print("Testing attr.has(42):")
try:
    result = attr.has(42)
    print(f"  Result: {result}")
except TypeError as e:
    print(f"  Raised TypeError: {e}")

# Test with string
print("\nTesting attr.has('not a class'):")
try:
    result = attr.has("not a class")
    print(f"  Result: {result}")
except TypeError as e:
    print(f"  Raised TypeError: {e}")

# Test with None
print("\nTesting attr.has(None):")
try:
    result = attr.has(None)
    print(f"  Result: {result}")
except TypeError as e:
    print(f"  Raised TypeError: {e}")

# Test with a real attrs class for comparison
@attr.s
class MyClass:
    x = attr.ib()

print("\nTesting attr.has(MyClass) - should work:")
try:
    result = attr.has(MyClass)
    print(f"  Result: {result}")
except TypeError as e:
    print(f"  Raised TypeError: {e}")

# Test with a regular class
class RegularClass:
    pass

print("\nTesting attr.has(RegularClass) - non-attrs class:")
try:
    result = attr.has(RegularClass)
    print(f"  Result: {result}")
except TypeError as e:
    print(f"  Raised TypeError: {e}")