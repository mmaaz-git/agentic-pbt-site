"""Test to confirm Python's classmethod behavior with instances"""

class TestClass:
    value = "class_value"

    @classmethod
    def my_classmethod(cls, arg):
        print(f"cls: {cls}")
        print(f"cls.__name__: {cls.__name__}")
        print(f"arg: {arg}")
        return (cls, arg)

# Create an instance
instance = TestClass()

print("Calling as classmethod (TestClass.my_classmethod):")
result1 = TestClass.my_classmethod("test_arg")
print(f"Result: {result1}\n")

print("Calling as instance method (instance.my_classmethod):")
result2 = instance.my_classmethod("test_arg")
print(f"Result: {result2}\n")

print("Both calls should work identically - Python allows classmethods to be called on instances")
print(f"Results are equal: {result1 == result2}")

# Now test what happens with numpy Polynomial
print("\n" + "="*60)
print("Testing with numpy Polynomial:")

from numpy.polynomial import Polynomial
import numpy as np

# The correct way (as classmethod)
p = Polynomial([1, 2, 3])
print(f"\nOriginal polynomial: {p}")

print("\nTrying Polynomial.cast(p, Polynomial) - the 'correct' way:")
try:
    # This should work if cast expects to be called as Polynomial.cast(instance, TargetClass)
    result = Polynomial.cast(p, Polynomial)
    print(f"Success: {result}")
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")

print("\nTrying p.cast(Polynomial) - instance method style:")
try:
    # This crashes but SHOULD work since classmethods can be called on instances
    result = p.cast(Polynomial)
    print(f"Success: {result}")
except AttributeError as e:
    print(f"Failed with AttributeError: {e}")
    print("This SHOULD work but doesn't - confirming the bug")