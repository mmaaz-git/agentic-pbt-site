#!/usr/bin/env python3
"""Check Python documentation on equality"""

import pydoc

# Get help on object.__eq__
help_text = pydoc.getdoc(object.__eq__)
print("=== object.__eq__ documentation ===")
print(help_text)
print()

# Also check NotImplemented
print("=== NotImplemented documentation ===")
print(repr(NotImplemented))
print(type(NotImplemented))
print()

# Check if there's more info
import inspect
print("=== Checking object.__eq__ signature ===")
try:
    sig = inspect.signature(object.__eq__)
    print(f"Signature: {sig}")
except:
    print("Unable to get signature")

# Test behavior
print("\n=== Testing NotImplemented behavior ===")
class A:
    def __eq__(self, other):
        print(f"A.__eq__ called with {other}")
        return NotImplemented

class B:
    def __eq__(self, other):
        print(f"B.__eq__ called with {other}")
        return NotImplemented

a = A()
b = B()

print("\nTesting a == b:")
result = a == b
print(f"Result: {result}")

print("\nTesting b == a:")
result = b == a
print(f"Result: {result}")