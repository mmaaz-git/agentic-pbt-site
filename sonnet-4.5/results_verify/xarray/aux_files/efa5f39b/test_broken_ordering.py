#!/usr/bin/env python3
"""Test to show the broken behavior with always-true __gt__"""

import functools

@functools.total_ordering
class BrokenAlwaysGreaterThan:
    def __gt__(self, other):
        return True  # Always returns True

    def __eq__(self, other):
        return isinstance(other, type(self))

# Test with two instances
a = BrokenAlwaysGreaterThan()
b = BrokenAlwaysGreaterThan()

print("BrokenAlwaysGreaterThan - what functools.total_ordering generates:")
print(f"a == b: {a == b}")  # Should be True
print(f"a > b: {a > b}")    # Returns True (violates total ordering!)
print(f"a < b: {a < b}")    # What does this give?
print(f"a >= b: {a >= b}")  # What does this give?
print(f"a <= b: {a <= b}")  # What does this give?

print("\nLet's trace how functools.total_ordering generates these methods:")
print("Given __gt__ and __eq__, it generates:")
print("  __lt__(self, other) = not (self == other) and not (self > other)")
print("  __le__(self, other) = self == other or not (self > other)")
print("  __ge__(self, other) = self == other or self > other")

print("\nSo for a == b (both instances of same class):")
print(f"  a == b = isinstance(b, type(a)) = True")
print(f"  a > b = True (always returns True)")
print(f"  a < b = not (a == b) and not (a > b) = not True and not True = False")
print(f"  a <= b = (a == b) or not (a > b) = True or not True = True or False = True")
print(f"  a >= b = (a == b) or (a > b) = True or True = True")

print("\nBut wait, that doesn't match! Let me check the actual implementation...")

# Check what functools.total_ordering actually does
print("\nActual implementation check:")
print(f"__lt__ method: {BrokenAlwaysGreaterThan.__lt__}")
print(f"__le__ method: {BrokenAlwaysGreaterThan.__le__}")
print(f"__ge__ method: {BrokenAlwaysGreaterThan.__ge__}")