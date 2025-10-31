#!/usr/bin/env python3
"""Test to understand how functools.total_ordering works"""

import functools

@functools.total_ordering
class TestClass:
    def __init__(self, value):
        self.value = value

    def __gt__(self, other):
        if isinstance(other, TestClass):
            return self.value > other.value
        return True  # Greater than everything else

    def __eq__(self, other):
        if isinstance(other, TestClass):
            return self.value == other.value
        return False

# Test with equal values
a = TestClass(5)
b = TestClass(5)

print("Testing TestClass with equal values:")
print(f"a == b: {a == b}")  # Should be True
print(f"a > b: {a > b}")    # Should be False (since they're equal)
print(f"a < b: {a < b}")    # Should be False (since they're equal)
print(f"a >= b: {a >= b}")  # Should be True (since they're equal)
print(f"a <= b: {a <= b}")  # Should be True (since they're equal)

print("\nTesting TestClass with different values:")
c = TestClass(3)
d = TestClass(7)
print(f"c == d: {c == d}")  # Should be False
print(f"c > d: {c > d}")    # Should be False
print(f"c < d: {c < d}")    # Should be True
print(f"c >= d: {c >= d}")  # Should be False
print(f"c <= d: {c <= d}")  # Should be True