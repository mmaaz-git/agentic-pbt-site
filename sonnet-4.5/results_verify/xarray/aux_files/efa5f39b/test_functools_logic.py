#!/usr/bin/env python3
"""Understanding functools.total_ordering implementation"""

import inspect
import functools

# Let's see what functools.total_ordering actually generates
# by looking at the source

@functools.total_ordering
class TestGT:
    def __gt__(self, other):
        return True
    def __eq__(self, other):
        return isinstance(other, type(self))

# Let's trace through the logic manually
a = TestGT()
b = TestGT()

print("Analyzing functools.total_ordering implementation for __gt__ and __eq__:")
print()

# From Python's functools source, when we have __gt__:
# __lt__ = lambda self, other: other > self
# __le__ = lambda self, other: not self > other or self == other
# __ge__ = lambda self, other: not other > self or self == other

print("When we define __gt__ and __eq__, functools generates:")
print("  __lt__(self, other) = other > self")
print("  __le__(self, other) = not self > other or self == other")
print("  __ge__(self, other) = not other > self or self == other")
print()

print("For two instances a and b where a == b is True:")
print(f"  a == b: {a == b} (both instances of same class)")
print(f"  a > b: {a > b} (always returns True)")
print()

print("Generated methods:")
print(f"  a < b = b > a = True (since b.__gt__ always returns True)")
# Wait, that's wrong. Let me verify:
print(f"  Actual a < b: {a < b}")

print(f"  a <= b = not (a > b) or (a == b) = not True or True = False or True = True")
# But we got False. Let's check:
print(f"  Actual a <= b: {a <= b}")

print(f"  a >= b = not (b > a) or (a == b) = not True or True = False or True = True")
print(f"  Actual a >= b: {a >= b}")

print("\nLet me check the actual Python source...")
print("\nInspecting the generated __lt__ method:")
# The actual implementation uses NotImplemented cleverly
print("Actually, functools.total_ordering uses a more complex implementation")
print("that handles NotImplemented returns properly.")