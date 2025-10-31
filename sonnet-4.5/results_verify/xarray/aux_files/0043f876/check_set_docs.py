#!/usr/bin/env python3
"""Check Python's set documentation and behavior"""

import collections.abc

# Check docstrings
print("Python's built-in set.discard docstring:")
print("-" * 50)
print(set.discard.__doc__)
print()

print("Python's built-in set.remove docstring:")
print("-" * 50)
print(set.remove.__doc__)
print()

# Test behavior
print("Testing behavior of set.discard vs set.remove:")
print("-" * 50)

s = {1, 2, 3}
print(f"Starting set: {s}")

# Test discard on non-existent element
print("\nTesting s.discard(999):")
try:
    s.discard(999)
    print("  No error raised (expected)")
except Exception as e:
    print(f"  Error raised: {e}")

print(f"Set after discard: {s}")

# Test remove on non-existent element
print("\nTesting s.remove(999):")
try:
    s.remove(999)
    print("  No error raised")
except KeyError as e:
    print(f"  KeyError raised (expected): {e}")

# Check MutableSet abstract methods
print("\nMutableSet abstract methods:")
print("-" * 50)
print(f"MutableSet abstract methods: {collections.abc.MutableSet.__abstractmethods__}")

# Check if MutableSet has default implementations
import inspect
print("\nMutableSet.discard source (if available):")
try:
    source = inspect.getsource(collections.abc.MutableSet.discard)
    print(source)
except:
    print("  No default implementation found")

print("\nMutableSet.remove source (if available):")
try:
    source = inspect.getsource(collections.abc.MutableSet.remove)
    print(source)
except:
    print("  No default implementation found")