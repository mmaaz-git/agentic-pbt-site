#!/usr/bin/env python3
"""Minimal reproduction of Cython.Tempita.bunch delattr bug."""

import Cython.Tempita as tempita

# Create a bunch object with some initial attributes
b = tempita.bunch(x=1, y=2)

# Verify initial attributes work
print(f"Initial state: b.x={b.x}, b.y={b.y}")
assert b.x == 1
assert b.y == 2

# Verify setattr works (modify existing attribute)
b.x = 10
print(f"After setattr: b.x={b.x}")
assert b.x == 10

# Verify setattr works (add new attribute)
setattr(b, 'z', 100)
print(f"After adding z: b.z={b.z}")
assert b.z == 100

# Verify getattr works
val = getattr(b, 'y')
print(f"Using getattr: getattr(b, 'y')={val}")
assert val == 2

# Check that the attribute exists before deletion
print(f"Has attribute 'x' before deletion: {hasattr(b, 'x')}")
assert hasattr(b, 'x')

# Try to delete attribute using delattr (THIS SHOULD FAIL)
print("\nAttempting to delete attribute 'x' using delattr...")
try:
    delattr(b, 'x')
    print("SUCCESS: Attribute deleted")
except AttributeError as e:
    print(f"FAILED: AttributeError raised: {e}")
    print(f"Has attribute 'x' after failed deletion: {hasattr(b, 'x')}")
    print(f"Value of b.x after failed deletion: {b.x}")