#!/usr/bin/env python3
"""Full test to verify the Field.__repr__ inconsistency"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Utility.Dataclasses import field, Field

print("=" * 60)
print("Testing Field.__repr__ attribute name consistency")
print("=" * 60)

# Test 1: Basic reproduction
print("\n1. Basic reproduction:")
f = field(kw_only=True)
repr_str = repr(f)
print(f"   Created field with kw_only=True")
print(f"   repr(field): {repr_str}")
print()

# Test 2: Check attribute access
print("2. Attribute access test:")
print(f"   f.kw_only exists: {hasattr(f, 'kw_only')}")
print(f"   f.kw_only value: {f.kw_only}")
print(f"   f.kwonly exists: {hasattr(f, 'kwonly')}")
if hasattr(f, 'kwonly'):
    print(f"   f.kwonly value: {f.kwonly}")
print()

# Test 3: Check what's in the repr
print("3. Repr content analysis:")
print(f"   'kw_only=True' in repr: {'kw_only=True' in repr_str}")
print(f"   'kwonly=True' in repr: {'kwonly=True' in repr_str}")
print()

# Test 4: Test with different values
print("4. Testing with different kw_only values:")
for val in [True, False, None]:
    f = field(kw_only=val)
    repr_str = repr(f)
    print(f"   kw_only={val}:")
    print(f"     repr contains 'kwonly={val}': {'kwonly=' + repr(val) in repr_str}")
    print(f"     repr contains 'kw_only={val}': {'kw_only=' + repr(val) in repr_str}")
    print(f"     Actual attribute f.kw_only={f.kw_only}")
print()

# Test 5: Check Field class directly
print("5. Direct Field class instantiation:")
f2 = Field(default=None, default_factory=None, init=True, repr=True,
           hash=None, compare=True, metadata=None, kw_only=True)
repr_str2 = repr(f2)
print(f"   Field(..., kw_only=True)")
print(f"   repr: {repr_str2}")
print(f"   'kwonly=True' in repr: {'kwonly=True' in repr_str2}")
print(f"   'kw_only=True' in repr: {'kw_only=True' in repr_str2}")
print()

# Test 6: Check the __slots__ definition
print("6. Field.__slots__ analysis:")
print(f"   Field.__slots__: {Field.__slots__}")
print(f"   'kw_only' in __slots__: {'kw_only' in Field.__slots__}")
print(f"   'kwonly' in __slots__: {'kwonly' in Field.__slots__}")
print()

print("=" * 60)
print("CONCLUSION:")
print("  The Field class uses 'kw_only' everywhere (slots, init, attribute)")
print("  But __repr__ outputs 'kwonly' (without underscore)")
print("  This is an inconsistency in the naming convention")
print("=" * 60)