#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/miniconda/lib/python3.13/site-packages')

from django.core.cache.utils import make_template_fragment_key

print("Testing cache key collisions:")
print("=" * 50)

# Test case 1
key1 = make_template_fragment_key("fragment", ["a:", "b"])
key2 = make_template_fragment_key("fragment", ["a", ":b"])

print(f"Test 1 - vary_on=['a:', 'b'] vs vary_on=['a', ':b']")
print(f"Key 1: {key1}")
print(f"Key 2: {key2}")
print(f"Keys are equal: {key1 == key2}")
print()

# Test case 2
key3 = make_template_fragment_key("test", ["x:y", "z"])
key4 = make_template_fragment_key("test", ["x", "y:z"])

print(f"Test 2 - vary_on=['x:y', 'z'] vs vary_on=['x', 'y:z']")
print(f"Key 3: {key3}")
print(f"Key 4: {key4}")
print(f"Keys are equal: {key3 == key4}")
print()

# Let me analyze what's happening step by step
import hashlib

print("=" * 50)
print("Manual hash calculation to understand the collision:")
print()

# For ["a:", "b"]
hasher1 = hashlib.md5(usedforsecurity=False)
hasher1.update(b"a:")  # First element
hasher1.update(b":")   # Separator
hasher1.update(b"b")   # Second element
hasher1.update(b":")   # Separator
print("Input for ['a:', 'b']: 'a:' + ':' + 'b' + ':' = 'a::b:'")
print(f"Hash: {hasher1.hexdigest()}")

# For ["a", ":b"]
hasher2 = hashlib.md5(usedforsecurity=False)
hasher2.update(b"a")   # First element
hasher2.update(b":")   # Separator
hasher2.update(b":b")  # Second element
hasher2.update(b":")   # Separator
print("Input for ['a', ':b']: 'a' + ':' + ':b' + ':' = 'a::b:'")
print(f"Hash: {hasher2.hexdigest()}")

print()
print("As we can see, both produce the exact same input to the hash: 'a::b:'")