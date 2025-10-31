#!/usr/bin/env python3
"""
Demonstration of Django cache incr_version bug when delta=0.

This script shows that calling incr_version with delta=0 incorrectly
deletes the cached value instead of preserving it at the same version.
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.core.cache.backends.locmem import LocMemCache

print("=== Django Cache incr_version Bug Demonstration ===")
print("Testing behavior when incrementing cache version by delta=0")
print()

# Initialize cache
cache = LocMemCache("test", {"timeout": 300})
cache.clear()

# Test case 1: Basic delta=0 scenario
print("Test 1: Basic delta=0 scenario")
print("-" * 40)
cache.set("mykey", 42, version=1)
print(f"Initial setup: set('mykey', 42, version=1)")
print(f"Value at version 1 before incr_version: {cache.get('mykey', version=1)}")

new_version = cache.incr_version("mykey", delta=0, version=1)
print(f"Called incr_version('mykey', delta=0, version=1)")
print(f"Returned new_version: {new_version}")

result = cache.get("mykey", version=new_version)
print(f"Value at version {new_version} after incr_version: {result}")
print(f"Expected: 42")
print(f"Actual: {result}")
print(f"BUG: Value was deleted! (got None instead of 42)")
print()

# Test case 2: Compare with non-zero delta values
print("Test 2: Comparing different delta values")
print("-" * 40)

test_cases = [
    ("delta=-1", -1),
    ("delta=0", 0),
    ("delta=1", 1),
    ("delta=2", 2),
]

for label, delta in test_cases:
    cache.clear()
    cache.set("testkey", 100, version=5)

    try:
        new_ver = cache.incr_version("testkey", delta=delta, version=5)
        val_at_new = cache.get("testkey", version=new_ver)
        val_at_old = cache.get("testkey", version=5)

        print(f"{label:10} -> new_version={new_ver}, value={val_at_new}, old_version_value={val_at_old}")

        if delta == 0 and val_at_new is None:
            print(f"           ^^ BUG: Value deleted when delta=0!")
    except Exception as e:
        print(f"{label:10} -> Error: {e}")

print()
print("=== Summary ===")
print("When delta=0, incr_version incorrectly deletes the cached value.")
print("This happens because the implementation:")
print("1. Sets the value at version + 0 (same version)")
print("2. Then deletes the value at the original version")
print("3. Since both operations target the same version, the value is lost")