#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Test 1: Reproduce the exact bug scenario
print("=" * 60)
print("TEST 1: Reproducing the bug with max_entries=1")
print("=" * 60)

from django.core.cache.backends.locmem import LocMemCache

cache = LocMemCache("test", {
    "timeout": 300,
    "max_entries": 1,
    "cull_frequency": 3
})

print(f"Initial cache size: {len(cache._cache)}")
print(f"Max entries: {cache._max_entries}")
print(f"Cull frequency: {cache._cull_frequency}")

cache.set("key1", "value1")
print(f"After adding key1, cache size: {len(cache._cache)}")
print(f"Cache contains: {list(cache._cache.keys())}")

cache.set("key2", "value2")
print(f"After adding key2, cache size: {len(cache._cache)}")
print(f"Cache contains: {list(cache._cache.keys())}")

print(f"\nBUG CHECK: Cache size ({len(cache._cache)}) > max_entries ({cache._max_entries})? {len(cache._cache) > cache._max_entries}")

# Test 2: Check the culling calculation
print("\n" + "=" * 60)
print("TEST 2: Understanding the culling calculation")
print("=" * 60)

cache2 = LocMemCache("test2", {
    "timeout": 300,
    "max_entries": 2,
    "cull_frequency": 3
})

print(f"Max entries: {cache2._max_entries}")
print(f"Cull frequency: {cache2._cull_frequency}")

# Fill to max_entries
cache2.set("a", 1)
cache2.set("b", 2)
print(f"Cache size after filling to max: {len(cache2._cache)}")

# Try to add one more
cache2.set("c", 3)
print(f"Cache size after adding one more: {len(cache2._cache)}")
print(f"Cache contains: {list(cache2._cache.keys())}")

# Calculate what _cull would do
print(f"\nCulling calculation: {len(cache2._cache)} // {cache2._cull_frequency} = {2 // 3}")
print("Since 2 // 3 = 0, no items would be removed in _cull()")

# Test 3: Test with different parameters
print("\n" + "=" * 60)
print("TEST 3: Testing various max_entries and cull_frequency combinations")
print("=" * 60)

test_cases = [
    (1, 2),  # max_entries=1, cull_freq=2
    (1, 3),  # max_entries=1, cull_freq=3
    (1, 4),  # max_entries=1, cull_freq=4
    (2, 3),  # max_entries=2, cull_freq=3
    (2, 4),  # max_entries=2, cull_freq=4
    (3, 4),  # max_entries=3, cull_freq=4
]

for max_entries, cull_freq in test_cases:
    cache_test = LocMemCache(f"test_{max_entries}_{cull_freq}", {
        "max_entries": max_entries,
        "cull_frequency": cull_freq
    })

    # Fill beyond max_entries
    for i in range(max_entries + 1):
        cache_test.set(f"key{i}", i)

    final_size = len(cache_test._cache)
    cull_calculation = max_entries // cull_freq
    exceeds = final_size > max_entries

    print(f"max_entries={max_entries}, cull_freq={cull_freq}: "
          f"final_size={final_size}, cull_calc={cull_calculation}, "
          f"exceeds_max={'YES' if exceeds else 'NO'}")