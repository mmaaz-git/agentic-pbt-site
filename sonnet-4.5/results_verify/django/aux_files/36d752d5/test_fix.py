#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.core.cache.backends.locmem import LocMemCache

# Monkey-patch the _cull method with the proposed fix
original_cull = LocMemCache._cull

def fixed_cull(self):
    if self._cull_frequency == 0:
        self._cache.clear()
        self._expire_info.clear()
    else:
        # The fix: ensure at least 1 item is removed
        count = max(1, len(self._cache) // self._cull_frequency)
        for i in range(count):
            key, _ = self._cache.popitem()
            del self._expire_info[key]

LocMemCache._cull = fixed_cull

print("Testing with the proposed fix applied...")
print("=" * 60)

# Test the original failing case
cache = LocMemCache("test_fixed", {
    "timeout": 300,
    "max_entries": 1,
    "cull_frequency": 3
})

cache.set("key1", "value1")
print(f"After adding key1, cache size: {len(cache._cache)}")

cache.set("key2", "value2")
print(f"After adding key2, cache size: {len(cache._cache)}")
print(f"Cache contains: {list(cache._cache.keys())}")
print(f"Does cache size exceed max_entries? {len(cache._cache) > cache._max_entries}")

# Test multiple scenarios
print("\n" + "=" * 60)
print("Testing various scenarios with the fix:")

test_cases = [
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
]

for max_entries, cull_freq in test_cases:
    cache_test = LocMemCache(f"test_fixed_{max_entries}_{cull_freq}", {
        "max_entries": max_entries,
        "cull_frequency": cull_freq
    })

    # Fill beyond max_entries
    for i in range(max_entries + 1):
        cache_test.set(f"key{i}", i)

    final_size = len(cache_test._cache)
    exceeds = final_size > max_entries

    print(f"max_entries={max_entries}, cull_freq={cull_freq}: "
          f"final_size={final_size}, exceeds_max={'YES' if exceeds else 'NO'}")