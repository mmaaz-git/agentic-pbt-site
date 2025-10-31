#!/usr/bin/env python3
"""Test standard LRU cache behavior for comparison."""

from collections import OrderedDict

print("Testing Python's OrderedDict as LRU:")
print("="*60)

class SimpleLRU:
    """A simple LRU implementation for comparison"""
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.cache = OrderedDict()

    def __setitem__(self, key, value):
        # Standard LRU behavior
        if key in self.cache:
            # Update existing key - just update value and move to end
            del self.cache[key]
            self.cache[key] = value
        else:
            # New key - check if we need to evict
            if len(self.cache) >= self.maxsize:
                # Evict least recently used (first item)
                self.cache.popitem(last=False)
            self.cache[key] = value

    def __getitem__(self, key):
        # Move to end to mark as recently used
        value = self.cache[key]
        del self.cache[key]
        self.cache[key] = value
        return value

    def __len__(self):
        return len(self.cache)

    def __contains__(self, key):
        return key in self.cache

print("Standard LRU implementation behavior:")
lru = SimpleLRU(2)
lru["a"] = 1
lru["b"] = 2
print(f"Initial: {dict(lru.cache)}")

lru["a"] = 999  # Update existing
print(f"After updating 'a': {dict(lru.cache)}")
print(f"Length: {len(lru)}")
print(f"Both keys present: 'a' in lru: {'a' in lru}, 'b' in lru: {'b' in lru}")

print("\n" + "="*60)
print("Comparing with dask's LRU:")

from dask.dataframe.dask_expr._util import LRU

dask_lru = LRU(2)
dask_lru["a"] = 1
dask_lru["b"] = 2
print(f"Initial: {dict(dask_lru)}")

dask_lru["a"] = 999  # Update existing
print(f"After updating 'a': {dict(dask_lru)}")
print(f"Length: {len(dask_lru)}")
print(f"Both keys present: 'a' in lru: {'a' in dask_lru}, 'b' in lru: {'b' in dask_lru}")

print("\n" + "="*60)
print("EXPECTED LRU BEHAVIOR:")
print("1. When updating an existing key, the cache should NOT evict anything")
print("2. The key should be updated in place or moved to 'most recently used'")
print("3. Eviction should only happen when adding a NEW key to a full cache")
print("\nDask's implementation violates this by evicting on every setitem when full,"
      "\neven if the key already exists.")