from django.core.cache.backends.locmem import LocMemCache

# Create a cache with max_entries=1 and cull_frequency=3
cache = LocMemCache("test", {
    "timeout": 300,
    "max_entries": 1,
    "cull_frequency": 3
})

print("Initial state:")
print(f"Cache size: {len(cache._cache)}")
print(f"Max entries: {cache._max_entries}")
print(f"Cull frequency: {cache._cull_frequency}")
print()

# Add the first item
cache.set("key1", "value1")
print("After adding key1:")
print(f"Cache size: {len(cache._cache)}")
print(f"Cache contents: {list(cache._cache.keys())}")
print()

# Add the second item - this should trigger culling
cache.set("key2", "value2")
print("After adding key2:")
print(f"Cache size: {len(cache._cache)}")
print(f"Max entries: {cache._max_entries}")
print(f"Cache contents: {list(cache._cache.keys())}")
print()

# Demonstrate the culling calculation
print("Culling calculation when cache size was 1:")
print(f"len(cache) // cull_frequency = {1} // {cache._cull_frequency} = {1 // cache._cull_frequency}")
print(f"Number of items to remove: {1 // cache._cull_frequency}")
print()

if len(cache._cache) > cache._max_entries:
    print(f"ERROR: Cache size ({len(cache._cache)}) exceeds max_entries ({cache._max_entries})")
else:
    print("Cache size is within limits")