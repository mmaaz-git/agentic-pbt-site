from dask.dataframe.dask_expr._util import LRU

# Create an LRU cache with maxsize of 3
lru = LRU(3)

# Fill the cache to capacity
lru['a'] = 1
lru['b'] = 2
lru['c'] = 3

print(f"Initial state after filling cache:")
print(f"  Keys: {list(lru.keys())}")
print(f"  Length: {len(lru)}")
print(f"  Expected length: 3")
print()

# Update an existing key
print(f"Updating existing key 'c' with new value 4...")
lru['c'] = 4

print(f"State after updating 'c':")
print(f"  Keys: {list(lru.keys())}")
print(f"  Length: {len(lru)}")
print(f"  Expected length: 3")
print()

# Check what happened
if len(lru) < 3:
    print(f"ERROR: Cache size dropped from 3 to {len(lru)} after updating existing key!")
    print(f"Missing key(s): {set(['a', 'b', 'c']) - set(lru.keys())}")
else:
    print(f"Cache size correctly maintained at {len(lru)}")