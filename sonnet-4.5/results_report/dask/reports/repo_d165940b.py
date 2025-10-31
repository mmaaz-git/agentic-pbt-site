from dask.dataframe.dask_expr._util import LRU

# Test 1: Basic update behavior
print("Test 1: Basic update behavior")
lru = LRU(2)
lru["a"] = 1
lru["b"] = 2
print(f"Initial state: {dict(lru)}")
print(f"Cache size: {len(lru)}")

# Update existing key "a"
lru["a"] = 999
print(f"After updating 'a': {dict(lru)}")
print(f"Cache size: {len(lru)}")

# What we expect: Both "a" and "b" should still be in the cache
# What actually happens: We'll see...
print(f"'a' in cache: {('a' in lru)}")
print(f"'b' in cache: {('b' in lru)}")
if 'a' in lru:
    print(f"Value of 'a': {lru['a']}")
if 'b' in lru:
    print(f"Value of 'b': {lru['b']}")

print("\n" + "="*50 + "\n")

# Test 2: Update in a size-1 cache
print("Test 2: Update in a size-1 cache")
lru = LRU(1)
lru["x"] = 100
print(f"Initial: {dict(lru)}")
lru["x"] = 200  # Update same key
print(f"After update: {dict(lru)}")
print(f"'x' in cache: {('x' in lru)}")
if 'x' in lru:
    print(f"Value of 'x': {lru['x']}")

print("\n" + "="*50 + "\n")

# Test 3: Demonstrating the eviction issue
print("Test 3: Eviction during update")
lru = LRU(3)
lru["first"] = 1
lru["second"] = 2
lru["third"] = 3
print(f"Full cache: {dict(lru)}")

# Access "first" to make it recently used
_ = lru["first"]
print(f"After accessing 'first': {list(lru.keys())}")

# Update "third" - should NOT evict anything since we're not adding a new key
lru["third"] = 300
print(f"After updating 'third': {dict(lru)}")
print(f"Cache size: {len(lru)}")

# Check what got evicted (if anything)
for key in ["first", "second", "third"]:
    print(f"'{key}' in cache: {(key in lru)}")