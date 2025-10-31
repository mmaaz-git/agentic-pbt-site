from dask.dataframe.dask_expr._util import LRU

lru = LRU(maxsize=2)
lru[0] = 0
lru[1] = 1
print(f"After adding 0 and 1: len={len(lru)}, keys={list(lru.keys())}")
assert len(lru) == 2, f"Expected len=2, got {len(lru)}"
assert 0 in lru, "Key 0 should be in cache"
assert 1 in lru, "Key 1 should be in cache"

# Update existing key 1
lru[1] = 1
print(f"After updating key 1: len={len(lru)}, keys={list(lru.keys())}")
assert len(lru) == 2, f"Expected len=2, got {len(lru)}"
assert 0 in lru, "Key 0 should still be in cache after updating key 1"
assert 1 in lru, "Key 1 should still be in cache"