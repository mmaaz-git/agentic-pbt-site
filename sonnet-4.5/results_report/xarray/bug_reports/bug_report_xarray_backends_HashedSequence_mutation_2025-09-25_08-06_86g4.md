# Bug Report: xarray.backends._HashedSequence Stale Hash After Mutation

**Target**: `xarray.backends.file_manager._HashedSequence`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_HashedSequence` class caches its hash value at initialization but inherits from `list`, making it mutable. If the list is mutated after creation, the cached hash becomes stale and no longer reflects the actual content, violating Python's hash contract and potentially causing cache corruption.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.backends.file_manager import _HashedSequence

@given(
    tuple_value=st.tuples(st.integers(), st.integers()),
)
def test_hashed_sequence_mutation_breaks_hash(tuple_value):
    hashed_seq = _HashedSequence(tuple_value)
    original_hash = hash(hashed_seq)

    hashed_seq.append(999)
    new_hash = hash(hashed_seq)

    assert original_hash == new_hash, \
        "Hash should not change even after mutation (cached hash bug)"

    actual_tuple_hash = hash(tuple(hashed_seq))
    assert new_hash == actual_tuple_hash, \
        "Cached hash should match the hash of current tuple value"
```

**Failing input**: `tuple_value=(0, 0)`

## Reproducing the Bug

```python
from xarray.backends.file_manager import _HashedSequence

original_tuple = (1, 2, 3)
hashed_seq = _HashedSequence(original_tuple)

print(f"Original: {list(hashed_seq)}, hash={hash(hashed_seq)}")

hashed_seq.append(999)

print(f"After mutation: {list(hashed_seq)}, hash={hash(hashed_seq)}")
print(f"Expected hash: {hash(tuple(hashed_seq))}")
print(f"Bug: hash is stale after mutation")
```

**Output**:
```
Original: [1, 2, 3], hash=529344067295497451
After mutation: [1, 2, 3, 999], hash=529344067295497451
Expected hash: 3713081631934410656
Bug: hash is stale after mutation
```

## Why This Is A Bug

The `_HashedSequence` class is used as a dictionary key in `CachingFileManager` (see line 211, 221, 232 in file_manager.py). Dictionary keys should be immutable, but `_HashedSequence` inherits from `list`, making it mutable.

While the class is based on Python's `functools._HashedSeq`, there's a critical difference: in xarray, the `_HashedSequence` is stored as `self._key` (line 148), potentially exposing it to mutation. If the key is mutated after being inserted into the cache, the cached hash no longer reflects the content, which can lead to:

1. Cache lookup failures
2. Orphaned cache entries
3. Violation of Python's hash invariant: objects used as dict keys should have stable hashes

Although the current code never mutates `_key`, the design is fragile and relies on implicit assumptions.

## Fix

The cleanest fix is to make `_HashedSequence` immutable by inheriting from `tuple` instead of `list`, or by preventing mutations:

```diff
--- a/xarray/backends/file_manager.py
+++ b/xarray/backends/file_manager.py
@@ -319,7 +319,7 @@ class _RefCounter:
         return count


-class _HashedSequence(list):
+class _HashedSequence(tuple):
     """Speedup repeated look-ups by caching hash values.

     Based on what Python uses internally in functools.lru_cache.
@@ -329,8 +329,8 @@ class _HashedSequence(list):
     """

     def __init__(self, tuple_value):
-        self[:] = tuple_value
-        self.hashvalue = hash(tuple_value)
+        # Can't modify tuple after creation, compute hash in __new__
+        pass

     def __hash__(self):
-        return self.hashvalue
+        if not hasattr(self, 'hashvalue'):
+            self.hashvalue = tuple.__hash__(self)
+        return self.hashvalue
+
+    def __new__(cls, tuple_value):
+        instance = tuple.__new__(cls, tuple_value)
+        instance.hashvalue = hash(tuple_value)
+        return instance
```

Alternatively, prevent mutations by overriding mutating methods to raise exceptions, or use `__slots__` to prevent attribute assignment.