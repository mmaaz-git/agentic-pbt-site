# Bug Report: xarray.backends.file_manager._HashedSequence Cached Hash Becomes Stale After Mutation

**Target**: `xarray.backends.file_manager._HashedSequence`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_HashedSequence` class caches its hash value at initialization but inherits from `list`, allowing mutations that make the cached hash stale and incorrect, violating Python's hash contract for objects used as dictionary keys.

## Property-Based Test

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

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

if __name__ == "__main__":
    test_hashed_sequence_mutation_breaks_hash()
```

<details>

<summary>
**Failing input**: `tuple_value=(0, 0)`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 26, in <module>
    test_hashed_sequence_mutation_breaks_hash()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 9, in test_hashed_sequence_mutation_breaks_hash
    tuple_value=st.tuples(st.integers(), st.integers()),
               ^^^
  File "/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 22, in test_hashed_sequence_mutation_breaks_hash
    assert new_hash == actual_tuple_hash, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Cached hash should match the hash of current tuple value
Falsifying example: test_hashed_sequence_mutation_breaks_hash(
    tuple_value=(0, 0),  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.backends.file_manager import _HashedSequence

# Create a _HashedSequence with an initial tuple
original_tuple = (1, 2, 3)
hashed_seq = _HashedSequence(original_tuple)

print(f"Original: {list(hashed_seq)}, hash={hash(hashed_seq)}")

# Mutate the _HashedSequence by appending an element
hashed_seq.append(999)

print(f"After mutation: {list(hashed_seq)}, hash={hash(hashed_seq)}")
print(f"Expected hash: {hash(tuple(hashed_seq))}")

# Demonstrate the bug: hash is stale after mutation
if hash(hashed_seq) != hash(tuple(hashed_seq)):
    print("Bug: hash is stale after mutation")
    print(f"Cached hash: {hash(hashed_seq)}")
    print(f"Correct hash for current content: {hash(tuple(hashed_seq))}")
else:
    print("No bug detected")
```

<details>

<summary>
Hash remains unchanged after list mutation, violating hash contract
</summary>
```
Original: [1, 2, 3], hash=529344067295497451
After mutation: [1, 2, 3, 999], hash=529344067295497451
Expected hash: -5051857752814232577
Bug: hash is stale after mutation
Cached hash: 529344067295497451
Correct hash for current content: -5051857752814232577
```
</details>

## Why This Is A Bug

This violates Python's fundamental hash contract: objects that are equal must have the same hash, and an object's hash should not change while it's being used as a dictionary key. The `_HashedSequence` class is used as a dictionary key in the `CachingFileManager` class (lines 211, 221, and 232 in file_manager.py), where it serves as the cache key for file objects.

The class inherits from `list` (line 322), making it mutable, but caches its hash value at initialization (line 333). When the list is mutated using methods like `append()`, `extend()`, or item assignment, the cached `hashvalue` attribute remains unchanged, causing:

1. **Dictionary corruption**: If used as a dict key and then mutated, the object becomes unfindable in the dict
2. **Cache misses**: The `CachingFileManager` could fail to find cached files if keys are mutated
3. **Memory leaks**: Orphaned cache entries that can never be retrieved or cleaned up

While the current xarray codebase never mutates `_HashedSequence` objects after creation (the `_key` is created once in line 148 and never modified), this design is fragile and relies on implicit assumptions that future code changes could violate.

## Relevant Context

The `_HashedSequence` class is based on Python's internal `functools._HashedSeq` (as noted in the docstring), which has the same issue. However, Python's version is truly internal and never exposed, while xarray's version is stored as an instance variable (`self._key`) that could theoretically be accessed and mutated.

The class is specifically designed to optimize repeated dictionary lookups in the file caching system. From the code at line 167, we can see it's created with a tuple containing file opener details, arguments, mode, and manager ID. This tuple is meant to uniquely identify a file for caching purposes.

Documentation: The class docstring (lines 323-329) mentions it's for "speedup repeated look-ups by caching hash values" but doesn't explicitly state the immutability requirement.

## Proposed Fix

The cleanest solution is to make `_HashedSequence` immutable by inheriting from `tuple` instead of `list`:

```diff
--- a/xarray/backends/file_manager.py
+++ b/xarray/backends/file_manager.py
@@ -319,15 +319,19 @@ class _RefCounter:
         return count


-class _HashedSequence(list):
+class _HashedSequence(tuple):
     """Speedup repeated look-ups by caching hash values.

     Based on what Python uses internally in functools.lru_cache.

     Python doesn't perform this optimization automatically:
     https://bugs.python.org/issue1462796
     """

-    def __init__(self, tuple_value):
-        self[:] = tuple_value
-        self.hashvalue = hash(tuple_value)
+    def __new__(cls, tuple_value):
+        instance = tuple.__new__(cls, tuple_value)
+        instance.hashvalue = hash(tuple_value)
+        return instance

     def __hash__(self):
         return self.hashvalue
```