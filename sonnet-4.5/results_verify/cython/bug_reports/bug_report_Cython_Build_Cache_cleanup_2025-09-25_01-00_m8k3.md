# Bug Report: Cython.Build.Cache.cleanup_cache Inverted LRU Deletion Order

**Target**: `Cython.Build.Cache.cleanup_cache`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `cleanup_cache` method deletes cache files in the wrong order for an LRU (Least Recently Used) cache. It keeps the oldest files and deletes the newest ones, which is backwards. An LRU cache should delete least recently accessed files (oldest) and keep most recently accessed files (newest).

## Property-Based Test

```python
from hypothesis import given, strategies as st
import os
import tempfile
from Cython.Build.Cache import Cache


@given(st.integers(min_value=1, max_value=10))
def test_cleanup_cache_lru_order(num_files):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Cache(tmpdir, cache_size=50)

        for i in range(num_files):
            filepath = os.path.join(tmpdir, f"file_{i}.c")
            with open(filepath, "w") as f:
                f.write("x" * 100)
            atime = 1000 + i * 1000
            os.utime(filepath, (atime, atime))

        cache.cleanup_cache(ratio=0.5)

        remaining = os.listdir(tmpdir)
        if remaining:
            most_recent_file = f"file_{num_files-1}.c"
            least_recent_file = f"file_0.c"

            assert most_recent_file in remaining, \
                "LRU cache should keep most recently accessed files"
            assert least_recent_file not in remaining or len(remaining) == num_files, \
                "LRU cache should delete least recently accessed files first"
```

**Failing input**: Any number of files ≥ 2, e.g., `num_files=3`

## Reproducing the Bug

```python
import os
import tempfile
from Cython.Build.Cache import Cache

with tempfile.TemporaryDirectory() as tmpdir:
    cache = Cache(tmpdir, cache_size=100)

    for i, name in enumerate(['old', 'mid', 'new']):
        filepath = os.path.join(tmpdir, f"{name}_file.c")
        with open(filepath, "w") as f:
            f.write("x" * 50)
        atime = 1000 + i * 1000
        os.utime(filepath, (atime, atime))

    print("Before cleanup:", sorted(os.listdir(tmpdir)))

    cache.cleanup_cache(ratio=0.8)

    print("After cleanup:", sorted(os.listdir(tmpdir)))
```

Output:
```
Before cleanup: ['mid_file.c', 'new_file.c', 'old_file.c']
After cleanup: ['old_file.c']
```

The OLDEST file is kept, while the NEWEST files are deleted - backwards for LRU!

## Why This Is A Bug

The method is intended to implement an LRU cache eviction policy (evidenced by its use of `st_atime` - access time). LRU caches should:
1. Keep frequently/recently accessed items
2. Delete least recently accessed items first

However, the current code on line 195:
```python
for time, size, file in reversed(sorted(all)):
```

Where `all` contains tuples `(st_atime, st_size, path)`:
1. `sorted(all)` sorts by access time (ascending): oldest first
2. `reversed(...)` reverses to: newest first
3. Loop deletes in this order: newest → oldest

This is inverted LRU behavior. The cache keeps old, rarely-accessed files and deletes recent, frequently-accessed ones. This defeats the purpose of caching, as:
- Hot compilation artifacts (recently built) get evicted
- Cold artifacts (old, unused) stay in cache
- Build times increase as frequently-used cache entries are repeatedly evicted

The impact is moderate because while it doesn't break functionality, it significantly degrades cache effectiveness, leading to unnecessary recompilations.

## Fix

Remove the `reversed()` call to delete in correct LRU order (oldest first):

```diff
--- a/Cython/Build/Cache.py
+++ b/Cython/Build/Cache.py
@@ -192,7 +192,7 @@ class Cache:
             total_size += s.st_size
             all.append((s.st_atime, s.st_size, path))
         if total_size > self.cache_size:
-            for time, size, file in reversed(sorted(all)):
+            for time, size, file in sorted(all):
                 os.unlink(file)
                 total_size -= size
                 if total_size < self.cache_size * ratio:
```

This change deletes files in order from oldest access time to newest, correctly implementing LRU eviction.