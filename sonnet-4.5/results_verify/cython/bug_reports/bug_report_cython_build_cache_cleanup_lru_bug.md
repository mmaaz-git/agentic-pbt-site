# Bug Report: Cython.Build.Cache cleanup_cache Removes Newest Files Instead of Oldest

**Target**: `Cython.Build.Cache.cleanup_cache`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `cleanup_cache()` method removes the most recently accessed files first instead of the least recently accessed files, implementing the opposite of the intended LRU (Least Recently Used) cache eviction policy.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from Cython.Build.Cache import Cache
import os
import tempfile
import time


@given(st.integers(min_value=3, max_value=10))
def test_cache_cleanup_preserves_recently_accessed_files(num_files):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Cache(tmpdir, cache_size=500)

        files = []
        for i in range(num_files):
            filepath = os.path.join(tmpdir, f"file{i}.txt")
            with open(filepath, 'w') as f:
                f.write('x' * 200)
            time.sleep(0.01)
            os.utime(filepath, None)
            files.append(filepath)

        total_size = sum(os.path.getsize(f) for f in files)

        if total_size > cache.cache_size:
            cache.cleanup_cache(ratio=0.5)
            remaining = set(os.listdir(tmpdir))

            oldest_file = os.path.basename(files[0])
            newest_file = os.path.basename(files[-1])

            assert newest_file in remaining, "Most recently accessed file should be kept"
```

**Failing input**: `num_files=3` (any value >= 3 triggers the bug when total size exceeds cache size)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Build.Cache import Cache
import os
import tempfile
import time

with tempfile.TemporaryDirectory() as tmpdir:
    cache = Cache(tmpdir, cache_size=500)

    old_file = os.path.join(tmpdir, "old.txt")
    with open(old_file, 'w') as f:
        f.write('x' * 200)

    time.sleep(0.1)

    new_file = os.path.join(tmpdir, "new.txt")
    with open(new_file, 'w') as f:
        f.write('x' * 200)

    time.sleep(0.1)

    newest_file = os.path.join(tmpdir, "newest.txt")
    with open(newest_file, 'w') as f:
        f.write('x' * 200)

    print(f"Files before cleanup: {sorted(os.listdir(tmpdir))}")

    cache.cleanup_cache(ratio=0.5)

    remaining = set(os.listdir(tmpdir))
    print(f"Files after cleanup: {sorted(remaining)}")

    if "old.txt" in remaining and "newest.txt" not in remaining:
        print("BUG CONFIRMED: Oldest file kept, newest file removed (opposite of LRU)")
```

## Why This Is A Bug

An LRU (Least Recently Used) cache should evict the least recently accessed files first to maximize cache utility. The current implementation sorts files by access time and then reverses the list before deletion, causing it to remove the most recently accessed files instead. This defeats the purpose of an LRU cache and will lead to poor cache performance.

## Fix

```diff
--- a/Cython/Build/Cache.py
+++ b/Cython/Build/Cache.py
@@ -192,7 +192,7 @@ class Cache:
             all.append((s.st_atime, s.st_size, path))
         if total_size > self.cache_size:
-            for time, size, file in reversed(sorted(all)):
+            for time, size, file in sorted(all):
                 os.unlink(file)
                 total_size -= size
                 if total_size < self.cache_size * ratio:
```