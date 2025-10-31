# Bug Report: Cython.Build.Cache cleanup_cache Removes Newest Files Instead of Oldest

**Target**: `Cython.Build.Cache.cleanup_cache`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `cleanup_cache()` method removes the most recently accessed files first instead of the least recently accessed files, implementing the opposite of the intended LRU (Least Recently Used) cache eviction policy.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from Cython.Build.Cache import Cache
import os
import tempfile
import time


@given(st.integers(min_value=3, max_value=10))
@settings(max_examples=10)
def test_cache_cleanup_preserves_recently_accessed_files(num_files):
    """Test that cache cleanup removes oldest files first (LRU policy)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create cache with small size to force cleanup
        cache = Cache(tmpdir, cache_size=500)

        # Create files with increasing access times
        files = []
        for i in range(num_files):
            filepath = os.path.join(tmpdir, f"file{i}.txt")
            with open(filepath, 'w') as f:
                f.write('x' * 200)  # Each file is 200 bytes
            time.sleep(0.01)  # Ensure different timestamps
            os.utime(filepath, None)  # Update access time
            files.append(filepath)

        # Calculate total size
        total_size = sum(os.path.getsize(f) for f in files)

        # Only test cleanup if we exceed cache size
        if total_size > cache.cache_size:
            # Run cleanup with ratio=0.5 (keep 50% of cache size)
            cache.cleanup_cache(ratio=0.5)

            # Check what remains
            remaining = set(os.listdir(tmpdir))

            # Get the oldest and newest file names
            oldest_file = os.path.basename(files[0])
            newest_file = os.path.basename(files[-1])

            # The newest file should be kept (LRU policy)
            assert newest_file in remaining, f"Most recently accessed file '{newest_file}' should be kept, but was removed. Remaining files: {remaining}"


# Run the test
if __name__ == "__main__":
    test_cache_cleanup_preserves_recently_accessed_files()
```

<details>

<summary>
**Failing input**: `num_files=3`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 50, in <module>
    test_cache_cleanup_preserves_recently_accessed_files()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 12, in test_cache_cleanup_preserves_recently_accessed_files
    @settings(max_examples=10)
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 45, in test_cache_cleanup_preserves_recently_accessed_files
    assert newest_file in remaining, f"Most recently accessed file '{newest_file}' should be kept, but was removed. Remaining files: {remaining}"
           ^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Most recently accessed file 'file2.txt' should be kept, but was removed. Remaining files: {'file0.txt'}
Falsifying example: test_cache_cleanup_preserves_recently_accessed_files(
    num_files=3,
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Build.Cache import Cache
import os
import tempfile
import time

# Create a temporary directory for testing
with tempfile.TemporaryDirectory() as tmpdir:
    # Create a cache with a small size limit (500 bytes)
    cache = Cache(tmpdir, cache_size=500)

    # Create three files with timestamps spaced apart
    # Each file is 200 bytes, total will be 600 bytes (exceeds cache size)

    # Create the oldest file
    old_file = os.path.join(tmpdir, "old.txt")
    with open(old_file, 'w') as f:
        f.write('x' * 200)

    # Wait to ensure different timestamps
    time.sleep(0.1)

    # Create a newer file
    new_file = os.path.join(tmpdir, "new.txt")
    with open(new_file, 'w') as f:
        f.write('x' * 200)

    # Wait to ensure different timestamps
    time.sleep(0.1)

    # Create the newest file
    newest_file = os.path.join(tmpdir, "newest.txt")
    with open(newest_file, 'w') as f:
        f.write('x' * 200)

    # Show files before cleanup
    print(f"Files before cleanup: {sorted(os.listdir(tmpdir))}")

    # Get access times before cleanup
    files_with_times = []
    for fname in os.listdir(tmpdir):
        fpath = os.path.join(tmpdir, fname)
        atime = os.stat(fpath).st_atime
        files_with_times.append((fname, atime))
    files_with_times.sort(key=lambda x: x[1])

    print("\nFiles ordered by access time (oldest to newest):")
    for fname, atime in files_with_times:
        print(f"  {fname}: {atime}")

    # Run cleanup with ratio=0.5 (should keep 250 bytes, so only 1 file)
    cache.cleanup_cache(ratio=0.5)

    # Show files after cleanup
    remaining = set(os.listdir(tmpdir))
    print(f"\nFiles after cleanup: {sorted(remaining)}")

    # Check which files were kept vs removed
    if "old.txt" in remaining and "newest.txt" not in remaining:
        print("\nBUG CONFIRMED: Oldest file kept, newest file removed (opposite of LRU)")
        print("Expected behavior: Keep newest files, remove oldest files")
        print("Actual behavior: Kept oldest file, removed newest files")
    elif "newest.txt" in remaining and "old.txt" not in remaining:
        print("\nCorrect LRU behavior: Newest file kept, oldest file removed")
    else:
        print(f"\nUnexpected result - remaining files: {remaining}")
```

<details>

<summary>
Output showing bug: Oldest file kept, newest files removed
</summary>
```
Files before cleanup: ['new.txt', 'newest.txt', 'old.txt']

Files ordered by access time (oldest to newest):
  old.txt: 1758834212.1506245
  new.txt: 1758834212.250998
  newest.txt: 1758834212.351003

Files after cleanup: ['old.txt']

BUG CONFIRMED: Oldest file kept, newest file removed (opposite of LRU)
Expected behavior: Keep newest files, remove oldest files
Actual behavior: Kept oldest file, removed newest files
```
</details>

## Why This Is A Bug

This violates the expected behavior of an LRU (Least Recently Used) cache for several critical reasons:

1. **Inverse of LRU Policy**: The code explicitly tracks access times via `os.utime(cached, None)` when files are loaded from cache (lines 146, 151 in Cache.py), indicating the intent to implement recency-based eviction. However, the cleanup method sorts files by access time and then reverses the list with `reversed(sorted(all))` on line 195, causing it to delete the most recently accessed files first.

2. **Cache Purpose Defeated**: A cache that removes recently used items provides zero benefit. Users will experience constant cache misses for frequently used files, forcing unnecessary recompilation of the same Cython modules repeatedly.

3. **Performance Degradation**: This bug causes severe performance issues in build systems that rely on Cython caching. Instead of speeding up builds by reusing compiled modules, the cache actively slows down builds by discarding the most useful cached items.

4. **Clear Implementation Error**: The combination of sorting by access time (oldest first) and then reversing the order (newest first) before deletion is logically backwards. The code should iterate through files from oldest to newest when deleting to maintain LRU semantics.

## Relevant Context

The bug is located in `/Cython/Build/Cache.py` at line 195 in the `cleanup_cache` method. The cache is designed to store compiled Cython files to avoid recompilation. When the cache exceeds its size limit (default 100MB, configurable), it should remove the least recently used files to make room for new ones.

The cache implementation correctly:
- Updates access times when files are used (lines 146, 151)
- Tracks file access times in the cleanup method (line 193)
- Sorts files by access time (line 195)

But incorrectly:
- Reverses the sorted order before deletion, removing newest files first

This is particularly problematic for CI/CD systems and development workflows where the same core files are repeatedly compiled. Those frequently-used files will be constantly evicted, negating any caching benefit.

## Proposed Fix

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