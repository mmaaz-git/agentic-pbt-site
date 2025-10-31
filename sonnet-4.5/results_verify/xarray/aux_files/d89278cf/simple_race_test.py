#!/usr/bin/env python3
"""Simple test to check thread-safety in __delitem__"""
import threading
import sys
from xarray.backends.lru_cache import LRUCache

print("Testing thread-safety of LRUCache.__delitem__")
print("=" * 50)

# Check if __delitem__ uses lock
print("\n1. Code inspection:")
print("   Checking if __delitem__ uses the lock...")

import inspect
source = inspect.getsource(LRUCache.__delitem__)
print(f"   Source code of __delitem__:\n{source}")

if "_lock" in source or "with self._lock" in source:
    print("   ✓ __delitem__ DOES use the lock")
else:
    print("   ✗ __delitem__ DOES NOT use the lock")

# Check other methods for comparison
print("\n2. Comparing with other methods:")
for method_name in ['__getitem__', '__setitem__', '__delitem__']:
    method = getattr(LRUCache, method_name)
    source = inspect.getsource(method)
    uses_lock = "_lock" in source or "with self._lock" in source
    print(f"   {method_name}: {'Uses lock ✓' if uses_lock else 'No lock ✗'}")

# Simple concurrent test
print("\n3. Simple concurrent test:")
cache = LRUCache(maxsize=100)

# Fill cache
for i in range(100):
    cache[i] = f"value_{i}"

errors = []
completed = [0, 0]  # Track completions

def delete_items():
    try:
        for _ in range(100):
            for i in range(50):
                if i in cache:
                    del cache[i]
        completed[0] = 1
    except Exception as e:
        errors.append(f"Delete error: {type(e).__name__}: {e}")
        completed[0] = 1

def read_items():
    try:
        for _ in range(100):
            for i in range(50):
                if i in cache:
                    _ = cache[i]
        completed[1] = 1
    except Exception as e:
        errors.append(f"Read error: {type(e).__name__}: {e}")
        completed[1] = 1

t1 = threading.Thread(target=delete_items)
t2 = threading.Thread(target=read_items)

t1.start()
t2.start()
t1.join(timeout=5)
t2.join(timeout=5)

if not all(completed):
    print("   ✗ Threads did not complete (possible deadlock)")
elif errors:
    print(f"   ✗ Errors detected: {errors[0] if errors else ''}")
else:
    print("   ✓ No errors in this run (race conditions are probabilistic)")

print("\n4. Analysis:")
print("   Based on code inspection, __delitem__ lacks thread-safety.")
print("   It directly modifies self._cache without acquiring self._lock,")
print("   while all other mutating methods do use the lock.")