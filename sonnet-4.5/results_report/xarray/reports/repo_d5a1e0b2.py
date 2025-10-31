import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import threading
from xarray.backends.lru_cache import LRUCache

# Create an LRUCache instance
cache = LRUCache(maxsize=10)

# Add some items to the cache
for i in range(5):
    cache[i] = f"value_{i}"

# Function to delete items from the cache
def delete_items():
    for i in range(5):
        try:
            del cache[i]
        except KeyError:
            pass  # Item already deleted by another thread

# Create multiple threads that will try to delete the same items
threads = []
for _ in range(10):
    t = threading.Thread(target=delete_items)
    threads.append(t)
    t.start()

# Wait for all threads to complete
for t in threads:
    t.join()

print("Test completed - race condition may have occurred silently")

# Verify all items were deleted
remaining_items = list(cache.keys())
print(f"Remaining items in cache: {remaining_items}")
print(f"Cache size: {len(cache)}")