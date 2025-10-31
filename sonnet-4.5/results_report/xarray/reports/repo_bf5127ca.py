import threading
from xarray.backends.lru_cache import LRUCache

cache = LRUCache(maxsize=10)
cache[1] = "value1"
cache[2] = "value2"

errors = []

def delete_items():
    for i in range(100):
        try:
            if 1 in cache:
                del cache[1]
        except Exception as e:
            errors.append(e)

def set_items():
    for i in range(100):
        cache[1] = f"value_{i}"

threads = [threading.Thread(target=delete_items) for _ in range(3)]
threads.extend([threading.Thread(target=set_items) for _ in range(3)])

for t in threads:
    t.start()
for t in threads:
    t.join()

if errors:
    print(f"Race conditions detected: {errors}")
else:
    print("No race conditions detected (this doesn't mean the bug doesn't exist)")