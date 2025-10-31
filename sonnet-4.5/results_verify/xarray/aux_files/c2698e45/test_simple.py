import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import threading
from xarray.backends.lru_cache import LRUCache

cache = LRUCache(maxsize=10)
cache[1] = "test"

def delete_item():
    try:
        del cache[1]
    except KeyError:
        pass

threads = [threading.Thread(target=delete_item) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print("Test completed - may have race condition")