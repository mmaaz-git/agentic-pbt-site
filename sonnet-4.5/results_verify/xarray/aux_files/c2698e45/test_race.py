import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import threading
import time
from xarray.backends.lru_cache import LRUCache

# Test to demonstrate the race condition
def stress_test():
    cache = LRUCache(maxsize=100)
    errors = []

    # Pre-populate cache
    for i in range(50):
        cache[i] = f"value_{i}"

    def writer_thread():
        for _ in range(1000):
            for i in range(50):
                try:
                    cache[i] = f"new_value_{i}"
                except Exception as e:
                    errors.append(('write', e))

    def deleter_thread():
        for _ in range(1000):
            for i in range(50):
                try:
                    del cache[i]
                except KeyError:
                    # Expected when key doesn't exist
                    pass
                except Exception as e:
                    errors.append(('delete', e))
                # Re-add for next iteration
                try:
                    cache[i] = f"value_{i}"
                except Exception as e:
                    errors.append(('re-add', e))

    def reader_thread():
        for _ in range(1000):
            for i in range(50):
                try:
                    _ = cache.get(i)
                except Exception as e:
                    errors.append(('read', e))

    threads = []
    # Create multiple threads of each type
    for _ in range(3):
        threads.append(threading.Thread(target=writer_thread))
        threads.append(threading.Thread(target=deleter_thread))
        threads.append(threading.Thread(target=reader_thread))

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    return errors

print("Running stress test to detect race conditions...")
errors = stress_test()
if errors:
    print(f"Found {len(errors)} errors during stress test:")
    for op, err in errors[:5]:  # Show first 5 errors
        print(f"  Operation: {op}, Error: {err}")
else:
    print("No errors detected (race condition may still exist but not triggered)")