import threading
from xarray.backends.lru_cache import LRUCache
import time
import random

def test_aggressive_race():
    cache = LRUCache(maxsize=100)
    # Populate cache with many keys
    for i in range(50):
        cache[i] = f"value_{i}"

    errors = []
    race_detected = False

    def delete_random():
        for _ in range(1000):
            key = random.randint(0, 49)
            try:
                if key in cache:
                    del cache[key]
            except KeyError:
                pass  # Expected if key was already deleted
            except Exception as e:
                errors.append(('delete', e))

    def set_random():
        for _ in range(1000):
            key = random.randint(0, 49)
            cache[key] = f"new_value_{random.random()}"

    def get_random():
        for _ in range(1000):
            key = random.randint(0, 49)
            try:
                _ = cache[key]
            except KeyError:
                pass  # Expected if key was deleted
            except Exception as e:
                errors.append(('get', e))

    def iterate_cache():
        for _ in range(100):
            try:
                for k in cache:
                    pass
            except Exception as e:
                errors.append(('iterate', e))

    # Create many threads to increase contention
    threads = []
    threads.extend([threading.Thread(target=delete_random) for _ in range(5)])
    threads.extend([threading.Thread(target=set_random) for _ in range(5)])
    threads.extend([threading.Thread(target=get_random) for _ in range(5)])
    threads.extend([threading.Thread(target=iterate_cache) for _ in range(2)])

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        print(f"Errors detected: {errors[:5]}...")  # Show first 5 errors
        return True
    return False

print("Running aggressive race condition test...")
race_found = False
for run in range(5):
    print(f"Attempt {run + 1}...")
    if test_aggressive_race():
        race_found = True
        print("Race condition detected!")
        break

if not race_found:
    print("No race conditions detected, but the missing lock is still a bug.")