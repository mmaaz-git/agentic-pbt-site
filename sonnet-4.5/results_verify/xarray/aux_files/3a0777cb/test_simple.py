import threading
from xarray.backends.lru_cache import LRUCache
import time

def test_race_condition():
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
        return True
    else:
        print("No errors detected in this run")
        return False

# Run multiple times to increase chance of catching race condition
race_detected = False
for i in range(10):
    print(f"\nRun {i+1}:")
    if test_race_condition():
        race_detected = True
        break

if not race_detected:
    print("\nNo race conditions detected in 10 runs, but this doesn't mean the code is thread-safe.")