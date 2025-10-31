import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import threading
from xarray.backends.locks import CombinedLock

locks = [threading.Lock(), threading.Lock()]
combined = CombinedLock(locks)

print("All locks unlocked")
print(f"combined.locked() = {combined.locked()}")

result = combined.locked()
print(f"Expected: False (or a boolean)")
print(f"Actual type: {type(result)}")
print(f"Actual value (as bool): {bool(result)}")

# Additional test: lock one of the locks
print("\n--- Testing with one lock acquired ---")
locks[0].acquire()
try:
    print(f"Lock 0 acquired")
    print(f"combined.locked() = {combined.locked()}")
    print(f"locks[0].locked() = {locks[0].locked()}")
    print(f"locks[1].locked() = {locks[1].locked()}")
finally:
    locks[0].release()