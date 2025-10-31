import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import threading
from xarray.backends.locks import CombinedLock

# Test what lock.locked actually returns when not called
lock = threading.Lock()
print(f"type(lock.locked): {type(lock.locked)}")
print(f"lock.locked: {lock.locked}")
print(f"bool(lock.locked): {bool(lock.locked)}")  # Method objects are truthy
print(f"lock.locked(): {lock.locked()}")  # Actual call returns False

# Test what happens in CombinedLock
locks = [threading.Lock()]
combined = CombinedLock(locks)

# Debug what's happening in the any() call
print("\n--- Debugging any() behavior ---")
print(f"List comprehension: [lock.locked for lock in combined.locks] = {[lock.locked for lock in combined.locks]}")
print(f"Are these truthy? {[bool(lock.locked) for lock in combined.locks]}")
print(f"Result of any(): {any(lock.locked for lock in combined.locks)}")
print(f"Should be (with ()): {any(lock.locked() for lock in combined.locks)}")