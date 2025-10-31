import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import threading
from xarray.backends.locks import CombinedLock

# Test with unlocked locks
locks = [threading.Lock(), threading.Lock()]
combined = CombinedLock(locks)

print("Testing CombinedLock.locked() with all locks unlocked:")
print(f"  Lock 1 is locked: {locks[0].locked()}")
print(f"  Lock 2 is locked: {locks[1].locked()}")
print(f"  combined.locked() returns: {combined.locked()}")
print(f"  Expected: False")
print(f"  Actual type returned: {type(combined.locked())}")
print()

# Test showing the bug: it returns a method object when it shouldn't
print("Debugging the issue:")
print(f"  locks[0].locked (without parentheses): {locks[0].locked}")
print(f"  type(locks[0].locked): {type(locks[0].locked)}")
print(f"  bool(locks[0].locked): {bool(locks[0].locked)}")
print()

# Test with one lock locked
print("Testing CombinedLock.locked() with one lock locked:")
locks[0].acquire()
print(f"  Lock 1 is locked: {locks[0].locked()}")
print(f"  Lock 2 is locked: {locks[1].locked()}")
print(f"  combined.locked() returns: {combined.locked()}")
print(f"  Expected: True")
locks[0].release()