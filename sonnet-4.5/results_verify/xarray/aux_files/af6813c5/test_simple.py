import threading
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')
from xarray.backends.locks import CombinedLock

lock1 = threading.Lock()
lock2 = threading.Lock()
combined = CombinedLock([lock1, lock2])

print(f"combined.locked() returns: {combined.locked()}")
print(f"Expected: False (no locks are held)")

# Let's also check what lock.locked is
print(f"\nlock1.locked is: {lock1.locked}")
print(f"lock1.locked() is: {lock1.locked()}")

# Now let's acquire a lock and test again
lock1.acquire()
print(f"\nAfter acquiring lock1:")
print(f"lock1.locked() returns: {lock1.locked()}")
print(f"combined.locked() returns: {combined.locked()}")
print(f"Expected combined.locked(): True (lock1 is held)")
lock1.release()