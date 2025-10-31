import threading
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')
from xarray.backends.locks import CombinedLock

# Test with no locks
combined_empty = CombinedLock([])
print(f"Empty CombinedLock.locked(): {combined_empty.locked()}")

# Test with one unlocked lock
lock1 = threading.Lock()
combined1 = CombinedLock([lock1])
print(f"\nOne unlocked lock:")
print(f"  lock.locked(): {lock1.locked()}")
print(f"  CombinedLock.locked(): {combined1.locked()}")

# Test what lock.locked actually is (without parentheses)
print(f"\nWhat is lock.locked (no parentheses)? {lock1.locked}")
print(f"  Type: {type(lock1.locked)}")
print(f"  Truthy? {bool(lock1.locked)}")

# Test with one locked lock
lock2 = threading.Lock()
lock2.acquire()
combined2 = CombinedLock([lock2])
print(f"\nOne locked lock:")
print(f"  lock.locked(): {lock2.locked()}")
print(f"  CombinedLock.locked(): {combined2.locked()}")
lock2.release()

# Test with multiple locks (one locked)
lock3 = threading.Lock()
lock4 = threading.Lock()
lock3.acquire()
combined3 = CombinedLock([lock3, lock4])
print(f"\nTwo locks, first locked:")
print(f"  lock3.locked(): {lock3.locked()}")
print(f"  lock4.locked(): {lock4.locked()}")
print(f"  CombinedLock.locked(): {combined3.locked()}")
lock3.release()

# Examine the actual implementation
print("\n\nActual implementation in CombinedLock.locked():")
print("  return any(lock.locked for lock in self.locks)")
print("\nThis iterates and checks 'lock.locked' (method object) not 'lock.locked()' (method call)")
print("Since method objects are always truthy, any() with non-empty list returns True")