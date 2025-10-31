import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import threading

# Manually create a fixed version of CombinedLock
class FixedCombinedLock:
    """A combination of multiple locks - FIXED VERSION"""

    def __init__(self, locks):
        self.locks = tuple(set(locks))  # remove duplicates

    def locked(self):
        return any(lock.locked() for lock in self.locks)  # FIXED: Added ()

# Test the fixed version
print("Testing FIXED CombinedLock:")
locks = [threading.Lock(), threading.Lock()]
combined = FixedCombinedLock(locks)

print(f"All locks unlocked: combined.locked() = {combined.locked()}")
assert combined.locked() == False, "Should be False when all locks are unlocked"

# Lock one
locks[0].acquire()
print(f"One lock acquired: combined.locked() = {combined.locked()}")
assert combined.locked() == True, "Should be True when one lock is acquired"
locks[0].release()

# Lock both
locks[0].acquire()
locks[1].acquire()
print(f"Both locks acquired: combined.locked() = {combined.locked()}")
assert combined.locked() == True, "Should be True when both locks are acquired"
locks[0].release()
locks[1].release()

print("\nAll tests passed! The fix works correctly.")