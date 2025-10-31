import threading

class FixedCombinedLock:
    """Test the fixed version"""
    def __init__(self, locks):
        self.locks = tuple(set(locks))  # remove duplicates

    def locked(self):
        # Fixed version - calls the method with ()
        return any(lock.locked() for lock in self.locks)

# Test the fixed version
print("Testing the fixed version:\n")

# Test with no locks
combined_empty = FixedCombinedLock([])
print(f"Empty FixedCombinedLock.locked(): {combined_empty.locked()}")

# Test with one unlocked lock
lock1 = threading.Lock()
combined1 = FixedCombinedLock([lock1])
print(f"\nOne unlocked lock:")
print(f"  lock.locked(): {lock1.locked()}")
print(f"  FixedCombinedLock.locked(): {combined1.locked()}")
assert combined1.locked() == False, "Should be False when no locks are locked"

# Test with one locked lock
lock2 = threading.Lock()
lock2.acquire()
combined2 = FixedCombinedLock([lock2])
print(f"\nOne locked lock:")
print(f"  lock.locked(): {lock2.locked()}")
print(f"  FixedCombinedLock.locked(): {combined2.locked()}")
assert combined2.locked() == True, "Should be True when a lock is locked"
lock2.release()

# Test with multiple locks (one locked)
lock3 = threading.Lock()
lock4 = threading.Lock()
lock3.acquire()
combined3 = FixedCombinedLock([lock3, lock4])
print(f"\nTwo locks, first locked:")
print(f"  lock3.locked(): {lock3.locked()}")
print(f"  lock4.locked(): {lock4.locked()}")
print(f"  FixedCombinedLock.locked(): {combined3.locked()}")
assert combined3.locked() == True, "Should be True when any lock is locked"
lock3.release()

# Test with multiple locks (none locked)
lock5 = threading.Lock()
lock6 = threading.Lock()
combined4 = FixedCombinedLock([lock5, lock6])
print(f"\nTwo locks, none locked:")
print(f"  lock5.locked(): {lock5.locked()}")
print(f"  lock6.locked(): {lock6.locked()}")
print(f"  FixedCombinedLock.locked(): {combined4.locked()}")
assert combined4.locked() == False, "Should be False when no locks are locked"

print("\nAll tests pass with the fixed version!")