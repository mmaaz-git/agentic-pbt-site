import threading

# Let's manually test the fix
class CombinedLockFixed:
    """A combination of multiple locks - with fix applied."""

    def __init__(self, locks):
        self.locks = tuple(set(locks))  # remove duplicates

    def locked(self):
        # Fixed version with parentheses
        return any(lock.locked() for lock in self.locks)

lock1 = threading.Lock()
lock2 = threading.Lock()
combined = CombinedLockFixed([lock1, lock2])

print(f"Fixed combined.locked() returns: {combined.locked()}")
print(f"Expected: False (no locks are held)")

# Now let's acquire a lock and test again
lock1.acquire()
print(f"\nAfter acquiring lock1:")
print(f"lock1.locked() returns: {lock1.locked()}")
print(f"Fixed combined.locked() returns: {combined.locked()}")
print(f"Expected combined.locked(): True (lock1 is held)")
lock1.release()

print(f"\nAfter releasing lock1:")
print(f"lock1.locked() returns: {lock1.locked()}")
print(f"Fixed combined.locked() returns: {combined.locked()}")
print(f"Expected combined.locked(): False (no locks are held)")