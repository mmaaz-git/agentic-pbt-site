import threading
from xarray.backends.locks import CombinedLock

# Create a simple threading lock
lock = threading.Lock()

# Create a CombinedLock with the single lock
combined = CombinedLock([lock])

# Check if the individual lock is locked (should be False since we haven't acquired it)
print(f"lock.locked() = {lock.locked()}")

# Check if the CombinedLock is locked (should also be False, but will be True due to bug)
print(f"combined.locked() = {combined.locked()}")

# Now let's test with the lock actually acquired
print("\n--- After acquiring the lock ---")
lock.acquire()
print(f"lock.locked() = {lock.locked()}")
print(f"combined.locked() = {combined.locked()}")
lock.release()

# Test with multiple locks, none acquired
print("\n--- Multiple locks, none acquired ---")
lock1 = threading.Lock()
lock2 = threading.Lock()
lock3 = threading.Lock()
combined_multi = CombinedLock([lock1, lock2, lock3])
print(f"lock1.locked() = {lock1.locked()}")
print(f"lock2.locked() = {lock2.locked()}")
print(f"lock3.locked() = {lock3.locked()}")
print(f"combined_multi.locked() = {combined_multi.locked()}")

# Test with one lock acquired
print("\n--- Multiple locks, one acquired ---")
lock2.acquire()
print(f"lock1.locked() = {lock1.locked()}")
print(f"lock2.locked() = {lock2.locked()}")
print(f"lock3.locked() = {lock3.locked()}")
print(f"combined_multi.locked() = {combined_multi.locked()}")
lock2.release()