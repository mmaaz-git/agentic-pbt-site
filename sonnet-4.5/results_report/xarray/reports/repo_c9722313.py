import threading
from xarray.backends.locks import CombinedLock

# Create a single threading lock
lock = threading.Lock()

# Create a CombinedLock with that single lock
combined = CombinedLock([lock])

# Check if the individual lock is locked (should be False)
print(f"Individual lock is locked: {lock.locked()}")

# Check if CombinedLock reports being locked (should be False but returns True)
print(f"CombinedLock.locked(): {combined.locked()}")

print("\nNow acquiring the lock...")
lock.acquire()
print(f"Individual lock is locked after acquire: {lock.locked()}")
print(f"CombinedLock.locked() after acquire: {combined.locked()}")

lock.release()
print("\nAfter releasing the lock...")
print(f"Individual lock is locked after release: {lock.locked()}")
print(f"CombinedLock.locked() after release: {combined.locked()}")

# Test with empty CombinedLock
print("\nTesting with empty CombinedLock:")
empty_combined = CombinedLock([])
print(f"Empty CombinedLock.locked(): {empty_combined.locked()}")