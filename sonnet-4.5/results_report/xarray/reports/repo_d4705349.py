import threading
from xarray.backends.locks import CombinedLock

# Create two regular threading locks
lock1 = threading.Lock()
lock2 = threading.Lock()

# Create a CombinedLock from them
combined = CombinedLock([lock1, lock2])

# Check the status of individual locks (should be False as they're not acquired)
print(f"lock1.locked() = {lock1.locked()}")
print(f"lock2.locked() = {lock2.locked()}")

# Check the combined lock status (should be False but returns True due to bug)
print(f"combined.locked() = {combined.locked()}")

# This assertion should pass but fails due to the bug
assert combined.locked() == False, f"Expected combined.locked() to be False, but got {combined.locked()}"