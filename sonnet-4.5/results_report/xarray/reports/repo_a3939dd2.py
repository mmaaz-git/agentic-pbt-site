import threading
from xarray.backends.locks import CombinedLock

# Create two threading locks
lock1 = threading.Lock()
lock2 = threading.Lock()

# Create a CombinedLock with these locks
combined = CombinedLock([lock1, lock2])

# Test 1: Check if CombinedLock reports locked when no locks are acquired
print("Test 1: No locks acquired")
print(f"combined.locked() returns: {combined.locked()}")
print(f"Expected: False")
print()

# Test 2: Acquire one lock and check
lock1.acquire()
print("Test 2: lock1 acquired")
print(f"combined.locked() returns: {combined.locked()}")
print(f"Expected: True")
lock1.release()
print()

# Test 3: Show the actual bug - accessing lock.locked without parentheses
print("Test 3: Demonstrating the bug")
print(f"lock1.locked (without parentheses): {lock1.locked}")
print(f"lock1.locked() (with parentheses): {lock1.locked()}")
print()

# Test 4: Show what the bug actually evaluates
print("Test 4: What the buggy code evaluates")
print(f"any([lock1.locked, lock2.locked]): {any([lock1.locked, lock2.locked])}")
print(f"any([lock1.locked(), lock2.locked()]): {any([lock1.locked(), lock2.locked()])}")