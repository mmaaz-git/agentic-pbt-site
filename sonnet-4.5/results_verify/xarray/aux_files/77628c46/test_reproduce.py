import threading
from xarray.backends.locks import CombinedLock

print("Testing CombinedLock.locked() behavior...")

# Test 1: Basic test from the bug report
lock1 = threading.Lock()
lock2 = threading.Lock()
combined = CombinedLock([lock1, lock2])

print("\n1. Testing when no locks are acquired:")
print(f"lock1.locked(): {lock1.locked()}")
print(f"lock2.locked(): {lock2.locked()}")
print(f"combined.locked(): {combined.locked()}")

print("\n2. Testing when lock1 is acquired:")
lock1.acquire()
print(f"lock1.locked(): {lock1.locked()}")
print(f"lock2.locked(): {lock2.locked()}")
print(f"combined.locked() should be True: {combined.locked()}")
lock1.release()

print("\n3. Testing when lock2 is acquired:")
lock2.acquire()
print(f"lock1.locked(): {lock1.locked()}")
print(f"lock2.locked(): {lock2.locked()}")
print(f"combined.locked() should be True: {combined.locked()}")
lock2.release()

# Test what the buggy code actually returns
print("\n4. Testing what the current implementation actually does:")
lock3 = threading.Lock()
lock3.acquire()
print(f"lock3.locked (without parentheses) is: {lock3.locked}")
print(f"Type of lock3.locked: {type(lock3.locked)}")
print(f"bool(lock3.locked): {bool(lock3.locked)}")
lock3.release()

# Test to demonstrate the bug more clearly
print("\n5. Testing the actual bug - comparing correct vs incorrect behavior:")
locks = [threading.Lock(), threading.Lock()]
combined = CombinedLock(locks)

print(f"When no locks acquired:")
print(f"  Expected (using locked()): {any(lock.locked() for lock in locks)}")
print(f"  Actual (using locked): {any(lock.locked for lock in locks)}")

locks[0].acquire()
print(f"\nWhen first lock acquired:")
print(f"  Expected (using locked()): {any(lock.locked() for lock in locks)}")
print(f"  Actual (using locked): {any(lock.locked for lock in locks)}")
locks[0].release()