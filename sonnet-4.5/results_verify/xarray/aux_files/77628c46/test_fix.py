import threading

# Simulate the fixed version
def fixed_locked(locks):
    """This is what the fixed CombinedLock.locked() would do"""
    return any(lock.locked() for lock in locks)

# Test the fix
print("Testing the proposed fix...")

# Test 1: No locks acquired
lock1 = threading.Lock()
lock2 = threading.Lock()
locks = [lock1, lock2]

print("\n1. When no locks are acquired:")
result = fixed_locked(locks)
print(f"   fixed_locked() returns: {result}")
print(f"   Expected: False")
print(f"   Correct: {result == False}")

# Test 2: First lock acquired
lock1.acquire()
print("\n2. When first lock is acquired:")
result = fixed_locked(locks)
print(f"   fixed_locked() returns: {result}")
print(f"   Expected: True")
print(f"   Correct: {result == True}")
lock1.release()

# Test 3: Second lock acquired
lock2.acquire()
print("\n3. When second lock is acquired:")
result = fixed_locked(locks)
print(f"   fixed_locked() returns: {result}")
print(f"   Expected: True")
print(f"   Correct: {result == True}")
lock2.release()

# Test 4: Both locks acquired
lock1.acquire()
lock2.acquire()
print("\n4. When both locks are acquired:")
result = fixed_locked(locks)
print(f"   fixed_locked() returns: {result}")
print(f"   Expected: True")
print(f"   Correct: {result == True}")
lock1.release()
lock2.release()

print("\nAll tests pass with the proposed fix!")