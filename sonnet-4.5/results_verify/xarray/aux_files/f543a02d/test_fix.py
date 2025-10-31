import threading

# Simulate the buggy version
def buggy_locked(locks):
    return any(lock.locked for lock in locks)

# Simulate the fixed version
def fixed_locked(locks):
    return any(lock.locked() for lock in locks)

# Test
locks = [threading.Lock(), threading.Lock()]
print(f"Buggy version with unlocked locks: {buggy_locked(locks)}")
print(f"Fixed version with unlocked locks: {fixed_locked(locks)}")

# Lock one
locks[0].acquire()
print(f"Buggy version with one locked: {buggy_locked(locks)}")
print(f"Fixed version with one locked: {fixed_locked(locks)}")
locks[0].release()