import threading
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

# Let's verify the root cause: lock.locked is a method reference, not a call
lock = threading.Lock()

print(f"lock.locked = {lock.locked}")
print(f"lock.locked() = {lock.locked()}")
print(f"bool(lock.locked) = {bool(lock.locked)}")  # Method objects are truthy
print(f"bool(lock.locked()) = {bool(lock.locked())}")

print("\n--- Testing in a generator expression ---")
locks = [threading.Lock(), threading.Lock()]
print(f"Generator with method reference: any(lock.locked for lock in locks) = {any(lock.locked for lock in locks)}")
print(f"Generator with method call: any(lock.locked() for lock in locks) = {any(lock.locked() for lock in locks)}")

print("\n--- What gets evaluated ---")
for lock in locks:
    print(f"  lock.locked = {lock.locked} (truthy: {bool(lock.locked)})")
    print(f"  lock.locked() = {lock.locked()} (truthy: {bool(lock.locked())})")