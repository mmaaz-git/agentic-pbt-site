import threading
import sys
sys.path.insert(0, "/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages")
from xarray.backends.locks import CombinedLock

# Test 1: All locks unlocked
locks1 = [threading.Lock(), threading.Lock(), threading.Lock()]
combined1 = CombinedLock(locks1)
print(f"Test 1 - All unlocked: combined.locked() = {combined1.locked()}, Expected = False")

# Test 2: One lock locked
locks2 = [threading.Lock(), threading.Lock(), threading.Lock()]
locks2[1].acquire()  # Lock the second one
combined2 = CombinedLock(locks2)
print(f"Test 2 - One locked: combined.locked() = {combined2.locked()}, Expected = True")
locks2[1].release()

# Test 3: All locks locked
locks3 = [threading.Lock(), threading.Lock()]
for lock in locks3:
    lock.acquire()
combined3 = CombinedLock(locks3)
print(f"Test 3 - All locked: combined.locked() = {combined3.locked()}, Expected = True")
for lock in locks3:
    lock.release()

# Test 4: No locks (edge case)
combined4 = CombinedLock([])
print(f"Test 4 - No locks: combined.locked() = {combined4.locked()}, Expected = False")