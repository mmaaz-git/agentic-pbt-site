import threading
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')
from xarray.backends.locks import CombinedLock

lock1 = threading.Lock()
lock2 = threading.Lock()

combined = CombinedLock([lock1, lock2])

print(f"lock1.locked() = {lock1.locked()}")
print(f"lock2.locked() = {lock2.locked()}")
print(f"combined.locked() = {combined.locked()}")

print(f"\nExpected combined.locked() to be False, got {combined.locked()}")
assert combined.locked() == False, f"Bug confirmed: combined.locked() returned True when no locks are acquired"