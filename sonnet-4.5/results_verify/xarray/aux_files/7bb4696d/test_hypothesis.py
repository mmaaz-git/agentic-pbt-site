import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import threading
from hypothesis import given, strategies as st
from xarray.backends.locks import CombinedLock

@given(st.integers(min_value=1, max_value=10))
def test_combinedlock_locked_when_all_unlocked(n_locks):
    locks = [threading.Lock() for _ in range(n_locks)]
    combined = CombinedLock(locks)

    assert not combined.locked(), \
        f"CombinedLock should return False when all constituent locks are unlocked, but returned {combined.locked()}"

@given(st.integers(min_value=1, max_value=10))
def test_combinedlock_locked_when_one_locked(n_locks):
    locks = [threading.Lock() for _ in range(n_locks)]
    combined = CombinedLock(locks)

    locks[0].acquire()
    try:
        assert combined.locked(), \
            "CombinedLock should return True when any constituent lock is locked"
    finally:
        locks[0].release()

# Run tests
print("Testing with all locks unlocked...")
test_combinedlock_locked_when_all_unlocked()

print("Testing with one lock locked...")
test_combinedlock_locked_when_one_locked()