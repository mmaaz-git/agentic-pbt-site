import threading
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')
from hypothesis import given, strategies as st
from xarray.backends.locks import CombinedLock


@given(st.integers(min_value=1, max_value=10))
def test_combined_lock_locked_false_when_no_locks_held(num_locks):
    locks = [threading.Lock() for _ in range(num_locks)]
    combined = CombinedLock(locks)

    assert combined.locked() == False, (
        "CombinedLock.locked() should return False when no locks are held"
    )

if __name__ == "__main__":
    test_combined_lock_locked_false_when_no_locks_held()
