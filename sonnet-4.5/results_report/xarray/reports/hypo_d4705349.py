from hypothesis import given, strategies as st
import threading
from xarray.backends.locks import CombinedLock

@given(st.lists(st.booleans(), min_size=1, max_size=5))
def test_combined_lock_locked_reflects_actual_state(lock_states):
    locks = [threading.Lock() for _ in lock_states]

    for lock, should_lock in zip(locks, lock_states):
        if should_lock:
            lock.acquire()

    combined = CombinedLock(locks)
    result = combined.locked()

    for lock, should_lock in zip(locks, lock_states):
        if should_lock:
            lock.release()

    assert isinstance(result, bool), f"locked() returned {type(result)} instead of bool"
    assert result == any(lock_states), f"locked() returned {result} but expected {any(lock_states)}"

if __name__ == "__main__":
    test_combined_lock_locked_reflects_actual_state()