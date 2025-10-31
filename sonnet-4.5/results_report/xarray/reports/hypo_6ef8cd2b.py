import threading
from hypothesis import given, strategies as st, settings
from xarray.backends.locks import CombinedLock

@given(st.lists(st.booleans(), min_size=1, max_size=10))
@settings(max_examples=200)
def test_combined_lock_locked_state(lock_states):
    locks = []
    for locked in lock_states:
        lock = threading.Lock()
        if locked:
            lock.acquire()
        locks.append(lock)

    combined = CombinedLock(locks)

    expected_locked = any(lock_states)
    actual_locked = combined.locked()

    for i, lock in enumerate(locks):
        if lock_states[i]:
            lock.release()

    assert actual_locked == expected_locked

# Run the test
test_combined_lock_locked_state()