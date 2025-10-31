from hypothesis import given, strategies as st
from xarray.backends.locks import CombinedLock
import threading

@given(num_locks=st.integers(min_value=1, max_value=5))
def test_combined_lock_locked_returns_correct_state(num_locks):
    """Test that CombinedLock.locked() correctly returns False when no locks are acquired."""
    locks = [threading.Lock() for _ in range(num_locks)]
    combined = CombinedLock(locks)

    # When no locks are acquired, locked() should return False
    assert combined.locked() == False, f"CombinedLock.locked() returned True when no locks are acquired (expected False)"

    # Acquire the first lock
    locks[0].acquire()
    try:
        # When at least one lock is acquired, locked() should return True
        assert combined.locked() == True, f"CombinedLock.locked() returned False when a lock is acquired (expected True)"
    finally:
        locks[0].release()

    # After releasing, locked() should return False again
    assert combined.locked() == False, f"CombinedLock.locked() returned True after releasing all locks (expected False)"

if __name__ == "__main__":
    # Run the test with hypothesis
    test_combined_lock_locked_returns_correct_state()