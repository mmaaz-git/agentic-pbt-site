from hypothesis import given, settings, strategies as st
import threading
from xarray.backends.locks import CombinedLock

@settings(max_examples=200)
@given(locks_count=st.integers(min_value=0, max_value=10))
def test_combined_lock_locked_property(locks_count):
    locks = [threading.Lock() for _ in range(locks_count)]
    combined = CombinedLock(locks)

    assert not combined.locked()

    if locks:
        locks[0].acquire()
        assert combined.locked()
        locks[0].release()

if __name__ == "__main__":
    test_combined_lock_locked_property()