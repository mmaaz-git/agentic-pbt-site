from hypothesis import given, strategies as st
from xarray.backends.locks import CombinedLock
import threading

@given(num_locks=st.integers(min_value=1, max_value=5))
def test_combined_lock_locked_property(num_locks):
    locks = [threading.Lock() for _ in range(num_locks)]
    combined = CombinedLock(locks)

    result = combined.locked()
    assert isinstance(result, bool), "locked() should return a boolean"

# Run the test
if __name__ == "__main__":
    try:
        test_combined_lock_locked_property()
        print("Test passed (which is unexpected if there's a bug)")
    except Exception as e:
        print(f"Test failed with error: {e}")