from hypothesis import given, strategies as st, settings
import threading
from xarray.backends.lru_cache import LRUCache

@given(
    keys=st.lists(st.integers(min_value=0, max_value=100), min_size=10, max_size=20)
)
@settings(max_examples=10, deadline=None)
def test_lru_cache_delitem_thread_safety(keys):
    cache = LRUCache(maxsize=50)
    for k in keys:
        cache[k] = f"value_{k}"

    errors = []
    def delete_keys():
        for k in keys:
            try:
                if k in cache:
                    del cache[k]
            except Exception as e:
                errors.append(e)

    def set_keys():
        for k in keys:
            cache[k] = f"new_value_{k}"

    threads = [threading.Thread(target=delete_keys) for _ in range(5)]
    threads.extend([threading.Thread(target=set_keys) for _ in range(5)])

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Thread-safety violations: {errors}"

if __name__ == "__main__":
    test_lru_cache_delitem_thread_safety()
    print("Test completed")