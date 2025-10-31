import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import threading
from hypothesis import given, strategies as st
from xarray.backends.lru_cache import LRUCache

@given(st.integers(min_value=10, max_value=100))
def test_lrucache_delitem_thread_safe(n_threads):
    cache = LRUCache(maxsize=100)

    for i in range(10):
        cache[i] = f"value_{i}"

    errors = []

    def delete_all():
        for i in range(10):
            try:
                del cache[i]
            except KeyError:
                pass
            except Exception as e:
                errors.append(e)

    threads = [threading.Thread(target=delete_all) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread-safety violated: {errors}"

if __name__ == "__main__":
    test_lrucache_delitem_thread_safe()