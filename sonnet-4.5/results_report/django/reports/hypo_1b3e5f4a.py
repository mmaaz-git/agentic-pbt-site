import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, settings, strategies as st
from django.core.cache.backends.locmem import LocMemCache


def make_cache(max_entries=300, cull_frequency=3):
    return LocMemCache('test_edge', {
        'TIMEOUT': 300,
        'OPTIONS': {
            'MAX_ENTRIES': max_entries,
            'CULL_FREQUENCY': cull_frequency,
        },
        'KEY_PREFIX': 'test',
        'VERSION': 1,
    })


@given(st.integers(min_value=1, max_value=30), st.integers(min_value=1, max_value=10))
@settings(max_examples=300)
def test_exact_max_entries_boundary(max_entries, num_additional):
    cache = make_cache(max_entries=max_entries, cull_frequency=3)
    cache.clear()

    for i in range(max_entries + num_additional):
        cache.set(f"key_{i:04d}", i)

    final_size = len(cache._cache)
    assert final_size <= max_entries, \
        f"Cache size {final_size} exceeds max_entries {max_entries}"


if __name__ == "__main__":
    test_exact_max_entries_boundary()