import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

import django
from django.conf import settings
if not settings.configured:
    settings.configure(DEBUG=True, DATABASES={})
    django.setup()

from hypothesis import given, strategies as st, settings as hyp_settings
from django.core.cache.backends.locmem import LocMemCache


@hyp_settings(max_examples=200)
@given(
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=2, max_value=10)
)
def test_max_entries_never_exceeded(max_entries, cull_frequency):
    cache = LocMemCache(f'test_{max_entries}_{cull_frequency}', {
        'OPTIONS': {'MAX_ENTRIES': max_entries, 'CULL_FREQUENCY': cull_frequency}
    })
    cache.clear()

    num_to_add = max_entries * 2
    for i in range(num_to_add):
        cache.set(f'key_{i}', i)
        cache_size = len(cache._cache)
        assert cache_size <= max_entries, f"Cache size {cache_size} exceeds MAX_ENTRIES {max_entries}"


if __name__ == "__main__":
    print("Running Hypothesis test for LocMemCache MAX_ENTRIES constraint...")
    print("-" * 60)
    test_max_entries_never_exceeded()
    print("✓ All tests passed!")