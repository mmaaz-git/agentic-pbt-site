import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

import django
from django.conf import settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        DATABASES={},
        CACHES={
            'default': {
                'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
                'LOCATION': 'test',
            }
        }
    )
    django.setup()

from hypothesis import given, strategies as st, settings as hypo_settings
from django.core.cache.backends.locmem import LocMemCache
import time

@given(st.text(min_size=1, max_size=100))
@hypo_settings(max_examples=10)
def test_delete_maintains_cache_expire_info_consistency(key):
    cache = LocMemCache('test', {})
    cache.clear()

    cache_key = cache.make_key(key, version=1)

    # Create inconsistent state: key only in _expire_info
    cache._expire_info[cache_key] = time.time() + 100

    cache._delete(cache_key)

    # Property: After _delete(), key should not be in either dictionary
    assert cache_key not in cache._cache, f"Key {cache_key} should not be in _cache"
    assert cache_key not in cache._expire_info, f"Key {cache_key} should not be in _expire_info"

print("Running Hypothesis property-based test...")
try:
    test_delete_maintains_cache_expire_info_consistency()
    print("All tests passed!")
except AssertionError as e:
    print(f"Test failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")