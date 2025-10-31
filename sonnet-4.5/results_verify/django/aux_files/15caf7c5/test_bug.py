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

from django.core.cache.backends.locmem import LocMemCache
import time

print("=== Testing _delete() method behavior ===")
print()

cache = LocMemCache('test', {})
cache_key = ':1:testkey'

# Set up the inconsistent state: key in _expire_info but not in _cache
cache._expire_info[cache_key] = time.time() + 100

print(f"Initial state:")
print(f"  key in _cache = {cache_key in cache._cache}")
print(f"  key in _expire_info = {cache_key in cache._expire_info}")
print()

print("Calling _delete(cache_key)...")
result = cache._delete(cache_key)
print(f"  _delete() returned: {result}")
print()

print(f"After _delete():")
print(f"  key in _cache = {cache_key in cache._cache}")
print(f"  key in _expire_info = {cache_key in cache._expire_info}")
print()

print("Expected: key should not be in _expire_info")
print(f"Actual: key {'IS' if cache_key in cache._expire_info else 'is NOT'} in _expire_info")
print()

if cache_key in cache._expire_info:
    print("BUG CONFIRMED: _expire_info was not cleaned up!")
else:
    print("No bug: _expire_info was properly cleaned up")