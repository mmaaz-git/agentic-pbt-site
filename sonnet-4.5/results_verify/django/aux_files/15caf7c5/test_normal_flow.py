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

print("=== Testing normal cache operations ===")
print()

cache = LocMemCache('test_normal', {})

# Test 1: Normal set and delete
print("Test 1: Normal set and delete")
cache.set('key1', 'value1')
print(f"  After set: key in _cache={':1:key1' in cache._cache}, key in _expire_info={':1:key1' in cache._expire_info}")
cache.delete('key1')
print(f"  After delete: key in _cache={':1:key1' in cache._cache}, key in _expire_info={':1:key1' in cache._expire_info}")
print()

# Test 2: _cull operation
print("Test 2: _cull() operation (lines 92-100)")
cache.set('key2', 'value2')
cache.set('key3', 'value3')
print(f"  Before _cull: _cache keys={list(cache._cache.keys())}, _expire_info keys={list(cache._expire_info.keys())}")
cache._cull_frequency = 2  # Make it cull half the items
cache._cull()
print(f"  After _cull: _cache keys={list(cache._cache.keys())}, _expire_info keys={list(cache._expire_info.keys())}")
print("  Note: _cull() properly maintains consistency (line 100: del self._expire_info[key])")
print()

# Test 3: Expired key handling in various methods
print("Test 3: How other methods handle expired keys")
cache.clear()
cache.set('key4', 'value4', timeout=0.001)  # Set with very short timeout
time.sleep(0.002)  # Wait for it to expire
print(f"  After expiry: key in _cache={':1:key4' in cache._cache}, key in _expire_info={':1:key4' in cache._expire_info}")

# get() method calls _delete on expired keys (line 39)
result = cache.get('key4')
print(f"  After get(): key in _cache={':1:key4' in cache._cache}, key in _expire_info={':1:key4' in cache._expire_info}")
print()

# Test 4: Check if _delete is ever called with inconsistent state in normal operations
print("Test 4: Can inconsistent state happen in normal operations?")
print("  All public methods use locks (lines 29, 37, 55, 60, 68, 82, 112, 116)")
print("  All methods that call _delete() first check _has_expired() which requires key in _expire_info")
print("  Methods calling _delete(): get (line 39), incr (line 70), has_key (line 84)")
print("  These all check _has_expired first, which returns False if key not in _expire_info")
print()

print("Conclusion: In normal operations, _cache and _expire_info stay synchronized")
print("The bug scenario (key in _expire_info but not in _cache) shouldn't occur naturally")