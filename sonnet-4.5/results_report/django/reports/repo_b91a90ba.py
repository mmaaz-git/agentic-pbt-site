import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

import django
from django.conf import settings
if not settings.configured:
    settings.configure(DEBUG=True, DATABASES={})
    django.setup()

from django.core.cache.backends.locmem import LocMemCache

# Create a cache with MAX_ENTRIES=2 and CULL_FREQUENCY=3
cache = LocMemCache('test', {
    'OPTIONS': {'MAX_ENTRIES': 2, 'CULL_FREQUENCY': 3}
})
cache.clear()

print(f"Configuration: MAX_ENTRIES={cache._max_entries}, CULL_FREQUENCY={cache._cull_frequency}")
print("-" * 60)

# Add 3 items and observe the cache size after each addition
for i in range(3):
    cache.set(f'key_{i}', f'value_{i}')
    cache_size = len(cache._cache)
    print(f"After adding key_{i}: cache size = {cache_size}")
    if cache_size > cache._max_entries:
        print(f"  ⚠️  Cache size ({cache_size}) exceeds MAX_ENTRIES ({cache._max_entries})!")

print("-" * 60)
print(f"Final state:")
print(f"  - Cache has {len(cache._cache)} entries")
print(f"  - MAX_ENTRIES is {cache._max_entries}")
print(f"  - Cache contents: {list(cache._cache.keys())}")

if len(cache._cache) > cache._max_entries:
    print(f"\n❌ BUG CONFIRMED: Cache exceeded MAX_ENTRIES limit!")
else:
    print(f"\n✓ Cache respects MAX_ENTRIES limit")