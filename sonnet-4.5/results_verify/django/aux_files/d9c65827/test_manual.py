import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.core.cache.backends.locmem import LocMemCache

cache = LocMemCache('reproduce_bug', {
    'TIMEOUT': 300,
    'OPTIONS': {
        'MAX_ENTRIES': 1,
        'CULL_FREQUENCY': 3,
    },
    'KEY_PREFIX': 'test',
    'VERSION': 1,
})
cache.clear()

cache.set("key1", "value1")
print(f"After setting key1: cache size = {len(cache._cache)}")

cache.set("key2", "value2")
print(f"After setting key2: cache size = {len(cache._cache)}")

assert len(cache._cache) <= cache._max_entries, \
    f"BUG: Cache has {len(cache._cache)} entries, exceeding max_entries={cache._max_entries}"