#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.core.cache.backends.locmem import LocMemCache
import itertools

counter = itertools.count()

@settings(max_examples=200)
@given(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=10)
)
def test_max_entries_enforcement(max_entries, extra):
    cache = LocMemCache(f"test_{next(counter)}", {
        "timeout": 300,
        "max_entries": max_entries,
        "cull_frequency": 3
    })

    total_items = max_entries + extra
    for i in range(total_items):
        cache.set(f"key_{i}", i)

    current_size = len(cache._cache)
    assert current_size <= max_entries, \
        f"Cache size {current_size} exceeds max_entries {max_entries}"

# Run the test
print("Running Hypothesis test...")
try:
    test_max_entries_enforcement()
    print("All tests passed!")
except AssertionError as e:
    print(f"Test failed: {e}")
    print("\nThis confirms the bug exists.")
except Exception as e:
    print(f"Unexpected error: {e}")