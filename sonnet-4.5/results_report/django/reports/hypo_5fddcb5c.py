#!/usr/bin/env python3
"""
Hypothesis-based property test for Django cache incr_version method.

This test verifies that incr_version correctly preserves the cached value
at the new version for all valid delta values, including delta=0.
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st
from django.core.cache.backends.locmem import LocMemCache

@given(st.text(min_size=1), st.integers(), st.integers(min_value=-10, max_value=10))
def test_incr_version_with_delta(key, value, delta):
    cache = LocMemCache("test", {"timeout": 300})
    cache.clear()

    initial_version = 1
    cache.set(key, value, version=initial_version)

    new_version = cache.incr_version(key, delta=delta, version=initial_version)

    assert new_version == initial_version + delta

    result_new = cache.get(key, version=new_version)
    assert result_new == value, f"New version: Expected {value}, got {result_new}"

    result_old = cache.get(key, default="MISSING", version=initial_version)
    assert result_old == "MISSING", f"Old version should be deleted, got {result_old}"

if __name__ == "__main__":
    # Run the test
    test_incr_version_with_delta()