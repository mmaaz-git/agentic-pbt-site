#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

# Test 1: Run the hypothesis test
print("=" * 60)
print("Running hypothesis test...")
print("=" * 60)

from hypothesis import given, strategies as st, settings
from django.core.cache.backends.locmem import LocMemCache

@given(st.text(min_size=1), st.integers(), st.integers(min_value=-10, max_value=10))
@settings(max_examples=100)
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

# Run the hypothesis test
try:
    test_incr_version_with_delta()
    print("Hypothesis test passed for all examples")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")
    print("Stopping after first failure")

# Test 2: Run the specific failing case
print("\n" + "=" * 60)
print("Running specific failing case with key='0', value=0, delta=0")
print("=" * 60)

cache = LocMemCache("test", {"timeout": 300})
cache.clear()

initial_version = 1
cache.set('0', 0, version=initial_version)

print(f"Initial state - Value at version {initial_version}: {cache.get('0', version=initial_version)}")

new_version = cache.incr_version('0', delta=0, version=initial_version)
print(f"After incr_version with delta=0, returned new_version: {new_version}")

result_new = cache.get('0', version=new_version)
print(f"Value at new version {new_version}: {result_new}")

result_old = cache.get('0', default="MISSING", version=initial_version)
print(f"Value at old version {initial_version}: {result_old}")

print(f"\nExpected at version {new_version}: 0")
print(f"Actual at version {new_version}: {result_new}")

if result_new != 0:
    print("\n*** BUG CONFIRMED: Value is None instead of 0 ***")
else:
    print("\n*** No bug: Value is correctly preserved ***")

# Test 3: Run the reproducing example from the bug report
print("\n" + "=" * 60)
print("Running the bug report's reproducing example")
print("=" * 60)

cache2 = LocMemCache("test", {"timeout": 300})
cache2.clear()

cache2.set("mykey", 42, version=1)
new_version2 = cache2.incr_version("mykey", delta=0, version=1)

result = cache2.get("mykey", version=new_version2)

print(f"Expected: 42")
print(f"Actual: {result}")

if result != 42:
    print("\n*** BUG CONFIRMED: Value is lost when delta=0 ***")
else:
    print("\n*** No bug: Value is correctly preserved ***")

# Test 4: Test with different delta values for comparison
print("\n" + "=" * 60)
print("Testing with different delta values for comparison")
print("=" * 60)

for delta in [-1, 0, 1, 2]:
    cache3 = LocMemCache("test", {"timeout": 300})
    cache3.clear()

    cache3.set("testkey", 100, version=5)
    print(f"\nDelta={delta}:")
    print(f"  Initial: version=5, value={cache3.get('testkey', version=5)}")

    new_v = cache3.incr_version("testkey", delta=delta, version=5)
    print(f"  After incr_version: new_version={new_v}")

    old_val = cache3.get("testkey", default="DELETED", version=5)
    new_val = cache3.get("testkey", default="NOT_FOUND", version=new_v)

    print(f"  Value at old version 5: {old_val}")
    print(f"  Value at new version {new_v}: {new_val}")

    if delta == 0 and new_val != 100:
        print(f"  *** ISSUE: Value should be 100 but is {new_val}")