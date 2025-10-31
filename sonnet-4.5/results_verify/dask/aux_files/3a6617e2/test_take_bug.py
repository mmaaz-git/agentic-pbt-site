#!/usr/bin/env python3
"""Test script to reproduce the dask.bag.Bag.take bug"""

import dask
import dask.bag as db
import warnings

# Use single-threaded scheduler to avoid multiprocessing issues
dask.config.set(scheduler='synchronous')

print("=" * 60)
print("TEST 1: Reproducing the reported bug")
print("=" * 60)

items = [0, 0, 0]
bag = db.from_sequence(items, npartitions=2)

# Capture warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = bag.take(3, compute=True)

    print(f"Items: {items}")
    print(f"Number of partitions: 2")
    print(f"Requested: 3 elements")
    print(f"Got: {result}")
    print(f"Length: {len(result)}")

    if w:
        print(f"Warning message: {w[0].message}")

print("\n" + "=" * 60)
print("TEST 2: Testing with npartitions=-1 (use all partitions)")
print("=" * 60)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result_all = bag.take(3, npartitions=-1, compute=True)

    print(f"Items: {items}")
    print(f"Requested: 3 elements with npartitions=-1")
    print(f"Got: {result_all}")
    print(f"Length: {len(result_all)}")

    if w:
        print(f"Warning message: {w[0].message}")
    else:
        print("No warning raised")

print("\n" + "=" * 60)
print("TEST 3: Testing with npartitions=2")
print("=" * 60)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result_2 = bag.take(3, npartitions=2, compute=True)

    print(f"Items: {items}")
    print(f"Requested: 3 elements with npartitions=2")
    print(f"Got: {result_2}")
    print(f"Length: {len(result_2)}")

    if w:
        print(f"Warning message: {w[0].message}")
    else:
        print("No warning raised")

print("\n" + "=" * 60)
print("TEST 4: Running property-based test from bug report")
print("=" * 60)

from hypothesis import given, strategies as st, settings, assume

@settings(deadline=None, max_examples=50)
@given(
    st.lists(st.integers(), min_size=1, max_size=50),
    st.integers(min_value=1, max_value=10)
)
def test_take_returns_k_elements(items, k):
    assume(k <= len(items))

    bag = db.from_sequence(items, npartitions=2)

    taken = bag.take(k, compute=True)

    assert len(taken) == k, f"Expected {k} elements, got {len(taken)} from {items}"

try:
    test_take_returns_k_elements()
    print("Property test passed!")
except AssertionError as e:
    print(f"Property test failed: {e}")
except Exception as e:
    print(f"Property test error: {e}")

print("\n" + "=" * 60)
print("TEST 5: Exploring partition distribution")
print("=" * 60)

items2 = [0, 1, 2, 3, 4]
bag2 = db.from_sequence(items2, npartitions=2)

# Check what's in each partition
print(f"Items: {items2}")
print(f"Number of partitions: 2")

# Get each partition
for i in range(2):
    partition_contents = bag2.pluck(0, default=None).map_partitions(list).to_delayed()[i].compute()
    print(f"Partition {i}: {partition_contents}")

# Now test take with different npartitions values
print("\nTake with different npartitions:")
for np in [1, 2, -1]:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = bag2.take(4, npartitions=np, compute=True)
        print(f"  npartitions={np}: got {len(result)} elements: {result}")
        if w:
            print(f"    Warning: {w[0].message}")