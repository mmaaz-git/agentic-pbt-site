#!/usr/bin/env python3
"""Test file to reproduce the dask.bag.from_sequence bug."""

from hypothesis import given, strategies as st, settings
import dask.bag as db


print("=" * 60)
print("TEST 1: Hypothesis Property-Based Test")
print("=" * 60)

@settings(max_examples=100)
@given(st.lists(st.integers(), min_size=1, max_size=20),
       st.integers(min_value=1, max_value=10))
def test_from_sequence_respects_npartitions(seq, npartitions):
    bag = db.from_sequence(seq, npartitions=npartitions)

    expected = npartitions if len(seq) >= npartitions else len(seq)

    assert bag.npartitions == expected, \
        f"from_sequence(seq of len {len(seq)}, npartitions={npartitions}) " \
        f"produced {bag.npartitions} partitions, expected {expected}"

try:
    test_from_sequence_respects_npartitions()
    print("Property test PASSED")
except AssertionError as e:
    print(f"Property test FAILED: {e}")
except Exception as e:
    print(f"Property test ERROR: {e}")

print("\n" + "=" * 60)
print("TEST 2: Manual Test Case from Bug Report")
print("=" * 60)

seq = [1, 2, 3, 4]
requested_npartitions = 3

bag = db.from_sequence(seq, npartitions=requested_npartitions)

print(f"Sequence: {seq}")
print(f"Requested npartitions: {requested_npartitions}")
print(f"Actual npartitions: {bag.npartitions}")

try:
    assert bag.npartitions == requested_npartitions
    print("Manual test PASSED")
except AssertionError:
    print(f"Manual test FAILED: Expected {requested_npartitions} partitions, got {bag.npartitions}")

print("\n" + "=" * 60)
print("TEST 3: Specific Failing Case from Report")
print("=" * 60)

seq = [0, 0, 0, 0, 0]
npartitions = 4

bag = db.from_sequence(seq, npartitions=npartitions)

print(f"Sequence: {seq} (length={len(seq)})")
print(f"Requested npartitions: {npartitions}")
print(f"Actual npartitions: {bag.npartitions}")

try:
    assert bag.npartitions == npartitions
    print("Specific case test PASSED")
except AssertionError:
    print(f"Specific case test FAILED: Expected {npartitions} partitions, got {bag.npartitions}")

print("\n" + "=" * 60)
print("TEST 4: Downstream zip() Problem")
print("=" * 60)

try:
    bag1 = db.from_sequence([0, 0, 0], npartitions=3)
    bag2 = db.from_sequence([0, 0, 0, 0], npartitions=3)

    print(f"bag1 with [0,0,0] and npartitions=3: {bag1.npartitions} partitions")
    print(f"bag2 with [0,0,0,0] and npartitions=3: {bag2.npartitions} partitions")

    zipped = db.zip(bag1, bag2)
    print("ZIP test PASSED")
except AssertionError as e:
    print(f"ZIP test FAILED with AssertionError: {e}")
except Exception as e:
    print(f"ZIP test ERROR: {e}")