#!/usr/bin/env python3
"""Test script to reproduce the dask.bag.Bag.mean empty sequence bug"""

import dask
import dask.bag as db

# Configure dask to use synchronous scheduler
dask.config.set(scheduler='synchronous')

print("Testing dask.bag.Bag.mean with empty sequence...")
print("-" * 50)

# Test 1: Empty bag
print("\nTest 1: Empty bag")
try:
    b = db.from_sequence([], npartitions=1)
    result = b.mean().compute()
    print(f"Result: {result}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError caught: {e}")
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")

# Test 2: Non-empty bag (for comparison)
print("\nTest 2: Non-empty bag [1, 2, 3]")
try:
    b = db.from_sequence([1, 2, 3], npartitions=1)
    result = b.mean().compute()
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

# Test 3: Run the property-based test
print("\nTest 3: Property-based test")
from hypothesis import given, strategies as st

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=0, max_size=100))
def test_mean_no_crash(data):
    b = db.from_sequence(data if data else [0], npartitions=1)
    try:
        mean = b.mean().compute()
        return True
    except ZeroDivisionError:
        assert len(data) > 0, "mean() should not crash with ZeroDivisionError on empty sequence"
        return False

# Test with empty list specifically
print("Testing with empty list []...")
try:
    test_mean_no_crash([])
    print("Test passed")
except AssertionError as e:
    print(f"AssertionError: {e}")
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")

# Test 4: The actual bug reproduction code from the report
print("\n" + "-" * 50)
print("Test 4: Exact reproduction from bug report")
print("-" * 50)

try:
    b = db.from_sequence([], npartitions=1)
    result = b.mean().compute()
    print(f"Result: {result}")
except ZeroDivisionError as e:
    print(f"Confirmed: ZeroDivisionError - {e}")
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")

# Test 5: Check count() behavior with empty bag
print("\nTest 5: count() behavior with empty bag (for comparison)")
try:
    b = db.from_sequence([], npartitions=1)
    count = b.count().compute()
    print(f"count() result: {count}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")