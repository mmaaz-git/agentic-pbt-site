#!/usr/bin/env python3
"""Test script to reproduce the reported bug"""

from hypothesis import given, strategies as st, settings
import dask.bag as db
import dask
import traceback

dask.config.set(scheduler='synchronous')

# First, run the hypothesis test
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=0, max_size=10))
@settings(max_examples=50)
def test_variance_no_crash(data):
    b = db.from_sequence(data if data else [0], npartitions=1)
    for ddof in range(0, len(data) + 2):
        try:
            var = b.var(ddof=ddof).compute()
            print(f"  data={data[:3] if len(data) > 3 else data}..., ddof={ddof}, n={len(data)}, result={var}")
        except ZeroDivisionError as e:
            print(f"  FAILED: ZeroDivisionError for data={data[:3] if len(data) > 3 else data}..., ddof={ddof}, n={len(data)}")
            return False
        except Exception as e:
            print(f"  Other error: {type(e).__name__}: {e} for data={data[:3] if len(data) > 3 else data}..., ddof={ddof}, n={len(data)}")
    return True

print("Running hypothesis test...")
try:
    test_variance_no_crash()
    print("Hypothesis test passed")
except Exception as e:
    print(f"Hypothesis test failed: {e}")

# Test specific failing cases mentioned in the bug report
print("\n=== Testing specific failing cases ===")

print("\n1. Testing [1.0, 2.0] with ddof=2:")
try:
    b = db.from_sequence([1.0, 2.0], npartitions=1)
    result = b.var(ddof=2).compute()
    print(f"   Result: {result}")
except ZeroDivisionError as e:
    print(f"   ZeroDivisionError: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"   Other error: {type(e).__name__}: {e}")

print("\n2. Testing empty sequence [] with ddof=0:")
try:
    b = db.from_sequence([], npartitions=1)
    result = b.var(ddof=0).compute()
    print(f"   Result: {result}")
except ZeroDivisionError as e:
    print(f"   ZeroDivisionError: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"   Other error: {type(e).__name__}: {e}")

print("\n3. Testing [1.0, 2.0, 3.0] with ddof=3 (edge case where ddof == n):")
try:
    b = db.from_sequence([1.0, 2.0, 3.0], npartitions=1)
    result = b.var(ddof=3).compute()
    print(f"   Result: {result}")
except ZeroDivisionError as e:
    print(f"   ZeroDivisionError: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"   Other error: {type(e).__name__}: {e}")

print("\n4. Testing [1.0, 2.0, 3.0] with ddof=4 (edge case where ddof > n):")
try:
    b = db.from_sequence([1.0, 2.0, 3.0], npartitions=1)
    result = b.var(ddof=4).compute()
    print(f"   Result: {result}")
except ZeroDivisionError as e:
    print(f"   ZeroDivisionError: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"   Other error: {type(e).__name__}: {e}")

print("\n5. Testing normal case [1.0, 2.0, 3.0] with ddof=1 (should work):")
try:
    b = db.from_sequence([1.0, 2.0, 3.0], npartitions=1)
    result = b.var(ddof=1).compute()
    print(f"   Result: {result}")
except Exception as e:
    print(f"   Error: {type(e).__name__}: {e}")