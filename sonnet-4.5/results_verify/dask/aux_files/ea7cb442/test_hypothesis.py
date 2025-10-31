#!/usr/bin/env python3
"""Run the hypothesis-based test from the bug report"""

from hypothesis import given, strategies as st
import dask.bag as db
import dask

dask.config.set(scheduler='synchronous')

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=0, max_size=100))
def test_mean_no_crash(data):
    """The test from the bug report"""
    # Note: The test modifies the input to avoid empty lists by using [0] when empty
    # This actually prevents the bug from being triggered in the test!
    b = db.from_sequence(data if data else [0], npartitions=1)
    try:
        mean = b.mean().compute()
    except ZeroDivisionError:
        # This assertion will fail if we get ZeroDivisionError on non-empty data
        assert len(data) > 0, "mean() should not crash with ZeroDivisionError on empty sequence"

# Run the test
print("Running hypothesis test...")
try:
    test_mean_no_crash()
    print("All hypothesis tests passed!")
except Exception as e:
    print(f"Test failed: {e}")

# Now test what would happen if we actually pass an empty list without the workaround
print("\nNow testing with actual empty list (without the workaround):")
try:
    b = db.from_sequence([], npartitions=1)
    mean = b.mean().compute()
    print(f"Mean of empty sequence: {mean}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {e}")
    print("This confirms the bug exists!")