#!/usr/bin/env python3
"""Run the hypothesis test from the bug report."""

from hypothesis import given, strategies as st
import numpy.rec as rec
import traceback

@given(st.lists(st.tuples(st.integers(), st.integers()), min_size=0, max_size=10))
def test_fromrecords_empty_list(records):
    if len(records) == 0:
        r = rec.fromrecords(records, names='x,y')
        assert len(r) == 0
    else:
        r = rec.fromrecords(records, names='x,y')
        assert len(r) == len(records)

print("Running hypothesis test...")
print("="*60)

try:
    test_fromrecords_empty_list()
    print("All hypothesis tests passed!")
except Exception as e:
    print(f"Hypothesis test failed: {e}")
    traceback.print_exc()

print("\nTesting specific failing case: records=[]")
try:
    test_fromrecords_empty_list(records=[])
    print("Test passed for empty list!")
except Exception as e:
    print(f"Test failed for empty list: {e}")
    traceback.print_exc()