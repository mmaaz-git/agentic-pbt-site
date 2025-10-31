#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from pandas.compat.numpy.function import ARGSORT_DEFAULTS, validate_argsort


@given(st.sampled_from([None, "quicksort", "mergesort", "heapsort", "stable"]))
def test_validate_argsort_kind_consistency(kind_value):
    print(f"Testing with kind_value={kind_value}")
    try:
        if kind_value == "quicksort":
            validate_argsort((), {"kind": kind_value})
            print("  No error for quicksort")
    except ValueError as e:
        print(f"  ValueError: {e}")
        if kind_value == "quicksort":
            raise  # Re-raise to fail the hypothesis test

# Run the test
test_validate_argsort_kind_consistency()