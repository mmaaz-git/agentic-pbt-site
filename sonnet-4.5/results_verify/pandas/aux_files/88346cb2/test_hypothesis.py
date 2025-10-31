#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings
import pandas.util.version as pv

def test_infinity_equals_itself():
    inf = pv.Infinity
    assert inf == inf
    assert not (inf != inf)
    assert not (inf < inf)
    assert not (inf > inf)
    assert inf <= inf
    assert inf >= inf

if __name__ == "__main__":
    try:
        test_infinity_equals_itself()
        print("All assertions passed!")
    except AssertionError as e:
        print(f"Assertion failed: {e}")
        import traceback
        traceback.print_exc()