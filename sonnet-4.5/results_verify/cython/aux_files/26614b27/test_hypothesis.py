#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env')

from hypothesis import given, strategies as st, settings
import math
from Cython.Utils import normalise_float_repr

# Test with the provided property-based test
@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e100, max_value=1e100))
@settings(max_examples=100)
def test_normalise_float_repr_round_trip(f):
    float_str = str(f)
    result = normalise_float_repr(float_str)
    try:
        parsed = float(result)
        assert math.isclose(parsed, f, rel_tol=1e-15), f"Failed for {f}: {result} -> {parsed}"
    except ValueError as e:
        raise AssertionError(f"Failed to parse result for {f}: {result} - {e}")

if __name__ == "__main__":
    print("Running property-based test with Hypothesis...")
    print("Testing normalise_float_repr round-trip property...")
    try:
        test_normalise_float_repr_round_trip()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()