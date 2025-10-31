#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from Cython.Utils import normalise_float_repr

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e100, max_value=1e100))
@settings(max_examples=1000)
def test_normalise_float_repr_round_trip(x):
    if x == 0.0:
        return

    float_str = str(x)
    normalized = normalise_float_repr(float_str)

    assert float(normalized) == float(float_str), (
        f"Round-trip failed: {float_str} -> {normalized} "
        f"({float(float_str)} != {float(normalized)})"
    )

if __name__ == "__main__":
    test_normalise_float_repr_round_trip()