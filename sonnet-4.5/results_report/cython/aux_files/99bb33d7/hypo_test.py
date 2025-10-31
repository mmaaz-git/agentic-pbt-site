#!/usr/bin/env python3
"""Property-based test for Cython.Utils.normalise_float_repr"""

from hypothesis import given, strategies as st
from Cython.Utils import normalise_float_repr

@given(st.floats(allow_nan=False, allow_infinity=False))
def test_normalise_float_repr_value_preservation(f):
    """Test that normalise_float_repr preserves the float value."""
    float_str = str(f)
    result = normalise_float_repr(float_str)
    assert float(float_str) == float(result), f"Value not preserved: {float_str} -> {result}"

# Run the test
if __name__ == "__main__":
    test_normalise_float_repr_value_preservation()