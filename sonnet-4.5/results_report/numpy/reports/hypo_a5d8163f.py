#!/usr/bin/env python3
"""Hypothesis test for numpy.f2py.symbolic power operator bug"""

from hypothesis import given, settings, strategies as st
import numpy.f2py.symbolic as sym

simple_expr_chars = st.sampled_from(['x', 'y', 'z', 'a', 'b', 'c'])

@given(st.integers(min_value=1, max_value=10), simple_expr_chars)
@settings(max_examples=200)
def test_power_operator_roundtrip(exp, var):
    expr_str = f'{var}**{exp}'
    e = sym.fromstring(expr_str)
    s = e.tostring()

    assert '**' in s or 'pow' in s.lower() or '^' in s, \
        f"Power operator lost in tostring: {expr_str} -> {s}"

    e2 = sym.fromstring(s)
    assert e == e2, f"Power round-trip failed: {expr_str} -> {s} -> {e2.tostring()}"

if __name__ == "__main__":
    test_power_operator_roundtrip()